"""Training and feature extraction for the player matchup neural net.

Training
========
Trains on historical game outcomes (regular season + tournament). For each
game, looks up both teams' rotation players from BartTorvik season-level stats
and learns to predict P(Team A wins) from the individual player features.

Two loss signals:
  1. BCE loss on win/loss (primary)
  2. MSE loss on normalized score margin (auxiliary — richer gradient)

Feature Extraction
==================
After training, generates two types of features:
  1. Team embeddings (team-level): 64-dim vectors from self-attention encoder.
     Added to team features → pipeline creates _A, _B, _delta automatically.
  2. Matchup prediction (matchup-level): P(Team A wins) from full model.
     Added directly to matchup rows.

Usage:
    from models.player_train import train_player_model, PlayerNNExtractor

    # Train (saves checkpoint to data/external/player_impact/player_nn.pt)
    train_player_model(data_dir, max_train_season=2024)

    # Extract features
    extractor = PlayerNNExtractor(data_dir)
    team_features = extractor.team_embeddings(season=2026)
    matchup_preds = extractor.matchup_predictions(team_a_ids, team_b_ids, season)
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.player_model import (
    PlayerMatchupModel, PLAYER_FEATURES, MAX_PLAYERS, N_PLAYER_FEATURES,
)


CHECKPOINT_DIR = "player_impact"  # relative to data/external/
CHECKPOINT_NAME = "player_nn.pt"
NORM_STATS_NAME = "player_nn_norm.json"
MODEL_CONFIG_NAME = "player_nn_config.json"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _build_name_to_id(data_dir: Path) -> dict:
    name_to_id = {}
    spellings = pd.read_csv(data_dir / "MTeamSpellings.csv", encoding="latin-1")
    for _, row in spellings.iterrows():
        name_to_id[str(row["TeamNameSpelling"]).lower().strip()] = row["TeamID"]
    teams = pd.read_csv(data_dir / "MTeams.csv")
    for _, row in teams.iterrows():
        name_to_id[row["TeamName"].lower().strip()] = row["TeamID"]
    return name_to_id


NAME_OVERRIDES = {"Queens": "Queens NC"}


def _load_player_data(data_dir: Path):
    """Load all player data from player_impact cache."""
    ext_dir = data_dir / "external" / CHECKPOINT_DIR
    frames = []
    for f in sorted(ext_dir.glob("players_full_*.csv")):
        frames.append(pd.read_csv(f))
    if not frames:
        raise FileNotFoundError(f"No player data in {ext_dir}. Run player_impact fetch first.")
    players = pd.concat(frames, ignore_index=True)

    name_to_id = _build_name_to_id(data_dir)
    players["_name"] = players["team"].apply(
        lambda x: NAME_OVERRIDES.get(str(x).strip(), str(x).strip())
    )
    players["TeamID"] = players["_name"].str.lower().str.strip().map(name_to_id)
    players = players.dropna(subset=["TeamID", "min_pct"])
    players["TeamID"] = players["TeamID"].astype(int)
    return players


def _compute_norm_stats(players: pd.DataFrame) -> dict:
    """Compute global mean/std for each player feature."""
    stats = {}
    for feat in PLAYER_FEATURES:
        vals = players[feat].dropna()
        stats[feat] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return stats


def _build_roster_lookup(players: pd.DataFrame, norm_stats: dict):
    """Build (TeamID, Season) → (features_tensor, mask_tensor) lookup.

    For each team-season, selects top MAX_PLAYERS by minutes and creates
    a normalized feature tensor. Pads with zeros if fewer players available.
    """
    lookup = {}

    for (tid, year), grp in players.groupby(["TeamID", "year"]):
        top = grp.nlargest(MAX_PLAYERS, "min_pct")

        feats = np.zeros((MAX_PLAYERS, N_PLAYER_FEATURES), dtype=np.float32)
        mask = np.ones(MAX_PLAYERS, dtype=bool)  # True = padded

        for i, (_, row) in enumerate(top.iterrows()):
            if i >= MAX_PLAYERS:
                break
            for j, feat_name in enumerate(PLAYER_FEATURES):
                val = row.get(feat_name, np.nan)
                if pd.notna(val):
                    s = norm_stats[feat_name]
                    std = s["std"] if s["std"] > 0 else 1.0
                    feats[i, j] = (float(val) - s["mean"]) / std
            mask[i] = False

        lookup[(int(tid), int(year))] = (
            torch.tensor(feats),
            torch.tensor(mask),
        )

    return lookup


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GameDataset(Dataset):
    """PyTorch dataset of game matchups with score margins and augmentation.

    Augmentation (training only):
      - Random player dropout: mask 0-2 extra players per team, simulating
        injuries/rotation changes. Forces the model to learn robust team
        representations instead of memorizing exact lineups.
      - Feature noise: add gaussian noise to player features, regularizing
        against overfitting to exact stat lines.
    """

    def __init__(self, games_df, roster_lookup, augment=False,
                 player_drop_prob=0.15, feature_noise_std=0.1):
        self.samples = []
        self.augment = augment
        self.player_drop_prob = player_drop_prob
        self.feature_noise_std = feature_noise_std

        for _, game in games_df.iterrows():
            season = int(game["Season"])
            w_key = (int(game["WTeamID"]), season)
            l_key = (int(game["LTeamID"]), season)

            if w_key not in roster_lookup or l_key not in roster_lookup:
                continue

            w_feat, w_mask = roster_lookup[w_key]
            l_feat, l_mask = roster_lookup[l_key]

            # Score margin (normalized by typical game spread ~12 pts)
            w_score = game["WScore"] if "WScore" in game.index else np.nan
            l_score = game["LScore"] if "LScore" in game.index else np.nan
            if pd.notna(w_score) and pd.notna(l_score):
                margin = (w_score - l_score) / 12.0
            else:
                margin = 0.0  # unknown margin — neutral signal

            # Winner perspective: label=1, margin=+
            self.samples.append((w_feat, l_feat, w_mask, l_mask, 1.0, margin))
            # Loser perspective: label=0, margin=-
            self.samples.append((l_feat, w_feat, l_mask, w_mask, 0.0, -margin))

    def __len__(self):
        return len(self.samples)

    def _augment_team(self, feat, mask):
        """Apply random player dropout and feature noise."""
        feat = feat.clone()
        mask = mask.clone()

        # Random player dropout: mask out non-padded players with some prob
        # but always keep at least 3 players
        if self.player_drop_prob > 0:
            n_real = (~mask).sum().item()
            if n_real > 3:
                drop = torch.rand(mask.shape) < self.player_drop_prob
                # Only drop non-padded players
                drop = drop & ~mask
                # Ensure we keep at least 3
                if (~mask & ~drop).sum() < 3:
                    drop = torch.zeros_like(mask)
                mask = mask | drop
                feat[drop] = 0.0

        # Feature noise
        if self.feature_noise_std > 0:
            noise = torch.randn_like(feat) * self.feature_noise_std
            noise[mask] = 0.0  # no noise on padded slots
            feat = feat + noise

        return feat, mask

    def __getitem__(self, idx):
        a_feat, b_feat, a_mask, b_mask, label, margin = self.samples[idx]

        if self.augment:
            a_feat, a_mask = self._augment_team(a_feat, a_mask)
            b_feat, b_mask = self._augment_team(b_feat, b_mask)

        return (
            a_feat, b_feat, a_mask, b_mask,
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(margin, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_player_model(
    data_dir: Path,
    max_train_season: int = 2024,
    val_season: int = 2025,
    epochs: int = 500,
    batch_size: int = 512,
    lr: float = 3e-4,
    patience: int = 75,
    device: str = None,
    time_limit: int = None,
    label_smoothing: float = 0.05,
    margin_weight: float = 0.3,
):
    """Train the player matchup model and save checkpoint.

    Args:
        data_dir: Path to data directory.
        max_train_season: Train on games from seasons <= this.
        val_season: Validate on this season (for early stopping).
        epochs: Maximum training epochs.
        batch_size: Training batch size.
        lr: Peak learning rate (after warmup).
        patience: Early stopping patience (epochs without improvement).
        device: 'cuda' or 'cpu'. Auto-detected if None.
        time_limit: Max training time in seconds. If set, overrides epochs.
        label_smoothing: Smooth binary labels towards 0.5 by this amount.
        margin_weight: Weight of margin MSE loss relative to BCE loss.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training player matchup model on {device}")
    print(f"  Train seasons: ≤ {max_train_season}, Val season: {val_season}")
    if time_limit:
        print(f"  Time limit: {time_limit}s ({time_limit/3600:.1f}h)")

    # Load data
    players = _load_player_data(data_dir)
    norm_stats = _compute_norm_stats(players)
    roster_lookup = _build_roster_lookup(players, norm_stats)
    print(f"  {len(roster_lookup)} team-seasons in roster lookup")

    # Load game results
    games = []
    for csv_name in [
        "MNCAATourneyCompactResults.csv",
        "MConferenceTourneyGames.csv",
        "MRegularSeasonCompactResults.csv",
    ]:
        path = data_dir / csv_name
        if path.exists():
            games.append(pd.read_csv(path))
    all_games = pd.concat(games, ignore_index=True)

    train_games = all_games[all_games["Season"] <= max_train_season]
    val_games = all_games[all_games["Season"] == val_season]
    print(f"  Train games: {len(train_games)}, Val games: {len(val_games)}")

    train_ds = GameDataset(train_games, roster_lookup, augment=True,
                           player_drop_prob=0.15, feature_noise_std=0.1)
    val_ds = GameDataset(val_games, roster_lookup, augment=False)
    print(f"  Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=(device == "cuda"))
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device == "cuda"))

    # Model
    model = PlayerMatchupModel(dropout=0.25).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-3)

    # Cosine annealing with warm restarts — multiple cycles let the model
    # escape local minima and train productively for longer
    restart_period = 50  # epochs per cosine cycle
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        # Cosine with restarts
        cycle_epoch = (epoch - warmup_epochs) % restart_period
        return 0.5 * (1 + np.cos(np.pi * cycle_epoch / restart_period))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    margin_loss_fn = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")
    print(f"  Label smoothing: {label_smoothing}, Margin weight: {margin_weight}")
    print(f"  Augmentation: player_drop=0.15, feature_noise=0.1")
    print(f"  Cosine restarts every {restart_period} epochs")

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    start_time = time.time()

    for epoch in range(epochs):
        # Time limit check
        if time_limit and (time.time() - start_time) >= time_limit:
            elapsed = time.time() - start_time
            print(f"  Time limit reached at epoch {epoch} ({elapsed:.0f}s)")
            break

        # Train
        model.train()
        train_bce = 0.0
        train_margin = 0.0
        n_train = 0
        for a_feat, b_feat, a_mask, b_mask, labels, margins in train_dl:
            a_feat = a_feat.to(device)
            b_feat = b_feat.to(device)
            a_mask = a_mask.to(device)
            b_mask = b_mask.to(device)
            labels = labels.to(device)
            margins = margins.to(device)

            # Label smoothing
            smooth_labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing

            logits, pred_margin, _, _, _ = model(a_feat, b_feat, a_mask, b_mask)
            loss_bce = bce_loss_fn(logits, smooth_labels)
            loss_margin = margin_loss_fn(pred_margin, margins)
            loss = loss_bce + margin_weight * loss_margin

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_bce += loss_bce.item() * len(labels)
            train_margin += loss_margin.item() * len(labels)
            n_train += len(labels)

        scheduler.step()
        train_bce /= max(n_train, 1)
        train_margin /= max(n_train, 1)

        # Validate
        model.eval()
        val_bce = 0.0
        val_margin = 0.0
        val_correct = 0
        n_val = 0
        with torch.no_grad():
            for a_feat, b_feat, a_mask, b_mask, labels, margins in val_dl:
                a_feat = a_feat.to(device)
                b_feat = b_feat.to(device)
                a_mask = a_mask.to(device)
                b_mask = b_mask.to(device)
                labels = labels.to(device)
                margins = margins.to(device)

                logits, pred_margin, _, _, _ = model(a_feat, b_feat, a_mask, b_mask)
                val_bce += bce_loss_fn(logits, labels).item() * len(labels)
                val_margin += margin_loss_fn(pred_margin, margins).item() * len(labels)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                n_val += len(labels)

        val_bce /= max(n_val, 1)
        val_margin /= max(n_val, 1)
        val_loss = val_bce + margin_weight * val_margin
        val_acc = val_correct / max(n_val, 1)

        elapsed = time.time() - start_time
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d} ({elapsed:.0f}s): "
                  f"train_bce={train_bce:.4f} margin={train_margin:.4f}  "
                  f"val_bce={val_bce:.4f} margin={val_margin:.4f}  "
                  f"val_acc={val_acc:.3f}  lr={scheduler.get_last_lr()[0]:.6f}")

        # Early stopping
        if val_bce < best_val_loss:
            best_val_loss = val_bce
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best val_bce={best_val_loss:.4f})")
                break

    # Save
    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    save_dir = data_dir / "external" / CHECKPOINT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(best_state, save_dir / CHECKPOINT_NAME)
    with open(save_dir / NORM_STATS_NAME, "w") as f:
        json.dump(norm_stats, f)

    # Save model config so we can reconstruct with the right hyperparams
    model_config = {
        "embed_dim": model.embed_dim,
        "n_layers": model.n_layers,
        "n_params": n_params,
        "best_val_bce": best_val_loss,
        "epochs_trained": epoch + 1,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "time_seconds": time.time() - start_time,
    }
    with open(save_dir / MODEL_CONFIG_NAME, "w") as f:
        json.dump(model_config, f, indent=2)

    print(f"  Saved checkpoint to {save_dir / CHECKPOINT_NAME}")
    print(f"  Best val_bce: {best_val_loss:.4f}")
    total_time = time.time() - start_time
    print(f"  Total training time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    return model


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class PlayerNNExtractor:
    """Extract features from a trained player matchup model."""

    def __init__(self, data_dir: Path, device: str = None):
        self.data_dir = data_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        ckpt_dir = data_dir / "external" / CHECKPOINT_DIR
        ckpt_path = ckpt_dir / CHECKPOINT_NAME
        norm_path = ckpt_dir / NORM_STATS_NAME

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"No trained model at {ckpt_path}. Run train_player_model() first."
            )

        # Load model
        self.model = PlayerMatchupModel().to(self.device)
        state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

        # Load normalization stats
        with open(norm_path) as f:
            self.norm_stats = json.load(f)

        # Build roster lookup
        players = _load_player_data(data_dir)
        self.roster_lookup = _build_roster_lookup(players, self.norm_stats)

    def _get_roster(self, team_id: int, season: int):
        """Get (features, mask) for a team-season. Returns zeros if missing."""
        key = (team_id, season)
        if key in self.roster_lookup:
            return self.roster_lookup[key]
        return (
            torch.zeros(MAX_PLAYERS, N_PLAYER_FEATURES),
            torch.ones(MAX_PLAYERS, dtype=torch.bool),
        )

    def team_embeddings(self, season: int) -> pd.DataFrame:
        """Generate team-level embeddings for all teams in a season.

        Returns DataFrame with columns: Season, TeamID, pnn_emb_0..pnn_emb_{d-1}.
        These are self-attention-only team embeddings (no opponent needed).
        """
        # Find all teams in this season
        team_ids = [
            tid for (tid, yr) in self.roster_lookup if yr == season
        ]

        if not team_ids:
            return pd.DataFrame(columns=["Season", "TeamID"])

        # Batch encode all teams (using self-attention only, no cross-attention)
        feats_list = []
        masks_list = []
        for tid in team_ids:
            f, m = self._get_roster(tid, season)
            feats_list.append(f)
            masks_list.append(m)

        feats_batch = torch.stack(feats_list).to(self.device)
        masks_batch = torch.stack(masks_list).to(self.device)

        with torch.no_grad():
            team_embs, _ = self.model.encode_team(feats_batch, masks_batch)
            team_embs = team_embs.cpu().numpy()

        rows = []
        for i, tid in enumerate(team_ids):
            row = {"Season": season, "TeamID": tid}
            for j in range(team_embs.shape[1]):
                row[f"pnn_emb_{j}"] = team_embs[i, j]
            rows.append(row)

        return pd.DataFrame(rows)

    def matchup_predictions(
        self,
        team_a_ids: np.ndarray,
        team_b_ids: np.ndarray,
        seasons: np.ndarray,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """Generate matchup predictions for given team pairs.

        Runs the full model (with cross-attention) to produce P(Team A wins).

        Args:
            team_a_ids: Array of Team A IDs.
            team_b_ids: Array of Team B IDs.
            seasons: Array of seasons.
            batch_size: Inference batch size.

        Returns:
            Array of predicted probabilities, same length as inputs.
        """
        n = len(team_a_ids)
        preds = np.full(n, np.nan, dtype=np.float32)

        # Track which matchups have valid rosters for both teams
        valid_indices = []
        a_feats_all, a_masks_all, b_feats_all, b_masks_all = [], [], [], []

        for i in range(n):
            a_key = (int(team_a_ids[i]), int(seasons[i]))
            b_key = (int(team_b_ids[i]), int(seasons[i]))

            if a_key not in self.roster_lookup or b_key not in self.roster_lookup:
                continue

            af, am = self.roster_lookup[a_key]
            bf, bm = self.roster_lookup[b_key]
            valid_indices.append(i)
            a_feats_all.append(af)
            a_masks_all.append(am)
            b_feats_all.append(bf)
            b_masks_all.append(bm)

        if not valid_indices:
            preds[:] = 0.5
            return preds

        # Batch inference on valid matchups only
        valid_indices = np.array(valid_indices)
        for start in range(0, len(valid_indices), batch_size):
            end = min(start + batch_size, len(valid_indices))

            a_batch = torch.stack(a_feats_all[start:end]).to(self.device)
            b_batch = torch.stack(b_feats_all[start:end]).to(self.device)
            am_batch = torch.stack(a_masks_all[start:end]).to(self.device)
            bm_batch = torch.stack(b_masks_all[start:end]).to(self.device)

            with torch.no_grad():
                logits, _, _, _, _ = self.model(a_batch, b_batch, am_batch, bm_batch)
                batch_preds = torch.sigmoid(logits).cpu().numpy()

            preds[valid_indices[start:end]] = batch_preds

        # Replace any remaining NaN (missing rosters) with 0.5
        preds = np.where(np.isfinite(preds), preds, 0.5)

        return preds


# ---------------------------------------------------------------------------
# Standalone submission generation (no AutoGluon)
# ---------------------------------------------------------------------------

def generate_pnn_submission(
    data_dir: Path,
    season: int,
    output_path: Path,
    device: str = None,
):
    """Generate a Kaggle submission using only the player NN predictions."""
    extractor = PlayerNNExtractor(data_dir, device=device)

    sample_sub = pd.read_csv(data_dir / "SampleSubmissionStage2.csv")
    ids = sample_sub["ID"].str.split("_", expand=True).astype(int)
    sample_sub["Season"] = ids[0]
    sample_sub["TeamA"] = ids[1]
    sample_sub["TeamB"] = ids[2]

    # Filter to requested season and men's (TeamID < 3000)
    mask = (sample_sub["Season"] == season) & (sample_sub["TeamA"] < 3000)
    men = sample_sub[mask].copy()

    if len(men) == 0:
        print(f"  No men's matchups found for season {season}")
        return

    preds = extractor.matchup_predictions(
        men["TeamA"].values, men["TeamB"].values,
        men["Season"].values,
    )
    men["Pred"] = preds

    # For women's games, use seed-based fallback
    women_mask = (sample_sub["Season"] == season) & (sample_sub["TeamA"] >= 3000)
    women = sample_sub[women_mask].copy()

    # Load seeds for women's fallback
    seeds = pd.read_csv(data_dir / "WNCAATourneySeeds.csv")
    seeds["seed_num"] = seeds["Seed"].str.extract(r"(\d+)").astype(int)
    seed_map = dict(zip(
        zip(seeds["Season"], seeds["TeamID"]),
        seeds["seed_num"]
    ))

    women_preds = []
    for _, row in women.iterrows():
        sa = seed_map.get((row["Season"], row["TeamA"]), 8)
        sb = seed_map.get((row["Season"], row["TeamB"]), 8)
        # Simple seed-based probability
        diff = sb - sa
        p = 1 / (1 + 10 ** (-diff / 5))
        women_preds.append(np.clip(p, 0.05, 0.95))
    women["Pred"] = women_preds

    result = pd.concat([men[["ID", "Pred"]], women[["ID", "Pred"]]])
    # Merge with full sample to keep any seasons we didn't predict
    final = sample_sub[["ID"]].merge(result, on="ID", how="left")
    final["Pred"] = final["Pred"].fillna(0.5)

    final.to_csv(output_path, index=False)
    n_valid = np.isfinite(preds).sum()
    print(f"  Submission saved to {output_path}")
    print(f"  Men's games: {len(men)} ({n_valid} with player data)")
    print(f"  Women's games: {len(women)} (seed-based fallback)")
    print(f"  Pred range: [{preds.min():.3f}, {preds.max():.3f}], mean={preds.mean():.3f}")
