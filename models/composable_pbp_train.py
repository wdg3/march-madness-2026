"""Composable PBP model — data loading, training, and prediction.

Builds per-player play indices from PBP data, trains the composable model
where teams are represented as compositions of individual players.

Usage:
    python run.py cpbp train [--time-limit 14400]
    python run.py cpbp submit --tag cpbp_v1
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models.pbp_model import PBPMatchupModel, PlayEncoder, N_PLAY_CONTEXT
from models.pbp_train import (
    load_pbp_data, split_games, _pad_plays, _ckpt_dir, _load_checkpoint,
    N_PLAY_TYPES, PLAY_TYPES, PTYPE_TO_IDX, MAX_PLAYS,
)
from models.composable_pbp_model import (
    ComposablePBPModel, MAX_PLAYERS, MAX_PLAYS_PER_PLAYER,
)


# =====================================================================
#  PER-PLAYER PLAY INDEX
# =====================================================================

def build_player_game_index(games, play_tensors):
    """Build an index: which players appeared in which games.

    For each game, extracts the set of CBBD player IDs that were on court
    for each team. This uses the player IDs stored in the play tensors
    (the `our` and `their` arrays).

    Returns:
        game_rosters: dict[game_id → {team → set of cbbd_player_idx}]
            Player indices (from player_to_idx), not raw CBBD IDs.
        player_games: dict[(player_idx, team, season) → [game_ids by date]]
            Which games each player appeared in, chronologically.
    """
    game_rosters: dict[int, dict[str, set[int]]] = {}
    player_games: dict[tuple, list[int]] = defaultdict(list)

    for (gid, team), pt in play_tensors.items():
        if gid not in games:
            continue
        g = games[gid]
        season = g["season"]

        # Extract unique player indices from the play tensor
        # `our` has shape (n_plays, 5) — our 5 players per play
        our_players = set(pt["our"].flatten().tolist()) - {0}  # 0 = padding

        if gid not in game_rosters:
            game_rosters[gid] = {}
        game_rosters[gid][team] = our_players

        for pid in our_players:
            player_games[(pid, team, season)].append(gid)

    # Sort each player's games by date
    for key in player_games:
        player_games[key] = sorted(
            player_games[key], key=lambda g: games[g]["date"]
        )

    return game_rosters, player_games


def get_player_play_indices(play_tensor: dict, player_idx: int) -> np.ndarray:
    """Get indices of plays where a specific player was on court."""
    our = play_tensor["our"]  # (n_plays, 5)
    mask = (our == player_idx).any(axis=1)
    return np.where(mask)[0]


# =====================================================================
#  PLAY ENCODING + PER-PLAYER PRECOMPUTATION
# =====================================================================

@torch.no_grad()
def encode_all_plays(play_encoder, play_tensors, device, batch_size=64):
    """Encode all plays through the PlayEncoder → cached embeddings.

    Returns:
        play_embs: dict[(game_id, team) → tensor of shape (n_plays, play_dim)]
    """
    play_encoder.eval()
    keys = list(play_tensors.keys())
    play_embs = {}

    for i in range(0, len(keys), batch_size):
        chunk = keys[i:i + batch_size]
        dicts = [play_tensors[k] for k in chunk]
        our, their, pt, ctx, mask = _pad_plays(dicts, device=device)

        embs = play_encoder(our, their, pt, ctx)  # (B, T, D)

        for j, k in enumerate(chunk):
            n = dicts[j]["n_plays"]
            play_embs[k] = embs[j, :n].cpu()

    play_encoder.train()
    return play_embs


def precompute_player_season_embs(
    play_embs, play_tensors, games, player_games,
):
    """Precompute per-player season play embeddings with game boundaries.

    For each (player_idx, team, season), concatenates play embeddings from
    all games (in date order) where the player was on court. Stores
    cumulative game boundary indices so that "plays before game N" is a
    simple slice: player_embs[:boundaries[N]].

    No trimming here — __getitem__ subsamples to MAX_PLAYS_PER_PLAYER
    after slicing to the causal boundary.

    Returns:
        player_season_embs: dict[(pid, team, season) → torch.Tensor (total_plays, D)]
        player_game_bounds: dict[(pid, team, season) → list[int]]
            Cumulative play counts after each game. boundaries[i] = total plays
            through game i. Plays before game i = embs[:boundaries[i-1]] (or
            embs[:0] = empty for the first game).
    """
    player_season_embs = {}
    player_game_bounds = {}

    for (pid, team, season), gids in player_games.items():
        emb_chunks = []
        bounds = []
        total = 0

        for gid in gids:  # already sorted by date
            key = (gid, team)
            if key not in play_embs or key not in play_tensors:
                bounds.append(total)
                continue
            ge = play_embs[key]
            pt = play_tensors[key]
            indices = get_player_play_indices(pt, pid)
            if len(indices) > 0:
                emb_chunks.append(ge[indices])
                total += len(indices)
            bounds.append(total)

        if total == 0:
            continue

        player_season_embs[(pid, team, season)] = torch.cat(emb_chunks, dim=0)
        player_game_bounds[(pid, team, season)] = bounds

    return player_season_embs, player_game_bounds


# =====================================================================
#  DATASET
# =====================================================================

class ComposableMatchupDataset(Dataset):
    """Dataset of game matchups for the composable PBP model.

    Uses precomputed per-player season embeddings with game boundaries
    for fast causal slicing. __getitem__ is pure tensor indexing — no
    Python loops over plays.

    Training augmentation: random player dropout (mask 0-2 players per team).
    """

    def __init__(
        self,
        game_ids: list[int],
        games: dict,
        game_rosters: dict,
        player_games: dict,
        player_season_embs: dict,
        player_game_bounds: dict,
        max_players: int = MAX_PLAYERS,
        player_drop_prob: float = 0.0,
        min_players: int = 5,
    ):
        self.games = games
        self.game_rosters = game_rosters
        self.player_games = player_games
        self.pse = player_season_embs
        self.pgb = player_game_bounds
        self.max_p = max_players
        self.player_drop_prob = player_drop_prob
        self.min_players = min_players
        self.training = False

        # Pre-build: for each game, map each player to their game index
        # in the player_games list (for boundary lookup)
        self._player_game_idx: dict[tuple, dict[int, int]] = {}
        for key, gids in player_games.items():
            gid_to_idx = {g: i for i, g in enumerate(gids)}
            self._player_game_idx[key] = gid_to_idx

        # Filter to games where both teams have enough players with prior data
        self.examples = []
        skipped = 0
        for gid in game_ids:
            g = games[gid]
            ht, at = g["home_team"], g["away_team"]
            r = game_rosters.get(gid, {})
            if ht not in r or at not in r:
                skipped += 1
                continue

            ok = True
            for team in (ht, at):
                n_valid = 0
                for pid in r[team]:
                    key = (pid, team, g["season"])
                    gi = self._player_game_idx.get(key, {}).get(gid)
                    # gi > 0 means this isn't their first game (they have prior plays)
                    if gi is not None and gi > 0:
                        n_valid += 1
                if n_valid < min_players:
                    ok = False
                    break

            if ok:
                self.examples.append((
                    gid, ht, at,
                    g["home_win"],
                    (g["home_score"] - g["away_score"]) / 20.0,
                ))
            else:
                skipped += 1

        if skipped > 0:
            print(f"    Skipped {skipped} games (insufficient prior play data)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        gid, ht, at, label, margin = self.examples[idx]
        g = self.games[gid]
        season = g["season"]

        result = {}
        for side, team in [("a", ht), ("b", at)]:
            roster = sorted(self.game_rosters[gid][team])

            # Training augmentation: drop random players
            if self.training and self.player_drop_prob > 0 and len(roster) > self.min_players:
                roster = [p for p in roster if random.random() > self.player_drop_prob]
                if len(roster) < self.min_players:
                    roster = sorted(self.game_rosters[gid][team])[:self.min_players]

            # Cap to max_players by prior play count
            if len(roster) > self.max_p:
                roster = sorted(
                    roster,
                    key=lambda p: self.pgb.get(
                        (p, team, season), [0])[-1] if (p, team, season) in self.pgb else 0,
                    reverse=True,
                )[:self.max_p]

            # Slice precomputed embeddings (pure tensor ops)
            player_plays = []
            for pid in roster:
                key = (pid, team, season)
                if key not in self.pse:
                    continue
                gi = self._player_game_idx.get(key, {}).get(gid)
                if gi is None or gi == 0:
                    continue
                bound = self.pgb[key][gi - 1]
                if bound <= 0:
                    continue
                player_plays.append(self.pse[key][:bound])

            result[f"plays_{side}"] = player_plays

        result["label"] = label
        result["margin"] = margin
        return result


def collate_composable(batch):
    """Collate variable-length rosters and play sequences into padded tensors."""
    B = len(batch)
    D = batch[0]["plays_a"][0].shape[-1] if batch[0]["plays_a"] else 64

    tensors = {}
    for side in ["a", "b"]:
        plays_key = f"plays_{side}"
        roster_key = f"roster_{side}"

        max_p = max(len(b[plays_key]) for b in batch)
        max_p = max(max_p, 1)  # at least 1
        max_t = max(
            (emb.shape[0] for b in batch for emb in b[plays_key]),
            default=1,
        )

        plays = torch.zeros(B, max_p, max_t, D)
        play_mask = torch.ones(B, max_p, max_t, dtype=torch.bool)
        roster_mask = torch.ones(B, max_p, dtype=torch.bool)

        for i, b in enumerate(batch):
            for j, emb in enumerate(b[plays_key]):
                n = emb.shape[0]
                plays[i, j, :n] = emb
                play_mask[i, j, :n] = False
                roster_mask[i, j] = False

        tensors[f"plays_{side}"] = plays
        tensors[f"play_mask_{side}"] = play_mask
        tensors[f"roster_mask_{side}"] = roster_mask

    labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
    margins = torch.tensor([b["margin"] for b in batch], dtype=torch.float32)

    return tensors, labels, margins


# =====================================================================
#  CHECKPOINT I/O
# =====================================================================

CPBP_CKPT_DIR = "cpbp_model"

def _cpbp_ckpt_dir(data_dir, version=None):
    if version:
        d = data_dir / "external" / CPBP_CKPT_DIR / version
    else:
        d = data_dir / "external" / CPBP_CKPT_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_cpbp_checkpoint(model, data_dir, tag, p2i,
                          optimizer=None, scheduler=None,
                          epoch=None, best_val=None, patience=None,
                          version=None):
    d = _cpbp_ckpt_dir(data_dir, version)
    torch.save(model.state_dict(), d / f"cpbp_{tag}.pt")
    with open(d / "player_index.json", "w") as f:
        json.dump({str(k): v for k, v in p2i.items()}, f)
    cfg = dict(
        play_dim=model.play_dim,
        d_player=model.d_player,
    )
    with open(d / "config.json", "w") as f:
        json.dump(cfg, f)
    if optimizer is not None:
        train_state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "best_val": best_val,
            "patience": patience,
        }
        torch.save(train_state, d / f"cpbp_{tag}_train_state.pt")


# =====================================================================
#  MAIN TRAINING FUNCTION
# =====================================================================

def train_composable_pbp(
    data_dir: Path,
    time_limit: int = 14400,
    play_dim: int = 64,
    d_player: int = 64,
    n_heads: int = 4,
    n_self_layers: int = 2,
    n_cross_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    margin_weight: float = 0.3,
    label_smoothing: float = 0.05,
    player_drop_prob: float = 0.15,
    max_epochs: int = 100,
    batch_size: int = 8,
    accum_steps: int = 4,
    device: str | None = None,
    version: str | None = None,
):
    t0 = time.time()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Training Composable PBP model on {device}"
          f"{f' (version: {version})' if version else ''}")
    print(f"  Time limit: {time_limit}s ({time_limit/3600:.1f}h)")
    print(f"  Effective batch: {batch_size} x {accum_steps} = {batch_size * accum_steps}")

    # ── Load PBP data ─────────────────────────────────────────────
    print("\n[1/4] Loading PBP data...")
    games, p2i, ts_games, ptensors = load_pbp_data(data_dir, seasons=[2024, 2025, 2026])
    train_ids, val_ids = split_games(games)
    print(f"  Train games: {len(train_ids)}, Val games: {len(val_ids)}")

    # ── Build per-player index ────────────────────────────────────
    print("\n[2/4] Building per-player play index...")
    game_rosters, player_games = build_player_game_index(games, ptensors)
    n_players_total = len({pid for (pid, _, _) in player_games})
    print(f"  {len(game_rosters)} games with rosters, {n_players_total} unique players")

    # ── Build model with warm-started PlayEncoder ─────────────────
    print("\n[3/4] Building model...")
    n_cbbd_players = len(p2i)
    play_encoder = PlayEncoder(
        n_cbbd_players, player_dim=8, n_play_types=N_PLAY_TYPES,
        ptype_dim=8, out_dim=play_dim, dropout=dropout,
    )

    # Warm-start from existing PBP model checkpoint
    pbp_ckpt = _ckpt_dir(data_dir) / "pbp_best.pt"
    if pbp_ckpt.exists():
        print(f"  Warm-starting PlayEncoder from {pbp_ckpt}")
        state = torch.load(pbp_ckpt, map_location="cpu", weights_only=True)
        pe_state = {
            k.replace("play_encoder.", ""): v
            for k, v in state.items() if k.startswith("play_encoder.")
        }
        play_encoder.load_state_dict(pe_state, strict=False)
    else:
        print("  No PBP checkpoint found — training PlayEncoder from scratch")

    model = ComposablePBPModel(
        play_encoder=play_encoder,
        play_dim=play_dim,
        d_player=d_player,
        n_heads=n_heads,
        n_self_layers=n_self_layers,
        n_cross_layers=n_cross_layers,
        dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} parameters")

    # ── Encode all plays (cached, no gradients) ───────────────────
    print("\n  Encoding all plays through PlayEncoder...")
    play_embs = encode_all_plays(model.play_encoder, ptensors, device)
    print(f"  {len(play_embs)} game-team play tensors encoded")

    # ── Precompute per-player season embeddings ─────────────────
    print("\n  Precomputing per-player season play embeddings...")
    pse, pgb = precompute_player_season_embs(
        play_embs, ptensors, games, player_games,
    )
    print(f"  {len(pse)} player-team-seasons precomputed")

    # ── Build datasets ────────────────────────────────────────────
    print("\n[4/4] Training...")
    train_ds = ComposableMatchupDataset(
        train_ids, games, game_rosters, player_games, pse, pgb,
        player_drop_prob=player_drop_prob,
    )
    val_ds = ComposableMatchupDataset(
        val_ids, games, game_rosters, player_games, pse, pgb,
    )
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_composable, num_workers=0,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_composable, num_workers=0,
    )

    # ── Optimizer ─────────────────────────────────────────────────
    # Play encoder at lower LR since it's warm-started
    pe_params = list(model.play_encoder.parameters())
    pe_ids = set(id(p) for p in pe_params)
    other_params = [p for p in model.parameters() if id(p) not in pe_ids]

    optimizer = torch.optim.AdamW([
        {"params": pe_params, "lr": lr * 0.3},
        {"params": other_params, "lr": lr},
    ], weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_val = float("inf")
    patience, patience_limit = 0, 15

    for epoch in range(max_epochs):
        if time.time() - t0 > time_limit * 0.85:
            print(f"  Time budget reached at epoch {epoch}")
            break

        # ── Train ─────────────────────────────────────────────────
        model.train()
        train_ds.training = True
        ep_loss, ep_n = 0.0, 0
        optimizer.zero_grad()

        for step, (tensors, labels, margins) in enumerate(train_dl):
            # Move to device
            plays_a = tensors["plays_a"].to(device)
            play_mask_a = tensors["play_mask_a"].to(device)
            roster_mask_a = tensors["roster_mask_a"].to(device)
            plays_b = tensors["plays_b"].to(device)
            play_mask_b = tensors["play_mask_b"].to(device)
            roster_mask_b = tensors["roster_mask_b"].to(device)
            labels = labels.to(device)
            margins = margins.to(device)

            logit, margin_pred = model(
                plays_a, play_mask_a, roster_mask_a,
                plays_b, play_mask_b, roster_mask_b,
            )

            tgt = labels * (1 - label_smoothing) + 0.5 * label_smoothing
            bce = F.binary_cross_entropy_with_logits(logit, tgt)
            m_loss = F.mse_loss(margin_pred, margins)
            loss = (bce + margin_weight * m_loss) / accum_steps

            loss.backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            ep_loss += bce.item() * labels.size(0)
            ep_n += labels.size(0)

        scheduler.step()

        # ── Validate ──────────────────────────────────────────────
        model.eval()
        train_ds.training = False
        v_loss, v_acc, v_n = 0.0, 0, 0
        with torch.no_grad():
            for tensors, labels, _ in val_dl:
                plays_a = tensors["plays_a"].to(device)
                play_mask_a = tensors["play_mask_a"].to(device)
                roster_mask_a = tensors["roster_mask_a"].to(device)
                plays_b = tensors["plays_b"].to(device)
                play_mask_b = tensors["play_mask_b"].to(device)
                roster_mask_b = tensors["roster_mask_b"].to(device)
                labels = labels.to(device)

                logit, _ = model(
                    plays_a, play_mask_a, roster_mask_a,
                    plays_b, play_mask_b, roster_mask_b,
                )
                bce = F.binary_cross_entropy_with_logits(logit, labels)
                v_loss += bce.item() * labels.size(0)
                v_acc += ((logit > 0).float() == labels).sum().item()
                v_n += labels.size(0)

        v_loss /= max(v_n, 1)
        v_acc /= max(v_n, 1)
        elapsed = time.time() - t0

        if epoch % 5 == 0 or v_loss < best_val:
            print(f"  Epoch {epoch:3d} ({elapsed:.0f}s): "
                  f"train_bce={ep_loss/max(ep_n,1):.4f}  "
                  f"val_bce={v_loss:.4f}  val_acc={v_acc:.3f}  "
                  f"lr={scheduler.get_last_lr()[1]:.6f}")

        if v_loss < best_val:
            best_val = v_loss
            patience = 0
            _save_cpbp_checkpoint(model, data_dir, "best", p2i,
                                  optimizer, scheduler, epoch, best_val, patience,
                                  version=version)
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"  Early stopping at epoch {epoch}")
                break

        # Periodic checkpoint
        if epoch % 5 == 0:
            _save_cpbp_checkpoint(model, data_dir, "latest", p2i,
                                  optimizer, scheduler, epoch, best_val, patience,
                                  version=version)

    total = time.time() - t0
    print(f"\nDone! Best val_bce: {best_val:.4f}")
    print(f"  Total training time: {total:.0f}s ({total/3600:.1f}h)")

    return model, p2i, games, ts_games, ptensors, play_embs, game_rosters, player_games


# =====================================================================
#  PREDICTION / SUBMISSION
# =====================================================================

@torch.no_grad()
def generate_cpbp_predictions(
    data_dir: Path,
    season: int = 2026,
    device: str | None = None,
    version: str | None = None,
    exclusion_file: str | None = None,
):
    """Generate pairwise win probabilities using the composable PBP model.

    Args:
        exclusion_file: Path to JSON file with player exclusions.
            Format: {"exclusions": [{"team": "Duke", "player": "Caleb Foster", ...}]}
    """
    import pandas as pd
    from models.pbp_train import _build_team_name_map

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Generating composable PBP predictions for {season}...")

    # Load data
    games, p2i, ts_games, ptensors = load_pbp_data(
        data_dir, seasons=[2024, 2025, 2026],
    )
    game_rosters, player_games = build_player_game_index(games, ptensors)
    name_to_id = _build_team_name_map(data_dir)

    # Load model
    ckpt_dir = _cpbp_ckpt_dir(data_dir, version)
    with open(ckpt_dir / "config.json") as f:
        cfg = json.load(f)

    play_encoder = PlayEncoder(
        len(p2i), player_dim=8, n_play_types=N_PLAY_TYPES,
        ptype_dim=8, out_dim=cfg["play_dim"], dropout=0.0,
    )
    model = ComposablePBPModel(
        play_encoder=play_encoder,
        play_dim=cfg["play_dim"],
        d_player=cfg["d_player"],
    )
    state = torch.load(ckpt_dir / "cpbp_best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    # Encode all plays
    print("  Encoding plays...")
    play_embs = encode_all_plays(model.play_encoder, ptensors, device)

    # Precompute per-player embeddings
    print("  Precomputing player embeddings...")
    pse, pgb = precompute_player_season_embs(play_embs, ptensors, games, player_games)

    # Build player game index for boundary lookup
    _pgi = {}
    for key, gids in player_games.items():
        _pgi[key] = {g: i for i, g in enumerate(gids)}

    # Load exclusions if provided
    excluded_pids: dict[str, set[int]] = {}  # team_name -> set of excluded player indices
    if exclusion_file:
        # Build CBBD player name -> pid mapping from raw PBP files
        pid_to_name = {}
        pbp_dir = data_dir / "external" / "pbp"
        for fpath in sorted(pbp_dir.glob(f"plays_{season}_*.json")):
            with open(fpath) as f:
                plays_raw = json.load(f)
            for p in plays_raw:
                for pl in p.get("onFloor") or []:
                    if pl["id"] in p2i:
                        pid_to_name[p2i[pl["id"]]] = (pl["name"], pl["team"])

        with open(exclusion_file) as f:
            excl_data = json.load(f)

        n_excluded = 0
        for e in excl_data.get("exclusions", []):
            if e.get("status") not in ("out", "suspended"):
                continue
            for pid, (name, pteam) in pid_to_name.items():
                if (e["player"].lower() in name.lower() or name.lower() in e["player"].lower()):
                    if (e["team"].lower() in pteam.lower() or pteam.lower() in e["team"].lower()):
                        excluded_pids.setdefault(pteam, set()).add(pid)
                        n_excluded += 1
                        print(f"    Excluding {name} ({pteam}) - {e['reason']}")
        print(f"  {n_excluded} players excluded")

    # Build team → full season player embeddings (with exclusions applied)
    def get_team_players(team, season):
        """Get full-season play embeddings for each available player."""
        excluded = excluded_pids.get(team, set())
        players = []
        for (pid, t, s) in pse:
            if t == team and s == season and pid not in excluded:
                players.append(pse[(pid, t, s)])
        return players

    # Get all teams with PBP data for the prediction season
    season_teams = set()
    for (team, s) in ts_games:
        if s == season:
            tid = name_to_id.get(team.lower().strip())
            if tid is not None:
                season_teams.add((team, tid))

    # Precompute full-season player embeddings per team
    team_players = {}
    for team, tid in season_teams:
        plays = get_team_players(team, season)
        if len(plays) >= 5:
            team_players[tid] = plays

    print(f"  {len(team_players)} teams with PBP data")

    # Generate all pairwise predictions
    teams = pd.read_csv(data_dir / "MTeams.csv")
    seeds = pd.read_csv(data_dir / "MNCAATourneySeeds.csv")
    season_seeds = seeds[seeds["Season"] == season]
    tourney_tids = set(season_seeds["TeamID"].tolist())

    # All pairs
    all_tids = sorted(tourney_tids | set(team_players.keys()))
    rows = []
    n_predicted = 0

    for i, tid_a in enumerate(all_tids):
        for tid_b in all_tids[i + 1:]:
            pred = 0.5
            if tid_a in team_players and tid_b in team_players:
                plays_a = team_players[tid_a]
                plays_b = team_players[tid_b]

                # Pad to same player count
                max_p = max(len(plays_a), len(plays_b))
                max_t = max(
                    max(e.shape[0] for e in plays_a),
                    max(e.shape[0] for e in plays_b),
                )
                D = plays_a[0].shape[-1]

                def _pad_team(plays_list, max_p, max_t, D):
                    pa = torch.zeros(1, max_p, max_t, D)
                    pm = torch.ones(1, max_p, max_t, dtype=torch.bool)
                    rm = torch.ones(1, max_p, dtype=torch.bool)
                    for j, emb in enumerate(plays_list):
                        n = emb.shape[0]
                        pa[0, j, :n] = emb
                        pm[0, j, :n] = False
                        rm[0, j] = False
                    return pa.to(device), pm.to(device), rm.to(device)

                pa, pma, rma = _pad_team(plays_a, max_p, max_t, D)
                pb, pmb, rmb = _pad_team(plays_b, max_p, max_t, D)

                logit, _ = model(pa, pma, rma, pb, pmb, rmb)
                pred = torch.sigmoid(logit).item()
                n_predicted += 1

            rows.append({
                "ID": f"{season}_{tid_a}_{tid_b}",
                "Pred": max(0.01, min(0.99, pred)),
            })

    preds = pd.DataFrame(rows)
    print(f"  {n_predicted}/{len(rows)} matchups predicted (rest = 0.5)")
    return preds
