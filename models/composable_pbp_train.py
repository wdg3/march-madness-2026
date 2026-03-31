"""Composable PBP model — end-to-end training with player-level representations.

Trains the full model (PlayEncoder → per-player pooling → team self-attention
→ cross-attention → prediction) end-to-end so gradients flow all the way
back to player embeddings.

Usage:
    python run.py cpbp train [--time-limit 21600] [--version v1]
    python run.py cpbp submit --tag cpbp_v1 [--version v1] [--exclusions file.json]
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

from models.pbp_model import PlayEncoder, N_PLAY_CONTEXT
from models.pbp_train import (
    load_pbp_data, split_games, _pad_plays, _ckpt_dir, _load_checkpoint,
    N_PLAY_TYPES, PLAY_TYPES, PTYPE_TO_IDX, MAX_PLAYS,
)
from models.composable_pbp_model import (
    ComposablePBPModel, MAX_PLAYERS, MAX_PLAYS_PER_PLAYER,
)


# =====================================================================
#  PER-PLAYER PLAY INDEX (raw tensor data, not embeddings)
# =====================================================================

def build_player_game_index(games, play_tensors):
    """Build index of which players appeared in which games."""
    game_rosters: dict[int, dict[str, set[int]]] = {}
    player_games: dict[tuple, list[int]] = defaultdict(list)

    for (gid, team), pt in play_tensors.items():
        if gid not in games:
            continue
        season = games[gid]["season"]
        our_players = set(pt["our"].flatten().tolist()) - {0}

        if gid not in game_rosters:
            game_rosters[gid] = {}
        game_rosters[gid][team] = our_players

        for pid in our_players:
            player_games[(pid, team, season)].append(gid)

    for key in player_games:
        player_games[key] = sorted(
            player_games[key], key=lambda g: games[g]["date"]
        )

    return game_rosters, player_games


def get_player_play_indices(play_tensor: dict, player_idx: int) -> np.ndarray:
    """Get indices of plays where a specific player was on court."""
    our = play_tensor["our"]
    mask = (our == player_idx).any(axis=1)
    return np.where(mask)[0]


def precompute_player_raw_plays(play_tensors, games, player_games):
    """Precompute per-player raw play tensor data with game boundaries.

    Stores the raw numpy arrays (our, their, ptypes, ctx) per player so
    the PlayEncoder can encode them on-the-fly during training with gradients.

    Returns:
        player_raw_plays: dict[(pid, team, season) → dict with keys:
            our (N,5), their (N,5), ptypes (N,), ctx (N,6)]
        player_game_bounds: dict[(pid, team, season) → list[int]]
    """
    player_raw_plays = {}
    player_game_bounds = {}

    for (pid, team, season), gids in player_games.items():
        our_l, their_l, pt_l, ctx_l = [], [], [], []
        bounds = []
        total = 0

        for gid in gids:
            key = (gid, team)
            if key not in play_tensors:
                bounds.append(total)
                continue
            pt = play_tensors[key]
            indices = get_player_play_indices(pt, pid)
            if len(indices) > 0:
                our_l.append(pt["our"][indices])
                their_l.append(pt["their"][indices])
                pt_l.append(pt["ptypes"][indices])
                ctx_l.append(pt["ctx"][indices])
                total += len(indices)
            bounds.append(total)

        if total == 0:
            continue

        our_cat = np.concatenate(our_l)
        their_cat = np.concatenate(their_l)
        pt_cat = np.concatenate(pt_l)
        ctx_cat = np.concatenate(ctx_l)

        player_raw_plays[(pid, team, season)] = {
            "our": our_cat,
            "their": their_cat,
            "ptypes": pt_cat,
            "ctx": ctx_cat,
        }
        player_game_bounds[(pid, team, season)] = bounds

    return player_raw_plays, player_game_bounds


# =====================================================================
#  DATASET (returns raw play tensors for end-to-end encoding)
# =====================================================================

class ComposableMatchupDataset(Dataset):
    """Dataset returning raw play tensor data per player for E2E training."""

    def __init__(
        self,
        game_ids: list[int],
        games: dict,
        game_rosters: dict,
        player_games: dict,
        player_raw_plays: dict,
        player_game_bounds: dict,
        max_players: int = MAX_PLAYERS,
        player_drop_prob: float = 0.0,
        min_players: int = 5,
    ):
        self.games = games
        self.game_rosters = game_rosters
        self.player_games = player_games
        self.prp = player_raw_plays
        self.pgb = player_game_bounds
        self.max_p = max_players
        self.player_drop_prob = player_drop_prob
        self.min_players = min_players
        self.training = False

        self._player_game_idx: dict[tuple, dict[int, int]] = {}
        for key, gids in player_games.items():
            self._player_game_idx[key] = {g: i for i, g in enumerate(gids)}

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
        season = self.games[gid]["season"]

        result = {}
        for side, team in [("a", ht), ("b", at)]:
            roster = sorted(self.game_rosters[gid][team])

            if self.training and self.player_drop_prob > 0 and len(roster) > self.min_players:
                roster = [p for p in roster if random.random() > self.player_drop_prob]
                if len(roster) < self.min_players:
                    roster = sorted(self.game_rosters[gid][team])[:self.min_players]

            if len(roster) > self.max_p:
                roster = sorted(
                    roster,
                    key=lambda p: self.pgb.get(
                        (p, team, season), [0])[-1] if (p, team, season) in self.pgb else 0,
                    reverse=True,
                )[:self.max_p]

            # Gather raw play tensor data per player (causal slice)
            player_data = []  # list of dicts with our/their/ptypes/ctx
            for pid in roster:
                key = (pid, team, season)
                if key not in self.prp:
                    continue
                gi = self._player_game_idx.get(key, {}).get(gid)
                if gi is None or gi == 0:
                    continue
                bound = self.pgb[key][gi - 1]
                if bound <= 0:
                    continue
                rp = self.prp[key]
                player_data.append({
                    "our": rp["our"][:bound],
                    "their": rp["their"][:bound],
                    "ptypes": rp["ptypes"][:bound],
                    "ctx": rp["ctx"][:bound],
                })

            result[f"players_{side}"] = player_data

        result["label"] = label
        result["margin"] = margin
        return result


def collate_e2e(batch):
    """Collate raw play data into padded tensors for E2E training.

    Returns per-team: (our, their, ptypes, ctx, play_mask, roster_mask)
    all shaped for the PlayEncoder and downstream attention.
    """
    B = len(batch)

    tensors = {}
    for side in ["a", "b"]:
        key = f"players_{side}"
        players_per_item = [b[key] for b in batch]

        max_p = max(len(ps) for ps in players_per_item)
        max_p = max(max_p, 1)
        max_t = 1
        for ps in players_per_item:
            for p in ps:
                max_t = max(max_t, len(p["our"]))

        our = np.zeros((B, max_p, max_t, 5), dtype=np.int32)
        their = np.zeros((B, max_p, max_t, 5), dtype=np.int32)
        ptypes = np.zeros((B, max_p, max_t), dtype=np.int32)
        ctx = np.zeros((B, max_p, max_t, N_PLAY_CONTEXT), dtype=np.float32)
        play_mask = np.ones((B, max_p, max_t), dtype=bool)
        roster_mask = np.ones((B, max_p), dtype=bool)

        for i, ps in enumerate(players_per_item):
            for j, p in enumerate(ps):
                n = len(p["our"])
                our[i, j, :n] = p["our"]
                their[i, j, :n] = p["their"]
                ptypes[i, j, :n] = p["ptypes"]
                ctx[i, j, :n] = p["ctx"]
                play_mask[i, j, :n] = False
                roster_mask[i, j] = False

        tensors[f"our_{side}"] = torch.from_numpy(our).long()
        tensors[f"their_{side}"] = torch.from_numpy(their).long()
        tensors[f"ptypes_{side}"] = torch.from_numpy(ptypes).long()
        tensors[f"ctx_{side}"] = torch.from_numpy(ctx).float()
        tensors[f"play_mask_{side}"] = torch.from_numpy(play_mask)
        tensors[f"roster_mask_{side}"] = torch.from_numpy(roster_mask)

    labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
    margins = torch.tensor([b["margin"] for b in batch], dtype=torch.float32)
    return tensors, labels, margins


def e2e_forward(model, tensors, device):
    """End-to-end forward: raw plays → PlayEncoder → pooler → attention → pred.

    Encodes plays through PlayEncoder WITH gradients.
    """
    results = {}
    for side in ["a", "b"]:
        our = tensors[f"our_{side}"].to(device)       # (B, P, T, 5)
        their = tensors[f"their_{side}"].to(device)    # (B, P, T, 5)
        pt = tensors[f"ptypes_{side}"].to(device)      # (B, P, T)
        ctx = tensors[f"ctx_{side}"].to(device)        # (B, P, T, 6)
        pmask = tensors[f"play_mask_{side}"].to(device) # (B, P, T)
        rmask = tensors[f"roster_mask_{side}"].to(device) # (B, P)

        B, P, T, _ = our.shape

        # Flatten B*P for PlayEncoder, then reshape back
        flat_our = our.reshape(B * P, T, 5)
        flat_their = their.reshape(B * P, T, 5)
        flat_pt = pt.reshape(B * P, T)
        flat_ctx = ctx.reshape(B * P, T, N_PLAY_CONTEXT)

        # PlayEncoder: (B*P, T, 5) → (B*P, T, D) — WITH gradients
        play_embs = model.play_encoder(flat_our, flat_their, flat_pt, flat_ctx)
        play_embs = play_embs.reshape(B, P, T, -1)  # (B, P, T, D)

        results[f"plays_{side}"] = play_embs
        results[f"play_mask_{side}"] = pmask
        results[f"roster_mask_{side}"] = rmask

    logit, margin = model(
        results["plays_a"], results["play_mask_a"], results["roster_mask_a"],
        results["plays_b"], results["play_mask_b"], results["roster_mask_b"],
    )
    return logit, margin


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
#  MAIN TRAINING FUNCTION (END-TO-END)
# =====================================================================

def train_composable_pbp(
    data_dir: Path,
    time_limit: int = 21600,
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
    batch_size: int = 4,
    accum_steps: int = 8,
    device: str | None = None,
    version: str | None = None,
):
    t0 = time.time()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Training Composable PBP model E2E on {device}"
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

    # ── Precompute raw play data per player ───────────────────────
    print("\n  Precomputing per-player raw play data...")
    prp, pgb = precompute_player_raw_plays(ptensors, games, player_games)
    print(f"  {len(prp)} player-team-seasons precomputed")

    # ── Build model ───────────────────────────────────────────────
    print("\n[3/4] Building model...")
    n_cbbd_players = len(p2i)
    play_encoder = PlayEncoder(
        n_cbbd_players, player_dim=8, n_play_types=N_PLAY_TYPES,
        ptype_dim=8, out_dim=play_dim, dropout=dropout,
    )

    # Warm-start PlayEncoder
    pbp_ckpt = _ckpt_dir(data_dir) / "pbp_best.pt"
    if pbp_ckpt.exists():
        print(f"  Warm-starting PlayEncoder from {pbp_ckpt}")
        state = torch.load(pbp_ckpt, map_location="cpu", weights_only=True)
        pe_state = {
            k.replace("play_encoder.", ""): v
            for k, v in state.items() if k.startswith("play_encoder.")
        }
        play_encoder.load_state_dict(pe_state, strict=False)

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
    print(f"  Model: {n_params:,} parameters (E2E training)")

    # ── Build datasets ────────────────────────────────────────────
    print("\n[4/4] Training...")
    train_ds = ComposableMatchupDataset(
        train_ids, games, game_rosters, player_games, prp, pgb,
        player_drop_prob=player_drop_prob,
    )
    val_ds = ComposableMatchupDataset(
        val_ids, games, game_rosters, player_games, prp, pgb,
    )
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_e2e, num_workers=0,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_e2e, num_workers=0,
    )

    # ── Optimizer (differential LR: play encoder slower) ──────────
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
            labels = labels.to(device)
            margins = margins.to(device)

            logit, margin_pred = e2e_forward(model, tensors, device)

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
                labels = labels.to(device)
                logit, _ = e2e_forward(model, tensors, device)
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

        if epoch % 5 == 0:
            _save_cpbp_checkpoint(model, data_dir, "latest", p2i,
                                  optimizer, scheduler, epoch, best_val, patience,
                                  version=version)

    total = time.time() - t0
    print(f"\nDone! Best val_bce: {best_val:.4f}")
    print(f"  Total training time: {total:.0f}s ({total/3600:.1f}h)")

    return model, p2i


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
    """Generate pairwise win probabilities using the composable PBP model."""
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

    # Encode all plays through PlayEncoder (for prediction, no gradients needed)
    print("  Encoding plays...")
    from models.composable_pbp_train import encode_all_plays, precompute_player_season_embs
    play_embs = encode_all_plays(model.play_encoder, ptensors, device)

    print("  Precomputing player embeddings...")
    pse, pgb = precompute_player_season_embs(play_embs, ptensors, games, player_games)

    # Load exclusions if provided
    excluded_pids: dict[str, set[int]] = {}
    if exclusion_file:
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

    # Build team → player embeddings (using pooler from trained model)
    def get_team_player_embs(team, season):
        excluded = excluded_pids.get(team, set())
        embs = []
        for (pid, t, s) in pse:
            if t == team and s == season and pid not in excluded:
                # Pool through the model's pbp_pooler
                pe = pse[(pid, t, s)].unsqueeze(0).to(device)
                pm = torch.zeros(1, pe.shape[1], dtype=torch.bool, device=device)
                player_emb = model.pbp_pooler(pe, pm)
                embs.append(player_emb.squeeze(0).cpu())
        return embs

    # Get all teams
    season_teams = set()
    for (team, s) in ts_games:
        if s == season:
            tid = name_to_id.get(team.lower().strip())
            if tid is not None:
                season_teams.add((team, tid))

    team_embs = {}
    for team, tid in season_teams:
        embs = get_team_player_embs(team, season)
        if len(embs) >= 5:
            team_embs[tid] = embs

    print(f"  {len(team_embs)} teams with PBP data")

    # Generate all pairwise predictions
    seeds = pd.read_csv(data_dir / "MNCAATourneySeeds.csv")
    season_seeds = seeds[seeds["Season"] == season]
    tourney_tids = set(season_seeds["TeamID"].tolist())

    all_tids = sorted(tourney_tids | set(team_embs.keys()))
    rows = []
    n_predicted = 0

    def _pad_player_embs(embs_list, max_p, D, device):
        """Pad list of player embeddings into (1, max_p, D) tensor + mask."""
        pa = torch.zeros(1, max_p, D, device=device)
        rm = torch.ones(1, max_p, dtype=torch.bool, device=device)
        for j, emb in enumerate(embs_list):
            pa[0, j] = emb.to(device)
            rm[0, j] = False
        return pa, rm

    for i, tid_a in enumerate(all_tids):
        for tid_b in all_tids[i + 1:]:
            pred = 0.5
            if tid_a in team_embs and tid_b in team_embs:
                embs_a = team_embs[tid_a]
                embs_b = team_embs[tid_b]
                max_p = max(len(embs_a), len(embs_b))
                D = model.d_player

                a_embs, rma = _pad_player_embs(embs_a, max_p, D, device)
                b_embs, rmb = _pad_player_embs(embs_b, max_p, D, device)

                # Self-attention
                for layer in model.team_self_layers:
                    a_embs = layer(a_embs, key_padding_mask=rma)
                    b_embs = layer(b_embs, key_padding_mask=rmb)
                a_embs = model.self_norm(a_embs)
                b_embs = model.self_norm(b_embs)

                # Cross-attention
                a_cross, b_cross = a_embs, b_embs
                for layer in model.cross_layers:
                    a_cross = layer(a_cross, kv=b_cross, key_padding_mask=rmb)
                    b_cross = layer(b_cross, kv=a_cross, key_padding_mask=rma)
                a_cross = model.cross_norm(a_cross)
                b_cross = model.cross_norm(b_cross)

                # Prediction attention
                a_side = a_cross + model.side_emb.weight[0]
                b_side = b_cross + model.side_emb.weight[1]
                all_players = torch.cat([a_side, b_side], dim=1)
                all_mask = torch.cat([rma, rmb], dim=1)

                query = model.pred_query.expand(1, -1, -1)
                safe_mask = all_mask.clone()
                if safe_mask.all(dim=1).any():
                    safe_mask[0, 0] = False
                pred_ctx, _ = model.pred_attn(
                    query, all_players, all_players, key_padding_mask=safe_mask,
                )
                pred_ctx = model.pred_norm(pred_ctx.squeeze(1))
                logit = model.head(pred_ctx).squeeze(-1)
                pred = torch.sigmoid(logit).item()
                n_predicted += 1

            rows.append({
                "ID": f"{season}_{tid_a}_{tid_b}",
                "Pred": max(0.01, min(0.99, pred)),
            })

    preds = pd.DataFrame(rows)
    print(f"  {n_predicted}/{len(rows)} matchups predicted (rest = 0.5)")
    return preds


# Keep for backward compat with frozen-embedding predictions
@torch.no_grad()
def encode_all_plays(play_encoder, play_tensors, device, batch_size=64):
    """Encode all plays through the PlayEncoder → cached embeddings."""
    play_encoder.eval()
    keys = list(play_tensors.keys())
    play_embs = {}

    for i in range(0, len(keys), batch_size):
        chunk = keys[i:i + batch_size]
        dicts = [play_tensors[k] for k in chunk]
        our, their, pt, ctx, mask = _pad_plays(dicts, device=device)
        embs = play_encoder(our, their, pt, ctx)
        for j, k in enumerate(chunk):
            n = dicts[j]["n_plays"]
            play_embs[k] = embs[j, :n].cpu()

    play_encoder.train()
    return play_embs


def precompute_player_season_embs(play_embs, play_tensors, games, player_games):
    """Precompute per-player season play embeddings (for prediction only)."""
    player_season_embs = {}
    player_game_bounds = {}

    for (pid, team, season), gids in player_games.items():
        emb_chunks = []
        bounds = []
        total = 0

        for gid in gids:
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
