"""PBP deep model — data loading, training, and prediction.

Unified end-to-end training:
    Each forward pass encodes plays → game embeddings → season embeddings
    → matchup prediction.  An auxiliary game-level loss gives the play
    encoder dense supervision from epoch 0 (solves cold start), while
    the matchup loss propagates quality signal through the full chain
    back to player embeddings.

Usage:
    python run.py pbp train [--time-limit 7200]
    python run.py pbp submit --tag pbp_v1
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

from models.pbp_model import PBPMatchupModel, N_PLAY_CONTEXT

# ── Play-type vocabulary ──────────────────────────────────────────
PLAY_TYPES = sorted([
    "Block Shot", "Coach's Challenge (Stands)", "Dead Ball Rebound",
    "Defensive Rebound", "DunkShot", "End Game", "End Period",
    "Jumpball", "JumpShot", "LayUpShot", "Lost Ball Turnover",
    "MadeFreeThrow", "MissedFreeThrow", "Offensive Rebound",
    "OfficialTVTimeOut", "PersonalFoul", "RegularTimeOut",
    "ShortTimeOut", "Steal", "Substitution", "Technical Foul",
    "TipShot",
])
PTYPE_TO_IDX = {t: i + 1 for i, t in enumerate(PLAY_TYPES)}  # 0 = unknown
N_PLAY_TYPES = len(PLAY_TYPES) + 1  # include unknown

MAX_PLAYS = 500  # truncate games longer than this


# =====================================================================
#  DATA LOADING  (single-pass: load JSON → encode tensors → discard raw)
# =====================================================================

def _encode_plays_raw(plays: list[dict], team: str, p2i: dict) -> dict | None:
    """Encode raw play dicts from *team*'s perspective into numpy arrays."""
    our_l, their_l, pt_l, ctx_l = [], [], [], []

    for p in plays:
        floor = p.get("onFloor") or []
        if len(floor) < 10:
            continue
        ours, theirs = [], []
        for pl in floor:
            (ours if pl["team"] == team else theirs).append(p2i.get(pl["id"], 0))
        if len(ours) < 5 or len(theirs) < 5:
            continue

        our_l.append(ours[:5])
        their_l.append(theirs[:5])
        pt_l.append(PTYPE_TO_IDX.get(p.get("playType", ""), 0))

        period_frac = (p.get("period", 1) - 1) / 4.0
        time_frac = p.get("secondsRemaining", 600) / 1200.0
        is_scoring = float(p.get("scoringPlay", False))
        score_val = p.get("scoreValue", 0) / 3.0
        is_shooting = float(p.get("shootingPlay", False))
        our_action = (1.0 if p.get("team") == team
                      else (0.0 if p.get("team") else 0.5))
        ctx_l.append([period_frac, time_frac, is_scoring, score_val,
                      is_shooting, our_action])

    n = len(our_l)
    if n < 20:
        return None
    if n > MAX_PLAYS:
        our_l, their_l, pt_l, ctx_l = (
            our_l[:MAX_PLAYS], their_l[:MAX_PLAYS],
            pt_l[:MAX_PLAYS], ctx_l[:MAX_PLAYS],
        )
        n = MAX_PLAYS

    return dict(
        our=np.array(our_l, dtype=np.int32),
        their=np.array(their_l, dtype=np.int32),
        ptypes=np.array(pt_l, dtype=np.int8),
        ctx=np.array(ctx_l, dtype=np.float32),
        n_plays=n,
    )


def load_pbp_data(data_dir: Path, seasons: list[int] | None = None):
    """Load PBP JSON → game metadata + play tensors in one pass.

    Raw JSON is discarded after encoding — only compact numpy arrays
    and lightweight game metadata are kept in memory.

    Returns:
        games       : dict[game_id → metadata dict]  (NO raw plays)
        player_to_idx : dict[cbbd_player_id → int]
        ts_games    : dict[(team, season) → [game_ids by date]]
        play_tensors: dict[(game_id, team) → tensor_dict]
    """
    pbp_dir = data_dir / "external" / "pbp"
    player_to_idx: dict[int, int] = {}
    _next = [1]

    def _pidx(pid):
        if pid not in player_to_idx:
            player_to_idx[pid] = _next[0]
            _next[0] += 1
        return player_to_idx[pid]

    games: dict[int, dict] = {}
    ts_games: dict[tuple, list] = defaultdict(list)
    play_tensors: dict[tuple, dict] = {}
    n_files = 0

    for fpath in sorted(pbp_dir.glob("plays_*.json")):
        parts = fpath.stem.split("_")
        season = int(parts[1])
        if seasons and season not in seasons:
            continue

        with open(fpath) as f:
            plays = json.load(f)
        if not plays:
            continue
        n_files += 1

        # ── Pass 1: register all player IDs (needed before encoding) ──
        for p in plays:
            for pl in p.get("onFloor") or []:
                _pidx(pl["id"])

        # ── Pass 2: group by game, extract metadata + encode tensors ──
        by_game: dict[int, list] = defaultdict(list)
        for p in plays:
            by_game[p["gameId"]].append(p)

        for gid, gplays in by_game.items():
            # Determine teams for tensor encoding even if game already seen
            # (we need tensors from THIS team's file's perspective)
            gplays.sort(key=lambda p: (p["period"], -p.get("secondsRemaining", 0)))

            # Determine home/away once
            home = away = None
            for p in gplays:
                if p.get("isHomeTeam") is True and p.get("team"):
                    home = p["team"]
                elif p.get("isHomeTeam") is False and p.get("team"):
                    away = p["team"]
                if home and away:
                    break
            if not home or not away:
                floor_teams = sorted({
                    pl["team"]
                    for p in gplays for pl in (p.get("onFloor") or [])
                })
                if len(floor_teams) >= 2:
                    home, away = home or floor_teams[0], away or floor_teams[1]
            if not home or not away:
                continue

            # Encode play tensors for both teams (if not already done)
            for team in (home, away):
                if (gid, team) not in play_tensors:
                    t = _encode_plays_raw(gplays, team, player_to_idx)
                    if t is not None:
                        play_tensors[(gid, team)] = t

            # Store game metadata (once per game, NO raw plays)
            if gid not in games:
                fh = max(p["homeScore"] for p in gplays)
                fa = max(p["awayScore"] for p in gplays)
                games[gid] = dict(
                    id=gid,
                    home_team=home, away_team=away,
                    date=gplays[0].get("gameStartDate", ""),
                    season=season,
                    season_type=gplays[0].get("seasonType", "regular"),
                    tournament=gplays[0].get("tournament"),
                    home_score=fh, away_score=fa,
                    home_win=float(fh > fa),
                )
                ts_games[(home, season)].append(gid)
                ts_games[(away, season)].append(gid)

        # Raw JSON is now out of scope → GC can reclaim it
        del plays, by_game

    # sort each team-season by game date
    ts_sorted = {
        k: sorted(v, key=lambda g: games[g]["date"])
        for k, v in ts_games.items()
    }

    print(f"  {n_files} files → {len(games)} games, {len(player_to_idx)} players, "
          f"{len(play_tensors)} play tensors")
    return games, player_to_idx, ts_sorted, play_tensors


# Back-compat wrappers used by cmd_pbp submit
def load_pbp_games(data_dir, seasons=None):
    """Wrapper returning (games, p2i, ts_games) — play_tensors separate."""
    games, p2i, ts, pt = load_pbp_data(data_dir, seasons)
    return games, p2i, ts

def precompute_play_tensors(games, p2i):
    """No-op — tensors are already computed by load_pbp_data."""
    raise RuntimeError("Use load_pbp_data() instead — it returns play_tensors directly")


# =====================================================================
#  PHASE 1 — GAME-LEVEL DATASET
# =====================================================================

class GameDataset(Dataset):
    """Each example is one game — predict outcome from current game plays."""

    def __init__(self, game_ids, games, play_tensors):
        self.examples = []
        for gid in game_ids:
            g = games[gid]
            ht, at = g["home_team"], g["away_team"]
            if (gid, ht) in play_tensors and (gid, at) in play_tensors:
                self.examples.append((
                    gid, ht, at,
                    g["home_win"],
                    (g["home_score"] - g["away_score"]) / 20.0,
                ))
        self.play_tensors = play_tensors

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        gid, ht, at, label, margin = self.examples[idx]
        return self.play_tensors[(gid, ht)], self.play_tensors[(gid, at)], label, margin


def _pad_plays(tensor_dicts, device="cpu"):
    """Pad a list of play tensor dicts → batched tensors + mask."""
    B = len(tensor_dicts)
    T = max(d["n_plays"] for d in tensor_dicts)

    our = torch.zeros(B, T, 5, dtype=torch.long, device=device)
    their = torch.zeros(B, T, 5, dtype=torch.long, device=device)
    pt = torch.zeros(B, T, dtype=torch.long, device=device)
    ctx = torch.zeros(B, T, N_PLAY_CONTEXT, device=device)
    mask = torch.ones(B, T, dtype=torch.bool, device=device)

    for i, d in enumerate(tensor_dicts):
        n = d["n_plays"]
        our[i, :n] = torch.from_numpy(d["our"])
        their[i, :n] = torch.from_numpy(d["their"])
        pt[i, :n] = torch.from_numpy(d["ptypes"])
        ctx[i, :n] = torch.from_numpy(d["ctx"])
        mask[i, :n] = False

    return our, their, pt, ctx, mask


def collate_phase1(batch):
    home, away, labels, margins = zip(*batch)
    h_our, h_their, h_pt, h_ctx, h_mask = _pad_plays(home)
    a_our, a_their, a_pt, a_ctx, a_mask = _pad_plays(away)
    return (h_our, h_their, h_pt, h_ctx, h_mask,
            a_our, a_their, a_pt, a_ctx, a_mask,
            torch.tensor(labels, dtype=torch.float32),
            torch.tensor(margins, dtype=torch.float32))


# =====================================================================
#  PHASE 2 — END-TO-END SEASON-LEVEL DATASET
# =====================================================================

class SeasonDatasetE2E(Dataset):
    """End-to-end season dataset — returns game references for on-the-fly encoding.

    Each example stores lightweight references to prior games. The training
    loop encodes plays through the play encoder with full gradient flow,
    so the matchup loss backprops all the way to player embeddings.
    """

    # Each game ref: (game_id, team, opponent, won, margin)
    GameRef = tuple  # (int, str, str, float, float)

    def __init__(self, game_ids, games, ts_games, play_tensors,
                 min_prior=3, max_prior=20, game_drop=0.0):
        self.examples = []
        self.games = games
        self.ts_games = ts_games
        self.play_tensors = play_tensors
        self.max_prior = max_prior
        self.min_prior = min_prior
        self.game_drop = game_drop
        self.training = False

        for gid in game_ids:
            g = games[gid]
            ht, at = g["home_team"], g["away_team"]
            date, season = g["date"], g["season"]
            prior_h = self._prior_refs(ht, season, date)
            prior_a = self._prior_refs(at, season, date)
            if len(prior_h) >= min_prior and len(prior_a) >= min_prior:
                self.examples.append((
                    prior_h[-max_prior:],
                    prior_a[-max_prior:],
                    g["home_win"],
                    (g["home_score"] - g["away_score"]) / 20.0,
                ))

    def _prior_refs(self, team, season, before):
        """Return list of (game_id, team, opponent, won, margin) for prior games."""
        refs = []
        for gid in self.ts_games.get((team, season), []):
            g = self.games[gid]
            if g["date"] >= before:
                continue
            if (gid, team) not in self.play_tensors:
                continue
            opp = g["away_team"] if g["home_team"] == team else g["home_team"]
            won = float((g["home_win"] == 1.0) == (g["home_team"] == team))
            margin = (g["home_score"] - g["away_score"]) / 20.0
            if g["home_team"] != team:
                margin = -margin
            refs.append((gid, team, opp, won, margin))
        return refs

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        h_refs, a_refs, label, margin = self.examples[idx]
        if self.training and self.game_drop > 0:
            h_refs = [r for r in h_refs if random.random() > self.game_drop]
            a_refs = [r for r in a_refs if random.random() > self.game_drop]
            # Ensure minimum games remain
            if len(h_refs) < self.min_prior:
                h_refs = self.examples[idx][0][-self.min_prior:]
            if len(a_refs) < self.min_prior:
                a_refs = self.examples[idx][1][-self.min_prior:]
        return h_refs, a_refs, label, margin


def e2e_forward(batch, model, play_tensors, device):
    """Full end-to-end forward pass: plays → game embs → season embs → prediction.

    Gradient flows all the way from the matchup loss back to player embeddings.

    Returns:
        logit, margin_pred, labels, margins (all tensors on device)
    """
    B = len(batch)

    # 1. Collect all unique (game_id, team) pairs to encode
    all_keys = set()
    for h_refs, a_refs, _, _ in batch:
        for gid, team, opp, _, _ in h_refs + a_refs:
            all_keys.add((gid, team))
            if (gid, opp) in play_tensors:
                all_keys.add((gid, opp))
    all_keys = list(all_keys)
    key_to_idx = {k: i for i, k in enumerate(all_keys)}

    # 2. Batch-encode all games through the play encoder (WITH gradients)
    dicts = [play_tensors[k] for k in all_keys]
    our, their, pt, ctx, mask = _pad_plays(dicts, device=device)
    all_game_embs = model.encode_game(our, their, pt, ctx, mask)  # (N, D)

    D = model.embed_dim
    zero_emb = torch.zeros(D, device=device)

    # 3. Build enriched season sequences
    h_enriched, a_enriched = [], []
    labels_list, margins_list = [], []

    for h_refs, a_refs, label, margin in batch:
        for refs, out_list in [(h_refs, h_enriched), (a_refs, a_enriched)]:
            embs = []
            for gid, team, opp, won, mgn in refs:
                our_emb = all_game_embs[key_to_idx[(gid, team)]]
                opp_key = (gid, opp)
                opp_emb = all_game_embs[key_to_idx[opp_key]] if opp_key in key_to_idx else zero_emb
                outcome = torch.tensor([won, mgn], device=device)
                embs.append(torch.cat([our_emb, opp_emb, outcome]))
            out_list.append(torch.stack(embs))
        labels_list.append(label)
        margins_list.append(margin)

    # 4. Pad sequences
    D_e = 2 * D + 2
    mh = max(s.size(0) for s in h_enriched)
    ma = max(s.size(0) for s in a_enriched)

    hp = torch.zeros(B, mh, D_e, device=device)
    ap = torch.zeros(B, ma, D_e, device=device)
    hm = torch.ones(B, mh, dtype=torch.bool, device=device)
    am = torch.ones(B, ma, dtype=torch.bool, device=device)

    for i in range(B):
        nh, na = h_enriched[i].size(0), a_enriched[i].size(0)
        hp[i, :nh] = h_enriched[i]
        ap[i, :na] = a_enriched[i]
        hm[i, :nh] = False
        am[i, :na] = False

    # 5. Project + season encode + predict matchup
    sa = model.encode_season_enriched(hp[:, :, :D], hp[:, :, D:2*D],
                                       hp[:, :, 2*D:], hm)
    sb = model.encode_season_enriched(ap[:, :, :D], ap[:, :, D:2*D],
                                       ap[:, :, 2*D:], am)
    logit, margin_pred = model.predict(sa, sb)

    labels_t = torch.tensor(labels_list, dtype=torch.float32, device=device)
    margins_t = torch.tensor(margins_list, dtype=torch.float32, device=device)
    return logit, margin_pred, labels_t, margins_t


# =====================================================================
#  GAME EMBEDDING COMPUTATION
# =====================================================================

@torch.no_grad()
def compute_game_embeddings(model, play_tensors, device, batch_size=64):
    """Forward all (game, team) pairs through the play encoder → mean pool."""
    model.eval()
    keys = list(play_tensors.keys())
    embs = {}

    for i in range(0, len(keys), batch_size):
        chunk = keys[i : i + batch_size]
        dicts = [play_tensors[k] for k in chunk]
        our, their, pt, ctx, mask = _pad_plays(dicts, device=device)
        ge = model.encode_game(our, their, pt, ctx, mask)  # (B, D)
        for j, k in enumerate(chunk):
            embs[k] = ge[j].cpu()

    model.train()
    return embs


# =====================================================================
#  TRAIN / VAL SPLIT
# =====================================================================

def split_games(games):
    """Split into train / val game-id lists.

    Train : 2024 non-NCAA, 2025 non-NCAA, 2026 all
    Val   : 2024 + 2025 NCAA tournament games (~130 games)

    Using two tournament years for validation gives a more stable
    signal for early stopping and model selection.
    """
    train_ids, val_ids = [], []
    for gid, g in games.items():
        is_ncaa = g.get("tournament") == "NCAA"
        if is_ncaa and g["season"] in (2024, 2025):
            val_ids.append(gid)
        else:
            train_ids.append(gid)
    return train_ids, val_ids


def _log_grad_norms(model, epoch):
    """Print gradient norms per module for diagnostics."""
    groups = {
        "player_emb": model.play_encoder.player_emb,
        "lineup_attn": model.play_encoder.lineup_attn,
        "play_mlp": model.play_encoder.mlp,
        "game_gru": model.game_gru,
        "season_enc": model.season_encoder,
        "head": model.head,
    }
    parts = []
    for name, mod in groups.items():
        grads = [p.grad for p in mod.parameters() if p.grad is not None]
        if grads:
            norm = torch.sqrt(sum(g.norm() ** 2 for g in grads)).item()
            parts.append(f"{name}={norm:.4f}")
        else:
            parts.append(f"{name}=none")
    print(f"    grad_norms @{epoch}: {', '.join(parts)}")


# =====================================================================
#  MAIN TRAINING FUNCTION
# =====================================================================

def train_pbp_model(
    data_dir: Path,
    time_limit: int = 7200,
    embed_dim: int = 64,
    player_dim: int = 8,
    n_heads: int = 4,
    n_season_layers: int = 2,
    dropout: float = 0.3,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    margin_weight: float = 0.3,
    label_smoothing: float = 0.05,
    game_drop: float = 0.2,
    max_epochs: int = 100,
    batch_size: int = 32,
    device: str | None = None,
    resume: bool = False,
    version: str | None = None,
    loss_fn: str = "brier",
):
    t0 = time.time()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Training PBP deep model on {device}"
          f"{f' (version: {version})' if version else ''}")
    print(f"  Time limit: {time_limit}s ({time_limit/3600:.1f}h)")
    print(f"  Loss: {loss_fn}")

    # ── load data ──────────────────────────────────────────────────
    print("\n[1/3] Loading PBP data + encoding play tensors...")
    games, p2i, ts_games, ptensors = load_pbp_data(data_dir, seasons=[2024, 2025, 2026])

    train_ids, val_ids = split_games(games)
    print(f"  Train games: {len(train_ids)}, Val games: {len(val_ids)}")

    # ── build model ────────────────────────────────────────────────
    n_players = len(p2i)
    model = PBPMatchupModel(
        n_players=n_players, embed_dim=embed_dim, player_dim=player_dim,
        n_play_types=N_PLAY_TYPES, ptype_dim=8,
        n_heads=n_heads, n_season_layers=n_season_layers, dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {n_params:,} parameters ({n_players} players)")

    # ── unified end-to-end training ────────────────────────────────
    print(f"\n[2/3] Unified E2E training ({max_epochs} epochs)")
    print(f"  Game dropout: {game_drop}, Dropout: {dropout}, "
          f"Weight decay: {weight_decay}, Label smoothing: {label_smoothing}")

    eff_batch = 32
    accum_steps = 2

    train_ds = SeasonDatasetE2E(train_ids, games, ts_games, ptensors,
                                game_drop=game_drop)
    val_ds = SeasonDatasetE2E(val_ids, games, ts_games, ptensors)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"  Batch size: {eff_batch} (accum {accum_steps}x = "
          f"effective {eff_batch * accum_steps})")

    train_dl = DataLoader(train_ds, batch_size=eff_batch, shuffle=True,
                          collate_fn=lambda x: x, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=eff_batch, shuffle=False,
                        collate_fn=lambda x: x, num_workers=0)

    # Differential learning rates calibrated to observed gradient norms
    # so that effective updates (grad_norm * lr) are balanced across modules.
    #
    # Observed grad norms @epoch10: player_emb=0.015, lineup_attn=0.06,
    #   play_mlp=0.25, game_gru=0.16, season_enc=0.14, head=0.08
    #
    # LR multipliers set ~inversely proportional to grad norm:
    #   emb:       10x  (sparse, tiny grads)
    #   play_attn:  2x  (small grads)
    #   play_mlp:  0.3x (large grads, overfits fast)
    #   downstream: 1x  (moderate grads)
    emb_params = (
        list(model.play_encoder.player_emb.parameters())
        + list(model.play_encoder.ptype_emb.parameters())
        + list(model.play_encoder.side_emb.parameters())
    )
    emb_ids = set(id(p) for p in emb_params)
    attn_params = list(model.play_encoder.lineup_attn.parameters())
    attn_ids = set(id(p) for p in attn_params)
    mlp_params = [
        p for p in model.play_encoder.parameters()
        if id(p) not in emb_ids and id(p) not in attn_ids
    ]
    mlp_ids = set(id(p) for p in mlp_params)
    downstream_params = [
        p for p in model.parameters()
        if id(p) not in emb_ids and id(p) not in attn_ids and id(p) not in mlp_ids
    ]

    optimizer = torch.optim.AdamW([
        {"params": emb_params, "lr": lr * 10, "weight_decay": weight_decay * 5},
        {"params": attn_params, "lr": lr * 2},
        {"params": mlp_params, "lr": lr * 0.3},
        {"params": downstream_params, "lr": lr},
    ], weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_val = float("inf")
    patience, patience_limit = 0, 12
    start_epoch = 0

    if resume:
        _load_checkpoint(model, data_dir, "latest", device, version)
        train_state = _load_train_state(data_dir, "latest", device, version)
        if train_state is not None:
            optimizer.load_state_dict(train_state["optimizer"])
            if train_state["scheduler"] is not None:
                scheduler.load_state_dict(train_state["scheduler"])
            start_epoch = train_state["epoch"] + 1
            best_val = train_state["best_val"]
            patience = train_state["patience"]
            print(f"  Resumed from epoch {start_epoch} "
                  f"(best_val={best_val:.4f}, patience={patience})")
        else:
            print("  --resume specified but no checkpoint found, starting fresh")

    for epoch in range(start_epoch, max_epochs):
        if time.time() - t0 > time_limit * 0.85:
            print(f"  Time budget reached at epoch {epoch}")
            break

        model.train()
        train_ds.training = True
        ep_bce, ep_n = 0.0, 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_dl):
            logit, margin_pred, labels, margins = e2e_forward(
                batch, model, ptensors, device,
            )

            tgt = labels * (1 - label_smoothing) + 0.5 * label_smoothing
            if loss_fn == "brier":
                pred_prob = torch.sigmoid(logit)
                main_loss = F.mse_loss(pred_prob, tgt)
            else:
                main_loss = F.binary_cross_entropy_with_logits(logit, tgt)
            m_loss = F.mse_loss(margin_pred, margins)
            loss = (main_loss + margin_weight * m_loss) / accum_steps

            loss.backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # Log gradient norms on first accumulation step of select epochs
                if epoch % 10 == 0 and step < accum_steps:
                    _log_grad_norms(model, epoch)
                optimizer.step()
                optimizer.zero_grad()

            ep_bce += main_loss.item() * labels.size(0)
            ep_n += labels.size(0)

        scheduler.step()

        # validation
        model.eval()
        train_ds.training = False
        v_bce, v_acc, v_n = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_dl:
                logit, _, labels, _ = e2e_forward(
                    batch, model, ptensors, device,
                )
                if loss_fn == "brier":
                    v_loss = F.mse_loss(torch.sigmoid(logit), labels)
                else:
                    v_loss = F.binary_cross_entropy_with_logits(logit, labels)
                v_bce += v_loss.item() * labels.size(0)
                v_acc += ((logit > 0).float() == labels).sum().item()
                v_n += labels.size(0)

        v_bce /= max(v_n, 1)
        v_acc /= max(v_n, 1)
        elapsed = time.time() - t0

        if epoch % 5 == 0 or v_bce < best_val:
            print(f"  Epoch {epoch:3d} ({elapsed:.0f}s): "
                  f"train_bce={ep_bce/max(ep_n,1):.4f}  "
                  f"val_bce={v_bce:.4f}  val_acc={v_acc:.3f}  "
                  f"lr_emb={scheduler.get_last_lr()[0]:.6f}  "
                  f"lr_base={scheduler.get_last_lr()[3]:.6f}")

        if v_bce < best_val:
            best_val = v_bce
            patience = 0
            _save_checkpoint(model, data_dir, "best", p2i,
                             optimizer, scheduler, epoch, best_val, patience,
                             version=version)
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"  Early stopping at epoch {epoch}")
                break

        # Periodic checkpoint every 5 epochs for resumability
        if epoch % 5 == 0 or epoch == max_epochs - 1:
            _save_checkpoint(model, data_dir, "latest", p2i,
                             optimizer, scheduler, epoch, best_val, patience,
                             version=version)

    # reload best
    _load_checkpoint(model, data_dir, "best", device, version)

    total = time.time() - t0
    print(f"\n[3/3] Done! Best val_bce: {best_val:.4f}")
    print(f"  Total training time: {total:.0f}s ({total/3600:.1f}h)")

    return model, p2i, games, ts_games, ptensors


# =====================================================================
#  CHECKPOINT I/O
# =====================================================================

def _ckpt_dir(data_dir, version=None):
    if version:
        d = data_dir / "external" / "pbp_model" / version
    else:
        d = data_dir / "external" / "pbp_model"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_checkpoint(model, data_dir, tag, p2i,
                     optimizer=None, scheduler=None,
                     epoch=None, best_val=None, patience=None,
                     version=None):
    d = _ckpt_dir(data_dir, version)
    torch.save(model.state_dict(), d / f"pbp_{tag}.pt")
    with open(d / "player_index.json", "w") as f:
        json.dump({str(k): v for k, v in p2i.items()}, f)
    # save model config for reload
    pe = model.play_encoder
    cfg = dict(
        n_players=len(p2i),
        embed_dim=model.embed_dim,
        player_dim=pe.player_emb.embedding_dim,
        n_play_types=N_PLAY_TYPES,
        ptype_dim=pe.ptype_emb.embedding_dim,
        n_heads=model.season_encoder.enc.layers[0].self_attn.num_heads,
        n_season_layers=len(model.season_encoder.enc.layers),
    )
    with open(d / "config.json", "w") as f:
        json.dump(cfg, f)
    # save training state for resumption
    if optimizer is not None:
        train_state = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "best_val": best_val,
            "patience": patience,
        }
        torch.save(train_state, d / f"pbp_{tag}_train_state.pt")


def _load_checkpoint(model, data_dir, tag, device, version=None):
    path = _ckpt_dir(data_dir, version) / f"pbp_{tag}.pt"
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        return True
    # Fall back to unversioned dir
    path = _ckpt_dir(data_dir) / f"pbp_{tag}.pt"
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        return True
    return False


def _load_train_state(data_dir, tag, device, version=None):
    """Load optimizer/scheduler/epoch state for resumption. Returns dict or None."""
    path = _ckpt_dir(data_dir, version) / f"pbp_{tag}_train_state.pt"
    if path.exists():
        return torch.load(path, map_location=device, weights_only=False)
    return None


# =====================================================================
#  PREDICTION / SUBMISSION
# =====================================================================

def _build_team_name_map(data_dir: Path):
    """Build lowercase team name → Kaggle TeamID map.

    Uses MTeamSpellings.csv (many variants) + MTeams.csv + manual aliases
    for CBBD API names that differ from Kaggle names.
    """
    import pandas as pd
    name_to_id: dict[str, int] = {}

    sp = data_dir / "MTeamSpellings.csv"
    if sp.exists():
        for _, row in pd.read_csv(sp, encoding="latin-1").iterrows():
            name_to_id[str(row["TeamNameSpelling"]).lower().strip()] = row["TeamID"]

    mt = data_dir / "MTeams.csv"
    if mt.exists():
        for _, row in pd.read_csv(mt).iterrows():
            name_to_id[row["TeamName"].lower().strip()] = row["TeamID"]

    # CBBD API → Kaggle name aliases (names the API uses that differ from Kaggle)
    cbbd_aliases = {
        "queens university": "queens nc",
        "st johns": "st john's",
        "saint marys": "st mary's ca",
        "saint josephs": "St Joseph's PA",
        "saint peters": "St Peter's",
        "miami": "miami fl",
        "mount st marys": "Mt St Mary's",
        "app state": "Appalachian St",
        "san josé state": "San Jose St",
        "st thomas-minnesota": "St Thomas MN",
        "stephen f austin": "SF Austin",
        "ualbany": "Albany NY",
        "ul monroe": "UL Monroe",
        "ut rio grande valley": "UT Rio Grande Valley",
    }
    for cbbd_name, kaggle_name in cbbd_aliases.items():
        kaggle_lower = kaggle_name.lower().strip()
        if kaggle_lower in name_to_id:
            name_to_id[cbbd_name.lower().strip()] = name_to_id[kaggle_lower]

    return name_to_id


@torch.no_grad()
def generate_predictions(
    model, p2i, games, ts_games, ptensors, data_dir,
    prediction_season=2026, device="cpu",
):
    """Generate pairwise predictions for all tournament teams.

    Returns DataFrame with columns [ID, Pred].
    """
    import pandas as pd

    model.eval()
    model.to(device)

    # Compute game embeddings for prediction season
    season_ptensors = {
        k: v for k, v in ptensors.items()
        if games.get(k[0], {}).get("season") == prediction_season
    }
    game_embs = compute_game_embeddings(model, season_ptensors, device)

    # Build enriched season embeddings for each team
    D = model.embed_dim
    season_embs = {}
    for (team, season), gids in ts_games.items():
        if season != prediction_season:
            continue
        enriched_list = []
        for gid in gids:
            if (gid, team) not in game_embs:
                continue
            g = games[gid]
            opp = g["away_team"] if g["home_team"] == team else g["home_team"]
            our_emb = game_embs[(gid, team)]
            opp_emb = game_embs.get((gid, opp), torch.zeros(D))
            won = float((g["home_win"] == 1.0) == (g["home_team"] == team))
            margin = (g["home_score"] - g["away_score"]) / 20.0
            if g["home_team"] != team:
                margin = -margin
            outcome = torch.tensor([won, margin])
            enriched_list.append(torch.cat([our_emb, opp_emb, outcome]))
        if len(enriched_list) < 3:
            continue
        enriched_list = enriched_list[-20:]
        seq = torch.stack(enriched_list).unsqueeze(0).to(device)  # (1, G, 2D+2)
        se = model.encode_season_enriched(
            seq[:, :, :D], seq[:, :, D:2*D], seq[:, :, 2*D:],
        )
        season_embs[team] = se.squeeze(0).cpu()

    print(f"  {len(season_embs)} teams with season embeddings")

    # Map to Kaggle IDs (lowercase matching)
    name_to_id = _build_team_name_map(data_dir)

    # Build pairwise predictions
    teams = sorted(season_embs.keys())
    rows = []
    unmapped = []
    for i, ta in enumerate(teams):
        tid_a = name_to_id.get(ta.lower().strip())
        if tid_a is None:
            unmapped.append(ta)
            continue
        for tb in teams[i + 1:]:
            tid_b = name_to_id.get(tb.lower().strip())
            if tid_b is None:
                continue

            lo, hi = min(tid_a, tid_b), max(tid_a, tid_b)
            sa = season_embs[ta].unsqueeze(0).to(device)
            sb = season_embs[tb].unsqueeze(0).to(device)

            if lo == tid_a:
                logit, _ = model.predict(sa, sb)
            else:
                logit, _ = model.predict(sb, sa)

            pred = torch.sigmoid(logit).item()
            rows.append({"ID": f"{prediction_season}_{lo}_{hi}", "Pred": pred})

    if unmapped:
        print(f"  WARNING: {len(set(unmapped))} teams unmapped to Kaggle IDs")
        for t in sorted(set(unmapped))[:10]:
            print(f"    {t}")

    preds = pd.DataFrame(rows)

    # Merge with sample submission for complete coverage
    sample_path = data_dir / "SampleSubmissionStage2.csv"
    if not sample_path.exists():
        sample_path = data_dir / "SampleSubmissionStage1.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        preds = sample[["ID"]].merge(preds, on="ID", how="left")
        preds["Pred"] = preds["Pred"].fillna(0.5).clip(0.01, 0.99)
        print(f"  Merged with sample submission: {len(preds)} rows "
              f"({(preds['Pred'] != 0.5).sum()} predicted)")
    else:
        preds["Pred"] = preds["Pred"].clip(0.01, 0.99)

    return preds
