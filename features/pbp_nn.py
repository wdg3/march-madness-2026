"""PBP deep model features — season embeddings + matchup predictions.

Team-level features (from season encoder):
    pbp_emb_0..63: 64-dim learned season representation from play-by-play data.
    Pipeline creates _A, _B, _delta columns automatically.

Matchup-level features (from matchup head):
    pbp_pred: P(Team A wins) from the full PBP matchup model.
    Added at matchup construction time, like travel features.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from features.base import FeatureSource


class PBPExtractor:
    """Load a trained PBP model and extract embeddings / predictions."""

    def __init__(self, data_dir: Path, version: str | None = None):
        import json
        from models.pbp_train import (
            _ckpt_dir, _load_checkpoint, _build_team_name_map,
            load_pbp_data, compute_game_embeddings, N_PLAY_TYPES,
        )
        from models.pbp_model import PBPMatchupModel

        self._data_dir = data_dir
        self._name_to_id = _build_team_name_map(data_dir)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt = _ckpt_dir(data_dir, version)
        with open(ckpt / "config.json") as f:
            cfg = json.load(f)

        self._model = PBPMatchupModel(
            n_players=cfg["n_players"],
            embed_dim=cfg.get("embed_dim", 64),
            player_dim=cfg.get("player_dim", 8),
            n_play_types=cfg.get("n_play_types", N_PLAY_TYPES),
            ptype_dim=cfg.get("ptype_dim", 8),
            n_heads=cfg.get("n_heads", 4),
            n_season_layers=cfg.get("n_season_layers", 2),
        )
        _load_checkpoint(self._model, data_dir, "best", self._device, version)
        self._model.to(self._device)
        self._model.eval()

        # Load PBP data
        self._games, self._p2i, self._ts_games, self._ptensors = load_pbp_data(
            data_dir, seasons=[2024, 2025, 2026],
        )
        self._compute_game_embeddings = compute_game_embeddings
        self._game_embs_cache: dict[int, dict] = {}

    def _game_embs_for_season(self, season: int) -> dict:
        if season not in self._game_embs_cache:
            ptensors = {
                k: v for k, v in self._ptensors.items()
                if self._games.get(k[0], {}).get("season") == season
            }
            self._game_embs_cache[season] = self._compute_game_embeddings(
                self._model, ptensors, self._device,
            )
        return self._game_embs_cache[season]

    def _season_embedding(self, team: str, season: int, game_embs: dict):
        """Compute season embedding for one team."""
        D = self._model.embed_dim
        gids = self._ts_games.get((team, season), [])
        enriched_list = []
        for gid in gids:
            if (gid, team) not in game_embs:
                continue
            g = self._games[gid]
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
            return None
        enriched_list = enriched_list[-20:]
        seq = torch.stack(enriched_list).unsqueeze(0).to(self._device)
        se = self._model.encode_season_enriched(
            seq[:, :, :D], seq[:, :, D:2*D], seq[:, :, 2*D:],
        )
        return se.squeeze(0).cpu()

    @torch.no_grad()
    def team_embeddings(self, season: int) -> pd.DataFrame:
        """Return DataFrame [Season, TeamID, pbp_emb_0..D-1] for all teams."""
        game_embs = self._game_embs_for_season(season)
        D = self._model.embed_dim
        rows = []
        for (team, s) in self._ts_games:
            if s != season:
                continue
            tid = self._name_to_id.get(team.lower().strip())
            if tid is None:
                continue
            emb = self._season_embedding(team, season, game_embs)
            if emb is None:
                continue
            row = {"Season": season, "TeamID": tid}
            for i in range(D):
                row[f"pbp_emb_{i}"] = emb[i].item()
            rows.append(row)
        return pd.DataFrame(rows)

    def _build_season_cache(self, seasons: set[int]):
        """Build season embedding cache keyed by (Kaggle TeamID, season)."""
        # Map: (kaggle_id, season) → embedding
        tid_cache: dict[tuple[int, int], torch.Tensor] = {}
        for season in seasons:
            game_embs = self._game_embs_for_season(season)
            for (team, s) in self._ts_games:
                if s != season:
                    continue
                tid = self._name_to_id.get(team.lower().strip())
                if tid is None:
                    continue
                if (tid, season) not in tid_cache:
                    emb = self._season_embedding(team, season, game_embs)
                    if emb is not None:
                        tid_cache[(tid, season)] = emb

        def _get_emb(tid, season):
            return tid_cache.get((tid, season))

        return _get_emb

    @torch.no_grad()
    def matchup_embeddings(
        self, team_a_ids: np.ndarray, team_b_ids: np.ndarray,
        seasons: np.ndarray,
    ) -> np.ndarray:
        """Extract penultimate matchup embeddings (64-dim) for arrays of matchups.

        Returns (N, 64) array with NaN rows for unknown teams.
        """
        D = self._model.embed_dim
        unique_seasons = set(int(s) for s in seasons)
        _get_emb = self._build_season_cache(unique_seasons)

        result = np.full((len(team_a_ids), D), np.nan)
        for i in range(len(team_a_ids)):
            ea = _get_emb(int(team_a_ids[i]), int(seasons[i]))
            eb = _get_emb(int(team_b_ids[i]), int(seasons[i]))
            if ea is not None and eb is not None:
                sa = ea.unsqueeze(0).to(self._device)
                sb = eb.unsqueeze(0).to(self._device)
                emb = self._model.matchup_embedding(sa, sb)
                result[i] = emb.squeeze(0).cpu().numpy()
        return result


class PBPTeamFeatures(FeatureSource):
    """Placeholder — pbp_nn features are matchup-level, not team-level.

    This exists so the feature name can be listed in configs to enable
    the matchup-level embeddings via add_pbp_to_matchups().
    """

    def name(self) -> str:
        return "pbp_nn"

    def build(self, data_dir: Path, gender: str = "M") -> pd.DataFrame:
        # No team-level features — matchup embeddings are added in pipeline
        return pd.DataFrame(columns=["Season", "TeamID"])


def add_pbp_to_matchups(
    matchups: pd.DataFrame,
    data_dir: Path,
) -> pd.DataFrame:
    """Add PBP matchup embeddings (penultimate layer) to matchup rows.

    Adds 64 columns (pbp_mu_0..63): the hidden representation from the
    matchup head that captures how two teams interact, before it gets
    collapsed to a win probability scalar.
    """
    ckpt_path = data_dir / "external" / "pbp_model" / "pbp_best.pt"
    if not ckpt_path.exists():
        return matchups

    print("    Adding PBP matchup embeddings...")
    extractor = PBPExtractor(data_dir)

    team_a = matchups["TeamID_A"].values.astype(int)
    team_b = matchups["TeamID_B"].values.astype(int)
    seasons = matchups["Season"].values.astype(int)

    embs = extractor.matchup_embeddings(team_a, team_b, seasons)
    matchups = matchups.copy()
    D = embs.shape[1]
    for i in range(D):
        matchups[f"pbp_mu_{i}"] = embs[:, i]

    n_valid = np.isfinite(embs[:, 0]).sum()
    n_total = len(embs)
    print(f"    pbp_mu: {n_valid}/{n_total} valid, {D} dims")

    return matchups
