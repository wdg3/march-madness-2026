"""Seed-based prior for tournament predictions.

Computes historical P(TeamA wins | seed_A, seed_B) from past tournaments,
then blends with model predictions via log-odds weighting. This anchors
predictions in tournament reality — seeds encode massive expert knowledge
from the selection committee — while letting the model adjust for team-specific
quality differences.

Usage:
    prior = SeedPrior(data_dir, max_season=2025)
    prior.tune_alpha(model_probs, actuals)    # find optimal blend weight
    blended = prior.blend(model_probs, seed_a, seed_b)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_seed(seed_str: str) -> int:
    """Extract numeric seed from Kaggle seed string (e.g., 'W01', 'X16a' -> 1, 16)."""
    return int(re.findall(r"\d+", seed_str)[0])


def _logit(p: np.ndarray) -> np.ndarray:
    """Log-odds transform, clipped to avoid infinities."""
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


class SeedPrior:
    """Historical seed-matchup win probabilities for tournament games.

    Builds a lookup table P(lower seed wins | seed_A, seed_B) from all
    tournament games before a given season to avoid forward contamination.
    """

    def __init__(self, data_dir: Path, max_season: int, gender: str = "M"):
        """Build seed prior from tournaments in [first_available, max_season).

        Args:
            data_dir: Path to data directory with Kaggle CSVs.
            max_season: Exclusive upper bound — uses tournaments BEFORE this season.
            gender: 'M' or 'W'.
        """
        seeds = pd.read_csv(data_dir / f"{gender}NCAATourneySeeds.csv")
        seeds["seed_num"] = seeds["Seed"].apply(_parse_seed)
        seed_lookup = dict(zip(zip(seeds["Season"], seeds["TeamID"]), seeds["seed_num"]))

        results = pd.read_csv(data_dir / f"{gender}NCAATourneyCompactResults.csv")
        results = results[results["Season"] < max_season]

        # Build win counts by seed matchup
        wins = {}   # (seed_a, seed_b) -> wins for seed_a
        total = {}  # (seed_a, seed_b) -> total games

        for _, game in results.iterrows():
            s = game["Season"]
            w_seed = seed_lookup.get((s, game["WTeamID"]))
            l_seed = seed_lookup.get((s, game["LTeamID"]))
            if w_seed is None or l_seed is None:
                continue

            # Store both directions
            for sa, sb, won in [(w_seed, l_seed, True), (l_seed, w_seed, False)]:
                key = (sa, sb)
                if key not in total:
                    wins[key] = 0
                    total[key] = 0
                total[key] += 1
                if won:
                    wins[key] += 1

        # Compute probabilities with Laplace smoothing (add 1 win and 1 loss)
        self._prior = {}
        for key in total:
            self._prior[key] = (wins[key] + 1) / (total[key] + 2)

        self._seed_lookup = seed_lookup
        self._alpha = 0.5  # default blend weight, tuned later
        self.max_season = max_season

        n_matchups = len([k for k in total if k[0] <= k[1]])
        n_games = sum(v for k, v in total.items() if k[0] <= k[1])
        print(f"  Seed prior: {n_games} games, {n_matchups} unique matchups (seasons < {max_season})")

    @property
    def alpha(self) -> float:
        return self._alpha

    def get_prior(self, seed_a: int, seed_b: int) -> float:
        """P(team_a wins | seed_a, seed_b) from historical data."""
        if (seed_a, seed_b) in self._prior:
            return self._prior[(seed_a, seed_b)]
        # Fallback: simple logistic based on seed difference
        return _sigmoid(0.15 * (seed_b - seed_a))

    def get_seeds(self, season: int, team_a: int, team_b: int) -> tuple[int, int]:
        """Look up seeds for a matchup. Returns (seed_a, seed_b) or (None, None)."""
        sa = self._seed_lookup.get((season, team_a))
        sb = self._seed_lookup.get((season, team_b))
        return sa, sb

    def blend(self, model_prob: np.ndarray, seed_a: np.ndarray, seed_b: np.ndarray) -> np.ndarray:
        """Blend model predictions with seed prior via log-odds.

        logit(P_final) = alpha * logit(P_seed) + (1 - alpha) * logit(P_model)

        Where alpha controls how much the seed prior anchors the prediction.
        """
        seed_probs = np.array([self.get_prior(int(a), int(b)) for a, b in zip(seed_a, seed_b)])
        blended_logits = self._alpha * _logit(seed_probs) + (1 - self._alpha) * _logit(model_prob)
        return _sigmoid(blended_logits)

    def tune_alpha(
        self,
        model_probs: np.ndarray,
        seed_a: np.ndarray,
        seed_b: np.ndarray,
        actuals: np.ndarray,
    ) -> float:
        """Find optimal alpha by minimizing Brier score on validation data.

        Sweeps alpha from 0 to 1 and picks the value that produces the
        best-calibrated blended predictions.

        Args:
            model_probs: Raw model P(team_a wins) for validation games.
            seed_a: Seed of team A for each game.
            seed_b: Seed of team B for each game.
            actuals: Binary outcomes (1 = team A won).

        Returns:
            Optimal alpha value.
        """
        seed_probs = np.array([self.get_prior(int(a), int(b)) for a, b in zip(seed_a, seed_b)])
        seed_logits = _logit(seed_probs)
        model_logits = _logit(model_probs)

        best_alpha = 0.0
        best_brier = float("inf")

        for alpha_candidate in np.arange(0, 1.01, 0.01):
            blended = _sigmoid(alpha_candidate * seed_logits + (1 - alpha_candidate) * model_logits)
            brier = np.mean((blended - actuals) ** 2)
            if brier < best_brier:
                best_brier = brier
                best_alpha = alpha_candidate

        # Also report the extremes for context
        seed_only_brier = np.mean((seed_probs - actuals) ** 2)
        model_only_brier = np.mean((model_probs - actuals) ** 2)

        print(f"  Alpha tuning (on {len(actuals)} tournament games):")
        print(f"    Seed prior only  (α=1.0): Brier = {seed_only_brier:.4f}")
        print(f"    Model only       (α=0.0): Brier = {model_only_brier:.4f}")
        print(f"    Optimal blend    (α={best_alpha:.2f}): Brier = {best_brier:.4f}")

        self._alpha = best_alpha
        return best_alpha
