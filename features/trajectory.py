"""Season trajectory features: windowed stats and linear trends for regular season performance.

Captures how a team's performance evolves throughout the season, beyond
full-season aggregates. Includes windowed comparisons (early/mid/late),
linear trend coefficients, and volatility measures.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from features.base import FeatureSource
from features.regular_season_advanced import _unpivot_games


def _add_efficiency(games: pd.DataFrame) -> pd.DataFrame:
    """Add per-game offensive efficiency to unpivoted games."""
    games = games.copy()
    poss = games["FGA"] - games["OR"] + games["TO"] + 0.475 * games["FTA"]
    games["OffEff"] = games["Score"] / poss.replace(0, np.nan) * 100
    opp_poss = games["OppFGA"] - games["OppOR"] + games["OppTO"] + 0.475 * games["OppFTA"]
    games["DefEff"] = games["OppScore"] / opp_poss.replace(0, np.nan) * 100
    games["FGPct"] = games["FGM"] / games["FGA"].replace(0, np.nan)
    return games


def _season_thirds(games: pd.DataFrame) -> pd.DataFrame:
    """Assign each game to early/mid/late third of the season."""
    games = games.copy()
    cuts = games.groupby(["Season", "TeamID"])["DayNum"].transform(
        lambda x: pd.qcut(x, q=3, labels=["early", "mid", "late"], duplicates="drop")
    )
    games["period"] = cuts
    return games


class RegularSeasonTrajectoryFeatures(FeatureSource):
    """Windowed and trend-based trajectory features from regular season games."""

    def name(self) -> str:
        return "rs_trajectory"

    def build(self, data_dir: Path, gender: str = "M") -> pd.DataFrame:
        print("  Building regular season trajectory features...")
        games = _unpivot_games(data_dir, gender)
        games = _add_efficiency(games)
        games = _season_thirds(games)

        metrics = ["Margin", "Win", "OffEff", "DefEff", "FGPct"]
        group_key = ["Season", "TeamID"]

        # --- Windowed stats (early vs late third) ---
        windowed = self._build_windowed(games, group_key, metrics)

        # --- Linear trend (slope over game sequence) ---
        trends = self._build_trends(games, group_key, metrics)

        # --- Volatility (rolling std and residual variance) ---
        volatility = self._build_volatility(games, group_key, metrics)

        result = windowed.merge(trends, on=group_key, how="outer")
        result = result.merge(volatility, on=group_key, how="outer")
        return result

    def _build_windowed(self, games, group_key, metrics):
        early = games[games["period"] == "early"].groupby(group_key)
        late = games[games["period"] == "late"].groupby(group_key)

        early_mean = early[metrics].mean()
        late_mean = late[metrics].mean()
        early_std = early[metrics].std()
        late_std = late[metrics].std()

        early_mean.columns = [f"traj_{m.lower()}_early" for m in metrics]
        late_mean.columns = [f"traj_{m.lower()}_late" for m in metrics]
        early_std.columns = [f"traj_{m.lower()}_early_std" for m in metrics]
        late_std.columns = [f"traj_{m.lower()}_late_std" for m in metrics]

        merged = early_mean.join(late_mean, how="outer")
        merged = merged.join(early_std, how="outer")
        merged = merged.join(late_std, how="outer").reset_index()

        # Deltas: late - early (positive = improving)
        for m in metrics:
            col_e = f"traj_{m.lower()}_early"
            col_l = f"traj_{m.lower()}_late"
            merged[f"traj_{m.lower()}_delta"] = merged[col_l] - merged[col_e]

        return merged

    def _build_trends(self, games, group_key, metrics):
        # Sort by day within each team-season, assign sequential game number
        games = games.sort_values(group_key + ["DayNum"])
        games["game_num"] = games.groupby(group_key).cumcount()

        def _slope(group, col):
            x = group["game_num"].values.astype(float)
            y = group[col].values.astype(float)
            mask = np.isfinite(y)
            if mask.sum() < 3:
                return np.nan
            x, y = x[mask], y[mask]
            x_mean = x.mean()
            denom = ((x - x_mean) ** 2).sum()
            if denom == 0:
                return 0.0
            return ((x - x_mean) * (y - y.mean())).sum() / denom

        rows = []
        for (season, team_id), group in games.groupby(group_key):
            row = {"Season": season, "TeamID": team_id}
            for m in metrics:
                row[f"traj_{m.lower()}_slope"] = _slope(group, m)
            rows.append(row)

        return pd.DataFrame(rows)

    def _build_volatility(self, games, group_key, metrics):
        """Residual volatility: std of detrended values (variance around trend line).

        High residual volatility means a team is inconsistent even after
        accounting for improvement/decline — a sign of unpredictability.
        """
        games = games.sort_values(group_key + ["DayNum"])
        games["game_num"] = games.groupby(group_key).cumcount()

        rows = []
        for (season, team_id), group in games.groupby(group_key):
            row = {"Season": season, "TeamID": team_id}
            x = group["game_num"].values.astype(float)
            for m in metrics:
                y = group[m].values.astype(float)
                mask = np.isfinite(y)
                if mask.sum() < 3:
                    row[f"traj_{m.lower()}_resid_std"] = np.nan
                    continue
                xm, ym = x[mask], y[mask]
                # Fit linear trend and compute residual std
                coeffs = np.polyfit(xm, ym, 1)
                predicted = np.polyval(coeffs, xm)
                row[f"traj_{m.lower()}_resid_std"] = float(np.std(ym - predicted))
            rows.append(row)

        return pd.DataFrame(rows)
