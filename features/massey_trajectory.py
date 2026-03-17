"""Massey Ordinals trajectory features: ranking trends, volatility, and convergence.

Instead of only using the final ranking day, this captures how a team's
rankings evolve throughout the season across all ranking systems.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from features.base import FeatureSource


class MasseyTrajectoryFeatures(FeatureSource):
    """Trend and volatility features from Massey Ordinal rankings over time."""

    def name(self) -> str:
        return "massey_trajectory"

    def build(self, data_dir: Path, gender: str = "M") -> pd.DataFrame:
        print("  Building Massey trajectory features...")
        path = data_dir / f"{gender}MasseyOrdinals.csv"
        if not path.exists():
            return pd.DataFrame(columns=["Season", "TeamID"])
        rankings = pd.read_csv(path)

        group_key = ["Season", "TeamID"]

        # Compute mean rank across all systems for each ranking day
        daily_mean = (
            rankings.groupby(group_key + ["RankingDayNum"])["OrdinalRank"]
            .mean()
            .reset_index()
            .rename(columns={"OrdinalRank": "MeanRank"})
        )

        # --- Per-system trends aggregated ---
        system_trends = self._build_system_trends(rankings, group_key)

        # --- Aggregate ranking trajectory (mean across systems) ---
        agg_trends = self._build_aggregate_trends(daily_mean, group_key)

        # --- Cross-system volatility over time ---
        convergence = self._build_convergence(rankings, group_key)

        result = system_trends.merge(agg_trends, on=group_key, how="outer")
        result = result.merge(convergence, on=group_key, how="outer")
        return result

    def _build_system_trends(self, rankings, group_key):
        """Mean and std of per-system slopes across all ranking systems."""
        slopes = []
        for (season, team_id, system), group in rankings.groupby(
            group_key + ["SystemName"]
        ):
            if len(group) < 3:
                continue
            x = group["RankingDayNum"].values.astype(float)
            y = group["OrdinalRank"].values.astype(float)
            x_mean = x.mean()
            denom = ((x - x_mean) ** 2).sum()
            if denom == 0:
                continue
            slope = ((x - x_mean) * (y - y.mean())).sum() / denom
            slopes.append({
                "Season": season, "TeamID": team_id, "slope": slope,
            })

        if not slopes:
            return pd.DataFrame(columns=group_key)

        df = pd.DataFrame(slopes)
        result = df.groupby(group_key)["slope"].agg(
            mt_system_slope_mean="mean",
            mt_system_slope_std="std",
            mt_system_slope_median="median",
        ).reset_index()

        # Negative slope = improving (rank number going down)
        return result

    def _build_aggregate_trends(self, daily_mean, group_key):
        """Slope, volatility, and windowed stats on the mean rank across systems."""
        rows = []
        for (season, team_id), group in daily_mean.groupby(group_key):
            group = group.sort_values("RankingDayNum")
            x = group["RankingDayNum"].values.astype(float)
            y = group["MeanRank"].values.astype(float)

            row = {"Season": season, "TeamID": team_id}

            if len(group) < 3:
                row.update({
                    "mt_rank_slope": np.nan,
                    "mt_rank_resid_std": np.nan,
                    "mt_rank_early": np.nan,
                    "mt_rank_late": np.nan,
                    "mt_rank_delta": np.nan,
                    "mt_rank_best": np.nan,
                    "mt_rank_worst": np.nan,
                    "mt_rank_range": np.nan,
                })
                rows.append(row)
                continue

            # Linear trend
            coeffs = np.polyfit(x, y, 1)
            row["mt_rank_slope"] = coeffs[0]

            # Residual volatility
            predicted = np.polyval(coeffs, x)
            row["mt_rank_resid_std"] = float(np.std(y - predicted))

            # Windowed: first third vs last third
            n = len(y)
            third = max(n // 3, 1)
            row["mt_rank_early"] = float(y[:third].mean())
            row["mt_rank_late"] = float(y[-third:].mean())
            row["mt_rank_delta"] = row["mt_rank_late"] - row["mt_rank_early"]

            # Extremes
            row["mt_rank_best"] = float(y.min())
            row["mt_rank_worst"] = float(y.max())
            row["mt_rank_range"] = row["mt_rank_worst"] - row["mt_rank_best"]

            rows.append(row)

        return pd.DataFrame(rows)

    def _build_convergence(self, rankings, group_key):
        """How much ranking systems converge (or diverge) on a team over the season.

        Early-season rankings have high disagreement; late-season should converge.
        Teams where systems still disagree late are harder to evaluate.
        """
        # Cross-system std at each ranking day
        daily_std = (
            rankings.groupby(group_key + ["RankingDayNum"])["OrdinalRank"]
            .std()
            .reset_index()
            .rename(columns={"OrdinalRank": "RankStd"})
        )

        rows = []
        for (season, team_id), group in daily_std.groupby(group_key):
            group = group.sort_values("RankingDayNum")
            y = group["RankStd"].values.astype(float)

            row = {"Season": season, "TeamID": team_id}
            n = len(y)

            if n < 3:
                row["mt_convergence_slope"] = np.nan
                row["mt_disagree_early"] = np.nan
                row["mt_disagree_late"] = np.nan
                row["mt_disagree_delta"] = np.nan
            else:
                x = group["RankingDayNum"].values.astype(float)
                x_mean = x.mean()
                denom = ((x - x_mean) ** 2).sum()
                row["mt_convergence_slope"] = (
                    ((x - x_mean) * (y - y.mean())).sum() / denom if denom > 0 else 0.0
                )
                third = max(n // 3, 1)
                row["mt_disagree_early"] = float(y[:third].mean())
                row["mt_disagree_late"] = float(y[-third:].mean())
                row["mt_disagree_delta"] = row["mt_disagree_late"] - row["mt_disagree_early"]

            rows.append(row)

        return pd.DataFrame(rows)
