"""Location and travel features: road/neutral performance and home-court dependence."""

from pathlib import Path
import pandas as pd
import numpy as np
from features.base import FeatureSource


class LocationFeatures(FeatureSource):
    """How a team performs away from home — tournament games are all neutral site.

    Teams with a big gap between home and away performance are more likely to
    underperform in the tournament. Teams that play well on neutral courts
    are better prepared for March.
    """

    def name(self) -> str:
        return "location"

    def build(self, data_dir: Path, gender: str = "M") -> pd.DataFrame:
        print("  Building location features...")
        df = pd.read_csv(data_dir / f"{gender}RegularSeasonDetailedResults.csv")

        # WLoc: H = winner was home, A = winner was away, N = neutral
        # Unpivot into per-team rows with location context
        winners = pd.DataFrame({
            "Season": df["Season"],
            "TeamID": df["WTeamID"],
            "Win": 1,
            "Margin": df["WScore"] - df["LScore"],
            "Loc": df["WLoc"],  # H/A/N from winner's perspective
        })
        losers = pd.DataFrame({
            "Season": df["Season"],
            "TeamID": df["LTeamID"],
            "Win": 0,
            "Margin": df["LScore"] - df["WScore"],
            "Loc": df["WLoc"].map({"H": "A", "A": "H", "N": "N"}),  # flip for loser
        })
        all_games = pd.concat([winners, losers], ignore_index=True)

        # Overall stats
        overall = all_games.groupby(["Season", "TeamID"]).agg(
            total_win_pct=("Win", "mean"),
            total_margin=("Margin", "mean"),
        ).reset_index()

        # Home stats
        home = all_games[all_games["Loc"] == "H"].groupby(["Season", "TeamID"]).agg(
            home_win_pct=("Win", "mean"),
            home_margin=("Margin", "mean"),
            home_games=("Win", "count"),
        ).reset_index()

        # Away stats
        away = all_games[all_games["Loc"] == "A"].groupby(["Season", "TeamID"]).agg(
            away_win_pct=("Win", "mean"),
            away_margin=("Margin", "mean"),
            away_games=("Win", "count"),
        ).reset_index()

        # Neutral stats (most similar to tournament conditions)
        neutral = all_games[all_games["Loc"] == "N"].groupby(["Season", "TeamID"]).agg(
            neutral_win_pct=("Win", "mean"),
            neutral_margin=("Margin", "mean"),
            neutral_games=("Win", "count"),
        ).reset_index()

        # Merge all
        result = overall.merge(home, on=["Season", "TeamID"], how="left")
        result = result.merge(away, on=["Season", "TeamID"], how="left")
        result = result.merge(neutral, on=["Season", "TeamID"], how="left")

        # Derived features
        # Home-court dependence: how much worse they are away vs home
        result["loc_home_away_gap"] = (
            result["home_win_pct"].fillna(0) - result["away_win_pct"].fillna(0)
        )
        result["loc_margin_home_away_gap"] = (
            result["home_margin"].fillna(0) - result["away_margin"].fillna(0)
        )

        # Tournament readiness: neutral site performance relative to overall
        result["loc_neutral_delta"] = (
            result["neutral_win_pct"].fillna(result["total_win_pct"])
            - result["total_win_pct"]
        )

        output_cols = [
            "Season", "TeamID",
            "loc_away_win_pct", "loc_away_margin",
            "loc_neutral_win_pct", "loc_neutral_margin",
            "loc_home_away_gap", "loc_margin_home_away_gap",
            "loc_neutral_delta",
        ]

        # Rename for output
        result.rename(columns={
            "away_win_pct": "loc_away_win_pct",
            "away_margin": "loc_away_margin",
            "neutral_win_pct": "loc_neutral_win_pct",
            "neutral_margin": "loc_neutral_margin",
        }, inplace=True)

        return result[output_cols]
