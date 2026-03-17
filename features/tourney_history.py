from pathlib import Path
import pandas as pd
from features.base import FeatureSource


class TourneyHistoryFeatures(FeatureSource):
    def name(self) -> str:
        return "th"

    def build(self, data_dir: Path, gender: str = "M") -> pd.DataFrame:
        print("  Building tournament history features...")
        results = pd.read_csv(data_dir / f"{gender}NCAATourneyCompactResults.csv")
        seeds = pd.read_csv(data_dir / f"{gender}NCAATourneySeeds.csv")

        # Count wins per team per season
        wins = results.groupby(["Season", "WTeamID"]).size().reset_index(name="wins")
        wins = wins.rename(columns={"WTeamID": "TeamID"})

        # Get all team appearances (winners + losers)
        w = results[["Season", "WTeamID"]].rename(columns={"WTeamID": "TeamID"})
        l = results[["Season", "LTeamID"]].rename(columns={"LTeamID": "TeamID"})
        appearances = pd.concat([w, l]).drop_duplicates()
        appearances["appeared"] = 1

        # Merge to get per-season stats
        season_stats = appearances.merge(wins, on=["Season", "TeamID"], how="left")
        season_stats["wins"] = season_stats["wins"].fillna(0).astype(int)

        # For each target season, compute 5-year rolling lookback
        target_seasons = seeds[["Season", "TeamID"]].drop_duplicates()
        records = []

        for season in sorted(target_seasons["Season"].unique()):
            season_teams = target_seasons[target_seasons["Season"] == season]["TeamID"].values
            lookback = season_stats[
                (season_stats["Season"] >= season - 5)
                & (season_stats["Season"] < season)
            ]
            team_agg = lookback.groupby("TeamID").agg(
                th_appearances_5yr=("appeared", "sum"),
                th_wins_5yr=("wins", "sum"),
            ).reset_index()

            season_df = pd.DataFrame({"TeamID": season_teams})
            season_df["Season"] = season
            season_df = season_df.merge(team_agg, on="TeamID", how="left")
            season_df["th_appearances_5yr"] = season_df["th_appearances_5yr"].fillna(0).astype(int)
            season_df["th_wins_5yr"] = season_df["th_wins_5yr"].fillna(0).astype(int)
            records.append(season_df)

        return pd.concat(records, ignore_index=True)[["Season", "TeamID", "th_appearances_5yr", "th_wins_5yr"]]
