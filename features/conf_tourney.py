"""Conference tournament performance features from MConferenceTourneyGames.csv."""

from pathlib import Path
import pandas as pd
from features.base import FeatureSource


class ConfTourneyFeatures(FeatureSource):
    """How a team performed in their conference tournament right before March Madness."""

    def name(self) -> str:
        return "conf_tourney"

    def build(self, data_dir: Path, gender: str = "M") -> pd.DataFrame:
        print("  Building conference tournament features...")
        ct = pd.read_csv(data_dir / f"{gender}ConferenceTourneyGames.csv")

        # Count wins and losses per team per season
        wins = ct.groupby(["Season", "WTeamID"]).size().reset_index(name="ct_wins")
        wins.rename(columns={"WTeamID": "TeamID"}, inplace=True)

        losses = ct.groupby(["Season", "LTeamID"]).size().reset_index(name="ct_losses")
        losses.rename(columns={"LTeamID": "TeamID"}, inplace=True)

        # Find conference tournament champions: team with most wins that has no loss
        # in their conference. Actually simpler: the team that won the final game
        # (latest DayNum per conference per season)
        last_game_idx = ct.groupby(["Season", "ConfAbbrev"])["DayNum"].idxmax()
        champs = ct.loc[last_game_idx, ["Season", "WTeamID"]].copy()
        champs.rename(columns={"WTeamID": "TeamID"}, inplace=True)
        champs["ct_champion"] = 1

        # Merge
        # Start with all teams that appeared in conf tourney
        all_teams = pd.concat([
            ct[["Season", "WTeamID"]].rename(columns={"WTeamID": "TeamID"}),
            ct[["Season", "LTeamID"]].rename(columns={"LTeamID": "TeamID"}),
        ]).drop_duplicates()

        result = all_teams.merge(wins, on=["Season", "TeamID"], how="left")
        result = result.merge(losses, on=["Season", "TeamID"], how="left")
        result = result.merge(champs, on=["Season", "TeamID"], how="left")

        result["ct_wins"] = result["ct_wins"].fillna(0).astype(int)
        result["ct_losses"] = result["ct_losses"].fillna(0).astype(int)
        result["ct_champion"] = result["ct_champion"].fillna(0).astype(int)
        result["ct_games"] = result["ct_wins"] + result["ct_losses"]
        result["ct_win_pct"] = result["ct_wins"] / result["ct_games"].replace(0, 1)

        return result[["Season", "TeamID", "ct_wins", "ct_losses", "ct_champion", "ct_win_pct"]]
