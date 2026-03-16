"""Coach tournament experience features from MTeamCoaches.csv + MNCAATourneyCompactResults.csv."""

from pathlib import Path
import pandas as pd
import numpy as np
from features.base import FeatureSource


class CoachFeatures(FeatureSource):
    """Coach career tournament wins, appearances, and years of experience."""

    def name(self) -> str:
        return "coach"

    def build(self, data_dir: Path) -> pd.DataFrame:
        print("  Building coach features...")
        coaches = pd.read_csv(data_dir / "MTeamCoaches.csv")
        tourney = pd.read_csv(data_dir / "MNCAATourneyCompactResults.csv")

        # Get the coach for each team-season (use the coach active at end of season)
        end_of_season = coaches.groupby(["Season", "TeamID"])["LastDayNum"].idxmax()
        active_coaches = coaches.loc[end_of_season, ["Season", "TeamID", "CoachName"]]

        # Build coach tournament history: for each (season, coach), compute
        # career tournament stats from *prior* seasons only
        # First, find all tournament appearances and wins per coach per season
        tourney_teams = set()
        for _, row in tourney.iterrows():
            tourney_teams.add((row["Season"], row["WTeamID"]))
            tourney_teams.add((row["Season"], row["LTeamID"]))

        # Map (season, team) -> coach
        team_coach = {}
        for _, row in active_coaches.iterrows():
            team_coach[(row["Season"], row["TeamID"])] = row["CoachName"]

        # Build per-coach-per-season tournament record
        coach_season_wins = {}  # (coach, season) -> wins
        coach_season_apps = {}  # (coach, season) -> 1 if appeared
        for _, row in tourney.iterrows():
            s = row["Season"]
            w_coach = team_coach.get((s, row["WTeamID"]))
            l_coach = team_coach.get((s, row["LTeamID"]))
            if w_coach:
                coach_season_wins[(w_coach, s)] = coach_season_wins.get((w_coach, s), 0) + 1
                coach_season_apps[(w_coach, s)] = 1
            if l_coach:
                coach_season_apps[(l_coach, s)] = 1

        # For each team-season, compute coach's career stats from prior seasons
        rows = []
        for _, row in active_coaches.iterrows():
            season = row["Season"]
            team_id = row["TeamID"]
            coach = row["CoachName"]

            career_wins = 0
            career_apps = 0
            for s in range(1985, season):  # all prior seasons
                career_wins += coach_season_wins.get((coach, s), 0)
                career_apps += coach_season_apps.get((coach, s), 0)

            # Years as head coach (any team) prior to this season
            prior = coaches[(coaches["CoachName"] == coach) & (coaches["Season"] < season)]
            years_exp = prior["Season"].nunique()

            rows.append({
                "Season": season,
                "TeamID": team_id,
                "coach_tourney_wins": career_wins,
                "coach_tourney_apps": career_apps,
                "coach_win_rate": career_wins / max(career_apps, 1),
                "coach_years_exp": years_exp,
            })

        return pd.DataFrame(rows)
