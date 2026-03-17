"""Stub external feature sources — ready to implement when data sources are available.

These use the ExternalFeatureSource base class which fetches data from external
APIs/websites into data/external/<name>/ before building features.

To implement a stub:
1. Fill in fetch() to download data to self.external_data_dir(data_dir)
2. Fill in build() to read from that directory and return a DataFrame
3. Enable the feature name in config.py ENABLED_FEATURES
"""

from pathlib import Path
import pandas as pd
from features.base import ExternalFeatureSource


class VegasOddsFeatures(ExternalFeatureSource):
    """Pre-tournament futures odds and regular season ATS record.

    Data source ideas:
    - sportsoddshistory.com (historical futures)
    - covers.com/sport/basketball/ncaab/odds (ATS records)
    - Kaggle datasets with historical betting lines

    Features to build:
    - Championship futures odds (implied probability)
    - Regular season ATS record (wins vs spread)
    - Average margin vs spread (how often they cover)
    - Over/under tendencies
    """

    def name(self) -> str:
        return "vegas"

    def fetch(self, data_dir: Path) -> None:
        raise NotImplementedError(
            "Vegas odds fetcher not yet implemented. "
            "Place a CSV with columns [Season, TeamID, ...] "
            "at data/external/vegas/odds.csv"
        )

    def build(self, data_dir: Path, gender: str = "M") -> pd.DataFrame:
        csv_path = self.external_data_dir(data_dir) / "odds.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Vegas odds data not found at {csv_path}. "
                "Run with a Vegas data source or manually place the file."
            )
        return pd.read_csv(csv_path)


class RosterContinuityFeatures(ExternalFeatureSource):
    """Returning minutes, transfer portal impact, and roster experience.

    Data source ideas:
    - barttorvik.com (returning minutes %, transfer rankings)
    - 247sports.com (recruiting class rankings)
    - sports-reference.com (player stats by team)

    Features to build:
    - Percentage of minutes returning from prior season
    - Number of high-impact transfers (top 100)
    - Average years of experience on roster
    - Recruiting class ranking (talent proxy)
    """

    def name(self) -> str:
        return "roster"

    def fetch(self, data_dir: Path) -> None:
        raise NotImplementedError(
            "Roster continuity fetcher not yet implemented. "
            "Place a CSV with columns [Season, TeamID, ...] "
            "at data/external/roster/continuity.csv"
        )

    def build(self, data_dir: Path, gender: str = "M") -> pd.DataFrame:
        csv_path = self.external_data_dir(data_dir) / "continuity.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Roster data not found at {csv_path}. "
                "Run with a roster data source or manually place the file."
            )
        return pd.read_csv(csv_path)



# APPollTrajectoryFeatures has been replaced by APPollFeatures in features/kenpom.py
