from pathlib import Path
import pandas as pd
from features.base import FeatureSource


class MasseyFeatures(FeatureSource):
    def name(self) -> str:
        return "massey"

    def build(self, data_dir: Path, gender: str = "M") -> pd.DataFrame:
        print("  Building Massey Ordinals features...")
        path = data_dir / f"{gender}MasseyOrdinals.csv"
        if not path.exists():
            return pd.DataFrame(columns=["Season", "TeamID"])
        rankings = pd.read_csv(path)

        # Keep only the last available ranking day per (Season, TeamID, SystemName)
        idx = rankings.groupby(["Season", "TeamID", "SystemName"])["RankingDayNum"].idxmax()
        latest = rankings.loc[idx]

        # Pivot: one column per ranking system
        pivoted = latest.pivot_table(
            index=["Season", "TeamID"],
            columns="SystemName",
            values="OrdinalRank",
            aggfunc="first",
        ).reset_index()

        # Prefix columns
        pivoted.columns = [
            f"massey_{c}" if c not in ("Season", "TeamID") else c
            for c in pivoted.columns
        ]

        return pivoted
