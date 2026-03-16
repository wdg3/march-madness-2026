"""Features derived from Massey Ordinals metadata: ranking disagreement and seed delta."""

from pathlib import Path
import re
import pandas as pd
import numpy as np
from features.base import FeatureSource


class RankingDisagreementFeatures(FeatureSource):
    """Standard deviation and range across all ranking systems for each team.

    High disagreement means the team is hard to evaluate — more upset potential.
    """

    def name(self) -> str:
        return "rank_disagree"

    def build(self, data_dir: Path) -> pd.DataFrame:
        print("  Building ranking disagreement features...")
        rankings = pd.read_csv(data_dir / "MMasseyOrdinals.csv")

        # Keep latest ranking per (Season, TeamID, SystemName)
        idx = rankings.groupby(["Season", "TeamID", "SystemName"])["RankingDayNum"].idxmax()
        latest = rankings.loc[idx]

        g = latest.groupby(["Season", "TeamID"])["OrdinalRank"]
        result = pd.DataFrame({
            "rd_rank_std": g.std(),
            "rd_rank_range": g.max() - g.min(),
            "rd_rank_mean": g.mean(),
            "rd_rank_median": g.median(),
            "rd_num_systems": g.count(),
        }).reset_index()

        return result


class SeedRankDeltaFeatures(FeatureSource):
    """Delta between tournament seed and average ranking system position.

    A low seed (e.g. 12) with a high average rank (e.g. 25th) is underseeded.
    """

    def name(self) -> str:
        return "seed_rank_delta"

    def build(self, data_dir: Path) -> pd.DataFrame:
        print("  Building seed-rank delta features...")
        # Get seeds
        seeds = pd.read_csv(data_dir / "MNCAATourneySeeds.csv")
        seeds["seed_num"] = seeds["Seed"].apply(lambda s: int(re.findall(r"\d+", s)[0]))

        # Get average ranking
        rankings = pd.read_csv(data_dir / "MMasseyOrdinals.csv")
        idx = rankings.groupby(["Season", "TeamID", "SystemName"])["RankingDayNum"].idxmax()
        latest = rankings.loc[idx]
        avg_rank = latest.groupby(["Season", "TeamID"])["OrdinalRank"].mean().reset_index()
        avg_rank.rename(columns={"OrdinalRank": "srd_avg_rank"}, inplace=True)

        # Merge and compute delta
        merged = seeds[["Season", "TeamID", "seed_num"]].merge(
            avg_rank, on=["Season", "TeamID"], how="left"
        )

        # Seed position in ranking terms: seed 1 ~ rank 1-4, seed 16 ~ rank 61-64
        # Simple mapping: expected rank ~ seed * 4
        merged["srd_expected_rank"] = merged["seed_num"] * 4
        merged["srd_delta"] = merged["srd_avg_rank"] - merged["srd_expected_rank"]
        # Negative delta = team is better than their seed suggests (underseeded)

        return merged[["Season", "TeamID", "srd_avg_rank", "srd_delta"]]
