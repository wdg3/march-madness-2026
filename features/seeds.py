import re
from pathlib import Path
import pandas as pd
from features.base import FeatureSource


class SeedFeatures(FeatureSource):
    def name(self) -> str:
        return "seed"

    def build(self, data_dir: Path) -> pd.DataFrame:
        print("  Building seed features...")
        seeds = pd.read_csv(data_dir / "MNCAATourneySeeds.csv")
        seeds["seed_num"] = seeds["Seed"].apply(lambda s: int(re.findall(r"\d+", s)[0]))
        return seeds[["Season", "TeamID", "seed_num"]]
