from pathlib import Path
import pandas as pd
from features.base import FeatureSource


class ConferenceFeatures(FeatureSource):
    def name(self) -> str:
        return "conf"

    def build(self, data_dir: Path, gender: str = "M") -> pd.DataFrame:
        print("  Building conference features...")
        conf = pd.read_csv(data_dir / f"{gender}TeamConferences.csv")
        # Label encode conferences
        conf["conf_id"] = conf["ConfAbbrev"].astype("category").cat.codes
        return conf[["Season", "TeamID", "conf_id"]]
