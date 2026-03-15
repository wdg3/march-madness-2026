from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd


class FeatureSource(ABC):
    """Base class for all feature sources.

    Each source loads its own raw data and returns a DataFrame
    with columns [Season, TeamID, ...features].
    Feature columns should be prefixed with the source name to avoid collisions.
    """

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def build(self, data_dir: Path) -> pd.DataFrame:
        ...
