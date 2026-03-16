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


class ExternalFeatureSource(FeatureSource):
    """Base class for feature sources that fetch data from external APIs/websites.

    Subclasses implement fetch() to download data to data/external/<name>/,
    and build() to construct features from the cached data.
    fetch() is called before build() and is skipped if data already exists
    (unless force_fetch=True).
    """

    def external_data_dir(self, data_dir: Path) -> Path:
        return data_dir / "external" / self.name()

    def is_fetched(self, data_dir: Path) -> bool:
        """Check if external data has already been downloaded."""
        ext_dir = self.external_data_dir(data_dir)
        return ext_dir.exists() and any(ext_dir.iterdir())

    @abstractmethod
    def fetch(self, data_dir: Path) -> None:
        """Download external data to data/external/<name>/."""
        ...

    def ensure_fetched(self, data_dir: Path, force: bool = False) -> None:
        """Fetch data if not already present."""
        if force or not self.is_fetched(data_dir):
            ext_dir = self.external_data_dir(data_dir)
            ext_dir.mkdir(parents=True, exist_ok=True)
            self.fetch(data_dir)
