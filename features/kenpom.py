"""KenPom and BartTorvik adjusted efficiency features from external Kaggle dataset.

Source: https://www.kaggle.com/datasets/nishaanamin/march-madness-data
These are opponent-adjusted stats that go beyond raw box scores — they account
for strength of schedule and pace, which our raw regular season features don't.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from features.base import ExternalFeatureSource


# Manual fixes for team names that don't match MTeamSpellings
NAME_OVERRIDES = {
    "Queens": "Queens NC",
}


def _build_name_to_id(data_dir: Path) -> dict:
    """Build lowercase team name -> TeamID mapping from MTeamSpellings + MTeams."""
    name_to_id = {}
    spellings = pd.read_csv(data_dir / "MTeamSpellings.csv", encoding="latin-1")
    for _, row in spellings.iterrows():
        name_to_id[str(row["TeamNameSpelling"]).lower().strip()] = row["TeamID"]
    teams = pd.read_csv(data_dir / "MTeams.csv")
    for _, row in teams.iterrows():
        name_to_id[row["TeamName"].lower().strip()] = row["TeamID"]
    return name_to_id


def _map_team_ids(ext_df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Map external dataset team names to our TeamID system."""
    name_to_id = _build_name_to_id(data_dir)
    ext_df = ext_df.copy()
    ext_df["_name"] = ext_df["TEAM"].apply(
        lambda x: NAME_OVERRIDES.get(x.strip(), x.strip())
    )
    ext_df["TeamID"] = ext_df["_name"].str.lower().str.strip().map(name_to_id)
    ext_df["Season"] = ext_df["YEAR"]
    return ext_df


class KenPomFeatures(ExternalFeatureSource):
    """KenPom/BartTorvik adjusted efficiency metrics.

    These are the gold standard for college basketball analytics — adjusted
    offensive and defensive efficiency, tempo, and derived metrics like
    BARTHAG (win probability vs average team).
    """

    def name(self) -> str:
        return "kenpom"

    def fetch(self, data_dir: Path) -> None:
        """Copy from kaggle_mm download to our external data dir."""
        src = data_dir / "external" / "kaggle_mm" / "KenPom Barttorvik.csv"
        if not src.exists():
            raise FileNotFoundError(
                f"KenPom data not found at {src}. "
                "Download from: kaggle datasets download -d nishaanamin/march-madness-data"
            )
        ext_dir = self.external_data_dir(data_dir)
        import shutil
        shutil.copy(src, ext_dir / "kenpom_barttorvik.csv")

    def build(self, data_dir: Path, gender: str = "M") -> pd.DataFrame:
        print("  Building KenPom/BartTorvik features...")
        if gender != "M":
            return pd.DataFrame(columns=["Season", "TeamID"])
        csv_path = self.external_data_dir(data_dir) / "kenpom_barttorvik.csv"
        df = pd.read_csv(csv_path)
        df = _map_team_ids(df, data_dir)

        # Drop rows we can't map
        unmapped = df["TeamID"].isna().sum()
        if unmapped > 0:
            print(f"    Warning: {unmapped} rows couldn't be mapped to TeamID")
        df = df.dropna(subset=["TeamID"])
        df["TeamID"] = df["TeamID"].astype(int)

        # Select the most valuable adjusted features
        result = pd.DataFrame({
            "Season": df["Season"],
            "TeamID": df["TeamID"],
            # KenPom adjusted metrics
            "kp_adj_tempo": df["KADJ T"],
            "kp_adj_off": df["KADJ O"],
            "kp_adj_def": df["KADJ D"],
            "kp_adj_em": df["KADJ EM"],
            # BartTorvik adjusted metrics
            "kp_badj_em": df["BADJ EM"],
            "kp_badj_off": df["BADJ O"],
            "kp_badj_def": df["BADJ D"],
            "kp_barthag": df["BARTHAG"],
            # Four factors (offense)
            "kp_efg_pct": df["EFG%"],
            "kp_tov_pct": df["TOV%"],
            "kp_oreb_pct": df["OREB%"],
            "kp_ftr": df["FTR"],
            # Four factors (defense)
            "kp_efg_pct_d": df["EFG%D"],
            "kp_tov_pct_d": df["TOV%D"],
            "kp_dreb_pct": df["DREB%"],
            "kp_ftrd": df["FTRD"],
            # Talent and experience
            "kp_talent": df["TALENT"],
            "kp_experience": df["EXP"],
            # Strength of schedule
            "kp_elite_sos": df["ELITE SOS"],
            "kp_wab": df["WAB"],  # Wins Above Bubble
        })

        return result


class APPollFeatures(ExternalFeatureSource):
    """AP Poll trajectory features — captures perception changes over the season."""

    def name(self) -> str:
        return "ap_poll"

    def fetch(self, data_dir: Path) -> None:
        src = data_dir / "external" / "kaggle_mm" / "AP Poll Data.csv"
        if not src.exists():
            raise FileNotFoundError(
                f"AP Poll data not found at {src}. "
                "Download from: kaggle datasets download -d nishaanamin/march-madness-data"
            )
        ext_dir = self.external_data_dir(data_dir)
        import shutil
        shutil.copy(src, ext_dir / "ap_poll.csv")

    def build(self, data_dir: Path, gender: str = "M") -> pd.DataFrame:
        print("  Building AP Poll trajectory features...")
        if gender != "M":
            return pd.DataFrame(columns=["Season", "TeamID"])
        csv_path = self.external_data_dir(data_dir) / "ap_poll.csv"
        df = pd.read_csv(csv_path)

        name_to_id = _build_name_to_id(data_dir)
        df["_name"] = df["TEAM"].apply(
            lambda x: NAME_OVERRIDES.get(str(x).strip(), str(x).strip())
        )
        df["TeamID"] = df["_name"].str.lower().str.strip().map(name_to_id)
        df["Season"] = df["YEAR"]
        df = df.dropna(subset=["TeamID"])
        df["TeamID"] = df["TeamID"].astype(int)

        # For each team-season, compute trajectory features
        rows = []
        for (season, team_id), group in df.groupby(["Season", "TeamID"]):
            group = group.sort_values("WEEK")
            ranks = group["AP RANK"].values
            votes = group["AP VOTES"].values
            weeks_ranked = (group["RANK?"] == 1).sum()
            total_weeks = len(group)

            preseason_rank = ranks[0] if len(ranks) > 0 else np.nan
            final_rank = ranks[-1] if len(ranks) > 0 else np.nan

            # Use votes for unranked teams (rank might be NaN but they could have votes)
            max_votes = votes.max() if len(votes) > 0 else 0

            rows.append({
                "Season": season,
                "TeamID": team_id,
                "ap_weeks_ranked": weeks_ranked,
                "ap_pct_weeks_ranked": weeks_ranked / max(total_weeks, 1),
                "ap_preseason_rank": preseason_rank,
                "ap_final_rank": final_rank,
                "ap_best_rank": np.nanmin(ranks) if len(ranks) > 0 else np.nan,
                "ap_rank_std": np.nanstd(ranks) if len(ranks) > 1 else 0,
                "ap_trajectory": (preseason_rank - final_rank) if not (np.isnan(preseason_rank) or np.isnan(final_rank)) else 0,
                "ap_max_votes": max_votes,
            })

        return pd.DataFrame(rows)


class PublicPicksFeatures(ExternalFeatureSource):
    """ESPN/public bracket pick percentages — crowd wisdom proxy.

    What percentage of public brackets pick this team to reach each round.
    This is essentially a free proxy for Vegas odds — it reflects public
    perception of team strength beyond what the seed alone conveys.
    """

    def name(self) -> str:
        return "public_picks"

    def fetch(self, data_dir: Path) -> None:
        src = data_dir / "external" / "kaggle_mm" / "Public Picks.csv"
        if not src.exists():
            raise FileNotFoundError(
                f"Public picks data not found at {src}. "
                "Download from: kaggle datasets download -d nishaanamin/march-madness-data"
            )
        ext_dir = self.external_data_dir(data_dir)
        import shutil
        shutil.copy(src, ext_dir / "public_picks.csv")

    def build(self, data_dir: Path, gender: str = "M") -> pd.DataFrame:
        print("  Building public picks features...")
        if gender != "M":
            return pd.DataFrame(columns=["Season", "TeamID"])
        csv_path = self.external_data_dir(data_dir) / "public_picks.csv"
        df = pd.read_csv(csv_path)

        name_to_id = _build_name_to_id(data_dir)
        df["_name"] = df["TEAM"].apply(
            lambda x: NAME_OVERRIDES.get(str(x).strip(), str(x).strip())
        )
        df["TeamID"] = df["_name"].str.lower().str.strip().map(name_to_id)
        df["Season"] = df["YEAR"]
        df = df.dropna(subset=["TeamID"])
        df["TeamID"] = df["TeamID"].astype(int)

        # Convert percentage strings to floats
        for col in ["R64", "R32", "S16", "E8", "F4", "FINALS"]:
            df[col] = df[col].astype(str).str.replace("%", "").astype(float) / 100

        result = pd.DataFrame({
            "Season": df["Season"],
            "TeamID": df["TeamID"],
            "pp_r64": df["R64"],
            "pp_r32": df["R32"],
            "pp_s16": df["S16"],
            "pp_e8": df["E8"],
            "pp_f4": df["F4"],
            "pp_finals": df["FINALS"],
        })

        return result
