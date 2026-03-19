"""BartTorvik adjusted efficiency features from pre-tournament snapshots.

Source: BartTorvik Time Machine (barttorvik.com/timemachine/)
Snapshots taken on Selection Sunday (before NCAA tournament begins) to avoid
tournament result contamination in adjusted metrics.

The previous Kaggle dataset (nishaanamin/march-madness-data) used post-tournament
BartTorvik ratings, which inflated KADJ EM by ~2-3 points for deep-run teams.
"""

import gzip
import io
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd
import numpy as np
from features.base import ExternalFeatureSource


# Manual fixes for team names that don't match MTeamSpellings
NAME_OVERRIDES = {
    "Arkansas Little Rock": "ualr",
    "Arkansas Pine Bluff": "Ark Pine Bluff",
    "Bethune Cookman": "Bethune-Cookman",
    "Cal St. Bakersfield": "CS Bakersfield",
    "Dixie St.": "Dixie St",
    "Illinois Chicago": "IL Chicago",
    "Louisiana Lafayette": "Louisiana",
    "Louisiana Monroe": "La-Monroe",
    "Mississippi Valley St.": "Mississippi Valley State",
    "Queens": "Queens NC",
    "Saint Francis": "St Francis PA",
    "Southeast Missouri St.": "Southeast Missouri",
    "St. Francis NY": "St Francis NY",
    "St. Francis PA": "St Francis PA",
    "Tarleton St.": "Tarleton St",
    "Tennessee Martin": "UT Martin",
    "Texas A&M Commerce": "East Texas A&M",
    "Texas A&M Corpus Chris": "A&M-Corpus Christi",
    "UT Rio Grande Valley": "UTRGV",
    "Winston Salem St.": "Winston-Salem",
}

# Selection Sunday dates by season (YYYYMMDD format)
# These are the last dates before the NCAA tournament begins
_SELECTION_SUNDAYS = {
    2010: "20100314",
    2011: "20110313",
    2012: "20120311",
    2013: "20130310",
    2014: "20140316",
    2015: "20150315",
    2016: "20160313",
    2017: "20170312",
    2018: "20180311",
    2019: "20190317",
    2021: "20210314",
    2022: "20220313",
    2023: "20230312",
    2024: "20240317",
    2025: "20250316",
    2026: "20260315",
}

# Column names for the BartTorvik time machine JSON (list of lists, no header)
_TORVIK_COLS = [
    "rank", "team", "conf", "record", "adjoe", "oe_rank", "adjde", "de_rank",
    "barthag", "rank2", "proj_w", "proj_l", "pro_con_w", "pro_con_l", "con_rec",
    "sos", "ncsos", "consos", "proj_sos", "proj_ncsos", "proj_consos",
    "elite_sos", "elite_ncsos", "opp_oe", "opp_de", "opp_proj_oe", "opp_proj_de",
    "con_adj_oe", "con_adj_de", "qual_o", "qual_d", "qual_barthag", "qual_games",
    "fun", "conpf", "conpa", "conposs", "conoe", "conde", "consos_remain",
    "conf_winpct", "wab", "wab_rk", "fun_rk", "adjt",
]


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


def _fetch_timemachine(date_str: str) -> pd.DataFrame:
    """Fetch a BartTorvik time machine snapshot for a given date."""
    url = f"https://barttorvik.com/timemachine/team_results/{date_str}_team_results.json.gz"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    data = gzip.decompress(raw)
    j = json.loads(data)
    return pd.DataFrame(j, columns=_TORVIK_COLS[:len(j[0])])


class KenPomFeatures(ExternalFeatureSource):
    """BartTorvik adjusted efficiency metrics from pre-tournament snapshots.

    Uses Selection Sunday snapshots from BartTorvik's Time Machine to ensure
    no NCAA tournament results contaminate the adjusted metrics.
    """

    def name(self) -> str:
        return "kenpom"

    def fetch(self, data_dir: Path) -> None:
        """Fetch pre-tournament BartTorvik snapshots for all available years."""
        ext_dir = self.external_data_dir(data_dir)
        ext_dir.mkdir(parents=True, exist_ok=True)

        for season, date_str in sorted(_SELECTION_SUNDAYS.items()):
            cache_path = ext_dir / f"barttorvik_{season}.csv"
            if cache_path.exists():
                print(f"    Cached: {season}")
                continue
            print(f"    Fetching {season} (Selection Sunday {date_str})...")
            try:
                df = _fetch_timemachine(date_str)
                df["season"] = season
                df.to_csv(cache_path, index=False)
                time.sleep(1)
            except Exception as e:
                print(f"    Failed {season}: {e}")

    def build(self, data_dir: Path, gender: str = "M") -> pd.DataFrame:
        print("  Building BartTorvik features (pre-tournament snapshots)...")
        if gender != "M":
            return pd.DataFrame(columns=["Season", "TeamID"])

        ext_dir = self.external_data_dir(data_dir)
        name_to_id = _build_name_to_id(data_dir)

        all_dfs = []
        for season in sorted(_SELECTION_SUNDAYS.keys()):
            cache_path = ext_dir / f"barttorvik_{season}.csv"
            if not cache_path.exists():
                continue
            df = pd.read_csv(cache_path)
            df["Season"] = season
            all_dfs.append(df)

        if not all_dfs:
            # Fall back to old Kaggle data if no time machine data available
            kaggle_path = ext_dir / "kenpom_barttorvik.csv"
            if kaggle_path.exists():
                print("    Warning: using Kaggle KenPom data (may contain post-tournament stats)")
                return self._build_from_kaggle(kaggle_path, data_dir)
            raise FileNotFoundError("No BartTorvik data found. Run fetch first.")

        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"    Loaded {len(combined)} team-seasons from {len(all_dfs)} years")

        # Map team names to TeamIDs
        combined["_name"] = combined["team"].apply(
            lambda x: NAME_OVERRIDES.get(str(x).strip(), str(x).strip())
        )
        combined["TeamID"] = combined["_name"].str.lower().str.strip().map(name_to_id)

        unmapped = combined["TeamID"].isna().sum()
        if unmapped > 0:
            print(f"    Warning: {unmapped} rows couldn't be mapped to TeamID")
        combined = combined.dropna(subset=["TeamID"])
        combined["TeamID"] = combined["TeamID"].astype(int)

        # Compute adjusted efficiency margin
        combined["adj_em"] = combined["adjoe"] - combined["adjde"]

        result = pd.DataFrame({
            "Season": combined["Season"],
            "TeamID": combined["TeamID"],
            "kp_adj_tempo": combined["adjt"],
            "kp_adj_off": combined["adjoe"],
            "kp_adj_def": combined["adjde"],
            "kp_adj_em": combined["adj_em"],
            "kp_barthag": combined["barthag"],
            "kp_elite_sos": combined["elite_sos"],
            "kp_wab": combined["wab"],
            "kp_sos": combined["sos"],
        })

        print(f"    Built {len(result)} rows, seasons {result['Season'].min()}-{result['Season'].max()}")
        return result

    def _build_from_kaggle(self, csv_path: Path, data_dir: Path) -> pd.DataFrame:
        """Legacy builder from Kaggle dataset (post-tournament, for fallback only)."""
        df = pd.read_csv(csv_path)
        name_to_id = _build_name_to_id(data_dir)
        df = df.copy()
        df["_name"] = df["TEAM"].apply(
            lambda x: NAME_OVERRIDES.get(x.strip(), x.strip())
        )
        df["TeamID"] = df["_name"].str.lower().str.strip().map(name_to_id)
        df["Season"] = df["YEAR"]
        df = df.dropna(subset=["TeamID"])
        df["TeamID"] = df["TeamID"].astype(int)

        return pd.DataFrame({
            "Season": df["Season"],
            "TeamID": df["TeamID"],
            "kp_adj_tempo": df["KADJ T"],
            "kp_adj_off": df["KADJ O"],
            "kp_adj_def": df["KADJ D"],
            "kp_adj_em": df["KADJ EM"],
            "kp_barthag": df["BARTHAG"],
            "kp_elite_sos": df["ELITE SOS"],
            "kp_wab": df["WAB"],
            "kp_sos": np.nan,
        })


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

        # Filter to pre-tournament weeks only (week 19 = post-conf-tourney, pre-NCAA)
        # Weeks 20+ contain NCAA tournament results
        df = df[df["WEEK"] <= 19].copy()

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
