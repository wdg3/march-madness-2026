"""Roster continuity features from BartTorvik player-level data.

Fetches player stats from barttorvik.com/getadvstats.php for consecutive years,
then computes returning minutes % and related roster stability metrics.

Features:
- roster_returning_min_pct: % of prior season's minutes played by returning players
- roster_new_min_pct: % of current season's minutes played by new players
- roster_avg_class: average class year (1=Fr, 2=So, 3=Jr, 4=Sr)
- roster_upperclass_min_pct: % of minutes played by Jr/Sr
"""

import io
import time
from pathlib import Path

import pandas as pd
import numpy as np

from features.base import ExternalFeatureSource

# BartTorvik player CSV columns (no header row)
# Verified from getadvstats.php?year=XXXX&csv=1
PLAYER_NAME_COL = 0
TEAM_NAME_COL = 1
GAMES_COL = 3
MIN_PCT_COL = 4  # % of team minutes played by this player
CLASS_COL = 25   # Fr, So, Jr, Sr
PLAYER_ID_COL = 32
YEAR_COL = 31

CLASS_TO_NUM = {"Fr": 1, "So": 2, "Jr": 3, "Sr": 4}

# Years where BartTorvik player data is available
FIRST_YEAR = 2008
LAST_YEAR = 2026


def _fetch_year(year: int) -> pd.DataFrame:
    """Fetch player-level stats for a single year from BartTorvik."""
    import urllib.request

    url = f"https://barttorvik.com/getadvstats.php?year={year}&csv=1"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")

    df = pd.read_csv(io.StringIO(raw), header=None)
    return pd.DataFrame({
        "player_name": df[PLAYER_NAME_COL].astype(str).str.strip().str.strip('"'),
        "team": df[TEAM_NAME_COL].astype(str).str.strip().str.strip('"'),
        "min_pct": pd.to_numeric(df[MIN_PCT_COL], errors="coerce"),
        "class": df[CLASS_COL].astype(str).str.strip().str.strip('"'),
        "player_id": pd.to_numeric(df[PLAYER_ID_COL], errors="coerce"),
        "year": year,
    })


def _build_name_to_id(data_dir: Path) -> dict:
    """Build lowercase team name -> TeamID mapping."""
    name_to_id = {}
    spellings = pd.read_csv(data_dir / "MTeamSpellings.csv", encoding="latin-1")
    for _, row in spellings.iterrows():
        name_to_id[str(row["TeamNameSpelling"]).lower().strip()] = row["TeamID"]
    teams = pd.read_csv(data_dir / "MTeams.csv")
    for _, row in teams.iterrows():
        name_to_id[row["TeamName"].lower().strip()] = row["TeamID"]
    return name_to_id


NAME_OVERRIDES = {
    "Queens": "Queens NC",
}


class RosterContinuityFeatures(ExternalFeatureSource):
    """Roster continuity and experience features from BartTorvik player data."""

    def name(self) -> str:
        return "roster"

    def fetch(self, data_dir: Path) -> None:
        """Fetch player stats from BartTorvik for all available years."""
        ext_dir = self.external_data_dir(data_dir)
        ext_dir.mkdir(parents=True, exist_ok=True)

        for year in range(FIRST_YEAR, LAST_YEAR + 1):
            cache_path = ext_dir / f"players_{year}.csv"
            if cache_path.exists():
                print(f"    Cached: {year}")
                continue
            print(f"    Fetching {year}...")
            try:
                df = _fetch_year(year)
                df.to_csv(cache_path, index=False)
                time.sleep(1)  # rate limit
            except Exception as e:
                print(f"    Failed {year}: {e}")

    def build(self, data_dir: Path, gender: str = "M") -> pd.DataFrame:
        print("  Building roster continuity features...")
        if gender != "M":
            return pd.DataFrame(columns=["Season", "TeamID"])

        ext_dir = self.external_data_dir(data_dir)
        name_to_id = _build_name_to_id(data_dir)

        # Load all years of player data
        all_players = []
        for year in range(FIRST_YEAR, LAST_YEAR + 1):
            cache_path = ext_dir / f"players_{year}.csv"
            if not cache_path.exists():
                continue
            df = pd.read_csv(cache_path)
            all_players.append(df)

        if not all_players:
            raise FileNotFoundError(
                f"No player data found in {ext_dir}. Run fetch first."
            )

        players = pd.concat(all_players, ignore_index=True)

        # Map team names to TeamIDs
        players["_name"] = players["team"].apply(
            lambda x: NAME_OVERRIDES.get(str(x).strip(), str(x).strip())
        )
        players["TeamID"] = players["_name"].str.lower().str.strip().map(name_to_id)
        players = players.dropna(subset=["TeamID", "min_pct"])
        players["TeamID"] = players["TeamID"].astype(int)

        rows = []
        for year in range(FIRST_YEAR + 1, LAST_YEAR + 1):
            curr = players[players["year"] == year]
            prev = players[players["year"] == year - 1]

            if curr.empty or prev.empty:
                continue

            for team_id, team_curr in curr.groupby("TeamID"):
                team_prev = prev[prev["TeamID"] == team_id]

                # Find returning players by player_id
                if team_prev.empty:
                    returning_min_pct = 0.0
                else:
                    prev_ids = set(team_prev["player_id"].dropna().astype(int))
                    # Players who were on this team last year
                    returning = team_curr[
                        team_curr["player_id"].dropna().astype(int).isin(prev_ids)
                    ]
                    # Also check for transfers: players who were on ANY team last year
                    all_prev_ids = set(prev["player_id"].dropna().astype(int))
                    curr_ids = set(team_curr["player_id"].dropna().astype(int))

                    # Returning min pct = sum of current min_pct for players
                    # who were on THIS team last year
                    returning_min_pct = returning["min_pct"].sum()

                # New players' share of minutes
                all_prev_ids_set = set(prev["player_id"].dropna().astype(int))
                curr_valid = team_curr.dropna(subset=["player_id"])
                new_players = curr_valid[
                    ~curr_valid["player_id"].astype(int).isin(all_prev_ids_set)
                ]
                new_min_pct = new_players["min_pct"].sum()

                # Class composition
                class_nums = team_curr["class"].map(CLASS_TO_NUM).dropna()
                avg_class = class_nums.mean() if len(class_nums) > 0 else np.nan

                # Upperclassmen minutes (Jr + Sr)
                upper = team_curr[team_curr["class"].isin(["Jr", "Sr"])]
                upperclass_min_pct = upper["min_pct"].sum()
                total_min = team_curr["min_pct"].sum()
                upperclass_frac = (
                    upperclass_min_pct / total_min if total_min > 0 else np.nan
                )

                rows.append({
                    "Season": year,
                    "TeamID": team_id,
                    "roster_returning_min_pct": min(returning_min_pct, 100.0),
                    "roster_new_min_pct": min(new_min_pct, 100.0),
                    "roster_avg_class": avg_class,
                    "roster_upperclass_min_frac": upperclass_frac,
                })

        return pd.DataFrame(rows)
