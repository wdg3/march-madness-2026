"""Player-level impact features aggregated to team level.

Fetches full BartTorvik advanced stats (67 columns per player) and computes
team-level aggregates capturing star power, depth, roster composition, and
returning production. Complements the simpler roster continuity features.

Features (~20):
  Minutes-weighted means:  pi_mean_bpm, pi_mean_ortg, pi_mean_usg, pi_mean_efg
  Star concentration:      pi_top1_bpm, pi_top1_min_pct, pi_top3_min_pct, pi_max_usg
  Depth/variance:          pi_std_bpm, pi_depth_8th_min_pct
  Roster composition:      pi_avg_height, pi_height_std, pi_n_upperclass_top5,
                           pi_n_new_in_top8
  Returning production:    pi_returning_bpm_sum, pi_lost_bpm_sum,
                           pi_portal_incoming_bpm
"""

import io
import time
from pathlib import Path

import numpy as np
import pandas as pd

from features.base import ExternalFeatureSource

# BartTorvik CSV column indices (no header row, 67 columns)
COL = {
    "player_name": 0,
    "team": 1,
    "conf": 2,
    "games": 3,
    "min_pct": 4,
    "ortg": 5,
    "usg": 6,
    "efg": 7,
    "ts_pct": 8,
    "orb_pct": 9,
    "drb_pct": 10,
    "ast_pct": 11,
    "to_pct": 12,
    "class": 25,
    "height": 26,
    "bpm": 28,
    "year": 31,
    "player_id": 32,
    "position": 64,
}

CLASS_TO_NUM = {"Fr": 1, "So": 2, "Jr": 3, "Sr": 4}

FIRST_YEAR = 2008
LAST_YEAR = 2026


def _parse_height_inches(ht: str) -> float:
    """Convert '6-9' to 81.0 inches. Returns NaN on failure."""
    try:
        parts = str(ht).strip().strip('"').split("-")
        return int(parts[0]) * 12 + int(parts[1])
    except (ValueError, IndexError):
        return np.nan


def _fetch_year_full(year: int) -> pd.DataFrame:
    """Fetch all columns from BartTorvik for one year."""
    import urllib.request

    url = f"https://barttorvik.com/getadvstats.php?year={year}&csv=1"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")

    df = pd.read_csv(io.StringIO(raw), header=None)

    result = pd.DataFrame()
    for col_name, col_idx in COL.items():
        if col_idx < df.shape[1]:
            result[col_name] = df[col_idx]
        else:
            result[col_name] = np.nan

    # Clean string columns
    for col in ("player_name", "team", "class", "height", "position"):
        result[col] = result[col].astype(str).str.strip().str.strip('"')

    # Numeric conversions
    for col in ("games", "min_pct", "ortg", "usg", "efg", "ts_pct",
                "orb_pct", "drb_pct", "ast_pct", "to_pct", "bpm", "player_id"):
        result[col] = pd.to_numeric(result[col], errors="coerce")

    result["year"] = year
    result["height_inches"] = result["height"].apply(_parse_height_inches)
    result["class_num"] = result["class"].map(CLASS_TO_NUM)

    return result


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


# Team names that barttorvik uses differently from MTeamSpellings
NAME_OVERRIDES = {
    "Queens": "Queens NC",
}


def _weighted_mean(values, weights):
    """Minutes-weighted mean, handling edge cases."""
    mask = ~(np.isnan(values) | np.isnan(weights))
    if mask.sum() == 0:
        return np.nan
    return np.average(values[mask], weights=weights[mask])


class PlayerImpactFeatures(ExternalFeatureSource):
    """Team-level features derived from player-level BartTorvik advanced stats."""

    def name(self) -> str:
        return "player_impact"

    def fetch(self, data_dir: Path) -> None:
        """Fetch full player stats from BartTorvik."""
        ext_dir = self.external_data_dir(data_dir)
        ext_dir.mkdir(parents=True, exist_ok=True)

        for year in range(FIRST_YEAR, LAST_YEAR + 1):
            cache_path = ext_dir / f"players_full_{year}.csv"
            if cache_path.exists():
                print(f"    Cached: {year}")
                continue
            print(f"    Fetching {year}...")
            try:
                df = _fetch_year_full(year)
                df.to_csv(cache_path, index=False)
                time.sleep(1)
            except Exception as e:
                print(f"    Failed {year}: {e}")

    def build(self, data_dir: Path, gender: str = "M") -> pd.DataFrame:
        print("  Building player impact features...")
        if gender != "M":
            return pd.DataFrame(columns=["Season", "TeamID"])

        ext_dir = self.external_data_dir(data_dir)
        name_to_id = _build_name_to_id(data_dir)

        all_players = []
        for year in range(FIRST_YEAR, LAST_YEAR + 1):
            cache_path = ext_dir / f"players_full_{year}.csv"
            if not cache_path.exists():
                continue
            df = pd.read_csv(cache_path)
            all_players.append(df)

        if not all_players:
            raise FileNotFoundError(
                f"No player data in {ext_dir}. Run fetch first."
            )

        players = pd.concat(all_players, ignore_index=True)

        # Map team names to TeamIDs
        players["_name"] = players["team"].apply(
            lambda x: NAME_OVERRIDES.get(str(x).strip(), str(x).strip())
        )
        players["TeamID"] = (
            players["_name"].str.lower().str.strip().map(name_to_id)
        )
        players = players.dropna(subset=["TeamID", "min_pct"])
        players["TeamID"] = players["TeamID"].astype(int)

        # Build prior-year lookup for returning production
        prior_lookup = {}
        for year in players["year"].unique():
            yr_data = players[players["year"] == year]
            for tid, grp in yr_data.groupby("TeamID"):
                prior_lookup[(int(tid), int(year))] = grp

        rows = []
        for year in range(FIRST_YEAR + 1, LAST_YEAR + 1):
            curr_all = players[players["year"] == year]
            prev_all = players[players["year"] == year - 1]

            if curr_all.empty:
                continue

            all_prev_ids = set(prev_all["player_id"].dropna().astype(int))

            for team_id, team_curr in curr_all.groupby("TeamID"):
                row = self._compute_team_features(
                    team_curr, team_id, year, prev_all, all_prev_ids,
                    prior_lookup,
                )
                rows.append(row)

        result = pd.DataFrame(rows)
        print(f"    {len(result)} team-seasons, {len(result.columns)-2} features")
        return result

    def _compute_team_features(
        self, team_curr, team_id, year, prev_all, all_prev_ids, prior_lookup,
    ) -> dict:
        """Compute all player impact features for one team-season."""
        # Sort by minutes, take top 8 for rotation
        top8 = team_curr.nlargest(8, "min_pct")
        top5 = top8.head(5)
        top3 = top8.head(3)
        top1 = top8.head(1)

        mins = top8["min_pct"].values
        bpms = top8["bpm"].values

        row = {
            "Season": year,
            "TeamID": int(team_id),
        }

        # --- Minutes-weighted means (top 8) ---
        for col in ("bpm", "ortg", "usg", "efg"):
            vals = top8[col].values.astype(float)
            row[f"pi_mean_{col}"] = _weighted_mean(vals, mins)

        # --- Star concentration ---
        row["pi_top1_bpm"] = float(top1["bpm"].iloc[0]) if len(top1) > 0 else np.nan
        row["pi_top1_min_pct"] = float(top1["min_pct"].iloc[0]) if len(top1) > 0 else np.nan
        row["pi_top3_min_pct"] = float(top3["min_pct"].sum()) if len(top3) > 0 else np.nan
        row["pi_max_usg"] = float(top8["usg"].max()) if len(top8) > 0 else np.nan

        # --- Depth / variance ---
        row["pi_std_bpm"] = float(np.nanstd(bpms)) if len(bpms) > 1 else np.nan
        row["pi_depth_8th_min_pct"] = (
            float(top8["min_pct"].iloc[-1]) if len(top8) >= 8 else np.nan
        )

        # --- Roster composition ---
        ht = top5["height_inches"].dropna()
        row["pi_avg_height"] = float(ht.mean()) if len(ht) > 0 else np.nan
        row["pi_height_std"] = float(ht.std()) if len(ht) > 1 else np.nan

        upper = top5["class"].isin(["Jr", "Sr"])
        row["pi_n_upperclass_top5"] = int(upper.sum())

        # New players in top 8 (not on any team last year)
        curr_ids = top8["player_id"].dropna().astype(int)
        n_new = int((~curr_ids.isin(all_prev_ids)).sum())
        row["pi_n_new_in_top8"] = n_new

        # --- Returning production ---
        prev_team = prior_lookup.get((int(team_id), year - 1), pd.DataFrame())

        if not prev_team.empty:
            prev_ids = set(prev_team["player_id"].dropna().astype(int))
            prev_bpm_by_id = dict(
                zip(
                    prev_team["player_id"].dropna().astype(int),
                    prev_team["bpm"],
                )
            )
            prev_min_by_id = dict(
                zip(
                    prev_team["player_id"].dropna().astype(int),
                    prev_team["min_pct"],
                )
            )

            # Returning: players on this team in both years
            curr_valid = team_curr.dropna(subset=["player_id"])
            curr_pids = set(curr_valid["player_id"].astype(int))
            returning_ids = curr_pids & prev_ids

            # Min-weighted BPM of returning players (using prior year stats)
            ret_bpm_sum = sum(
                prev_bpm_by_id.get(pid, 0) * prev_min_by_id.get(pid, 0) / 100
                for pid in returning_ids
            )
            row["pi_returning_bpm_sum"] = ret_bpm_sum

            # Lost: on team last year, not on team this year
            lost_ids = prev_ids - curr_pids
            lost_bpm_sum = sum(
                prev_bpm_by_id.get(pid, 0) * prev_min_by_id.get(pid, 0) / 100
                for pid in lost_ids
            )
            row["pi_lost_bpm_sum"] = lost_bpm_sum

            # Portal incoming: on team now, was on a DIFFERENT team last year
            portal_incoming_ids = set()
            for pid in curr_pids:
                if pid in all_prev_ids and pid not in prev_ids:
                    portal_incoming_ids.add(pid)

            # Use current year BPM for portal players
            portal_bpm = 0.0
            for pid in portal_incoming_ids:
                p = curr_valid[curr_valid["player_id"].astype(int) == pid]
                if len(p) > 0:
                    portal_bpm += float(p["bpm"].iloc[0]) * float(p["min_pct"].iloc[0]) / 100
            row["pi_portal_incoming_bpm"] = portal_bpm
        else:
            row["pi_returning_bpm_sum"] = np.nan
            row["pi_lost_bpm_sum"] = np.nan
            row["pi_portal_incoming_bpm"] = np.nan

        return row
