"""Fetch closing consensus odds from The Odds API for tournament teams.

Uses ESPN scoreboard API to get exact tipoff times for every game, then
fetches closing odds from the Odds API at the right timestamp for each game.

Usage:
    python fetch_odds.py --api-key YOUR_KEY
    python fetch_odds.py --api-key YOUR_KEY --resume    # skip already-fetched games

Output: data/external/odds_api/ncaab_2026_closing.csv
"""
import argparse
import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from features.vegas import _build_name_to_id, _API_NAME_OVERRIDES

DATA_DIR = Path("data")
OUTPUT_CSV = DATA_DIR / "external" / "odds_api" / "ncaab_2026_closing.csv"
CHECKPOINT_DIR = DATA_DIR / "external" / "odds_api" / "_fetch_checkpoint_v2"
ESPN_CACHE_DIR = DATA_DIR / "external" / "odds_api" / "_espn_cache"

BASE_URL = "https://api.the-odds-api.com"
SPORT = "basketball_ncaab"

REQUEST_DELAY = 0.5


def _resolve_api_name(api_name: str, name_to_id: dict) -> int | None:
    """Resolve an Odds API / ESPN team name to a Kaggle TeamID."""
    if api_name in _API_NAME_OVERRIDES:
        return name_to_id.get(_API_NAME_OVERRIDES[api_name].lower())
    words = api_name.lower().split()
    for i in range(len(words), 0, -1):
        attempt = " ".join(words[:i])
        if attempt in name_to_id:
            return name_to_id[attempt]
    return None


def _api_get(endpoint: str, params: dict, retries: int = 3) -> tuple[dict, dict]:
    """Make a GET request to the Odds API with retry on timeout."""
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{BASE_URL}{endpoint}?{query}"
    for attempt in range(retries):
        req = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                headers = {k.lower(): v for k, v in resp.getheaders()}
                body = json.loads(resp.read().decode())
                return body, headers
        except urllib.error.HTTPError as e:
            body = e.read().decode() if e.fp else ""
            raise RuntimeError(f"API error {e.code}: {body}") from e
        except (TimeoutError, urllib.error.URLError) as e:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    Timeout (attempt {attempt + 1}/{retries}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Timeout after {retries} attempts: {e}") from e


def _ml_to_implied_prob(ml: float) -> float:
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    else:
        return 100 / (ml + 100)


def _implied_prob_to_ml(prob: float) -> float:
    if prob >= 0.5:
        return -(prob / (1 - prob)) * 100
    else:
        return ((1 - prob) / prob) * 100


def _compute_consensus(event: dict) -> dict | None:
    """Compute consensus odds from bookmaker data for one event."""
    home = event["home_team"]
    away = event["away_team"]
    bookmakers = event.get("bookmakers", [])

    if len(bookmakers) < 3:
        return None

    home_mls, away_mls = [], []
    home_spreads, away_spreads = [], []
    totals = []

    for bk in bookmakers:
        for mkt in bk.get("markets", []):
            outcomes = {o["name"]: o for o in mkt["outcomes"]}
            if mkt["key"] == "h2h":
                if home in outcomes and away in outcomes:
                    home_mls.append(outcomes[home]["price"])
                    away_mls.append(outcomes[away]["price"])
            elif mkt["key"] == "spreads":
                if home in outcomes and away in outcomes:
                    home_spreads.append(outcomes[home]["point"])
                    away_spreads.append(outcomes[away]["point"])
            elif mkt["key"] == "totals":
                if "Over" in outcomes:
                    totals.append(outcomes["Over"]["point"])

    result = {
        "date": event["commence_time"],
        "home_team": home,
        "away_team": away,
        "event_id": event["id"],
    }

    if len(home_spreads) >= 3:
        result["home_point_spread"] = round(np.median(home_spreads) * 2) / 2
        result["away_point_spread"] = -result["home_point_spread"]
    else:
        result["home_point_spread"] = np.nan
        result["away_point_spread"] = np.nan

    if len(home_mls) >= 3:
        home_probs = [_ml_to_implied_prob(ml) for ml in home_mls]
        away_probs = [_ml_to_implied_prob(ml) for ml in away_mls]
        result["home_money_line"] = round(_implied_prob_to_ml(np.median(home_probs)))
        result["away_money_line"] = round(_implied_prob_to_ml(np.median(away_probs)))
    else:
        result["home_money_line"] = np.nan
        result["away_money_line"] = np.nan

    if len(totals) >= 3:
        result["over_under"] = round(np.median(totals) * 2) / 2
    else:
        result["over_under"] = np.nan

    result["n_bookmakers"] = len(bookmakers)
    return result


# ── ESPN tipoff time discovery ──────────────────────────────────────────────


def _fetch_espn_scoreboard(date_str: str) -> list[dict]:
    """Fetch ESPN scoreboard for a date (YYYYMMDD). Returns list of games."""
    cache_path = ESPN_CACHE_DIR / f"{date_str}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    url = (f"https://site.api.espn.com/apis/site/v2/sports/basketball/"
           f"mens-college-basketball/scoreboard?dates={date_str}&limit=200&groups=50")
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read())

    games = []
    for event in data.get("events", []):
        comp = event["competitions"][0]
        teams = comp["competitors"]
        home = next((t for t in teams if t["homeAway"] == "home"), None)
        away = next((t for t in teams if t["homeAway"] == "away"), None)
        if home and away:
            games.append({
                "commence_time": event["date"],
                "home_team": home["team"]["displayName"],
                "away_team": away["team"]["displayName"],
                "espn_id": event["id"],
            })

    ESPN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(games, f)

    return games


def _build_game_schedule(name_to_id: dict, tourney_ids: set) -> pd.DataFrame:
    """Build complete schedule of tournament-team games with ESPN tipoff times.

    Cross-references Kaggle game data with ESPN scoreboards to get exact
    commence_time for every game.
    """
    seasons = pd.read_csv(DATA_DIR / "MSeasons.csv")
    day_zero = pd.Timestamp(seasons[seasons["Season"] == 2026]["DayZero"].values[0])

    # Load all 2026 games from Kaggle (regular season + conference tourney)
    rs = pd.read_csv(DATA_DIR / "MRegularSeasonDetailedResults.csv")
    rs_2026 = rs[rs["Season"] == 2026].copy()

    ct = pd.read_csv(DATA_DIR / "MConferenceTourneyGames.csv")
    ct_2026 = ct[ct["Season"] == 2026].copy()
    # Conference tourney has same columns we need
    all_games = pd.concat([rs_2026, ct_2026], ignore_index=True)

    # Filter to tournament teams
    mask = all_games["WTeamID"].isin(tourney_ids) | all_games["LTeamID"].isin(tourney_ids)
    tourney_games = all_games[mask].copy()

    # Convert DayNum to date
    tourney_games["game_date"] = (day_zero + pd.to_timedelta(tourney_games["DayNum"], unit="D")).dt.date

    print(f"  Kaggle tournament-team games: {len(tourney_games)}")

    # Build team ID -> set of possible ESPN names (reverse lookup)
    # We don't need this — we'll match by team ID after resolving ESPN names

    # Fetch ESPN scoreboards for each unique date
    unique_dates = sorted(tourney_games["game_date"].unique())
    print(f"  Unique game dates: {len(unique_dates)}")

    # For each Kaggle game, find the ESPN match to get commence_time
    schedule = []
    matched = 0
    unmatched_games = []

    for date in unique_dates:
        date_str = pd.Timestamp(date).strftime("%Y%m%d")
        espn_games = _fetch_espn_scoreboard(date_str)

        # Build lookup: frozenset(home_id, away_id) -> ESPN game
        espn_lookup = {}
        for eg in espn_games:
            hid = _resolve_api_name(eg["home_team"], name_to_id)
            aid = _resolve_api_name(eg["away_team"], name_to_id)
            if hid and aid:
                espn_lookup[frozenset([hid, aid])] = eg

        # Match Kaggle games
        kaggle_day = tourney_games[tourney_games["game_date"] == date]
        for _, kg in kaggle_day.iterrows():
            pair = frozenset([kg["WTeamID"], kg["LTeamID"]])
            espn_match = espn_lookup.get(pair)
            if espn_match:
                schedule.append({
                    "w_team_id": kg["WTeamID"],
                    "l_team_id": kg["LTeamID"],
                    "day_num": kg["DayNum"],
                    "game_date": date,
                    "commence_time": espn_match["commence_time"],
                    "home_team": espn_match["home_team"],
                    "away_team": espn_match["away_team"],
                })
                matched += 1
            else:
                unmatched_games.append((date, kg["WTeamID"], kg["LTeamID"]))

    print(f"  ESPN matched: {matched}/{len(tourney_games)}")
    if unmatched_games:
        print(f"  Unmatched: {len(unmatched_games)}")

    return pd.DataFrame(schedule)


# ── Main fetch logic ────────────────────────────────────────────────────────


def _load_existing_games() -> set:
    """Load set of (home_team, away_team) pairs already fetched."""
    existing = set()
    if not CHECKPOINT_DIR.exists():
        return existing
    for f in CHECKPOINT_DIR.glob("*.json"):
        with open(f) as fh:
            for row in json.load(fh):
                existing.add(frozenset([row.get("home_team_id"), row.get("away_team_id")]))
    return existing


def _load_all_checkpoints() -> list[dict]:
    rows = []
    if not CHECKPOINT_DIR.exists():
        return rows
    for f in sorted(CHECKPOINT_DIR.glob("*.json")):
        with open(f) as fh:
            rows.extend(json.load(fh))
    return rows


def fetch_odds(api_key: str, resume: bool = False):
    """Main fetch loop."""
    seeds = pd.read_csv(DATA_DIR / "MNCAATourneySeeds.csv")
    tourney_ids = set(seeds[seeds["Season"] == 2026]["TeamID"].values)
    name_to_id = _build_name_to_id(DATA_DIR)

    print(f"Tournament teams: {len(tourney_ids)}")

    # Step 1: Build complete game schedule with ESPN tipoff times
    print("\nStep 1: Building game schedule from ESPN...")
    schedule = _build_game_schedule(name_to_id, tourney_ids)

    if schedule.empty:
        print("No games found!")
        return

    # Step 2: Identify games already fetched (if resuming)
    existing = _load_existing_games() if resume else set()
    if existing:
        schedule["_pair"] = schedule.apply(
            lambda r: frozenset([r["w_team_id"], r["l_team_id"]]), axis=1
        )
        before = len(schedule)
        schedule = schedule[~schedule["_pair"].isin(existing)].drop(columns=["_pair"])
        print(f"\nStep 2: {before - len(schedule)} games already fetched, {len(schedule)} remaining")
    else:
        print(f"\nStep 2: {len(schedule)} games to fetch")

    if schedule.empty:
        print("All games already fetched!")
        all_rows = _load_all_checkpoints()
    else:
        # Step 3: Group by tipoff window (10 min) and fetch closing odds
        print("\nStep 3: Fetching closing odds...")

        schedule["_ct"] = pd.to_datetime(schedule["commence_time"])
        schedule["_window"] = schedule["_ct"].dt.floor("10min")

        windows = schedule.groupby("_window")
        total_credits = 0
        total_fetched = 0
        all_rows = _load_all_checkpoints() if resume else []

        for window_time, group in sorted(windows, key=lambda x: x[0]):
            closing_time = window_time - timedelta(minutes=5)

            # Fetch odds at this snapshot — no eventIds filter, get everything
            try:
                odds_resp, odds_h = _api_get(
                    f"/v4/historical/sports/{SPORT}/odds",
                    {
                        "apiKey": api_key,
                        "date": closing_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "regions": "us",
                        "markets": "h2h,spreads,totals",
                        "oddsFormat": "american",
                        "dateFormat": "iso",
                    },
                )
            except RuntimeError as e:
                print(f"  ERROR at {closing_time}: {e}")
                time.sleep(2)
                continue

            credits_used = int(odds_h.get("x-requests-last", 30))
            total_credits += credits_used
            remaining = odds_h.get("x-requests-remaining", "?")
            time.sleep(REQUEST_DELAY)

            # Build lookup of API events by team pair
            api_events = {}
            for ev in odds_resp.get("data", []):
                hid = _resolve_api_name(ev["home_team"], name_to_id)
                aid = _resolve_api_name(ev["away_team"], name_to_id)
                if hid and aid:
                    api_events[frozenset([hid, aid])] = ev

            # Match our target games in this window
            window_rows = []
            found = 0
            missed = 0
            for _, game in group.iterrows():
                pair = frozenset([game["w_team_id"], game["l_team_id"]])
                ev = api_events.get(pair)
                if ev:
                    consensus = _compute_consensus(ev)
                    if consensus:
                        hid = _resolve_api_name(ev["home_team"], name_to_id)
                        aid = _resolve_api_name(ev["away_team"], name_to_id)
                        consensus["home_team_id"] = hid
                        consensus["away_team_id"] = aid
                        consensus["snapshot_time"] = odds_resp.get("timestamp", "")
                        consensus.pop("_warning", None)
                        window_rows.append(consensus)
                        found += 1
                    else:
                        missed += 1
                else:
                    missed += 1

            total_fetched += found

            # Save checkpoint
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            ck_name = window_time.strftime("%Y%m%d_%H%M")
            with open(CHECKPOINT_DIR / f"{ck_name}.json", "w") as f:
                json.dump(window_rows, f)
            all_rows.extend(window_rows)

            date_str = window_time.strftime("%Y-%m-%d %H:%M")
            print(f"  {date_str}: {len(group)} target, {found} found, {missed} missed  "
                  f"[credits: {total_credits}, remaining: {remaining}]")

        print(f"\nTotal credits used: {total_credits}")
        print(f"Total games fetched: {total_fetched}")

    # Step 4: Write final CSV
    if all_rows:
        df = pd.DataFrame(all_rows)
        out_cols = [
            "date", "home_team", "away_team", "home_team_id", "away_team_id",
            "home_point_spread", "away_point_spread",
            "home_money_line", "away_money_line", "over_under",
            "n_bookmakers", "event_id", "snapshot_time",
        ]
        df = df[[c for c in out_cols if c in df.columns]]
        # Deduplicate by team pair (in case of resume overlaps)
        df["_pair"] = df.apply(
            lambda r: frozenset([r.get("home_team_id"), r.get("away_team_id")]), axis=1
        )
        df = df.drop_duplicates(subset=["_pair", "date"]).drop(columns=["_pair"])
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nWrote {len(df)} games to {OUTPUT_CSV}")

    # Step 5: Verify against ground truth
    print("\n=== VERIFICATION ===")
    seeds_ids = set(seeds[seeds["Season"] == 2026]["TeamID"].values)
    rs = pd.read_csv(DATA_DIR / "MRegularSeasonDetailedResults.csv")
    ct = pd.read_csv(DATA_DIR / "MConferenceTourneyGames.csv")
    all_kaggle = pd.concat([rs[rs["Season"] == 2026], ct[ct["Season"] == 2026]])
    kaggle_tourney = all_kaggle[
        all_kaggle["WTeamID"].isin(seeds_ids) | all_kaggle["LTeamID"].isin(seeds_ids)
    ]
    total_needed = len(kaggle_tourney)
    total_got = len(df) if all_rows else 0
    print(f"Kaggle tournament-team games: {total_needed}")
    print(f"Odds fetched: {total_got}")
    print(f"Coverage: {total_got/total_needed*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Fetch closing consensus odds from The Odds API")
    parser.add_argument("--api-key", required=True, help="Odds API key")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    fetch_odds(args.api_key, resume=args.resume)


if __name__ == "__main__":
    main()
