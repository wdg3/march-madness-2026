"""Play-by-play data fetcher for CBBD (CollegeBasketballData.com).

Fetches and caches raw play-by-play data with full on-court lineup
information (10 players per play) for use in the PBP matchup model.

API budget (free tier = 1,000 calls/month):
  - 2026 all teams: 365 calls
  - 2025 all teams: 364 calls
  - 2024 tournament teams: 68 calls
  - Overhead (teams/games endpoints): ~6 calls
  - Total: ~803 calls

Every response is cached to disk as JSON. The fetcher is fully
resumable — it skips any team-season already cached.

Usage:
    python run.py data fetch-pbp [--season 2025] [--team Duke]
"""

import json
import time
from pathlib import Path

import requests

CBBD_BASE = "https://api.collegebasketballdata.com"
CACHE_DIR_NAME = "pbp"  # relative to data/external/


def _cache_dir(data_dir: Path) -> Path:
    d = data_dir / "external" / CACHE_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path(data_dir: Path, season: int, team: str) -> Path:
    """Path for a cached team-season file."""
    safe_name = team.replace(" ", "_").replace(".", "").replace("'", "")
    return _cache_dir(data_dir) / f"plays_{season}_{safe_name}.json"


def _log_path(data_dir: Path) -> Path:
    return _cache_dir(data_dir) / "fetch_log.jsonl"


def _log_call(data_dir: Path, season: int, team: str, n_plays: int, status: int):
    """Append a line to the fetch log for budget tracking."""
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "season": season,
        "team": team,
        "n_plays": n_plays,
        "status": status,
    }
    with open(_log_path(data_dir), "a") as f:
        f.write(json.dumps(entry) + "\n")


def _count_calls(data_dir: Path) -> int:
    """Count total API calls made so far from the log."""
    log = _log_path(data_dir)
    if not log.exists():
        return 0
    with open(log) as f:
        return sum(1 for _ in f)


def _headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}"}


def get_teams(api_key: str, season: int) -> list[dict]:
    """Fetch all teams for a season. Returns list of team dicts."""
    r = requests.get(
        f"{CBBD_BASE}/teams",
        params={"season": season},
        headers=_headers(api_key),
    )
    r.raise_for_status()
    return r.json()


def get_tournament_teams(api_key: str, season: int) -> list[str]:
    """Fetch team names that appeared in the NCAA tournament."""
    r = requests.get(
        f"{CBBD_BASE}/games",
        params={"season": season, "tournament": "NCAA"},
        headers=_headers(api_key),
    )
    r.raise_for_status()
    games = r.json()
    teams = set()
    for g in games:
        teams.add(g["homeTeam"])
        teams.add(g["awayTeam"])
    return sorted(teams)


def fetch_team_plays(
    api_key: str,
    data_dir: Path,
    season: int,
    team: str,
    rate_limit: float = 1.0,
) -> list[dict]:
    """Fetch and cache all plays for a team-season.

    Returns plays from cache if already fetched. Otherwise makes one
    API call, caches the result, and returns it.

    Args:
        api_key: CBBD API key.
        data_dir: Data directory root.
        season: Season year (e.g. 2025).
        team: Team school name (e.g. "Duke", "UConn").
        rate_limit: Minimum seconds between API calls.
    """
    cache = _cache_path(data_dir, season, team)
    if cache.exists():
        with open(cache) as f:
            return json.load(f)

    # Rate limit
    time.sleep(rate_limit)

    r = requests.get(
        f"{CBBD_BASE}/plays/team",
        params={"season": season, "team": team},
        headers=_headers(api_key),
    )

    _log_call(data_dir, season, team, len(r.json()) if r.ok else 0, r.status_code)

    if r.status_code == 429:
        print(f"    Rate limited! Waiting 60s...")
        time.sleep(60)
        r = requests.get(
            f"{CBBD_BASE}/plays/team",
            params={"season": season, "team": team},
            headers=_headers(api_key),
        )
        _log_call(data_dir, season, team, len(r.json()) if r.ok else 0, r.status_code)

    r.raise_for_status()
    plays = r.json()

    # Cache immediately
    with open(cache, "w") as f:
        json.dump(plays, f)

    return plays


def fetch_all(
    api_key: str,
    data_dir: Path,
    rate_limit: float = 1.0,
):
    """Fetch all play-by-play data within our API budget.

    Fetch plan:
      - 2026: all D1 teams (365 calls)
      - 2025: all D1 teams (364 calls)
      - 2024: NCAA tournament teams only (68 calls)

    Fully resumable — skips cached team-seasons.
    """
    calls_before = _count_calls(data_dir)
    print(f"PBP fetch: {calls_before} API calls logged so far")

    # Phase 1: 2026 all teams (most important — prediction season)
    print("\n[1/3] Fetching 2026 all teams...")
    teams_2026 = get_teams(api_key, 2026)
    _log_call(data_dir, 2026, "_teams_endpoint", len(teams_2026), 200)
    team_names_2026 = [t["school"] for t in teams_2026]
    _fetch_team_list(api_key, data_dir, 2026, team_names_2026, rate_limit)

    # Phase 2: 2025 all teams (training + validation)
    print("\n[2/3] Fetching 2025 all teams...")
    teams_2025 = get_teams(api_key, 2025)
    _log_call(data_dir, 2025, "_teams_endpoint", len(teams_2025), 200)
    team_names_2025 = [t["school"] for t in teams_2025]
    _fetch_team_list(api_key, data_dir, 2025, team_names_2025, rate_limit)

    # Phase 3: 2024 tournament teams only
    print("\n[3/3] Fetching 2024 tournament teams...")
    tourney_2024 = get_tournament_teams(api_key, 2024)
    _log_call(data_dir, 2024, "_tourney_games_endpoint", len(tourney_2024), 200)
    _fetch_team_list(api_key, data_dir, 2024, tourney_2024, rate_limit)

    calls_after = _count_calls(data_dir)
    print(f"\nDone! {calls_after - calls_before} new API calls this run")
    print(f"Total calls logged: {calls_after}")


def _fetch_team_list(
    api_key: str,
    data_dir: Path,
    season: int,
    team_names: list[str],
    rate_limit: float,
):
    """Fetch plays for a list of teams, with progress and skip logic."""
    cached = 0
    fetched = 0
    errors = 0

    for i, team in enumerate(team_names):
        cache = _cache_path(data_dir, season, team)
        if cache.exists():
            cached += 1
            continue

        try:
            plays = fetch_team_plays(api_key, data_dir, season, team, rate_limit)
            fetched += 1
            n_with_floor = sum(
                1 for p in plays
                if p.get("onFloor") and len(p["onFloor"]) >= 10
            )
            if (fetched % 25 == 0) or (fetched <= 3):
                print(
                    f"    [{cached + fetched + errors}/{len(team_names)}] "
                    f"{team} {season}: {len(plays)} plays "
                    f"({n_with_floor} with lineup)"
                )
        except Exception as e:
            errors += 1
            print(f"    ERROR {team} {season}: {e}")

    print(
        f"  Season {season}: {cached} cached, {fetched} fetched, "
        f"{errors} errors, {len(team_names)} total"
    )


def fetch_status(data_dir: Path):
    """Print fetch status summary."""
    cache = _cache_dir(data_dir)
    if not cache.exists():
        print("  No PBP data fetched yet.")
        return

    calls = _count_calls(data_dir)
    print(f"  API calls logged: {calls}/1000")

    for season in [2024, 2025, 2026]:
        files = list(cache.glob(f"plays_{season}_*.json"))
        if files:
            total_plays = 0
            for f in files:
                with open(f) as fh:
                    total_plays += len(json.load(fh))
            print(f"  Season {season}: {len(files)} teams, {total_plays:,} plays")
        else:
            print(f"  Season {season}: not fetched")
