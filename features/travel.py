"""Matchup-level travel distance features.

For each potential game between two teams, computes the distance each team
would travel to the game venue. This is a matchup-level feature (not team-level)
because the venue depends on WHERE in the bracket two teams would meet.

For training data: we know the exact city from MGameCities.csv.
For prediction data: we determine the venue from the bracket structure
(seeds -> slot -> round -> venue city).

Uses cached geocoded coordinates from OpenStreetMap Nominatim.
"""

import json
import time
from math import radians, cos, sin, asin, sqrt
from pathlib import Path

import pandas as pd
import numpy as np

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles between two points."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 3956 * asin(sqrt(a))  # 3956 = Earth radius in miles


def geocode_nominatim(query, session):
    """Geocode a location string using Nominatim. Returns (lat, lng) or None."""
    try:
        resp = session.get(
            NOMINATIM_URL,
            params={"q": query, "format": "json", "limit": 1, "countrycodes": "us"},
            headers={"User-Agent": "MarchMadness2026/1.0"},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
    except Exception:
        pass
    return None


def ensure_geocoded(data_dir: Path) -> None:
    """Geocode all schools and venue cities if not already cached."""
    ext_dir = data_dir / "external" / "travel"
    ext_dir.mkdir(parents=True, exist_ok=True)

    school_cache_path = ext_dir / "school_coords.json"
    city_cache_path = ext_dir / "city_coords.json"

    if school_cache_path.exists() and city_cache_path.exists():
        return  # Already geocoded

    import requests
    session = requests.Session()

    # 1. Geocode schools
    if school_cache_path.exists():
        school_coords = json.loads(school_cache_path.read_text())
    else:
        school_coords = {}

    teams = pd.read_csv(data_dir / "MTeams.csv")
    team_queries = dict(zip(teams["TeamID"], teams["TeamName"]))
    to_geocode = {
        str(tid): name for tid, name in team_queries.items()
        if str(tid) not in school_coords
    }

    if to_geocode:
        print(f"    Geocoding {len(to_geocode)} schools...")
        for tid, name in to_geocode.items():
            query = f"{name} university basketball"
            result = geocode_nominatim(query, session)
            if result is None:
                result = geocode_nominatim(f"{name} university", session)
            if result:
                school_coords[tid] = {"lat": result[0], "lng": result[1]}
            time.sleep(1.1)
        school_cache_path.write_text(json.dumps(school_coords, indent=2))
        print(f"    Geocoded {len(school_coords)} schools total")

    # 2. Geocode venue cities
    if city_cache_path.exists():
        city_coords = json.loads(city_cache_path.read_text())
    else:
        city_coords = {}

    cities = pd.read_csv(data_dir / "Cities.csv")
    to_geocode_cities = {
        str(row["CityID"]): f"{row['City']}, {row['State']}"
        for _, row in cities.iterrows()
        if str(row["CityID"]) not in city_coords
    }

    if to_geocode_cities:
        print(f"    Geocoding {len(to_geocode_cities)} cities...")
        for cid, name in to_geocode_cities.items():
            result = geocode_nominatim(name, session)
            if result:
                city_coords[cid] = {"lat": result[0], "lng": result[1]}
            time.sleep(1.1)
        city_cache_path.write_text(json.dumps(city_coords, indent=2))
        print(f"    Geocoded {len(city_coords)} cities total")


def _load_coords(data_dir: Path):
    """Load cached school and city coordinates."""
    ext_dir = data_dir / "external" / "travel"
    school_coords = json.loads((ext_dir / "school_coords.json").read_text())
    city_coords = json.loads((ext_dir / "city_coords.json").read_text())
    return school_coords, city_coords


def _team_dist_to_city(team_id, city_lat, city_lng, school_coords):
    """Compute distance from a team's school to a city. Returns NaN if unknown."""
    tid = str(int(team_id))
    if tid not in school_coords:
        return np.nan
    return haversine(
        school_coords[tid]["lat"], school_coords[tid]["lng"],
        city_lat, city_lng,
    )


# ---------------------------------------------------------------------------
# Training: add travel features to historical matchups
# ---------------------------------------------------------------------------

def add_travel_to_matchups(matchups: pd.DataFrame, games: pd.DataFrame,
                           data_dir: Path) -> pd.DataFrame:
    """Add travel_dist_A, travel_dist_B, travel_advantage to training matchups.

    Uses MGameCities to look up the actual city for each historical game,
    then computes each team's distance to that venue.
    """
    print("  Adding travel distance features to matchups...")
    school_coords, city_coords = _load_coords(data_dir)

    # Build game -> city mapping from MGameCities
    game_cities = pd.read_csv(data_dir / "MGameCities.csv")
    ncaa = game_cities[game_cities["CRType"] == "NCAA"]
    cities_df = pd.read_csv(data_dir / "Cities.csv")

    # Create lookup: (Season, WTeamID, LTeamID) -> CityID
    game_city_map = {}
    for _, row in ncaa.iterrows():
        key = (row["Season"], row["WTeamID"], row["LTeamID"])
        game_city_map[key] = row["CityID"]

    dist_a_list = []
    dist_b_list = []

    for _, mrow in matchups.iterrows():
        s = mrow["Season"]
        team_a = mrow["TeamID_A"]
        team_b = mrow["TeamID_B"]
        label = mrow["Label"]

        # In our matchup construction, Label=1 means A=winner, B=loser
        # Label=0 means A=loser, B=winner
        if label == 1:
            key = (s, int(team_a), int(team_b))
        else:
            key = (s, int(team_b), int(team_a))

        city_id = game_city_map.get(key)
        if city_id is not None:
            cid = str(city_id)
            if cid in city_coords:
                clat = city_coords[cid]["lat"]
                clng = city_coords[cid]["lng"]
                dist_a_list.append(_team_dist_to_city(team_a, clat, clng, school_coords))
                dist_b_list.append(_team_dist_to_city(team_b, clat, clng, school_coords))
                continue

        dist_a_list.append(np.nan)
        dist_b_list.append(np.nan)

    matchups = matchups.copy()
    matchups["travel_dist_A"] = dist_a_list
    matchups["travel_dist_B"] = dist_b_list
    matchups["travel_advantage"] = matchups["travel_dist_B"] - matchups["travel_dist_A"]

    filled = matchups["travel_dist_A"].notna().sum()
    total = len(matchups)
    print(f"    Travel features: {filled}/{total} matchups have distance data")
    return matchups


# ---------------------------------------------------------------------------
# Prediction: determine venue from bracket structure
# ---------------------------------------------------------------------------

# 2026 venue mapping: (region_letter, seed_pod) -> city name for R1/R2
# Pod groupings: [1,16,8,9], [5,12,4,13], [6,11,3,14], [7,10,2,15]
# Region letters: W=East, X=South, Y=Midwest, Z=West

VENUE_CONFIGS = {
    2026: {
        "first_four": "Dayton, OH",
        "final_four": "Indianapolis, IN",
        "regionals": {  # Sweet 16 / Elite 8
            "W": "Washington, DC",   # East
            "X": "Houston, TX",      # South
            "Y": "Chicago, IL",      # Midwest
            "Z": "San Jose, CA",     # West
        },
        "pods": {  # R1/R2 sites: (region, pod) -> city
            # Pod A = seeds 1,16,8,9; Pod B = 5,12,4,13; Pod C = 6,11,3,14; Pod D = 7,10,2,15
            ("W", "A"): "Greenville, SC",     # East 1/16/8/9
            ("W", "B"): "San Diego, CA",      # East 5/12/4/13
            ("W", "C"): "Buffalo, NY",        # East 6/11/3/14
            ("W", "D"): "Philadelphia, PA",   # East 7/10/2/15
            ("X", "A"): "Tampa, FL",          # South 1/16/8/9
            ("X", "B"): "Oklahoma City, OK",  # South 5/12/4/13
            ("X", "C"): "Greenville, SC",     # South 6/11/3/14
            ("X", "D"): "Oklahoma City, OK",  # South 7/10/2/15
            ("Y", "A"): "Buffalo, NY",        # Midwest 1/16/8/9
            ("Y", "B"): "Tampa, FL",          # Midwest 5/12/4/13
            ("Y", "C"): "Philadelphia, PA",   # Midwest 6/11/3/14
            ("Y", "D"): "St. Louis, MO",      # Midwest 7/10/2/15
            ("Z", "A"): "San Diego, CA",      # West 1/16/8/9
            ("Z", "B"): "Portland, OR",       # West 5/12/4/13
            ("Z", "C"): "Portland, OR",       # West 6/11/3/14
            ("Z", "D"): "St. Louis, MO",      # West 7/10/2/15
        },
    }
}

# Seed number -> pod letter
SEED_TO_POD = {
    1: "A", 16: "A", 8: "A", 9: "A",
    5: "B", 12: "B", 4: "B", 13: "B",
    6: "C", 11: "C", 3: "C", 14: "C",
    7: "D", 10: "D", 2: "D", 15: "D",
}

# Which pods are in each bracket half (for determining Sweet 16 vs Elite 8)
# Top half: pods A + B (seeds 1,16,8,9,5,12,4,13)
# Bottom half: pods C + D (seeds 6,11,3,14,7,10,2,15)
TOP_HALF_PODS = {"A", "B"}
BOTTOM_HALF_PODS = {"C", "D"}


def _parse_seed(seed_str):
    """Parse seed string like 'W01' or 'X16a' -> (region_letter, seed_number)."""
    region = seed_str[0]
    seed_num = int(seed_str[1:3])
    return region, seed_num


def _get_venue_for_matchup(seed_a, seed_b, venue_config):
    """Determine venue city for a matchup given both teams' seed strings.

    Returns city string like 'Indianapolis, IN' or None if can't determine.
    """
    region_a, num_a = _parse_seed(seed_a)
    region_b, num_b = _parse_seed(seed_b)

    pod_a = SEED_TO_POD.get(num_a)
    pod_b = SEED_TO_POD.get(num_b)

    if region_a != region_b:
        # Cross-region: Final Four / Championship
        return venue_config["final_four"]

    # Same region
    region = region_a

    if pod_a == pod_b:
        # Same pod: R2 game at the pod's R1/R2 site
        return venue_config["pods"].get((region, pod_a))

    # Same region, different pods
    half_a = "top" if pod_a in TOP_HALF_PODS else "bottom"
    half_b = "top" if pod_b in TOP_HALF_PODS else "bottom"

    # Whether same half or different half, it's at the regional site (Sweet 16 or Elite 8)
    return venue_config["regionals"].get(region)


def _geocode_venue_city(city_str, city_coords_by_name):
    """Look up coordinates for a venue city string like 'Indianapolis, IN'."""
    return city_coords_by_name.get(city_str)


def _build_city_name_coords(data_dir: Path, city_coords: dict) -> dict:
    """Build city name -> (lat, lng) mapping for venue cities."""
    cities_df = pd.read_csv(data_dir / "Cities.csv")
    result = {}
    for _, row in cities_df.iterrows():
        cid = str(row["CityID"])
        if cid in city_coords:
            name = f"{row['City']}, {row['State']}"
            result[name] = (city_coords[cid]["lat"], city_coords[cid]["lng"])
    return result


# Hardcoded coordinates for 2026 venue cities (in case Cities.csv doesn't cover them)
VENUE_COORDS = {
    "Dayton, OH": (39.7589, -84.1916),
    "Greenville, SC": (34.8526, -82.3940),
    "San Diego, CA": (32.7157, -117.1611),
    "Buffalo, NY": (42.8864, -78.8784),
    "Philadelphia, PA": (39.9526, -75.1652),
    "Tampa, FL": (27.9506, -82.4572),
    "Oklahoma City, OK": (35.4676, -97.5164),
    "Portland, OR": (45.5152, -122.6784),
    "St. Louis, MO": (38.6270, -90.1994),
    "Washington, DC": (38.9072, -77.0369),
    "Houston, TX": (29.7604, -95.3698),
    "Chicago, IL": (41.8781, -87.6298),
    "San Jose, CA": (37.3382, -121.8863),
    "Indianapolis, IN": (39.7684, -86.1581),
}


def add_travel_to_predictions(pairs: pd.DataFrame, data_dir: Path,
                               season: int) -> pd.DataFrame:
    """Add travel features to prediction pairs using bracket structure.

    For each pair, determines where the game would be played based on
    both teams' seeds, then computes distances.
    """
    print(f"  Adding travel distance features to {season} predictions...")

    if season not in VENUE_CONFIGS:
        print(f"    Warning: No venue config for {season}, skipping travel features")
        pairs = pairs.copy()
        pairs["travel_dist_A"] = np.nan
        pairs["travel_dist_B"] = np.nan
        pairs["travel_advantage"] = np.nan
        return pairs

    venue_config = VENUE_CONFIGS[season]
    school_coords, _ = _load_coords(data_dir)

    # Build team -> seed mapping
    seeds_df = pd.read_csv(data_dir / "MNCAATourneySeeds.csv")
    seeds_season = seeds_df[seeds_df["Season"] == season]
    team_to_seed = dict(zip(seeds_season["TeamID"], seeds_season["Seed"]))

    dist_a_list = []
    dist_b_list = []

    for _, row in pairs.iterrows():
        team_a = int(row["TeamID_A"])
        team_b = int(row["TeamID_B"])

        seed_a = team_to_seed.get(team_a)
        seed_b = team_to_seed.get(team_b)

        if seed_a is None or seed_b is None:
            dist_a_list.append(np.nan)
            dist_b_list.append(np.nan)
            continue

        # Strip play-in suffixes for pod determination
        clean_a = seed_a[:3]
        clean_b = seed_b[:3]

        city = _get_venue_for_matchup(clean_a, clean_b, venue_config)
        if city is None or city not in VENUE_COORDS:
            dist_a_list.append(np.nan)
            dist_b_list.append(np.nan)
            continue

        clat, clng = VENUE_COORDS[city]
        dist_a_list.append(_team_dist_to_city(team_a, clat, clng, school_coords))
        dist_b_list.append(_team_dist_to_city(team_b, clat, clng, school_coords))

    pairs = pairs.copy()
    pairs["travel_dist_A"] = dist_a_list
    pairs["travel_dist_B"] = dist_b_list
    pairs["travel_advantage"] = pairs["travel_dist_B"] - pairs["travel_dist_A"]

    filled = pairs["travel_dist_A"].notna().sum()
    total = len(pairs)
    print(f"    Travel features: {filled}/{total} pairs have distance data")
    return pairs
