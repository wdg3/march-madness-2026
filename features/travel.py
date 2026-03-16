"""Travel distance features using geocoded school and venue locations.

Uses OpenStreetMap Nominatim for geocoding (free, no API key).
Computes average historical tournament travel distance per team as a
proxy for geographic advantage — teams from major metro areas near
common tournament venues travel less on average.
"""

import json
import time
from math import radians, cos, sin, asin, sqrt
from pathlib import Path

import pandas as pd
import numpy as np

from features.base import ExternalFeatureSource

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


class TravelFeatures(ExternalFeatureSource):
    """Average travel distance to tournament venues — closer = more fan support."""

    def name(self) -> str:
        return "travel"

    def fetch(self, data_dir: Path) -> None:
        """Geocode all schools and tournament venue cities."""
        import requests

        ext_dir = self.external_data_dir(data_dir)
        session = requests.Session()

        # 1. Geocode schools
        school_cache_path = ext_dir / "school_coords.json"
        if school_cache_path.exists():
            school_coords = json.loads(school_cache_path.read_text())
        else:
            school_coords = {}

        teams = pd.read_csv(data_dir / "MTeams.csv")
        spellings = pd.read_csv(data_dir / "MTeamSpellings.csv", encoding="latin-1")
        # Get the most common spelling for each team as a search query
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
                    # Try just the name + state
                    result = geocode_nominatim(f"{name} university", session)
                if result:
                    school_coords[tid] = {"lat": result[0], "lng": result[1]}
                time.sleep(1.1)  # Nominatim rate limit: 1 req/sec

            school_cache_path.write_text(json.dumps(school_coords, indent=2))
            print(f"    Geocoded {len(school_coords)} schools total")

        # 2. Geocode tournament venue cities
        city_cache_path = ext_dir / "city_coords.json"
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

    def build(self, data_dir: Path) -> pd.DataFrame:
        print("  Building travel distance features...")
        ext_dir = self.external_data_dir(data_dir)

        school_path = ext_dir / "school_coords.json"
        city_path = ext_dir / "city_coords.json"

        if not school_path.exists() or not city_path.exists():
            raise FileNotFoundError(
                "Travel data not fetched. Run with travel feature enabled "
                "and internet access to geocode locations."
            )

        school_coords = json.loads(school_path.read_text())
        city_coords = json.loads(city_path.read_text())

        # Load tournament game locations
        game_cities = pd.read_csv(data_dir / "MGameCities.csv")
        ncaa_games = game_cities[game_cities["CRType"] == "NCAA"]

        # Compute travel distance for each team in each tournament game
        distances = []
        for _, row in ncaa_games.iterrows():
            city_id = str(row["CityID"])
            if city_id not in city_coords:
                continue

            city_lat = city_coords[city_id]["lat"]
            city_lng = city_coords[city_id]["lng"]

            for team_id in [row["WTeamID"], row["LTeamID"]]:
                tid = str(team_id)
                if tid not in school_coords:
                    continue

                dist = haversine(
                    school_coords[tid]["lat"], school_coords[tid]["lng"],
                    city_lat, city_lng,
                )
                distances.append({
                    "Season": row["Season"],
                    "TeamID": team_id,
                    "Distance": dist,
                })

        if not distances:
            # Return empty frame with correct columns
            return pd.DataFrame(columns=[
                "Season", "TeamID", "travel_avg_dist", "travel_min_dist",
                "travel_max_dist",
            ])

        dist_df = pd.DataFrame(distances)

        # Aggregate: for each team, compute average distance across all their
        # historical tournament games (over all prior seasons) as a proxy for
        # how centrally located they are relative to typical tournament venues
        all_hist = dist_df.groupby("TeamID").agg(
            hist_avg_dist=("Distance", "mean"),
        ).reset_index()

        # Also compute per-season tournament travel (for seasons where available)
        per_season = dist_df.groupby(["Season", "TeamID"]).agg(
            travel_avg_dist=("Distance", "mean"),
            travel_min_dist=("Distance", "min"),
            travel_max_dist=("Distance", "max"),
        ).reset_index()

        # For teams/seasons without tournament game location data,
        # use their historical average
        # Build a full frame of all team-seasons
        all_teams = pd.read_csv(data_dir / "MTeams.csv")
        seasons = pd.read_csv(data_dir / "MSeasons.csv")
        # We only need seasons where teams exist
        team_seasons = pd.read_csv(data_dir / "MRegularSeasonCompactResults.csv")
        all_ts = pd.concat([
            team_seasons[["Season", "WTeamID"]].rename(columns={"WTeamID": "TeamID"}),
            team_seasons[["Season", "LTeamID"]].rename(columns={"LTeamID": "TeamID"}),
        ]).drop_duplicates()

        result = all_ts.merge(per_season, on=["Season", "TeamID"], how="left")
        result = result.merge(all_hist, on="TeamID", how="left")

        # Fill missing per-season travel with historical average
        for col in ["travel_avg_dist", "travel_min_dist", "travel_max_dist"]:
            result[col] = result[col].fillna(result["hist_avg_dist"])

        result.rename(columns={"hist_avg_dist": "travel_hist_avg_dist"}, inplace=True)

        return result[["Season", "TeamID", "travel_avg_dist", "travel_min_dist",
                        "travel_max_dist", "travel_hist_avg_dist"]]
