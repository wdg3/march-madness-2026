from pathlib import Path
import hashlib
import pandas as pd
from features import REGISTRY
from features.base import ExternalFeatureSource
from features.travel import ensure_geocoded, add_travel_to_matchups, add_travel_to_predictions

CACHE_DIR = Path(".cache")


def _cache_path(enabled: list[str], gender: str, data_dir: Path) -> Path:
    """Return path to disk cache file, keyed on feature names AND data content.

    The cache key includes modification times of all source data files so that
    changing any underlying CSV automatically invalidates the cache.
    """
    parts = [gender, ",".join(sorted(enabled))]

    # Include mtimes of external data directories used by enabled features
    team_sources = [n for n in enabled if n != "travel"]
    for name in sorted(team_sources):
        source = REGISTRY[name]()
        if isinstance(source, ExternalFeatureSource):
            ext_dir = source.external_data_dir(data_dir)
            if ext_dir.exists():
                for f in sorted(ext_dir.glob("*.csv")):
                    parts.append(f"{f.name}:{f.stat().st_mtime_ns}")

    # Include core Kaggle data file mtimes
    for f in sorted(data_dir.glob(f"{gender}*.csv")):
        parts.append(f"{f.name}:{f.stat().st_mtime_ns}")

    h = hashlib.md5("|".join(parts).encode()).hexdigest()[:12]
    return CACHE_DIR / f"team_features_{gender}_{h}.pkl"


def build_team_features(data_dir: Path, enabled: list[str], force_fetch: bool = False,
                        gender: str = "M") -> pd.DataFrame:
    """Build per-team-per-season feature matrix by merging all enabled sources.

    Results are cached to disk — repeated calls with the same enabled list,
    gender, and data content load from cache without rebuilding. The cache
    automatically invalidates when source data files change. Use force_fetch=True
    to bypass the cache and rebuild from scratch.
    """
    cache_file = _cache_path(enabled, gender, data_dir)

    if not force_fetch and cache_file.exists():
        cached = pd.read_pickle(cache_file)
        print(f"Loaded cached {'women' if gender == 'W' else 'men'}'s team features {cached.shape} from {cache_file}")
        return cached

    print(f"Building {'women' if gender == 'W' else 'men'}'s team features...")
    result = None

    # Filter out "travel" — it's now a matchup-level feature, not team-level
    team_sources = [name for name in enabled if name != "travel"]

    for name in team_sources:
        source = REGISTRY[name]()
        if isinstance(source, ExternalFeatureSource):
            if gender == "M":
                source.ensure_fetched(data_dir, force=force_fetch)
            elif not source.is_fetched(data_dir):
                # Skip external sources that haven't been fetched for women
                continue
        df = source.build(data_dir, gender=gender)
        if df.empty or list(df.columns) == ["Season", "TeamID"]:
            continue  # Skip empty feature sources (e.g., massey for women)
        if result is None:
            result = df
        else:
            result = result.merge(df, on=["Season", "TeamID"], how="left")

    print(f"  Team features shape: {result.shape}")

    # Save to disk cache
    CACHE_DIR.mkdir(exist_ok=True)
    result.to_pickle(cache_file)
    print(f"  Cached to {cache_file}")

    return result


def build_matchups(team_features: pd.DataFrame, games: pd.DataFrame,
                   data_dir: Path = None, travel: bool = False) -> pd.DataFrame:
    """Build pairwise matchup training data from historical tournament games.

    For each game, creates two rows:
      [TeamA_features | TeamB_features, Label=1]  (A is winner)
      [TeamB_features | TeamA_features, Label=0]  (A is loser)
    """
    print("Building matchup training data...")

    # Only keep seasons present in team features
    available_seasons = set(team_features["Season"].unique())
    games = games[games["Season"].isin(available_seasons)]

    # Prepare feature columns (everything except Season and TeamID)
    feat_cols = [c for c in team_features.columns if c not in ("Season", "TeamID")]

    rows = []
    for _, game in games.iterrows():
        s = game["Season"]
        sf = team_features[team_features["Season"] == s]

        w = sf[sf["TeamID"] == game["WTeamID"]]
        l = sf[sf["TeamID"] == game["LTeamID"]]

        if len(w) == 0 or len(l) == 0:
            continue

        w_feats = w[feat_cols].values[0]
        l_feats = l[feat_cols].values[0]
        delta = w_feats - l_feats

        # Winner as A (label=1), Loser as A (label=0)
        row1 = [s, game["WTeamID"], game["LTeamID"]] + list(w_feats) + list(l_feats) + list(delta) + [1]
        row0 = [s, game["LTeamID"], game["WTeamID"]] + list(l_feats) + list(w_feats) + list(-delta) + [0]
        rows.append(row1)
        rows.append(row0)

    col_names = (
        ["Season", "TeamID_A", "TeamID_B"]
        + [f"{c}_A" for c in feat_cols]
        + [f"{c}_B" for c in feat_cols]
        + [f"{c}_delta" for c in feat_cols]
        + ["Label"]
    )
    result = pd.DataFrame(rows, columns=col_names)

    # Add matchup-level travel features
    if travel and data_dir is not None:
        result = add_travel_to_matchups(result, games, data_dir)

    print(f"  Matchup data shape: {result.shape}")
    return result


def build_prediction_pairs(team_features: pd.DataFrame, season: int,
                           data_dir: Path = None, travel: bool = False) -> pd.DataFrame:
    """Build all pairwise matchups for prediction in a given season.

    Only includes teams that have a seed (tournament teams).
    """
    print(f"Building prediction pairs for {season}...")
    current = team_features[team_features["Season"] == season]
    feat_cols = [c for c in team_features.columns if c not in ("Season", "TeamID")]
    teams = current["TeamID"].unique()

    rows = []
    for i, a in enumerate(teams):
        for b in teams:
            if a != b:
                a_feats = current[current["TeamID"] == a][feat_cols].values[0]
                b_feats = current[current["TeamID"] == b][feat_cols].values[0]
                delta = a_feats - b_feats
                rows.append([season, a, b] + list(a_feats) + list(b_feats) + list(delta))

    col_names = (
        ["Season", "TeamID_A", "TeamID_B"]
        + [f"{c}_A" for c in feat_cols]
        + [f"{c}_B" for c in feat_cols]
        + [f"{c}_delta" for c in feat_cols]
    )
    result = pd.DataFrame(rows, columns=col_names)

    # Add matchup-level travel features
    if travel and data_dir is not None:
        result = add_travel_to_predictions(result, data_dir, season)

    print(f"  Prediction pairs shape: {result.shape}")
    return result
