from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from features import REGISTRY
from features.base import ExternalFeatureSource
from features.travel import ensure_geocoded, add_travel_to_matchups, add_travel_to_predictions
from features.player_nn import add_player_nn_to_matchups
from features.pbp_nn import add_pbp_to_matchups

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
                   data_dir: Path = None, travel: bool = False,
                   player_nn: bool = False, pbp_nn: bool = False) -> pd.DataFrame:
    """Build pairwise matchup rows from game results.

    For each game, creates two rows:
      [TeamA_features | TeamB_features | delta_features | Label=1]  (A is winner)
      [TeamA_features | TeamB_features | delta_features | Label=0]  (A is loser)

    Uses dict-based feature lookup for O(1) per-game instead of DataFrame filtering,
    which matters when processing ~80K+ regular season games.
    """
    print(f"  Building matchups from {len(games)} games...")

    available_seasons = set(team_features["Season"].unique())
    games = games[games["Season"].isin(available_seasons)]

    feat_cols = [c for c in team_features.columns if c not in ("Season", "TeamID")]

    # Dict lookup: (season, team_id) -> feature array — O(1) per game
    feat_lookup = {}
    for _, row in team_features.iterrows():
        feat_lookup[(row["Season"], row["TeamID"])] = row[feat_cols].values

    rows = []
    for _, game in games.iterrows():
        s = game["Season"]
        w_key = (s, game["WTeamID"])
        l_key = (s, game["LTeamID"])

        if w_key not in feat_lookup or l_key not in feat_lookup:
            continue

        w_feats = feat_lookup[w_key]
        l_feats = feat_lookup[l_key]
        delta = w_feats - l_feats

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

    if travel and data_dir is not None:
        result = add_travel_to_matchups(result, games, data_dir)

    if player_nn and data_dir is not None:
        result = add_player_nn_to_matchups(result, data_dir)

    if pbp_nn and data_dir is not None:
        result = add_pbp_to_matchups(result, data_dir)

    print(f"    {len(result)} matchup rows")
    return result


def build_training_data(
    team_features: pd.DataFrame,
    data_dir: Path,
    travel: bool = False,
    player_nn: bool = False,
    pbp_nn: bool = False,
    include_regular_season: bool = True,
    include_conf_tournament: bool = True,
    time_decay_half_life: float = 5.0,
    game_weights: dict = None,
) -> pd.DataFrame:
    """Build training data from multiple game sources with sample weights.

    Combines NCAA tournament, conference tournament, and regular season games.
    Adds game-type features (is_ncaa_tournament, is_conf_tournament) and computes
    sample_weight = time_decay * game_type_weight for AutoGluon sample weighting.

    Args:
        team_features: Per-team-per-season feature matrix.
        data_dir: Path to data directory.
        travel: Whether to add travel features (only for tournament games).
        include_regular_season: Include regular season games in training.
        include_conf_tournament: Include conference tournament games in training.
        time_decay_half_life: Half-life in years for exponential decay.
            A season this many years old gets half the weight of the most recent.
            Set to 0 to disable time decay (all seasons weighted equally).
        game_weights: Per-source importance weights.
            Defaults: tournament=1.0, conf_tournament=0.5, regular_season=0.15.
    """
    if game_weights is None:
        game_weights = {"tournament": 1.0, "conf_tournament": 0.5, "regular_season": 0.15}

    print("Building training data...")
    sources = []

    # NCAA tournament games (always included)
    tourney = pd.read_csv(data_dir / "MNCAATourneyCompactResults.csv")
    t_matchups = build_matchups(team_features, tourney, data_dir=data_dir, travel=travel, player_nn=player_nn, pbp_nn=pbp_nn)
    t_matchups["is_ncaa_tournament"] = 1
    t_matchups["is_conf_tournament"] = 0
    t_matchups["_game_weight"] = game_weights.get("tournament", 1.0)
    sources.append(t_matchups)

    # Conference tournament games
    if include_conf_tournament:
        conf_path = data_dir / "MConferenceTourneyGames.csv"
        if conf_path.exists():
            conf = pd.read_csv(conf_path)
            c_matchups = build_matchups(team_features, conf, data_dir=data_dir, travel=False, player_nn=player_nn, pbp_nn=pbp_nn)
            c_matchups["is_ncaa_tournament"] = 0
            c_matchups["is_conf_tournament"] = 1
            c_matchups["_game_weight"] = game_weights.get("conf_tournament", 0.5)
            sources.append(c_matchups)

    # Regular season games
    if include_regular_season:
        rs = pd.read_csv(data_dir / "MRegularSeasonCompactResults.csv")
        rs_matchups = build_matchups(team_features, rs, data_dir=data_dir, travel=False, player_nn=player_nn, pbp_nn=pbp_nn)
        rs_matchups["is_ncaa_tournament"] = 0
        rs_matchups["is_conf_tournament"] = 0
        rs_matchups["_game_weight"] = game_weights.get("regular_season", 0.15)
        sources.append(rs_matchups)

    combined = pd.concat(sources, ignore_index=True)

    # Sample weight = time_decay * game_type_weight
    max_season = combined["Season"].max()
    if time_decay_half_life and time_decay_half_life > 0:
        time_decay = np.float_power(0.5, (max_season - combined["Season"]) / time_decay_half_life)
        combined["sample_weight"] = combined["_game_weight"] * time_decay
    else:
        combined["sample_weight"] = combined["_game_weight"]

    combined = combined.drop(columns=["_game_weight"])

    print(f"  Total training data: {len(combined)} rows")
    print(f"  Sample weight range: [{combined['sample_weight'].min():.4f}, {combined['sample_weight'].max():.4f}]")
    return combined


def build_prediction_pairs(team_features: pd.DataFrame, season: int,
                           data_dir: Path = None, travel: bool = False,
                           player_nn: bool = False, pbp_nn: bool = False) -> pd.DataFrame:
    """Build all pairwise matchups for prediction in a given season.

    Adds game-type features (is_ncaa_tournament=1, is_conf_tournament=0) for
    consistency with training data — predictions are for tournament games.
    """
    print(f"Building prediction pairs for {season}...")
    current = team_features[team_features["Season"] == season]
    feat_cols = [c for c in team_features.columns if c not in ("Season", "TeamID")]
    teams = current["TeamID"].unique()

    # Dict lookup for O(1) per pair
    feat_lookup = {}
    for _, row in current.iterrows():
        feat_lookup[row["TeamID"]] = row[feat_cols].values

    rows = []
    for a in teams:
        a_feats = feat_lookup[a]
        for b in teams:
            if a != b:
                b_feats = feat_lookup[b]
                delta = a_feats - b_feats
                rows.append([season, a, b] + list(a_feats) + list(b_feats) + list(delta))

    col_names = (
        ["Season", "TeamID_A", "TeamID_B"]
        + [f"{c}_A" for c in feat_cols]
        + [f"{c}_B" for c in feat_cols]
        + [f"{c}_delta" for c in feat_cols]
    )
    result = pd.DataFrame(rows, columns=col_names)

    # Game-type features: predictions are for tournament games
    result["is_ncaa_tournament"] = 1
    result["is_conf_tournament"] = 0

    if travel and data_dir is not None:
        result = add_travel_to_predictions(result, data_dir, season)

    if player_nn and data_dir is not None:
        result = add_player_nn_to_matchups(result, data_dir)

    if pbp_nn and data_dir is not None:
        result = add_pbp_to_matchups(result, data_dir)

    print(f"  Prediction pairs shape: {result.shape}")
    return result
