from pathlib import Path
import pandas as pd
from features import REGISTRY
from features.base import ExternalFeatureSource


def build_team_features(data_dir: Path, enabled: list[str], force_fetch: bool = False) -> pd.DataFrame:
    """Build per-team-per-season feature matrix by merging all enabled sources."""
    print("Building team features...")
    result = None

    for name in enabled:
        source = REGISTRY[name]()
        if isinstance(source, ExternalFeatureSource):
            source.ensure_fetched(data_dir, force=force_fetch)
        df = source.build(data_dir)
        if result is None:
            result = df
        else:
            result = result.merge(df, on=["Season", "TeamID"], how="left")

    print(f"  Team features shape: {result.shape}")
    return result


def build_matchups(team_features: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
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

        # Winner as A (label=1), Loser as A (label=0)
        row1 = [s, game["WTeamID"], game["LTeamID"]] + list(w_feats) + list(l_feats) + [1]
        row0 = [s, game["LTeamID"], game["WTeamID"]] + list(l_feats) + list(w_feats) + [0]
        rows.append(row1)
        rows.append(row0)

    col_names = (
        ["Season", "TeamID_A", "TeamID_B"]
        + [f"{c}_A" for c in feat_cols]
        + [f"{c}_B" for c in feat_cols]
        + ["Label"]
    )
    result = pd.DataFrame(rows, columns=col_names)
    print(f"  Matchup data shape: {result.shape}")
    return result


def build_prediction_pairs(team_features: pd.DataFrame, season: int) -> pd.DataFrame:
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
                rows.append([season, a, b] + list(a_feats) + list(b_feats))

    col_names = (
        ["Season", "TeamID_A", "TeamID_B"]
        + [f"{c}_A" for c in feat_cols]
        + [f"{c}_B" for c in feat_cols]
    )
    result = pd.DataFrame(rows, columns=col_names)
    print(f"  Prediction pairs shape: {result.shape}")
    return result
