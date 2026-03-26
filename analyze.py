"""Matchup analysis and diagnostics."""

import pandas as pd
import numpy as np
from pathlib import Path
from pipeline import build_team_features, build_prediction_pairs
from config import DATA_DIR, ENABLED_FEATURES


def _load_predictor(tag: str):
    from autogluon.tabular import TabularPredictor
    return TabularPredictor.load(f"AutogluonModels/{tag}")


def _load_names() -> dict:
    teams = pd.read_csv(DATA_DIR / "MTeams.csv")
    return dict(zip(teams["TeamID"], teams["TeamName"]))


def _load_seeds(season: int) -> pd.DataFrame:
    seeds = pd.read_csv(DATA_DIR / "MNCAATourneySeeds.csv")
    s = seeds[seeds["Season"] == season].copy()
    s["SeedNum"] = s["Seed"].str.extract(r"(\d+)").astype(int)
    s["Region"] = s["Seed"].str[0]
    return s


def cmd_matchups(args):
    """Analyze matchups for a given seed pairing (e.g. 8v9, 5v12)."""
    seed_a, seed_b = map(int, args.seeds.lower().split("v"))
    season = args.season
    tag = args.tag

    tf = build_team_features(DATA_DIR, ENABLED_FEATURES, gender="M")
    predictor = _load_predictor(tag)
    name_map = _load_names()
    seeds = _load_seeds(season)

    pairs = build_prediction_pairs(tf, season, data_dir=DATA_DIR, travel=True)
    id_cols = ["Season", "TeamID_A", "TeamID_B"]
    feat_cols = [c for c in tf.columns if c not in ("Season", "TeamID")]
    tf_season = tf[tf["Season"] == season]

    higher = seeds[seeds["SeedNum"] == seed_a]
    lower = seeds[seeds["SeedNum"] == seed_b]

    print(f"\n=== {seed_a} vs {seed_b} SEED MATCHUPS ({season}) ===\n")

    for _, h in higher.iterrows():
        region = h["Region"]
        l_row = lower[lower["Region"] == region]
        if len(l_row) == 0:
            continue
        l_row = l_row.iloc[0]

        a_id, b_id = int(h["TeamID"]), int(l_row["TeamID"])
        a_name = name_map.get(a_id, str(a_id))
        b_name = name_map.get(b_id, str(b_id))

        row = pairs[(pairs["TeamID_A"] == a_id) & (pairs["TeamID_B"] == b_id)]
        if len(row) == 0:
            print(f"{region}: {a_name} vs {b_name} -- MISSING")
            continue

        X = row.drop(columns=id_cols)
        proba = predictor.predict_proba(X)[1].values[0]
        print(f"{region}: {h['Seed']} {a_name} vs {l_row['Seed']} {b_name}  ->  P({a_name})={proba:.1%}")

        if args.detail:
            a_feats = tf_season[tf_season["TeamID"] == a_id][feat_cols].values[0]
            b_feats = tf_season[tf_season["TeamID"] == b_id][feat_cols].values[0]
            signed_delta = pd.Series(a_feats - b_feats, index=feat_cols)
            abs_delta = signed_delta.abs().sort_values(ascending=False)
            n = args.top_n
            for feat in abs_delta.head(n).index:
                idx = feat_cols.index(feat)
                print(f"    {feat:40s} delta={signed_delta[feat]:+8.2f}  "
                      f"({a_name}={a_feats[idx]:.2f}, {b_name}={b_feats[idx]:.2f})")
        print()


def cmd_team(args):
    """Show a team's feature profile and percentile ranks within its season."""
    season = args.season
    tag = args.tag

    tf = build_team_features(DATA_DIR, ENABLED_FEATURES, gender="M")
    name_map = _load_names()

    # Find team by name substring
    matches = {tid: name for tid, name in name_map.items() if args.team.lower() in name.lower()}
    if not matches:
        print(f"No team matching '{args.team}'")
        return
    if len(matches) > 1:
        # Try exact match first
        exact = {tid: name for tid, name in matches.items() if name.lower() == args.team.lower()}
        if len(exact) == 1:
            matches = exact
        else:
            print(f"Multiple matches: {matches}")
            return

    team_id = list(matches.keys())[0]
    team_name = matches[team_id]

    tf_season = tf[tf["Season"] == season]
    team_row = tf_season[tf_season["TeamID"] == team_id]

    if len(team_row) == 0:
        print(f"{team_name} ({team_id}) not found in {season}")
        return

    feat_cols = [c for c in tf.columns if c not in ("Season", "TeamID")]
    team_vals = team_row[feat_cols].values[0]

    # Compute percentile rank within the season
    percentiles = []
    for i, col in enumerate(feat_cols):
        vals = tf_season[col].dropna()
        if len(vals) == 0:
            percentiles.append(np.nan)
        else:
            pct = (vals < team_vals[i]).sum() / len(vals) * 100
            percentiles.append(pct)

    profile = pd.DataFrame({
        "Feature": feat_cols,
        "Value": team_vals,
        "Percentile": percentiles,
    })

    # Filter to non-NaN features
    profile = profile.dropna(subset=["Value"])

    print(f"\n=== {team_name} ({team_id}) — {season} Feature Profile ===")
    print(f"  {len(profile)} features with values\n")

    # Show extremes
    print("TOP 15 FEATURES (highest percentile):")
    top = profile.nlargest(15, "Percentile")
    for _, r in top.iterrows():
        print(f"  {r['Feature']:40s}  val={r['Value']:8.2f}  pctl={r['Percentile']:.0f}%")

    print(f"\nBOTTOM 15 FEATURES (lowest percentile):")
    bot = profile.nsmallest(15, "Percentile")
    for _, r in bot.iterrows():
        print(f"  {r['Feature']:40s}  val={r['Value']:8.2f}  pctl={r['Percentile']:.0f}%")

    # Non-massey features in extremes
    non_massey = profile[~profile["Feature"].str.startswith("massey_")]
    print(f"\nNON-MASSEY EXTREMES (top/bottom 10):")
    nm_top = non_massey.nlargest(10, "Percentile")
    nm_bot = non_massey.nsmallest(10, "Percentile")
    print("  High:")
    for _, r in nm_top.iterrows():
        print(f"    {r['Feature']:40s}  val={r['Value']:8.2f}  pctl={r['Percentile']:.0f}%")
    print("  Low:")
    for _, r in nm_bot.iterrows():
        print(f"    {r['Feature']:40s}  val={r['Value']:8.2f}  pctl={r['Percentile']:.0f}%")


def cmd_confidence(args):
    """Compare prediction confidence distributions between two seasons."""
    tag = args.tag
    s1, s2 = args.season1, args.season2

    tf = build_team_features(DATA_DIR, ENABLED_FEATURES, gender="M")
    predictor = _load_predictor(tag)
    seeds_df = pd.read_csv(DATA_DIR / "MNCAATourneySeeds.csv")

    results = {}
    for season in [s1, s2]:
        pairs = build_prediction_pairs(tf, season, data_dir=DATA_DIR, travel=True)
        id_cols = ["Season", "TeamID_A", "TeamID_B"]
        X = pairs.drop(columns=id_cols)
        proba = predictor.predict_proba(X)[1].values
        # Only keep one direction (A < B) for unique matchups
        mask = pairs["TeamID_A"] < pairs["TeamID_B"]
        results[season] = proba[mask.values]

    # Bucket into bins
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print(f"\n=== Prediction Confidence Distribution ===\n")
    print(f"{'Bin':>12s}  {s1:>8d}  {s2:>8d}  {'Diff':>8s}")
    print("-" * 45)
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        c1 = np.mean((results[s1] >= lo) & (results[s1] < hi)) * 100
        c2 = np.mean((results[s2] >= lo) & (results[s2] < hi)) * 100
        print(f"  [{lo:.1f},{hi:.1f})  {c1:7.1f}%  {c2:7.1f}%  {c2-c1:+7.1f}%")

    # Summary stats
    for season in [s1, s2]:
        p = results[season]
        fav = np.maximum(p, 1 - p)  # confidence = distance from 0.5
        print(f"\n{season}: mean fav prob={fav.mean():.3f}, "
              f"median={np.median(fav):.3f}, "
              f">80%={np.mean(fav > 0.8)*100:.1f}%, "
              f">90%={np.mean(fav > 0.9)*100:.1f}%")


def cmd_importance(args):
    """Show feature importance for a trained model.

    Uses permutation importance on validation data (2025 tournament)
    to measure how much each feature contributes to prediction quality.
    Optionally show a specific team's values for the top features.
    """
    import json as _json
    from autogluon.tabular import TabularPredictor
    from pipeline import build_team_features as _btf, build_training_data
    from config import load_config, VALIDATION_SEASON

    tag = args.tag
    predictor = TabularPredictor.load(f"AutogluonModels/{tag}")

    top_n = args.top_n

    # Extract feature importance from L1 tree-based models (fast, no data needed).
    # L1 models use the original features; L2+ use stacked predictions.
    print(f"\n=== Feature Importance ({tag}) ===")
    print(f"  Extracting from L1 tree-based model splits...\n")

    l1_tree_models = [m for m in predictor.model_names()
                      if "_BAG_L1" in m and not m.startswith(("NeuralNet", "Weighted"))]

    importance_totals = {}
    model_count = 0
    for model_name in l1_tree_models:
        try:
            bag_model = predictor._trainer.load_model(model_name)
            child_imps = []
            for child_key in bag_model.models:
                child = bag_model.load_child(child_key)
                inner = child.model
                if hasattr(inner, "feature_importance"):
                    fi = inner.feature_importance()
                    names = inner.feature_name()
                    child_imps.append(dict(zip(names, fi)))
                elif hasattr(inner, "feature_importances_"):
                    names = list(child._features) if hasattr(child, "_features") else [f"f{i}" for i in range(len(inner.feature_importances_))]
                    child_imps.append(dict(zip(names, inner.feature_importances_)))
                elif hasattr(inner, "get_feature_importance"):
                    names = list(child._features) if hasattr(child, "_features") else [f"f{i}" for i in range(len(inner.get_feature_importance()))]
                    child_imps.append(dict(zip(names, inner.get_feature_importance())))
            if child_imps:
                avg_imp = {}
                for d in child_imps:
                    for k, v in d.items():
                        avg_imp[k] = avg_imp.get(k, 0) + v / len(child_imps)
                for k, v in avg_imp.items():
                    importance_totals[k] = importance_totals.get(k, 0) + v
                model_count += 1
                print(f"  {model_name}: {len(child_imps)} folds")
        except Exception as e:
            print(f"  {model_name}: skipped ({e})")
            continue

    if model_count == 0:
        print("No tree-based models found with feature importance.")
        return

    # Average across models and sort
    importance = pd.Series({k: v / model_count for k, v in importance_totals.items()})
    importance = importance.sort_values(ascending=False)

    print(f"\n  Averaged across {model_count} L1 tree models\n")
    print(f"{'Rank':>4s}  {'Feature':45s}  {'Importance':>10s}")
    print("-" * 65)
    for rank, (feat, imp) in enumerate(importance.head(top_n).items(), 1):
        print(f"{rank:4d}  {feat:45s}  {imp:10.1f}")

    # Group by feature source
    source_imp = {}
    for feat, imp_val in importance.items():
        for prefix in ["massey_", "rs_", "kp_", "seed_", "conf_", "th_", "rd_",
                       "sr_", "srd_", "cg_", "sv_", "mom_", "tempo_", "coach_",
                       "ct_", "loc_", "travel_", "traj_", "mt_", "ap_", "pp_",
                       "roster_", "vg_"]:
            if feat.startswith(prefix):
                source = prefix.rstrip("_")
                break
        else:
            source = feat.split("_")[0]
        source_imp.setdefault(source, 0.0)
        source_imp[source] += max(imp_val, 0)

    print(f"\n{'Source':15s}  {'Total Importance':>16s}")
    print("-" * 35)
    for source, imp_val in sorted(source_imp.items(), key=lambda x: -x[1]):
        if imp_val > 0:
            print(f"  {source:13s}  {imp_val:16.1f}")

    # If --team specified, show that team's values for top features
    if args.team:
        tf = build_team_features(DATA_DIR, ENABLED_FEATURES, gender="M")
        name_map = _load_names()
        matches = {tid: name for tid, name in name_map.items()
                   if args.team.lower() in name.lower()}
        if len(matches) == 0:
            print(f"\nNo team matching '{args.team}'")
            return
        exact = {tid: n for tid, n in matches.items() if n.lower() == args.team.lower()}
        if exact:
            matches = exact
        if len(matches) > 1:
            print(f"\nMultiple matches: {matches}")
            return

        team_id = list(matches.keys())[0]
        team_name = matches[team_id]
        season = args.season

        tf_season = tf[tf["Season"] == season]
        team_row = tf_season[tf_season["TeamID"] == team_id]
        if len(team_row) == 0:
            print(f"\n{team_name} not found in {season}")
            return

        # Map _A/_B/_delta features back to team-level features
        team_feat_cols = [c for c in tf.columns if c not in ("Season", "TeamID")]
        team_vals = team_row[team_feat_cols].iloc[0]

        print(f"\n=== {team_name} — Top Feature Values ({season}) ===\n")
        print(f"{'Feature':45s}  {'Value':>8s}  {'Pctl':>5s}  {'Importance':>10s}")
        print("-" * 75)

        shown = set()
        for feat, imp_val in importance.head(top_n * 3).items():
            # Strip _A/_B/_delta suffix to get team-level feature
            for suffix in ("_delta", "_A", "_B"):
                if feat.endswith(suffix):
                    base = feat[: -len(suffix)]
                    break
            else:
                base = feat
            if base in shown or base not in team_feat_cols:
                continue
            shown.add(base)
            val = team_vals[base]
            if pd.isna(val):
                continue
            # Percentile
            col_vals = tf_season[base].dropna()
            pctl = (col_vals < val).sum() / len(col_vals) * 100 if len(col_vals) > 0 else 0
            print(f"  {base:43s}  {val:8.2f}  {pctl:4.0f}%  {imp_val:10.1f}")
            if len(shown) >= top_n:
                break


if __name__ == "__main__":
    print("Use 'python run.py analyze ...' instead. Run 'python run.py --help' for details.")
