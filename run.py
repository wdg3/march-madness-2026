"""March Madness 2026 — Unified CLI.

Usage:
    python run.py train --tag v1                      Train a model
    python run.py predict --tag v1                    Generate men's submission
    python run.py submit --tag v1                     Generate M+W Kaggle submission
    python run.py bracket --tag v1                    Monte Carlo bracket simulation
    python run.py bet --tag v1 --odds odds.csv        Head-to-head bet sheet
    python run.py futures --tag v1                    Kalshi futures bets
    python run.py backtest --tag v1 --season 2025     Score against actuals
    python run.py analyze matchups --tag v1 --seeds 8v9
    python run.py analyze team --tag v1 --team Duke
    python run.py analyze confidence --tag v1
    python run.py data fetch [--source kenpom roster vegas odds]
    python run.py data status
    python run.py full --tag v1                       Train -> predict -> bracket
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import (
    DATA_DIR, OUTPUT_DIR, TRAIN_SEASONS_END, VALIDATION_SEASON,
    PREDICTION_SEASON, AG_PRESETS, AG_TIME_LIMIT, AG_NUM_BAG_FOLDS,
    AG_NUM_STACK_LEVELS, ENABLED_FEATURES, load_config,
)

MODELS_ROOT = Path("./AutogluonModels")


def model_dir(tag: str) -> Path:
    return MODELS_ROOT / tag


def tag_output_dir(tag: str) -> Path:
    """Return output/{tag}/, creating it if needed."""
    d = OUTPUT_DIR / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_submission(args) -> str:
    """Find the submission CSV for a given tag."""
    if getattr(args, "submission", None):
        return args.submission
    tag_dir = tag_output_dir(args.tag)
    path = tag_dir / "submission.csv"
    if path.exists():
        return str(path)
    # Fall back to legacy flat naming
    legacy = OUTPUT_DIR / f"submission_{args.tag}.csv"
    if legacy.exists():
        return str(legacy)
    return str(path)


def _verify_manifest(tag: str, features: list[str]):
    """Warn if current features don't match what the model was trained with."""
    manifest_path = model_dir(tag) / "manifest.json"
    if not manifest_path.exists():
        return
    with open(manifest_path) as f:
        manifest = json.load(f)
    trained_features = set(manifest.get("features", []))
    current_features = set(features)
    if trained_features and trained_features != current_features:
        added = current_features - trained_features
        removed = trained_features - current_features
        print("\nWARNING: Feature mismatch with trained model!")
        if added:
            print(f"  Added since training:   {sorted(added)}")
        if removed:
            print(f"  Removed since training: {sorted(removed)}")
        print("  Predictions may not match training expectations.\n")


# ── Train ────────────────────────────────────────────────────────────────────


def cmd_train(args):
    """Train the model on men's historical tournament data."""
    from pipeline import build_team_features, build_matchups
    from features.travel import ensure_geocoded
    from training import train

    cfg = load_config(args.tag)
    features = cfg.get("features", ENABLED_FEATURES)
    train_cfg = cfg.get("training", {})

    out_dir = model_dir(args.tag)
    if out_dir.exists():
        print(f"Warning: model directory {out_dir} already exists, will overwrite.")

    team_features = build_team_features(DATA_DIR, features, gender="M")

    use_travel = "travel" in features
    if use_travel:
        ensure_geocoded(DATA_DIR)

    tourney = pd.read_csv(DATA_DIR / "MNCAATourneyCompactResults.csv")
    matchups = build_matchups(team_features, tourney, data_dir=DATA_DIR, travel=use_travel)

    start_season = train_cfg.get("train_seasons_start", 2010)
    end_season = train_cfg.get("train_seasons_end", TRAIN_SEASONS_END)
    val_season = train_cfg.get("validation_season", VALIDATION_SEASON)

    matchups = matchups[matchups["Season"] >= start_season]
    train_data = matchups[matchups["Season"] < end_season].copy()
    val_data = matchups[matchups["Season"] == val_season].copy()

    drop_cols = ["Season", "TeamID_A", "TeamID_B"]
    train_data = train_data.drop(columns=drop_cols)
    val_data = val_data.drop(columns=drop_cols)

    presets = train_cfg.get("presets", AG_PRESETS)
    time_limit = args.time_limit or train_cfg.get("time_limit", AG_TIME_LIMIT)
    num_bag_folds = train_cfg.get("num_bag_folds", AG_NUM_BAG_FOLDS)
    num_stack_levels = train_cfg.get("num_stack_levels", AG_NUM_STACK_LEVELS)

    train(
        train_data, val_data,
        presets=presets,
        time_limit=time_limit,
        num_bag_folds=num_bag_folds,
        num_stack_levels=num_stack_levels,
        output_dir=str(out_dir),
        tag=args.tag,
        config=cfg,
        features=features,
        data_dir=DATA_DIR,
    )
    print(f"\nTraining complete! Model saved to {out_dir}")


# ── Predict ──────────────────────────────────────────────────────────────────


def cmd_predict(args):
    """Generate men's-only submission CSV."""
    from autogluon.tabular import TabularPredictor
    from pipeline import build_team_features, build_prediction_pairs
    from features.travel import ensure_geocoded
    from submission import generate_submission

    cfg = load_config(args.tag)
    features = cfg.get("features", ENABLED_FEATURES)
    _verify_manifest(args.tag, features)

    predictor = TabularPredictor.load(str(model_dir(args.tag)))
    team_features = build_team_features(DATA_DIR, features, gender="M")

    use_travel = "travel" in features
    if use_travel:
        ensure_geocoded(DATA_DIR)

    pred_pairs = build_prediction_pairs(
        team_features, PREDICTION_SEASON, data_dir=DATA_DIR, travel=use_travel,
    )

    sample_sub = pd.read_csv(DATA_DIR / "SampleSubmissionStage2.csv")
    out_dir = tag_output_dir(args.tag)
    output_path = out_dir / "submission.csv"
    generate_submission(predictor, pred_pairs, sample_sub, output_path)
    print(f"\nMen's submission saved to {output_path}")


# ── Submit ───────────────────────────────────────────────────────────────────


def cmd_submit(args):
    """Generate full Kaggle submission with both men's and women's predictions."""
    from autogluon.tabular import TabularPredictor
    from pipeline import build_team_features, build_prediction_pairs
    from features.travel import ensure_geocoded
    from submission import generate_submission

    cfg = load_config(args.tag)
    features = cfg.get("features", ENABLED_FEATURES)
    _verify_manifest(args.tag, features)

    predictor = TabularPredictor.load(str(model_dir(args.tag)))
    use_travel = "travel" in features

    # Men's predictions
    print("=" * 60)
    print("  MEN'S PREDICTIONS")
    print("=" * 60)
    men_features = build_team_features(DATA_DIR, features, gender="M")
    if use_travel:
        ensure_geocoded(DATA_DIR)
    men_pairs = build_prediction_pairs(
        men_features, PREDICTION_SEASON, data_dir=DATA_DIR, travel=use_travel,
    )

    # Women's predictions
    print("\n" + "=" * 60)
    print("  WOMEN'S PREDICTIONS")
    print("=" * 60)
    women_features = build_team_features(DATA_DIR, features, gender="W")
    for col in men_features.columns:
        if col not in women_features.columns:
            women_features[col] = float("nan")
    women_features = women_features[men_features.columns]
    women_pairs = build_prediction_pairs(
        women_features, PREDICTION_SEASON, data_dir=DATA_DIR, travel=False,
    )

    all_pairs = pd.concat([men_pairs, women_pairs], ignore_index=True)
    sample_sub = pd.read_csv(DATA_DIR / "SampleSubmissionStage2.csv")
    out_dir = tag_output_dir(args.tag)
    output_path = out_dir / "submission.csv"
    generate_submission(predictor, all_pairs, sample_sub, output_path)

    sub = pd.read_csv(output_path)
    ids = sub["ID"].str.split("_", expand=True)
    n_men = (ids[1].astype(int) < 3000).sum()
    n_women = (ids[1].astype(int) >= 3000).sum()
    print(f"\nFull submission saved to {output_path}")
    print(f"  Men's pairs: {n_men}")
    print(f"  Women's pairs: {n_women}")
    print(f"  Total: {len(sub)}")


# ── Bracket ──────────────────────────────────────────────────────────────────


def cmd_bracket(args):
    """Run Monte Carlo bracket simulation."""
    import simulate

    submission = _resolve_submission(args)
    out_dir = tag_output_dir(args.tag)
    output_path = out_dir / "bracket.csv"

    simulate.run(
        submission, args.season, args.n_sims,
        output_path=str(output_path),
    )


# ── Bet ──────────────────────────────────────────────────────────────────────


def cmd_bet(args):
    """Generate Kelly Criterion bet sheet from model predictions and odds."""
    from betting import generate_bet_sheet, print_bet_sheet

    submission = _resolve_submission(args)

    bet_df = generate_bet_sheet(
        args.odds, submission, args.season,
        kelly_frac=args.kelly,
        min_edge=args.min_edge,
        max_bet_pct=args.max_bet,
        bankroll=args.bankroll,
        fee=args.fee,
    )

    print_bet_sheet(bet_df, args.bankroll)

    if len(bet_df) > 0:
        out_dir = tag_output_dir(args.tag)
        out_path = out_dir / "bets.csv"
        bet_df.to_csv(out_path, index=False)
        print(f"\nBet sheet saved to {out_path}")


# ── Futures ──────────────────────────────────────────────────────────────────


def cmd_futures(args):
    """Generate Kalshi futures bet sheet with YES/NO positions."""
    from futures import (
        fetch_kalshi_markets, load_kalshi_from_csv, build_name_to_id,
        compute_advancement_probs, generate_futures_bets, print_bet_sheet,
    )

    submission = _resolve_submission(args)

    if args.from_csv:
        print(f"Loading markets from {args.from_csv}...")
        markets = load_kalshi_from_csv(args.from_csv)
    else:
        print("Fetching markets from Kalshi API...")
        markets = fetch_kalshi_markets()
    print(f"  {len(markets)} markets loaded")

    name_to_id = build_name_to_id(DATA_DIR)

    print(f"Running {args.n_sims:,} bracket simulations...")
    adv_probs, team_names = compute_advancement_probs(
        submission, args.season, DATA_DIR, args.n_sims, seed=42
    )

    df = generate_futures_bets(
        markets, adv_probs, team_names, name_to_id,
        total_cost=args.total_cost,
        kelly_frac=args.kelly,
        min_edge=args.min_edge,
        max_bet_pct=args.max_bet,
        bankroll=args.bankroll,
    )

    print_bet_sheet(df, args.bankroll)

    if len(df) > 0:
        out_dir = tag_output_dir(args.tag)
        out_path = out_dir / "futures_bets.csv"
        df.to_csv(out_path, index=False)
        print(f"\nFutures bet sheet saved to {out_path}")


# ── Backtest ─────────────────────────────────────────────────────────────────


def cmd_backtest(args):
    """Score bracket picks against actual results using ESPN scoring."""
    from backtest import run_backtest

    submission = None
    if getattr(args, "submission", None):
        submission = args.submission
    elif (tag_output_dir(args.tag) / "submission.csv").exists():
        submission = str(tag_output_dir(args.tag) / "submission.csv")

    run_backtest(args.season, args.n_sims, submission=submission, tag=args.tag)


# ── Analyze ──────────────────────────────────────────────────────────────────


def cmd_analyze(args):
    """Matchup analysis and diagnostics."""
    from analyze import cmd_matchups, cmd_team, cmd_confidence

    if args.analyze_cmd == "matchups":
        cmd_matchups(args)
    elif args.analyze_cmd == "team":
        cmd_team(args)
    elif args.analyze_cmd == "confidence":
        cmd_confidence(args)
    else:
        print("Usage: python run.py analyze {matchups|team|confidence} --tag TAG ...")


# ── Data ─────────────────────────────────────────────────────────────────────


def cmd_data(args):
    """Data management: fetch and status."""
    if args.data_cmd == "fetch":
        _cmd_data_fetch(args)
    elif args.data_cmd == "status":
        _cmd_data_status(args)
    else:
        print("Usage: python run.py data {fetch|status}")


def _cmd_data_fetch(args):
    """Fetch/refresh external data sources."""
    from features import REGISTRY
    from features.base import ExternalFeatureSource

    sources = args.source if args.source else None

    for name, cls in sorted(REGISTRY.items()):
        source = cls()
        if not isinstance(source, ExternalFeatureSource):
            continue
        if sources and name not in sources:
            continue
        print(f"Fetching {name}...")
        source.ensure_fetched(DATA_DIR, force=True)

    # Odds API requires separate handling (not a FeatureSource)
    if sources is None or "odds" in (sources or []):
        if getattr(args, "api_key", None):
            from fetch_odds import fetch_odds
            fetch_odds(args.api_key, resume=getattr(args, "resume", False))
        elif sources and "odds" in sources:
            print("Error: --api-key required for odds fetch")


def _cmd_data_status(args):
    """Show freshness of all data sources."""
    from features import REGISTRY
    from features.base import ExternalFeatureSource

    print("\n=== Data Source Status ===\n")

    # External sources
    for name, cls in sorted(REGISTRY.items()):
        source = cls()
        if isinstance(source, ExternalFeatureSource):
            ext_dir = source.external_data_dir(DATA_DIR)
            if ext_dir.exists() and any(ext_dir.iterdir()):
                files = [f for f in ext_dir.glob("*") if f.is_file()]
                newest = max(f.stat().st_mtime for f in files)
                ts = datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M")
                print(f"  {name:20s}  fetched  ({len(files)} files, latest: {ts})")
            else:
                print(f"  {name:20s}  not fetched")

    # Kaggle data
    print()
    for prefix, label in [("M", "Men's"), ("W", "Women's")]:
        files = list(DATA_DIR.glob(f"{prefix}*.csv"))
        if files:
            newest = max(f.stat().st_mtime for f in files)
            ts = datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M")
            print(f"  Kaggle ({label}):    {len(files)} files, latest: {ts}")

    # Feature cache
    cache_dir = Path(".cache")
    if cache_dir.exists():
        cache_files = list(cache_dir.glob("*.pkl"))
        if cache_files:
            newest = max(f.stat().st_mtime for f in cache_files)
            ts = datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M")
            print(f"\n  Feature cache:       {len(cache_files)} files, latest: {ts}")

    # Trained models
    if MODELS_ROOT.exists():
        tags = sorted(d.name for d in MODELS_ROOT.iterdir() if d.is_dir())
        if tags:
            print(f"\n  Trained models:      {', '.join(tags)}")
        else:
            print(f"\n  Trained models:      none")


# ── Full ─────────────────────────────────────────────────────────────────────


def cmd_full(args):
    """Run the full pipeline: train -> predict -> bracket."""
    print("=" * 60)
    print(f"  FULL PIPELINE: {args.tag}")
    print("=" * 60)

    # Train
    print("\n[1/3] Training model...")
    cmd_train(args)

    # Predict
    print("\n[2/3] Generating submission...")
    cmd_predict(args)

    # Bracket
    print("\n[3/3] Running bracket simulation...")
    args.season = PREDICTION_SEASON
    args.n_sims = getattr(args, "n_sims", 10000)
    args.submission = None
    cmd_bracket(args)

    print("\n" + "=" * 60)
    print(f"  DONE! All outputs in output/{args.tag}/")
    print("=" * 60)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="March Madness 2026 Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python run.py train --tag v1              Train a model
  python run.py predict --tag v1            Generate submission
  python run.py bracket --tag v1            Simulate bracket
  python run.py full --tag v1               Train + predict + bracket
  python run.py bet --tag v1 --odds o.csv   Generate bet sheet
  python run.py data status                 Show data freshness
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── train ──
    p = subparsers.add_parser("train", help="Train the model")
    p.add_argument("--tag", required=True, help="Model tag (e.g. v1, brier_clean)")
    p.add_argument("--time-limit", type=int, help="Training time limit in seconds")
    p.set_defaults(func=cmd_train)

    # ── predict ──
    p = subparsers.add_parser("predict", help="Generate men's submission CSV")
    p.add_argument("--tag", required=True, help="Model tag to load")
    p.set_defaults(func=cmd_predict)

    # ── submit ──
    p = subparsers.add_parser("submit", help="Generate full Kaggle submission (M+W)")
    p.add_argument("--tag", required=True, help="Model tag to load")
    p.set_defaults(func=cmd_submit)

    # ── bracket ──
    p = subparsers.add_parser("bracket", help="Monte Carlo bracket simulation")
    p.add_argument("--tag", required=True, help="Model tag")
    p.add_argument("--n-sims", type=int, default=10000, help="Number of simulations")
    p.add_argument("--submission", default=None, help="Override submission CSV path")
    p.add_argument("--season", type=int, default=PREDICTION_SEASON)
    p.set_defaults(func=cmd_bracket)

    # ── bet ──
    p = subparsers.add_parser("bet", help="Kelly Criterion head-to-head bet sheet")
    p.add_argument("--tag", required=True, help="Model tag")
    p.add_argument("--odds", required=True, help="Path to odds CSV")
    p.add_argument("--season", type=int, default=PREDICTION_SEASON)
    p.add_argument("--bankroll", type=float, default=1000.0)
    p.add_argument("--kelly", type=float, default=0.25, help="Kelly fraction (default: 0.25)")
    p.add_argument("--min-edge", type=float, default=0.02, help="Min edge to bet (default: 2%%)")
    p.add_argument("--max-bet", type=float, default=0.05, help="Max bet %% of bankroll")
    p.add_argument("--fee", type=float, default=0.0, help="Platform fee per contract ($)")
    p.set_defaults(func=cmd_bet)

    # ── futures ──
    p = subparsers.add_parser("futures", help="Kalshi futures bet sheet (YES/NO)")
    p.add_argument("--tag", required=True, help="Model tag")
    p.add_argument("--season", type=int, default=PREDICTION_SEASON)
    p.add_argument("--bankroll", type=float, default=1000.0)
    p.add_argument("--kelly", type=float, default=0.25)
    p.add_argument("--min-edge", type=float, default=0.02)
    p.add_argument("--max-bet", type=float, default=0.05)
    p.add_argument("--n-sims", type=int, default=10000)
    p.add_argument("--total-cost", type=float, default=1.01, help="YES+NO total cost")
    p.add_argument("--from-csv", help="Load market data from CSV instead of API")
    p.set_defaults(func=cmd_futures)

    # ── backtest ──
    p = subparsers.add_parser("backtest", help="Score bracket against actual results")
    p.add_argument("--tag", required=True, help="Model tag")
    p.add_argument("--season", type=int, default=2025, help="Season to backtest")
    p.add_argument("--n-sims", type=int, default=10000)
    p.add_argument("--submission", default=None, help="Override submission CSV")
    p.set_defaults(func=cmd_backtest)

    # ── analyze ──
    p_analyze = subparsers.add_parser("analyze", help="Matchup analysis and diagnostics")
    analyze_sub = p_analyze.add_subparsers(dest="analyze_cmd")

    m = analyze_sub.add_parser("matchups", help="Analyze seed matchups")
    m.add_argument("--tag", required=True)
    m.add_argument("--seeds", required=True, help="Seed pairing (e.g. 8v9, 5v12)")
    m.add_argument("--season", type=int, default=PREDICTION_SEASON)
    m.add_argument("--detail", action="store_true", default=True)
    m.add_argument("--no-detail", action="store_false", dest="detail")
    m.add_argument("--top-n", type=int, default=10)

    t = analyze_sub.add_parser("team", help="Team feature profile")
    t.add_argument("--tag", required=True)
    t.add_argument("--team", required=True, help="Team name (substring match)")
    t.add_argument("--season", type=int, default=PREDICTION_SEASON)

    c = analyze_sub.add_parser("confidence", help="Compare confidence distributions")
    c.add_argument("--tag", required=True)
    c.add_argument("--season1", type=int, default=2025)
    c.add_argument("--season2", type=int, default=PREDICTION_SEASON)

    p_analyze.set_defaults(func=cmd_analyze)

    # ── data ──
    p_data = subparsers.add_parser("data", help="Data management")
    data_sub = p_data.add_subparsers(dest="data_cmd")

    f = data_sub.add_parser("fetch", help="Fetch/refresh external data")
    f.add_argument("--source", nargs="*", help="Sources to fetch (default: all)")
    f.add_argument("--api-key", help="Odds API key (for odds source)")
    f.add_argument("--resume", action="store_true")

    data_sub.add_parser("status", help="Show data freshness")

    p_data.set_defaults(func=cmd_data)

    # ── full ──
    p = subparsers.add_parser("full", help="Train -> predict -> bracket (one shot)")
    p.add_argument("--tag", required=True, help="Model tag")
    p.add_argument("--time-limit", type=int, help="Training time limit in seconds")
    p.add_argument("--n-sims", type=int, default=10000)
    p.set_defaults(func=cmd_full)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
