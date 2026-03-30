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
    """Train the model on men's historical data."""
    import json as _json
    from autogluon.tabular import TabularPredictor as _TabularPredictor
    from pipeline import build_team_features, build_training_data, build_matchups
    from features.travel import ensure_geocoded
    from seed_prior import SeedPrior
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

    use_player_nn = "player_nn" in features
    if use_player_nn:
        from models.player_train import train_player_model
        ckpt = DATA_DIR / "external" / "player_impact" / "player_nn.pt"
        if not ckpt.exists():
            val_season_cfg = train_cfg.get("validation_season", VALIDATION_SEASON)
            train_player_model(DATA_DIR, max_train_season=val_season_cfg - 1,
                               val_season=val_season_cfg)

    start_season = train_cfg.get("train_seasons_start", 2010)
    end_season = train_cfg.get("train_seasons_end", TRAIN_SEASONS_END)
    val_season = train_cfg.get("validation_season", VALIDATION_SEASON)

    # Build matchup rows from all game sources
    use_pbp_nn = "pbp_nn" in features
    print(f"\nBuilding matchup data (seasons {start_season}-{end_season})...")
    all_data = build_training_data(
        team_features, DATA_DIR,
        travel=use_travel,
        player_nn=use_player_nn,
        pbp_nn=use_pbp_nn,
        include_regular_season=train_cfg.get("include_regular_season", True),
        include_conf_tournament=train_cfg.get("include_conf_tournament", True),
        time_decay_half_life=train_cfg.get("time_decay_half_life", 5.0),
        game_weights=train_cfg.get("game_weights"),
    )
    all_data = all_data[all_data["Season"] >= start_season]

    # Drop rows without PBP embeddings if pbp_nn is enabled
    if use_pbp_nn and "pbp_mu_0" in all_data.columns:
        before = len(all_data)
        all_data = all_data[all_data["pbp_mu_0"].notna()].reset_index(drop=True)
        print(f"  PBP filter: {before} → {len(all_data)} rows "
              f"(dropped {before - len(all_data)} without embeddings)")

    # Split: validation = tournament games from val season
    #        training = everything else (regular season + conf tourney for all seasons)
    is_val = (all_data["Season"] == val_season) & (all_data["is_ncaa_tournament"] == 1)
    val_data = all_data[is_val].copy()
    val_data["sample_weight"] = 1.0
    train_data = all_data[~is_val & (all_data["Season"] <= end_season)].copy()

    # Summary
    print(f"\n  {'='*50}")
    print(f"  DATASET SUMMARY")
    print(f"  {'='*50}")
    print(f"  Training:   {len(train_data):>7,} rows")
    for s in sorted(train_data["Season"].unique()):
        n = len(train_data[train_data["Season"] == s])
        print(f"    {s}: {n:>6,} rows")
    print(f"  Validation: {len(val_data):>7,} rows "
          f"({val_season} NCAA tournament)")
    print(f"  Features:   {len([c for c in train_data.columns if c not in ('Season','TeamID_A','TeamID_B','Label','sample_weight','is_ncaa_tournament','is_conf_tournament','_game_weight')]):>7,} columns")
    print(f"  {'='*50}\n")

    drop_cols = ["Season", "TeamID_A", "TeamID_B"]
    train_data = train_data.drop(columns=drop_cols)
    val_data = val_data.drop(columns=drop_cols)

    presets = train_cfg.get("presets", AG_PRESETS)
    time_limit = args.time_limit or train_cfg.get("time_limit", AG_TIME_LIMIT)
    num_bag_folds = train_cfg.get("num_bag_folds", AG_NUM_BAG_FOLDS)
    num_stack_levels = train_cfg.get("num_stack_levels", AG_NUM_STACK_LEVELS)

    predictor = train(
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
        eval_metric=train_cfg.get("eval_metric", "brier_score"),
    )

    # Tune seed prior alpha on validation tournament games
    if train_cfg.get("use_seed_prior", True):
        print("\nTuning seed prior...")
        prior = SeedPrior(DATA_DIR, max_season=val_season)

        # Get model predictions on val tournament games
        tourney = pd.read_csv(DATA_DIR / "MNCAATourneyCompactResults.csv")
        val_tourney = build_matchups(team_features, tourney, data_dir=DATA_DIR, travel=use_travel, player_nn=use_player_nn, pbp_nn=use_pbp_nn)
        val_tourney["is_ncaa_tournament"] = 1
        val_tourney["is_conf_tournament"] = 0
        val_tourney = val_tourney[val_tourney["Season"] == val_season].copy()

        val_seasons = val_tourney["Season"].values
        val_team_a = val_tourney["TeamID_A"].values
        val_team_b = val_tourney["TeamID_B"].values

        val_X = val_tourney.drop(columns=["Season", "TeamID_A", "TeamID_B", "Label"])
        model_probs = predictor.predict_proba(val_X)[1].values
        actuals = val_tourney["Label"].values

        seed_a = [prior.get_seeds(int(s), int(a), int(b))[0] or 8
                  for s, a, b in zip(val_seasons, val_team_a, val_team_b)]
        seed_b = [prior.get_seeds(int(s), int(a), int(b))[1] or 8
                  for s, a, b in zip(val_seasons, val_team_a, val_team_b)]

        alpha = prior.tune_alpha(model_probs, seed_a, seed_b, actuals)

        # Save alpha alongside model
        alpha_path = out_dir / "seed_prior_alpha.json"
        with open(alpha_path, "w") as f:
            _json.dump({"alpha": alpha, "val_season": val_season}, f)
        print(f"  Alpha saved to {alpha_path}")

    print(f"\nTraining complete! Model saved to {out_dir}")


# ── Predict ──────────────────────────────────────────────────────────────────


def _load_seed_prior(tag: str):
    """Load seed prior with tuned alpha for a trained model, or None."""
    import json as _json
    from seed_prior import SeedPrior

    alpha_path = model_dir(tag) / "seed_prior_alpha.json"
    if not alpha_path.exists():
        return None

    with open(alpha_path) as f:
        meta = _json.load(f)

    # Build prior using all data up to prediction season (no contamination)
    prior = SeedPrior(DATA_DIR, max_season=PREDICTION_SEASON + 1)
    prior._alpha = meta["alpha"]
    print(f"  Loaded seed prior (α={meta['alpha']:.2f})")
    return prior


def cmd_predict(args):
    """Generate men's-only submission CSV."""
    from autogluon.tabular import TabularPredictor
    from pipeline import build_team_features, build_prediction_pairs
    from features.travel import ensure_geocoded
    from submission import generate_submission

    cfg = load_config(args.tag)
    features = cfg.get("features", ENABLED_FEATURES)
    _verify_manifest(args.tag, features)

    mdir = model_dir(args.tag)
    predictor = TabularPredictor.load(str(mdir))
    seed_prior = _load_seed_prior(args.tag)
    team_features = build_team_features(DATA_DIR, features, gender="M")

    use_travel = "travel" in features
    if use_travel:
        ensure_geocoded(DATA_DIR)
    use_player_nn = "player_nn" in features
    use_pbp_nn = "pbp_nn" in features

    pred_pairs = build_prediction_pairs(
        team_features, PREDICTION_SEASON, data_dir=DATA_DIR, travel=use_travel,
        player_nn=use_player_nn, pbp_nn=use_pbp_nn,
    )

    sample_sub = pd.read_csv(DATA_DIR / "SampleSubmissionStage2.csv")
    out_dir = tag_output_dir(args.tag)
    output_path = out_dir / "submission.csv"
    generate_submission(predictor, pred_pairs, sample_sub, output_path, seed_prior=seed_prior)
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

    mdir = model_dir(args.tag)
    predictor = TabularPredictor.load(str(mdir))
    seed_prior = _load_seed_prior(args.tag)
    use_travel = "travel" in features
    use_player_nn = "player_nn" in features
    use_pbp_nn = "pbp_nn" in features

    # Men's predictions
    print("=" * 60)
    print("  MEN'S PREDICTIONS")
    print("=" * 60)
    men_features = build_team_features(DATA_DIR, features, gender="M")
    if use_travel:
        ensure_geocoded(DATA_DIR)
    men_pairs = build_prediction_pairs(
        men_features, PREDICTION_SEASON, data_dir=DATA_DIR, travel=use_travel,
        player_nn=use_player_nn, pbp_nn=use_pbp_nn,
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
    generate_submission(predictor, all_pairs, sample_sub, output_path, seed_prior=seed_prior)

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
    adv_probs, team_names, tourney_teams = compute_advancement_probs(
        submission, args.season, DATA_DIR, args.n_sims, seed=42
    )

    df = generate_futures_bets(
        markets, adv_probs, team_names, name_to_id,
        total_cost=args.total_cost,
        kelly_frac=args.kelly,
        min_edge=args.min_edge,
        max_bet_pct=args.max_bet,
        bankroll=args.bankroll,
        fee=args.fee,
        tourney_teams=tourney_teams,
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
    from analyze import cmd_matchups, cmd_team, cmd_confidence, cmd_importance

    if args.analyze_cmd == "matchups":
        cmd_matchups(args)
    elif args.analyze_cmd == "team":
        cmd_team(args)
    elif args.analyze_cmd == "confidence":
        cmd_confidence(args)
    elif args.analyze_cmd == "importance":
        cmd_importance(args)
    else:
        print("Usage: python run.py analyze {matchups|team|confidence|importance} --tag TAG ...")


# ── Data ─────────────────────────────────────────────────────────────────────


def cmd_data(args):
    """Data management: fetch and status."""
    if args.data_cmd == "fetch":
        _cmd_data_fetch(args)
    elif args.data_cmd == "fetch-pbp":
        _cmd_data_fetch_pbp(args)
    elif args.data_cmd == "status":
        _cmd_data_status(args)
    else:
        print("Usage: python run.py data {fetch|fetch-pbp|status}")


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


def _cmd_data_fetch_pbp(args):
    """Fetch play-by-play data from CBBD API."""
    from features.pbp import fetch_all, fetch_team_plays, fetch_status

    api_key = args.api_key
    if not api_key:
        print("Error: --api-key required for PBP fetch")
        return

    if args.team:
        # Single team fetch (for testing)
        plays = fetch_team_plays(api_key, DATA_DIR, args.season, args.team)
        n_floor = sum(1 for p in plays if p.get("onFloor") and len(p["onFloor"]) >= 10)
        print(f"{args.team} {args.season}: {len(plays)} plays ({n_floor} with lineup)")
    else:
        # Full fetch
        fetch_all(api_key, DATA_DIR, rate_limit=args.rate_limit)

    print()
    fetch_status(DATA_DIR)


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


# ── Player NN ────────────────────────────────────────────────────────────────


def cmd_player_nn(args):
    """Train or inspect the player matchup neural net."""
    if args.pnn_cmd == "train":
        _cmd_pnn_train(args)
    elif args.pnn_cmd == "test":
        _cmd_pnn_test(args)
    elif args.pnn_cmd == "submit":
        _cmd_pnn_submit(args)
    elif args.pnn_cmd == "bracket":
        _cmd_pnn_bracket(args)
    else:
        print("Usage: python run.py player-nn {train|test|submit|bracket}")


def _cmd_pnn_train(args):
    """Train the player matchup model."""
    from models.player_train import train_player_model

    train_player_model(
        DATA_DIR,
        max_train_season=args.val_season - 1,
        val_season=args.val_season,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        time_limit=args.time_limit,
    )


def _cmd_pnn_test(args):
    """Test the trained model: show embeddings and matchup predictions."""
    from models.player_train import PlayerNNExtractor
    import numpy as np

    ext = PlayerNNExtractor(DATA_DIR, device=args.device)

    # Team embeddings summary
    emb = ext.team_embeddings(season=args.season)
    teams_df = pd.read_csv(DATA_DIR / "MTeams.csv")
    emb = emb.merge(teams_df[["TeamID", "TeamName"]], on="TeamID")
    emb_cols = [c for c in emb.columns if c.startswith("pnn_emb_")]
    print(f"Team embeddings for {args.season}: {len(emb)} teams × {len(emb_cols)} dims")

    # Show top teams by embedding norm (proxy for "model thinks this team is distinctive")
    emb["emb_norm"] = np.sqrt((emb[emb_cols] ** 2).sum(axis=1))
    print(f"\nTop 10 by embedding magnitude:")
    for _, r in emb.nlargest(10, "emb_norm").iterrows():
        print(f"  {r['TeamName']:25s} ‖emb‖={r['emb_norm']:.3f}")

    # Sample matchup predictions
    seeds = pd.read_csv(DATA_DIR / "MNCAATourneySeeds.csv")
    seeds_yr = seeds[seeds["Season"] == args.season].copy()
    seeds_yr["seed_num"] = seeds_yr["Seed"].str.extract(r"(\d+)").astype(int)
    seeds_yr = seeds_yr.merge(teams_df[["TeamID", "TeamName"]], on="TeamID")
    top_seeds = seeds_yr.nsmallest(8, "seed_num")

    if len(top_seeds) >= 2:
        print(f"\nSample matchup predictions ({args.season}):")
        ids = top_seeds["TeamID"].values
        names = dict(zip(top_seeds["TeamID"], top_seeds["TeamName"]))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pred = ext.matchup_predictions(
                    np.array([ids[i]]), np.array([ids[j]]),
                    np.array([args.season]),
                )
                a, b = names[ids[i]], names[ids[j]]
                print(f"  {a:20s} vs {b:20s}: P({a} wins) = {pred[0]:.3f}")


def _cmd_pnn_submit(args):
    """Generate a Kaggle submission from the player NN alone (no AutoGluon)."""
    from models.player_train import generate_pnn_submission

    out_dir = tag_output_dir(args.tag)
    output_path = out_dir / "submission.csv"
    generate_pnn_submission(DATA_DIR, args.season, output_path, device=args.device)


def _cmd_pnn_bracket(args):
    """Run bracket simulation from player NN submission."""
    import simulate

    out_dir = tag_output_dir(args.tag)
    submission = str(out_dir / "submission.csv")
    output_path = out_dir / "bracket.csv"
    simulate.run(submission, args.season, args.n_sims, output_path=str(output_path))


# ── PBP Deep Model ────────────────────────────────────────────────────────────


def cmd_pbp(args):
    """Train, predict, or simulate with the PBP deep model."""
    if args.pbp_cmd == "train":
        from models.pbp_train import train_pbp_model
        train_pbp_model(DATA_DIR, time_limit=args.time_limit, device=args.device,
                        resume=args.resume, version=args.version,
                        loss_fn=args.loss)
    elif args.pbp_cmd == "submit":
        from models.pbp_train import (
            load_pbp_data, generate_predictions,
            _load_checkpoint, N_PLAY_TYPES,
        )
        from models.pbp_model import PBPMatchupModel
        import torch
        import json as _json

        from models.pbp_train import _ckpt_dir
        ckpt = _ckpt_dir(DATA_DIR, getattr(args, "version", None))
        with open(ckpt / "player_index.json") as f:
            p2i = {int(k): v for k, v in _json.load(f).items()}
        with open(ckpt / "config.json") as f:
            cfg = _json.load(f)

        model = PBPMatchupModel(
            n_players=cfg["n_players"],
            embed_dim=cfg.get("embed_dim", 64),
            player_dim=cfg.get("player_dim", 32),
            n_play_types=cfg.get("n_play_types", N_PLAY_TYPES),
            ptype_dim=cfg.get("ptype_dim", 8),
            n_heads=cfg.get("n_heads", 4),
            n_season_layers=cfg.get("n_season_layers", 2),
        )
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        _load_checkpoint(model, DATA_DIR, "best", device,
                         version=getattr(args, "version", None))
        model.to(device)

        # Load data using checkpoint's player index for consistent embeddings
        games, _p2i_new, ts_games, ptensors = load_pbp_data(DATA_DIR, seasons=[2024, 2025, 2026])

        preds = generate_predictions(model, p2i, games, ts_games, ptensors,
                                     DATA_DIR, device=device)
        out_dir = tag_output_dir(args.tag)
        out_path = out_dir / "submission.csv"
        preds.to_csv(out_path, index=False)
        print(f"  Saved {len(preds)} predictions to {out_path}")
    elif args.pbp_cmd == "bracket":
        import simulate
        out_dir = tag_output_dir(args.tag)
        submission = str(out_dir / "submission.csv")
        output_path = out_dir / "bracket.csv"
        simulate.run(submission, args.season, args.n_sims, output_path=str(output_path))
    else:
        print("Usage: python run.py pbp {train|submit|bracket}")


# ── Composable PBP ───────────────────────────────────────────────────────────


def cmd_cpbp(args):
    """Composable PBP player-level model commands."""
    if args.cpbp_cmd == "train":
        from models.composable_pbp_train import train_composable_pbp
        train_composable_pbp(
            DATA_DIR, time_limit=args.time_limit,
            device=args.device, version=args.version,
        )
    elif args.cpbp_cmd == "submit":
        from models.composable_pbp_train import generate_cpbp_predictions
        preds = generate_cpbp_predictions(
            DATA_DIR, season=PREDICTION_SEASON,
            device=args.device, version=args.version,
            exclusion_file=getattr(args, "exclusions", None),
        )
        out_dir = Path("output") / args.tag
        out_dir.mkdir(parents=True, exist_ok=True)
        preds.to_csv(out_dir / "submission.csv", index=False)
        print(f"  Saved to {out_dir / 'submission.csv'}")
    elif args.cpbp_cmd == "bracket":
        from simulate import run as sim_run
        out_dir = Path("output") / args.tag
        sim_run(
            submission=str(out_dir / "submission.csv"),
            season=PREDICTION_SEASON,
            n_sims=args.n_sims,
            output_path=str(out_dir / "bracket.csv"),
        )
    else:
        print("Usage: python run.py cpbp {train|submit|bracket}")


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
    p.add_argument("--fee", type=float, default=0.02, help="Per-contract fee (default: 0.02)")
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

    fi = analyze_sub.add_parser("importance", help="Feature importance (permutation)")
    fi.add_argument("--tag", required=True)
    fi.add_argument("--top-n", type=int, default=30, help="Number of top features to show")
    fi.add_argument("--shuffles", type=int, default=5, help="Permutation shuffle sets")
    fi.add_argument("--team", default=None, help="Show team's values for top features")
    fi.add_argument("--season", type=int, default=PREDICTION_SEASON)

    p_analyze.set_defaults(func=cmd_analyze)

    # ── data ──
    p_data = subparsers.add_parser("data", help="Data management")
    data_sub = p_data.add_subparsers(dest="data_cmd")

    f = data_sub.add_parser("fetch", help="Fetch/refresh external data")
    f.add_argument("--source", nargs="*", help="Sources to fetch (default: all)")
    f.add_argument("--api-key", help="Odds API key (for odds source)")
    f.add_argument("--resume", action="store_true")

    pbp = data_sub.add_parser("fetch-pbp", help="Fetch play-by-play from CBBD API")
    pbp.add_argument("--api-key", required=True, help="CBBD API key")
    pbp.add_argument("--team", default=None, help="Single team (for testing)")
    pbp.add_argument("--season", type=int, default=2026, help="Season (with --team)")
    pbp.add_argument("--rate-limit", type=float, default=1.0,
                     help="Seconds between API calls (default: 1.0)")

    data_sub.add_parser("status", help="Show data freshness")

    p_data.set_defaults(func=cmd_data)

    # ── player-nn ──
    p_pnn = subparsers.add_parser("player-nn", help="Player matchup neural net")
    pnn_sub = p_pnn.add_subparsers(dest="pnn_cmd")

    pnn_train = pnn_sub.add_parser("train", help="Train the player matchup model")
    pnn_train.add_argument("--val-season", type=int, default=VALIDATION_SEASON,
                           help="Validation season (trains on prior seasons)")
    pnn_train.add_argument("--epochs", type=int, default=500)
    pnn_train.add_argument("--batch-size", type=int, default=512)
    pnn_train.add_argument("--device", default=None, help="cuda or cpu (auto-detected)")
    pnn_train.add_argument("--time-limit", type=int, default=None,
                           help="Max training time in seconds (e.g. 7200 for 2h)")

    pnn_test = pnn_sub.add_parser("test", help="Test trained model: embeddings & predictions")
    pnn_test.add_argument("--season", type=int, default=PREDICTION_SEASON)
    pnn_test.add_argument("--device", default=None)

    pnn_submit = pnn_sub.add_parser("submit", help="Generate submission from player NN only")
    pnn_submit.add_argument("--tag", required=True, help="Output tag (e.g. pnn_v1)")
    pnn_submit.add_argument("--season", type=int, default=PREDICTION_SEASON)
    pnn_submit.add_argument("--device", default=None)

    pnn_bracket = pnn_sub.add_parser("bracket", help="Bracket sim from player NN submission")
    pnn_bracket.add_argument("--tag", required=True, help="Output tag (must have submission)")
    pnn_bracket.add_argument("--season", type=int, default=PREDICTION_SEASON)
    pnn_bracket.add_argument("--n-sims", type=int, default=10000)

    p_pnn.set_defaults(func=cmd_player_nn)

    # ── pbp ──
    p_pbp = subparsers.add_parser("pbp", help="PBP deep matchup model")
    pbp_sub = p_pbp.add_subparsers(dest="pbp_cmd")

    pbp_tr = pbp_sub.add_parser("train", help="Train the PBP deep model")
    pbp_tr.add_argument("--time-limit", type=int, default=7200,
                        help="Max training time in seconds (default: 7200)")
    pbp_tr.add_argument("--device", default=None, help="cuda or cpu")
    pbp_tr.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    pbp_tr.add_argument("--version", default=None,
                        help="Model version tag (e.g. v4_brier). Saves to versioned dir.")
    pbp_tr.add_argument("--loss", default="brier", choices=["brier", "bce"],
                        help="Loss function: brier (MSE on probs) or bce (default: brier)")

    pbp_sub_cmd = pbp_sub.add_parser("submit", help="Generate submission from PBP model")
    pbp_sub_cmd.add_argument("--tag", required=True, help="Output tag")
    pbp_sub_cmd.add_argument("--version", default=None,
                             help="Model version to load (default: unversioned)")
    pbp_sub_cmd.add_argument("--device", default=None)

    pbp_bracket = pbp_sub.add_parser("bracket", help="Bracket from PBP submission")
    pbp_bracket.add_argument("--tag", required=True)
    pbp_bracket.add_argument("--season", type=int, default=PREDICTION_SEASON)
    pbp_bracket.add_argument("--n-sims", type=int, default=10000)

    p_pbp.set_defaults(func=cmd_pbp)

    # ── cpbp (composable PBP) ──
    p_cpbp = subparsers.add_parser("cpbp", help="Composable PBP player-level model")
    cpbp_sub = p_cpbp.add_subparsers(dest="cpbp_cmd")

    cpbp_tr = cpbp_sub.add_parser("train", help="Train the composable PBP model")
    cpbp_tr.add_argument("--time-limit", type=int, default=14400,
                         help="Max training time in seconds (default: 14400)")
    cpbp_tr.add_argument("--device", default=None, help="cuda or cpu")
    cpbp_tr.add_argument("--version", default=None,
                         help="Model version tag")

    cpbp_sub_cmd = cpbp_sub.add_parser("submit", help="Generate submission")
    cpbp_sub_cmd.add_argument("--tag", required=True, help="Output tag")
    cpbp_sub_cmd.add_argument("--device", default=None)
    cpbp_sub_cmd.add_argument("--version", default=None)
    cpbp_sub_cmd.add_argument("--exclusions", default=None,
                              help="JSON file with player exclusions")

    cpbp_br = cpbp_sub.add_parser("bracket", help="Simulate bracket")
    cpbp_br.add_argument("--tag", required=True, help="Output tag")
    cpbp_br.add_argument("--n-sims", type=int, default=10000)

    p_cpbp.set_defaults(func=cmd_cpbp)

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
