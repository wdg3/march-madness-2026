"""March Madness 2026 — CLI entry point.

Usage:
    python run.py train     Train the model on men's historical data
    python run.py predict   Generate men's-only submission CSV
    python run.py bracket   Run Monte Carlo bracket simulation
    python run.py submit    Generate full Kaggle submission (men's + women's)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from config import (
    DATA_DIR, OUTPUT_DIR, TRAIN_SEASONS_END, VALIDATION_SEASON,
    PREDICTION_SEASON, AG_PRESETS, AG_TIME_LIMIT, AG_NUM_BAG_FOLDS,
    AG_NUM_STACK_LEVELS, ENABLED_FEATURES,
)

MODEL_DIR = Path("./AutogluonModels")


def cmd_train(args):
    """Train the model on men's historical tournament data."""
    from pipeline import build_team_features, build_matchups
    from features.travel import ensure_geocoded
    from training import train

    # Build men's features
    team_features = build_team_features(DATA_DIR, ENABLED_FEATURES, gender="M")

    use_travel = "travel" in ENABLED_FEATURES
    if use_travel:
        ensure_geocoded(DATA_DIR)

    # Build matchups from men's tournament results
    tourney = pd.read_csv(DATA_DIR / "MNCAATourneyCompactResults.csv")
    matchups = build_matchups(team_features, tourney, data_dir=DATA_DIR, travel=use_travel)

    # Filter to 2010+ (where external features exist)
    matchups = matchups[matchups["Season"] >= 2010]

    train_data = matchups[matchups["Season"] < TRAIN_SEASONS_END].copy()
    val_data = matchups[matchups["Season"] == VALIDATION_SEASON].copy()

    drop_cols = ["Season", "TeamID_A", "TeamID_B"]
    train_data = train_data.drop(columns=drop_cols)
    val_data = val_data.drop(columns=drop_cols)

    train(
        train_data, val_data,
        presets=AG_PRESETS,
        time_limit=args.time_limit or AG_TIME_LIMIT,
        num_bag_folds=AG_NUM_BAG_FOLDS,
        num_stack_levels=AG_NUM_STACK_LEVELS,
        output_dir=str(MODEL_DIR),
    )
    print("\nTraining complete!")


def cmd_predict(args):
    """Generate men's-only submission CSV."""
    from autogluon.tabular import TabularPredictor
    from pipeline import build_team_features, build_prediction_pairs
    from features.travel import ensure_geocoded
    from submission import generate_submission

    predictor = TabularPredictor.load(str(MODEL_DIR))

    team_features = build_team_features(DATA_DIR, ENABLED_FEATURES, gender="M")

    use_travel = "travel" in ENABLED_FEATURES
    if use_travel:
        ensure_geocoded(DATA_DIR)

    pred_pairs = build_prediction_pairs(
        team_features, PREDICTION_SEASON, data_dir=DATA_DIR, travel=use_travel,
    )

    sample_sub = pd.read_csv(DATA_DIR / "SampleSubmissionStage2.csv")
    output_path = OUTPUT_DIR / "submission.csv"
    generate_submission(predictor, pred_pairs, sample_sub, output_path)
    print(f"\nMen's submission saved to {output_path}")


def cmd_bracket(args):
    """Run Monte Carlo bracket simulation."""
    import simulate
    simulate.run(args.submission, args.season, args.n_sims)


def cmd_submit(args):
    """Generate full Kaggle submission with both men's and women's predictions."""
    from autogluon.tabular import TabularPredictor
    from pipeline import build_team_features, build_prediction_pairs
    from features.travel import ensure_geocoded
    from submission import generate_submission

    predictor = TabularPredictor.load(str(MODEL_DIR))

    use_travel = "travel" in ENABLED_FEATURES

    # --- Men's predictions ---
    print("=" * 60)
    print("  MEN'S PREDICTIONS")
    print("=" * 60)
    men_features = build_team_features(DATA_DIR, ENABLED_FEATURES, gender="M")
    if use_travel:
        ensure_geocoded(DATA_DIR)
    men_pairs = build_prediction_pairs(
        men_features, PREDICTION_SEASON, data_dir=DATA_DIR, travel=use_travel,
    )

    # --- Women's predictions ---
    print("\n" + "=" * 60)
    print("  WOMEN'S PREDICTIONS")
    print("=" * 60)
    women_features = build_team_features(DATA_DIR, ENABLED_FEATURES, gender="W")

    # Ensure women's features have the same columns as men's (model expects this)
    for col in men_features.columns:
        if col not in women_features.columns:
            women_features[col] = float("nan")
    women_features = women_features[men_features.columns]

    women_pairs = build_prediction_pairs(
        women_features, PREDICTION_SEASON, data_dir=DATA_DIR, travel=False,
    )

    # Combine men's and women's prediction pairs
    all_pairs = pd.concat([men_pairs, women_pairs], ignore_index=True)

    sample_sub = pd.read_csv(DATA_DIR / "SampleSubmissionStage2.csv")
    output_path = OUTPUT_DIR / "submission.csv"
    generate_submission(predictor, all_pairs, sample_sub, output_path)
    print(f"\nFull submission saved to {output_path}")

    # Count men's and women's rows
    sub = pd.read_csv(output_path)
    ids = sub["ID"].str.split("_", expand=True)
    n_men = (ids[1].astype(int) < 3000).sum()
    n_women = (ids[1].astype(int) >= 3000).sum()
    print(f"  Men's pairs: {n_men}")
    print(f"  Women's pairs: {n_women}")
    print(f"  Total: {len(sub)}")


def main():
    parser = argparse.ArgumentParser(
        description="March Madness 2026 Prediction Pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = subparsers.add_parser("train", help="Train the model")
    p_train.add_argument("--time-limit", type=int, help="Training time limit in seconds")
    p_train.set_defaults(func=cmd_train)

    # predict
    p_predict = subparsers.add_parser("predict", help="Generate men's submission CSV")
    p_predict.set_defaults(func=cmd_predict)

    # bracket
    p_bracket = subparsers.add_parser("bracket", help="Monte Carlo bracket simulation")
    p_bracket.add_argument("--n-sims", type=int, default=10000, help="Number of simulations")
    p_bracket.add_argument("--submission", default="output/submission.csv")
    p_bracket.add_argument("--season", type=int, default=PREDICTION_SEASON)
    p_bracket.set_defaults(func=cmd_bracket)

    # submit
    p_submit = subparsers.add_parser("submit", help="Generate full Kaggle submission (men's + women's)")
    p_submit.set_defaults(func=cmd_submit)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
