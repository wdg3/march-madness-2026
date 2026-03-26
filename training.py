import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.core.metrics import make_scorer
from autogluon.tabular import TabularPredictor


def _brier_score(y_true, y_pred):
    """Brier score (lower is better)."""
    return np.mean((y_pred - y_true) ** 2)


brier_scorer = make_scorer(
    "brier_score",
    _brier_score,
    optimum=0,
    greater_is_better=False,
    needs_proba=True,
)


def _get_eval_metric(name: str):
    """Return the AutoGluon eval metric for the given name."""
    if name == "log_loss":
        return "log_loss"
    return brier_scorer


def _regularized_hyperparameters():
    """Hyperparameters tuned for small tournament datasets with high feature counts.

    Key constraints vs zeroshot defaults:
    - Capped tree depth to prevent memorizing noise
    - L1/L2 regularization to shrink noisy feature weights
    - Feature subsampling (colsample_bytree) so no single noisy feature
      dominates every tree
    - Higher min_child_samples/min_child_weight to avoid leaf overfitting

    5 model types: LightGBM, XGBoost, CatBoost, RandomForest, NeuralNet.
    Single config per type to avoid correlated models inflating ensemble weight.
    """
    return {
        "GBM": [
            {
                "max_depth": 6,
                "num_leaves": 31,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "min_child_samples": 20,
                "colsample_bytree": 0.7,
                "learning_rate": 0.05,
                "n_estimators": 2000,
                "ag_args_fit": {"num_gpus": 0},
            },
        ],
        "XGB": [
            {
                "max_depth": 5,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "colsample_bytree": 0.7,
                "min_child_weight": 5,
                "learning_rate": 0.05,
                "n_estimators": 2000,
                "ag_args_fit": {"num_gpus": 0},
            },
        ],
        "CAT": [
            {
                "depth": 5,
                "l2_leaf_reg": 3.0,
                "learning_rate": 0.05,
                "iterations": 2000,
                "rsm": 0.7,
                "ag_args_fit": {"num_gpus": 0},
            },
        ],
        "RF": [
            {
                "max_depth": 8,
                "min_samples_leaf": 10,
                "max_features": 0.7,
                "n_estimators": 500,
                "ag_args_fit": {"num_gpus": 0},
            },
        ],
        "FASTAI": [
            {
                "epochs": 200,
                "wd": 1e-3,
                "ps": 0.3,
                "ag_args_fit": {"num_gpus": 1},
            },
        ],
    }


def _git_sha() -> str:
    """Get current git commit SHA, or 'unknown' if not in a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _save_manifest(
    output_dir: str,
    tag: str,
    train_rows: int,
    val_rows: int,
    val_brier: float,
    feature_shape: tuple,
    features: list[str],
    config: dict,
    data_dir: Path = None,
):
    """Save a provenance manifest alongside the trained model."""
    data_files = {}
    if data_dir:
        for f in sorted(data_dir.glob("M*.csv")):
            data_files[f.name] = f.stat().st_mtime_ns
        ext_dir = data_dir / "external"
        if ext_dir.exists():
            for source_dir in sorted(ext_dir.iterdir()):
                if source_dir.is_dir():
                    for f in sorted(source_dir.glob("*.csv")):
                        data_files[f"external/{source_dir.name}/{f.name}"] = f.stat().st_mtime_ns

    manifest = {
        "tag": tag,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "features": features,
        "config": config,
        "feature_shape": list(feature_shape),
        "train_rows": train_rows,
        "val_rows": val_rows,
        "val_brier": val_brier,
        "data_files": data_files,
    }

    manifest_path = Path(output_dir) / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  Manifest saved to {manifest_path}")


def train(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    presets: str = "best_quality",
    time_limit: int = 3600,
    num_bag_folds: int = 10,
    num_stack_levels: int = 2,
    output_dir: str = "./AutogluonModels",
    tag: str = "",
    config: dict = None,
    features: list[str] = None,
    data_dir: Path = None,
    eval_metric: str = "brier_score",
) -> TabularPredictor:
    """Train AutoGluon TabularPredictor with configurable eval metric.

    Args:
        train_data: Training DataFrame. May include a 'sample_weight' column for
            per-row importance weighting (time decay * game type).
        val_data: Validation DataFrame (full season, equal weights for model selection).
        eval_metric: 'brier_score' or 'log_loss'. Log loss provides sharper gradients
            for model selection; Brier score matches Kaggle's grading.
    """
    print(f"Training on {len(train_data)} rows, validating on {len(val_data)} rows...")
    print(f"  Eval metric: {eval_metric}")

    has_weights = "sample_weight" in train_data.columns
    if has_weights:
        print(f"  Using sample weights (range: [{train_data['sample_weight'].min():.4f}, "
              f"{train_data['sample_weight'].max():.4f}])")

    metric = _get_eval_metric(eval_metric)

    predictor = TabularPredictor(
        label="Label",
        eval_metric=metric,
        path=output_dir,
        sample_weight="sample_weight" if has_weights else None,
    ).fit(
        train_data=train_data,
        tuning_data=val_data,
        presets=presets,
        time_limit=time_limit,
        num_bag_folds=num_bag_folds,
        num_stack_levels=num_stack_levels,
        use_bag_holdout=True,
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
        ag_args_fit={"num_gpus": 0},
        hyperparameters=_regularized_hyperparameters(),
    )

    print("\nLeaderboard:")
    lb = predictor.leaderboard(val_data)
    print(lb)

    val_brier = float(lb["score_val"].iloc[0]) if len(lb) > 0 else None
    _save_manifest(
        output_dir=output_dir,
        tag=tag,
        train_rows=len(train_data),
        val_rows=len(val_data),
        val_brier=val_brier,
        feature_shape=(len(train_data), len(train_data.columns)),
        features=features or [],
        config=config or {},
        data_dir=data_dir,
    )

    return predictor
