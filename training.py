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


def _regularized_hyperparameters():
    """Hyperparameters tuned for small tournament datasets with high feature counts.

    Key constraints vs zeroshot defaults:
    - Capped tree depth to prevent memorizing noise
    - L1/L2 regularization to shrink noisy feature weights
    - Feature subsampling (colsample_bytree) so no single noisy feature
      dominates every tree
    - Higher min_child_samples/min_child_weight to avoid leaf overfitting
    """
    return {
        "GBM": [
            # LightGBM config 1: moderate depth, strong regularization
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
            # LightGBM config 2: shallower, more aggressive subsampling
            {
                "max_depth": 4,
                "num_leaves": 15,
                "reg_alpha": 0.5,
                "reg_lambda": 2.0,
                "min_child_samples": 30,
                "colsample_bytree": 0.5,
                "subsample": 0.8,
                "learning_rate": 0.03,
                "n_estimators": 3000,
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
            {
                "max_depth": 3,
                "reg_alpha": 0.5,
                "reg_lambda": 2.0,
                "colsample_bytree": 0.5,
                "subsample": 0.8,
                "min_child_weight": 10,
                "learning_rate": 0.03,
                "n_estimators": 3000,
                "ag_args_fit": {"num_gpus": 0},
            },
        ],
        "CAT": [
            {
                "depth": 5,
                "l2_leaf_reg": 3.0,
                "learning_rate": 0.05,
                "iterations": 2000,
                "rsm": 0.7,  # CatBoost's colsample_bytree equivalent
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
        "XT": [
            {
                "max_depth": 8,
                "min_samples_leaf": 10,
                "max_features": 0.7,
                "n_estimators": 500,
                "ag_args_fit": {"num_gpus": 0},
            },
        ],
        "NN_TORCH": [
            {
                "num_epochs": 200,
                "weight_decay": 1e-3,
                "dropout_prob": 0.3,
                "ag_args_fit": {"num_gpus": 1},
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
) -> TabularPredictor:
    """Train AutoGluon TabularPredictor optimized for Brier score."""
    print(f"Training on {len(train_data)} rows, validating on {len(val_data)} rows...")

    predictor = TabularPredictor(
        label="Label",
        eval_metric=brier_scorer,
        path=output_dir,
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

    # Save provenance manifest
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
