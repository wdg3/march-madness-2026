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


def train(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    presets: str = "best_quality",
    time_limit: int = 3600,
    num_bag_folds: int = 10,
    num_stack_levels: int = 2,
    output_dir: str = "./AutogluonModels",
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
    print(predictor.leaderboard(val_data))

    return predictor
