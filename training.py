import copy

import numpy as np
import pandas as pd
from autogluon.core.metrics import make_scorer
from autogluon.tabular import TabularPredictor
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config


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


def _gpu_hyperparameters():
    """Get the zeroshot hyperparameters with GPU enabled only for neural nets."""
    hp = copy.deepcopy(get_hyperparameter_config("zeroshot"))
    for model_type in ("NN_TORCH", "FASTAI"):
        configs = hp.get(model_type, [])
        if isinstance(configs, dict):
            configs = [configs]
        for cfg in configs:
            cfg.setdefault("ag_args_fit", {})["num_gpus"] = 1
        hp[model_type] = configs
    return hp


def train(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    presets: str = "best_quality",
    time_limit: int = 3600,
    num_bag_folds: int = 8,
    num_stack_levels: int = 1,
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
        hyperparameters=_gpu_hyperparameters(),
    )

    print("\nLeaderboard:")
    print(predictor.leaderboard(val_data))

    return predictor
