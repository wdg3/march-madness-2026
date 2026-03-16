import pandas as pd
from autogluon.tabular import TabularPredictor


def train(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    presets: str = "best_quality",
    time_limit: int = 3600,
    num_bag_folds: int = 8,
    num_stack_levels: int = 1,
    output_dir: str = "./AutogluonModels",
) -> TabularPredictor:
    """Train AutoGluon TabularPredictor optimized for log loss."""
    print(f"Training on {len(train_data)} rows, validating on {len(val_data)} rows...")

    predictor = TabularPredictor(
        label="Label",
        eval_metric="log_loss",
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
        excluded_model_types=["NN_TORCH", "FASTAI"],
    )

    print("\nLeaderboard:")
    print(predictor.leaderboard(val_data))

    return predictor
