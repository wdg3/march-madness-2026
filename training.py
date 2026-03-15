import pandas as pd
from autogluon.tabular import TabularPredictor


def train(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    hyperparameters: dict,
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
        hyperparameters=hyperparameters,
    )

    print("\nLeaderboard:")
    print(predictor.leaderboard(val_data))

    return predictor
