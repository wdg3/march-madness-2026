from pathlib import Path

DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./output")

# Seasons
TRAIN_SEASONS_END = 2025  # train on seasons < this
VALIDATION_SEASON = 2025  # validate on this season
PREDICTION_SEASON = 2026

# AutoGluon
AG_HYPERPARAMETERS = {
    "NN_TORCH": {},
    "CAT": {},
    "FASTAI": {},
    "RF": [
        {"criterion": "gini", "ag_args": {"name_suffix": "Gini", "problem_types": ["binary", "multiclass"]}},
        {"criterion": "entropy", "ag_args": {"name_suffix": "Entr", "problem_types": ["binary", "multiclass"]}},
    ],
    "XT": [
        {"criterion": "gini", "ag_args": {"name_suffix": "Gini", "problem_types": ["binary", "multiclass"]}},
        {"criterion": "entropy", "ag_args": {"name_suffix": "Entr", "problem_types": ["binary", "multiclass"]}},
    ],
    "KNN": [
        {"weights": "uniform", "ag_args": {"name_suffix": "Unif"}},
        {"weights": "distance", "ag_args": {"name_suffix": "Dist"}},
    ],
}

# Feature sources to enable
ENABLED_FEATURES = [
    "massey",
    "seeds",
    "conference",
    "regular_season",
    "tourney_history",
]
