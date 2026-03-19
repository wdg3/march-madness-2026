from pathlib import Path

DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./output")

# Seasons
TRAIN_SEASONS_END = 2025  # train on seasons < this
VALIDATION_SEASON = 2025  # validate on this season
PREDICTION_SEASON = 2026

# AutoGluon
AG_PRESETS = "best_quality"
AG_TIME_LIMIT = 7200  # 2 hours
AG_NUM_BAG_FOLDS = 10
AG_NUM_STACK_LEVELS = 2

# Feature sources to enable
ENABLED_FEATURES = [
    "massey",
    "seeds",
    "conference",
    "regular_season",
    "tourney_history",
    "rank_disagree",
    "seed_rank_delta",
    "close_games",
    "scoring_variance",
    "momentum",
    "tempo",
    "coach",
    "conf_tourney",
    "location",
    "travel",
    "rs_trajectory",
    "massey_trajectory",
    # External sources (require data in data/external/):
    "kenpom",
    "ap_poll",
    "public_picks",
    "roster",
    "vegas",
]
