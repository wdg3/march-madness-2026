from pathlib import Path

DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./output")
CONFIGS_DIR = Path("./configs")

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


def load_config(tag: str) -> dict:
    """Load YAML config for a model tag.

    Looks for configs/<tag>.yaml first, falls back to configs/default.yaml,
    then falls back to hardcoded defaults in this module.

    Returns a dict with keys: features, training, prediction.
    """
    import yaml

    tag_path = CONFIGS_DIR / f"{tag}.yaml"
    default_path = CONFIGS_DIR / "default.yaml"

    if tag_path.exists():
        with open(tag_path) as f:
            return yaml.safe_load(f) or {}
    elif default_path.exists():
        with open(default_path) as f:
            return yaml.safe_load(f) or {}
    else:
        return {
            "features": ENABLED_FEATURES,
            "training": {
                "presets": AG_PRESETS,
                "time_limit": AG_TIME_LIMIT,
                "num_bag_folds": AG_NUM_BAG_FOLDS,
                "num_stack_levels": AG_NUM_STACK_LEVELS,
                "train_seasons_start": 2010,
                "train_seasons_end": TRAIN_SEASONS_END,
                "validation_season": VALIDATION_SEASON,
            },
            "prediction": {
                "season": PREDICTION_SEASON,
            },
        }
