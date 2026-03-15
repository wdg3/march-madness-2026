"""March Madness 2026 Prediction Pipeline"""

import pandas as pd
from config import (
    DATA_DIR, OUTPUT_DIR, TRAIN_SEASONS_END, VALIDATION_SEASON,
    PREDICTION_SEASON, AG_HYPERPARAMETERS, ENABLED_FEATURES,
)
from pipeline import build_team_features, build_matchups, build_prediction_pairs
from training import train
from submission import generate_submission

# 1. Build features
team_features = build_team_features(DATA_DIR, ENABLED_FEATURES)

# 2. Load tournament results
tourney = pd.read_csv(DATA_DIR / "MNCAATourneyCompactResults.csv")

# 3. Build pairwise training data
matchups = build_matchups(team_features, tourney)

# 4. Split train/validation
train_data = matchups[matchups["Season"] < TRAIN_SEASONS_END].copy()
val_data = matchups[matchups["Season"] == VALIDATION_SEASON].copy()

# Drop identifier columns
drop_cols = ["Season", "TeamID_A", "TeamID_B"]
train_data = train_data.drop(columns=drop_cols)
val_data = val_data.drop(columns=drop_cols)

# 5. Train
predictor = train(train_data, val_data, AG_HYPERPARAMETERS)

# 6. Generate submission
pred_pairs = build_prediction_pairs(team_features, PREDICTION_SEASON)
sample_sub = pd.read_csv(DATA_DIR / "SampleSubmissionStage2.csv")
generate_submission(predictor, pred_pairs, sample_sub, OUTPUT_DIR / "submission.csv")

print("Done!")
