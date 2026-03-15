from pathlib import Path
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor


def generate_submission(
    predictor: TabularPredictor,
    prediction_pairs: pd.DataFrame,
    sample_submission: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate Kaggle-format submission with symmetry enforcement."""
    print("Generating predictions...")

    # Columns to drop before prediction (identifiers)
    id_cols = ["Season", "TeamID_A", "TeamID_B"]
    X = prediction_pairs.drop(columns=id_cols)

    proba = predictor.predict_proba(X)
    # Get probability of class 1 (TeamA wins)
    prediction_pairs = prediction_pairs.copy()
    prediction_pairs["Pred"] = proba[1].values

    # Symmetry enforcement: average forward and backward predictions
    print("Enforcing symmetry...")
    season = prediction_pairs["Season"].iloc[0]

    # Build lookup: (TeamA, TeamB) -> prediction
    lookup = {}
    for _, row in prediction_pairs.iterrows():
        lookup[(int(row["TeamID_A"]), int(row["TeamID_B"]))] = row["Pred"]

    results = []
    seen = set()
    for (a, b), p_forward in lookup.items():
        pair = (min(a, b), max(a, b))
        if pair in seen:
            continue
        seen.add(pair)

        p_backward = lookup.get((b, a), 1 - p_forward)
        p_avg = (p_forward + (1 - p_backward)) / 2 if a < b else ((1 - p_forward) + p_backward) / 2

        # If a > b, we need to flip since Kaggle ID has lower ID first
        if a < b:
            results.append({"ID": f"{season}_{a}_{b}", "Pred": p_avg})
        else:
            results.append({"ID": f"{season}_{b}_{a}", "Pred": 1 - p_avg})

    sub = pd.DataFrame(results)

    # Merge with sample submission to ensure correct rows
    final = sample_submission[["ID"]].merge(sub, on="ID", how="left")
    final["Pred"] = final["Pred"].fillna(0.5)

    # Clip to avoid catastrophic log loss
    final["Pred"] = final["Pred"].clip(0.05, 0.95)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} ({len(final)} rows)")
