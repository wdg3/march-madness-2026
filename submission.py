from pathlib import Path
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor


def generate_submission(
    predictor: TabularPredictor,
    prediction_pairs: pd.DataFrame,
    sample_submission: pd.DataFrame,
    output_path: Path,
    seed_prior=None,
) -> None:
    """Generate Kaggle-format submission with optional seed prior blending and symmetry enforcement.

    Args:
        seed_prior: Optional SeedPrior instance. If provided, blends model predictions
            with historical seed-matchup win rates via log-odds weighting.
    """
    print("Generating predictions...")

    # Columns to drop before prediction (identifiers)
    id_cols = ["Season", "TeamID_A", "TeamID_B"]
    X = prediction_pairs.drop(columns=id_cols)

    proba = predictor.predict_proba(X)
    model_proba = proba[1].values

    # Blend with seed prior if available
    if seed_prior is not None:
        seasons = prediction_pairs["Season"].values
        teams_a = prediction_pairs["TeamID_A"].values
        teams_b = prediction_pairs["TeamID_B"].values

        seed_a = np.array([
            seed_prior.get_seeds(int(s), int(a), int(b))[0] or 8
            for s, a, b in zip(seasons, teams_a, teams_b)
        ])
        seed_b = np.array([
            seed_prior.get_seeds(int(s), int(a), int(b))[1] or 8
            for s, a, b in zip(seasons, teams_a, teams_b)
        ])

        model_proba = seed_prior.blend(model_proba, seed_a, seed_b)
        print(f"  Blended with seed prior (α={seed_prior.alpha:.2f})")

    prediction_pairs = prediction_pairs.copy()
    prediction_pairs["Pred"] = model_proba

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

    # Clip extreme probabilities (wide range for Brier score grading)
    final["Pred"] = final["Pred"].clip(0.01, 0.99)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} ({len(final)} rows)")
