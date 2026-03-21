"""Backtest bracket picks against actual tournament results.

Generates a bracket from model predictions, then scores it against
actual outcomes using ESPN bracket scoring rules.
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from config import DATA_DIR, OUTPUT_DIR
from simulate import build_bracket, simulate_once, resolve_team


ESPN_POINTS = {
    "R1": 10,   # Round of 64
    "R2": 20,   # Round of 32
    "R3": 40,   # Sweet 16
    "R4": 80,   # Elite 8
    "R5": 160,  # Final Four
    "R6": 320,  # Championship
}


def get_actual_results(season: int, data_dir) -> dict:
    """Build slot -> winning team ID from actual tournament results."""
    seeds_df = pd.read_csv(data_dir / "MNCAATourneySeeds.csv")
    slots_df = pd.read_csv(data_dir / "MNCAATourneySlots.csv")
    results_df = pd.read_csv(data_dir / "MNCAATourneyCompactResults.csv")

    seeds_df = seeds_df[seeds_df["Season"] == season]
    slots_df = slots_df[slots_df["Season"] == season]
    results_df = results_df[results_df["Season"] == season]

    seed_to_team = dict(zip(seeds_df["Seed"], seeds_df["TeamID"]))

    # Build a set of winners for each matchup
    winners = set()
    for _, row in results_df.iterrows():
        winners.add((row["WTeamID"], row["LTeamID"]))

    # Walk through slots in order, resolving actual winners
    def slot_sort_key(slot_name):
        if slot_name.startswith("R"):
            return (int(slot_name[1]), slot_name)
        return (0, slot_name)

    slots = []
    for _, row in slots_df.iterrows():
        slots.append({
            "slot": row["Slot"],
            "strong": row["StrongSeed"],
            "weak": row["WeakSeed"],
        })
    slots.sort(key=lambda s: slot_sort_key(s["slot"]))

    actual = {}
    for s in slots:
        team_a = resolve_team(s["strong"], actual, seed_to_team)
        team_b = resolve_team(s["weak"], actual, seed_to_team)
        if team_a is None or team_b is None:
            continue
        if (team_a, team_b) in winners:
            actual[s["slot"]] = team_a
        elif (team_b, team_a) in winners:
            actual[s["slot"]] = team_b

    return actual


def score_bracket(picks: dict, actual: dict, team_names: dict) -> int:
    """Score bracket picks against actual results using ESPN rules.

    Returns total points and prints round-by-round breakdown.
    """
    total = 0
    round_correct = defaultdict(int)
    round_total = defaultdict(int)
    round_points = defaultdict(int)

    details = []

    for slot, picked_team in picks.items():
        # Determine round
        rnd = None
        for prefix in ESPN_POINTS:
            if slot.startswith(prefix):
                rnd = prefix
                break
        if rnd is None:
            continue  # play-in, skip

        actual_team = actual.get(slot)
        if actual_team is None:
            continue

        correct = picked_team == actual_team
        pts = ESPN_POINTS[rnd] if correct else 0
        round_total[rnd] += 1
        if correct:
            round_correct[rnd] += 1
            round_points[rnd] += pts
        total += pts

        details.append({
            "slot": slot,
            "round": rnd,
            "picked": team_names.get(picked_team, picked_team),
            "actual": team_names.get(actual_team, actual_team),
            "correct": correct,
            "points": pts,
        })

    # Print round-by-round summary
    print(f"\n{'='*60}")
    print(f"  ESPN Bracket Scoring")
    print(f"{'='*60}")

    round_names = {
        "R1": "Round of 64", "R2": "Round of 32", "R3": "Sweet 16",
        "R4": "Elite 8", "R5": "Final Four", "R6": "Championship",
    }

    for rnd in ["R1", "R2", "R3", "R4", "R5", "R6"]:
        if rnd not in round_total:
            continue
        c = round_correct[rnd]
        t = round_total[rnd]
        p = round_points[rnd]
        ppg = ESPN_POINTS[rnd]
        print(f"  {round_names[rnd]:16s}  {c:2d}/{t:2d} correct  "
              f"× {ppg:3d} pts = {p:4d} pts")

    print(f"  {'─'*50}")
    print(f"  {'TOTAL':16s}  {sum(round_correct.values()):2d}/{sum(round_total.values()):2d} correct"
          f"           = {total:4d} pts")

    # Show misses
    misses = [d for d in details if not d["correct"]]
    if misses:
        print(f"\n  Incorrect picks:")
        for d in sorted(misses, key=lambda x: x["slot"]):
            print(f"    {d['slot']:8s}  Picked: {d['picked']:20s}  Actual: {d['actual']}")

    return total


def run_backtest(season: int, n_sims: int = 10000, submission: str = None, tag: str = None):
    """Simulate bracket and score against actuals using ESPN rules.

    If submission is provided, uses that file directly. Otherwise generates
    predictions from the current trained model.
    """
    from simulate import load_probabilities

    if submission:
        backtest_path = submission
        print(f"Using pre-built submission: {submission}")
    else:
        from autogluon.tabular import TabularPredictor
        from pipeline import build_team_features, build_prediction_pairs
        from features.travel import ensure_geocoded
        from submission import generate_submission

        model_path = f"./AutogluonModels/{tag}" if tag else "./AutogluonModels"
        predictor = TabularPredictor.load(model_path)

        from config import ENABLED_FEATURES
        use_travel = "travel" in ENABLED_FEATURES

        print(f"Building {season} predictions for backtesting...")
        team_features = build_team_features(DATA_DIR, ENABLED_FEATURES, gender="M")
        if use_travel:
            ensure_geocoded(DATA_DIR)

        pred_pairs = build_prediction_pairs(
            team_features, season, data_dir=DATA_DIR, travel=use_travel,
        )

        seeds = pd.read_csv(DATA_DIR / "MNCAATourneySeeds.csv")
        season_teams = seeds[seeds["Season"] == season]["TeamID"].unique()
        ids = []
        for a in sorted(season_teams):
            for b in sorted(season_teams):
                if a < b:
                    ids.append(f"{season}_{a}_{b}")
        sample_sub = pd.DataFrame({"ID": ids, "Pred": 0.5})

        backtest_path = OUTPUT_DIR / f"backtest_{season}.csv"
        generate_submission(predictor, pred_pairs, sample_sub, backtest_path)

    # Simulate bracket
    probs = load_probabilities(str(backtest_path), season)
    seed_to_team, slots, team_names = build_bracket(season, DATA_DIR)

    rng = np.random.default_rng(42)
    slot_counts = defaultdict(lambda: defaultdict(int))
    for _ in range(n_sims):
        results = simulate_once(slots, seed_to_team, probs, rng)
        for slot, team in results.items():
            slot_counts[slot][team] += 1

    # Pick most frequent winner per slot
    picks = {}
    for s in slots:
        slot_name = s["slot"]
        if slot_name not in slot_counts:
            continue
        counts = slot_counts[slot_name]
        picks[slot_name] = max(counts, key=counts.get)

    # Get actual results and score
    actual = get_actual_results(season, DATA_DIR)
    total = score_bracket(picks, actual, team_names)

    return total


if __name__ == "__main__":
    print("Use 'python run.py backtest ...' instead. Run 'python run.py --help' for details.")
