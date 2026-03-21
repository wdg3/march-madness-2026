"""Monte Carlo bracket simulation.

Uses pairwise win probabilities from the model to simulate the tournament
thousands of times, then picks the team that advances from each bracket
slot most often. This properly accounts for path probability — a team that
is 99% likely to win five games but only 49.9% in one will still be chosen
over a team that is 50.1% everywhere.

Usage:
    python simulate.py [--n-sims 10000] [--submission output/submission.csv]
"""

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from config import DATA_DIR, OUTPUT_DIR, PREDICTION_SEASON


def load_probabilities(submission_path: str, season: int) -> dict:
    """Load submission CSV into a lookup: (teamA, teamB) -> P(A beats B)."""
    sub = pd.read_csv(submission_path)
    probs = {}
    for _, row in sub.iterrows():
        parts = row["ID"].split("_")
        s, a, b = int(parts[0]), int(parts[1]), int(parts[2])
        if s != season:
            continue
        probs[(a, b)] = row["Pred"]
        probs[(b, a)] = 1.0 - row["Pred"]
    return probs


def build_bracket(season: int, data_dir):
    """Parse seeds and slots into bracket structure."""
    seeds_df = pd.read_csv(data_dir / "MNCAATourneySeeds.csv")
    slots_df = pd.read_csv(data_dir / "MNCAATourneySlots.csv")

    seeds_df = seeds_df[seeds_df["Season"] == season]
    slots_df = slots_df[slots_df["Season"] == season]

    # Map seed label -> team ID
    seed_to_team = dict(zip(seeds_df["Seed"], seeds_df["TeamID"]))

    # Map team ID -> team name (if available)
    teams_path = data_dir / "MTeams.csv"
    team_names = {}
    if teams_path.exists():
        teams_df = pd.read_csv(teams_path)
        team_names = dict(zip(teams_df["TeamID"], teams_df["TeamName"]))

    # Parse slots and sort so play-ins come before R1, R1 before R2, etc.
    # Play-in slots (e.g. "W16", "X11") lack the "R" prefix and must run first.
    def slot_sort_key(slot_name):
        if slot_name.startswith("R"):
            return (int(slot_name[1]), slot_name)
        return (0, slot_name)  # play-ins first

    slots = []
    for _, row in slots_df.iterrows():
        slots.append({
            "slot": row["Slot"],
            "strong": row["StrongSeed"],
            "weak": row["WeakSeed"],
        })
    slots.sort(key=lambda s: slot_sort_key(s["slot"]))

    return seed_to_team, slots, team_names


def resolve_team(label, results, seed_to_team):
    """Resolve a seed label or slot reference to a team ID."""
    if label in seed_to_team:
        return seed_to_team[label]
    return results.get(label)


def simulate_once(slots, seed_to_team, probs, rng):
    """Simulate one full tournament. Returns dict of slot -> winning team ID."""
    results = {}
    for s in slots:
        team_a = resolve_team(s["strong"], results, seed_to_team)
        team_b = resolve_team(s["weak"], results, seed_to_team)
        if team_a is None or team_b is None:
            continue
        p = probs.get((team_a, team_b), 0.5)
        winner = team_a if rng.random() < p else team_b
        results[s["slot"]] = winner
    return results


def simulate_tournament(
    submission_path: str,
    season: int,
    data_dir,
    n_sims: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run Monte Carlo simulation and return bracket picks."""
    print(f"Loading probabilities from {submission_path}...")
    probs = load_probabilities(submission_path, season)
    if not probs:
        raise ValueError(f"No probabilities found for season {season}")

    print(f"Building bracket for {season}...")
    seed_to_team, slots, team_names = build_bracket(season, data_dir)
    if not seed_to_team:
        raise ValueError(f"No seeds found for season {season}")

    print(f"Running {n_sims:,} simulations...")
    rng = np.random.default_rng(seed)

    # Count how often each team wins each slot
    slot_counts = defaultdict(lambda: defaultdict(int))
    for _ in range(n_sims):
        results = simulate_once(slots, seed_to_team, probs, rng)
        for slot, team in results.items():
            slot_counts[slot][team] += 1

    # For each slot, pick the team that won most often
    bracket = []
    for s in slots:
        slot_name = s["slot"]
        if slot_name not in slot_counts:
            continue
        counts = slot_counts[slot_name]
        best_team = max(counts, key=counts.get)
        best_pct = counts[best_team] / n_sims * 100
        team_name = team_names.get(best_team, str(best_team))
        bracket.append({
            "Slot": slot_name,
            "TeamID": best_team,
            "TeamName": team_name,
            "WinPct": round(best_pct, 1),
        })

    bracket_df = pd.DataFrame(bracket)
    return bracket_df, slot_counts, team_names


ROUND_NAMES = {
    "R1": "Round of 64",
    "R2": "Round of 32",
    "R3": "Sweet 16",
    "R4": "Elite 8",
    "R5": "Final Four",
    "R6": "Championship",
}

REGION_NAMES = {"W": "W", "X": "X", "Y": "Y", "Z": "Z"}


def format_bracket(bracket_df, slot_counts, team_names, n_sims):
    """Pretty-print the bracket grouped by round."""
    lines = []

    # Group slots by round
    round_order = ["W16", "X11", "X16", "Y11", "Y16", "Z16",
                   "R1", "R2", "R3", "R4", "R5", "R6"]

    def round_key(slot):
        for i, prefix in enumerate(round_order):
            if slot.startswith(prefix):
                return (i, slot)
        return (len(round_order), slot)

    bracket_df = bracket_df.copy()
    bracket_df["_sort"] = bracket_df["Slot"].apply(round_key)
    bracket_df = bracket_df.sort_values("_sort")

    current_round = None
    for _, row in bracket_df.iterrows():
        slot = row["Slot"]

        # Determine round name
        if slot in ("W16", "X11", "X16", "Y11", "Y16", "Z16"):
            rnd = "Play-in"
        else:
            rnd = ROUND_NAMES.get(slot[:2], slot[:2])

        if rnd != current_round:
            current_round = rnd
            lines.append(f"\n{'='*60}")
            lines.append(f"  {rnd}")
            lines.append(f"{'='*60}")

        # Show top 3 candidates for this slot
        counts = slot_counts.get(slot, {})
        sorted_teams = sorted(counts.items(), key=lambda x: -x[1])[:3]
        top_str = " | ".join(
            f"{team_names.get(t, t)} {c/n_sims*100:.0f}%"
            for t, c in sorted_teams
        )
        lines.append(f"  {slot:8s}  {row['TeamName']:25s} ({row['WinPct']:5.1f}%)  [{top_str}]")

    return "\n".join(lines)


def run(submission=None, season=None, n_sims=10000, seed=42, output_path=None):
    """Run bracket simulation with given parameters."""
    submission = submission or str(OUTPUT_DIR / "submission.csv")
    season = season or PREDICTION_SEASON

    bracket_df, slot_counts, team_names = simulate_tournament(
        submission, season, DATA_DIR, n_sims, seed,
    )

    print(format_bracket(bracket_df, slot_counts, team_names, n_sims))

    out_path = Path(output_path) if output_path else OUTPUT_DIR / "bracket.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bracket_df.drop(columns=["_sort"], errors="ignore").to_csv(out_path, index=False)
    print(f"\nBracket saved to {out_path}")

    return bracket_df, slot_counts, team_names


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo bracket simulation")
    parser.add_argument("--n-sims", type=int, default=10000,
                        help="Number of simulations (default: 10000)")
    parser.add_argument("--submission", type=str,
                        default=str(OUTPUT_DIR / "submission.csv"),
                        help="Path to submission CSV")
    parser.add_argument("--season", type=int, default=PREDICTION_SEASON,
                        help="Season to simulate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    run(args.submission, args.season, args.n_sims, args.seed)


if __name__ == "__main__":
    main()
