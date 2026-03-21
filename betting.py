"""Kelly Criterion bet sheet generator.

Takes model predictions and prediction market prices (implied probability %),
computes optimal wager allocation using fractional Kelly sizing.

Odds CSV format (prediction market style — prices as implied probability %):
    Season,TeamA,TeamB,PriceA,PriceB
    2026,1181,1369,95,8
    2026,1385,1314,60,42

Prices represent the market's implied probability for each team winning.
They may sum to >100% (overround/vig) or exactly 100% (no-vig market).
"""

from pathlib import Path

import pandas as pd
import numpy as np

from config import DATA_DIR, OUTPUT_DIR, PREDICTION_SEASON
from kelly import kelly_fraction


def load_model_probs(submission_path: str, season: int) -> dict:
    """Load submission CSV into lookup: (teamA, teamB) -> P(A beats B)."""
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


def _price_to_cost(price_pct: float, fee: float = 0.0) -> float:
    """Convert prediction market price (implied probability %) to cost (0-1)."""
    return price_pct / 100.0 + fee


def generate_bet_sheet(
    odds_path: str,
    submission_path: str,
    season: int,
    kelly_frac: float = 0.25,
    min_edge: float = 0.02,
    max_bet_pct: float = 0.05,
    bankroll: float = 1000.0,
    fee: float = 0.0,
) -> pd.DataFrame:
    """Generate a bet sheet from odds and model predictions.

    Args:
        odds_path: Path to CSV with columns [Season, TeamA, TeamB, PriceA, PriceB]
        submission_path: Path to model submission CSV
        season: Tournament season
        kelly_frac: Fraction of full Kelly to use (0.25 = quarter Kelly)
        min_edge: Minimum edge to place a bet (default 2%)
        max_bet_pct: Maximum bet as fraction of bankroll (default 5%)
        bankroll: Starting bankroll in dollars

    Returns:
        DataFrame with bet recommendations
    """
    probs = load_model_probs(submission_path, season)
    if not probs:
        raise ValueError(f"No predictions found for season {season}")

    odds_df = pd.read_csv(odds_path)

    # Load team names
    teams_path = DATA_DIR / "MTeams.csv"
    team_names = {}
    if teams_path.exists():
        teams = pd.read_csv(teams_path)
        team_names = dict(zip(teams["TeamID"], teams["TeamName"]))

    bets = []
    for _, row in odds_df.iterrows():
        s = int(row["Season"])
        if s != season:
            continue

        team_a = int(row["TeamA"])
        team_b = int(row["TeamB"])
        price_a = float(row["PriceA"])
        price_b = float(row["PriceB"])
        cost_a = _price_to_cost(price_a, fee)
        cost_b = _price_to_cost(price_b, fee)

        p_a = probs.get((team_a, team_b))
        if p_a is None:
            continue
        p_b = 1 - p_a

        # Check both sides of the bet
        for team, p, cost, opp, market_price in [
            (team_a, p_a, cost_a, team_b, price_a),
            (team_b, p_b, cost_b, team_a, price_b),
        ]:
            edge = p / cost - 1
            if edge < min_edge:
                continue

            odds = 1.0 / cost
            kf = kelly_fraction(p, cost) * kelly_frac
            bet_pct = min(kf, max_bet_pct)
            wager = round(bankroll * bet_pct, 2)
            payout = round(wager * odds, 2)
            profit = round(payout - wager, 2)
            ev = round(wager * edge, 2)

            bets.append({
                "Game": f"{team_names.get(team_a, team_a)} vs {team_names.get(team_b, team_b)}",
                "Bet On": team_names.get(team, team),
                "Model Prob": round(p, 4),
                "Market Price": round(market_price, 1),
                "Edge": round(edge, 4),
                "Kelly %": round(kf * 100, 2),
                "Bet %": round(bet_pct * 100, 2),
                "Wager": wager,
                "Payout": payout,
                "Profit (if win)": profit,
                "EV": ev,
            })

    bet_df = pd.DataFrame(bets)
    if len(bet_df) == 0:
        print("No positive-edge bets found.")
        return bet_df

    bet_df = bet_df.sort_values("Edge", ascending=False).reset_index(drop=True)
    return bet_df


def print_bet_sheet(bet_df: pd.DataFrame, bankroll: float):
    """Pretty-print the bet sheet."""
    if len(bet_df) == 0:
        return

    total_wagered = bet_df["Wager"].sum()
    total_ev = bet_df["EV"].sum()

    print(f"\n{'='*80}")
    print(f"  BET SHEET  (bankroll: ${bankroll:,.0f})")
    print(f"{'='*80}")
    print()

    for _, row in bet_df.iterrows():
        print(f"  {row['Game']}")
        print(f"    Bet: {row['Bet On']:20s}  Market: {row['Market Price']:.0f}¢   "
              f"Model: {row['Model Prob']:.1%}   Edge: {row['Edge']:.1%}")
        print(f"    Wager: ${row['Wager']:.2f} ({row['Bet %']:.1f}%)   "
              f"Payout: ${row['Payout']:.2f}   EV: ${row['EV']:.2f}")
        print()

    print(f"  {'─'*70}")
    print(f"  Total bets: {len(bet_df)}")
    print(f"  Total wagered: ${total_wagered:,.2f} ({total_wagered/bankroll*100:.1f}% of bankroll)")
    print(f"  Total EV: ${total_ev:,.2f}")
    print(f"  EV/wagered: {total_ev/total_wagered*100:.1f}%" if total_wagered > 0 else "")


if __name__ == "__main__":
    print("Use 'python run.py bet ...' instead. Run 'python run.py --help' for details.")
