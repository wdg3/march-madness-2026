"""Kelly Criterion bet sheet generator.

Takes model predictions and prediction market prices (implied probability %),
computes optimal wager allocation using fractional Kelly sizing.

Usage:
    python betting.py --odds odds.csv --submission output/submission_brier_ts_3hr.csv
    python betting.py --odds odds.csv --tag brier_ts_3hr

Odds CSV format (prediction market style — prices as implied probability %):
    Season,TeamA,TeamB,PriceA,PriceB
    2026,1181,1369,95,8
    2026,1385,1314,60,42

Prices represent the market's implied probability for each team winning.
They may sum to >100% (overround/vig) or exactly 100% (no-vig market).
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from config import DATA_DIR, OUTPUT_DIR, PREDICTION_SEASON


def price_to_decimal(price_pct: float, fee: float = 0.0) -> float:
    """Convert prediction market price (implied probability %) to decimal odds.

    Args:
        price_pct: Contract price as implied probability % (e.g. 52 means 52¢).
        fee: Platform fee per contract in dollars (e.g. 0.02 for 2¢).
    """
    cost = price_pct / 100.0 + fee  # total cost per contract in dollars
    return 1.0 / cost  # pays $1.00 when correct


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


def kelly_fraction(p: float, decimal_odds: float) -> float:
    """Compute Kelly fraction for a bet.

    Returns the fraction of bankroll to wager (0 if no edge).
    """
    b = decimal_odds - 1  # net payout per unit wagered
    if b <= 0:
        return 0.0
    edge = p * decimal_odds - 1
    if edge <= 0:
        return 0.0
    return edge / b


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
        odds_a = price_to_decimal(price_a, fee)
        odds_b = price_to_decimal(price_b, fee)

        p_a = probs.get((team_a, team_b))
        if p_a is None:
            continue
        p_b = 1 - p_a

        # Check both sides of the bet
        for team, p, odds, opp, market_price in [
            (team_a, p_a, odds_a, team_b, price_a),
            (team_b, p_b, odds_b, team_a, price_b),
        ]:
            edge = p * odds - 1
            if edge < min_edge:
                continue

            kf = kelly_fraction(p, odds) * kelly_frac
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate Kelly Criterion bet sheet from model predictions and book odds",
    )
    parser.add_argument("--odds", required=True, help="Path to odds CSV")
    parser.add_argument("--submission", help="Path to submission CSV")
    parser.add_argument("--tag", help="Model tag (uses output/submission_<tag>.csv)")
    parser.add_argument("--season", type=int, default=PREDICTION_SEASON)
    parser.add_argument("--bankroll", type=float, default=1000.0,
                        help="Starting bankroll (default: $1000)")
    parser.add_argument("--kelly", type=float, default=0.25,
                        help="Kelly fraction (default: 0.25 = quarter Kelly)")
    parser.add_argument("--min-edge", type=float, default=0.02,
                        help="Minimum edge to bet (default: 0.02 = 2%%)")
    parser.add_argument("--max-bet", type=float, default=0.05,
                        help="Max bet as fraction of bankroll (default: 0.05 = 5%%)")
    parser.add_argument("--fee", type=float, default=0.0,
                        help="Platform fee per contract in dollars (default: 0.00)")
    parser.add_argument("--output", help="Save bet sheet to CSV")
    args = parser.parse_args()

    if args.submission:
        submission = args.submission
    elif args.tag:
        submission = str(OUTPUT_DIR / f"submission_{args.tag}.csv")
    else:
        parser.error("Must provide --submission or --tag")

    bet_df = generate_bet_sheet(
        args.odds, submission, args.season,
        kelly_frac=args.kelly,
        min_edge=args.min_edge,
        max_bet_pct=args.max_bet,
        bankroll=args.bankroll,
        fee=args.fee,
    )

    print_bet_sheet(bet_df, args.bankroll)

    if args.output and len(bet_df) > 0:
        bet_df.to_csv(args.output, index=False)
        print(f"\nBet sheet saved to {args.output}")


if __name__ == "__main__":
    main()
