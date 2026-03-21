"""Futures bet sheet generator with YES and NO support.

Fetches market prices from Kalshi API, computes advancement probabilities
via Monte Carlo simulation, and generates Kelly-sized bet recommendations
for both YES (team advances) and NO (team doesn't advance) positions.
"""

import csv
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR, OUTPUT_DIR, PREDICTION_SEASON
from kelly import kelly_fraction
from simulate import load_probabilities, build_bracket, simulate_once


# Kalshi team name -> Kaggle TeamName mapping (only non-obvious ones)
KALSHI_NAME_MAP = {
    "Hawai'i": "Hawaii",
    "Miami (FL)": "Miami FL",
    "Miami (OH)": "Miami OH",
    "St. John's": "St John's",
    "St. Bonaventure": "St Bonaventure",
    "Saint Louis": "St Louis",
    "Saint Mary's": "St Mary's CA",
    "McNeese": "McNeese St",
    "Boise St.": "Boise St",
    "Iowa St.": "Iowa St",
    "Colorado St.": "Colorado St",
    "Michigan St.": "Michigan St",
    "North Carolina St.": "NC State",
    "Kansas St.": "Kansas St",
    "Oklahoma St.": "Oklahoma St",
    "Ohio St.": "Ohio St",
    "Utah St.": "Utah St",
    "Kennesaw St.": "Kennesaw St",
    "Murray St.": "Murray St",
    "Wichita St.": "Wichita St",
    "Tennessee St.": "Tennessee St",
    "Weber St.": "Weber St",
    "Indiana St.": "Indiana St",
    "San Diego St.": "San Diego St",
    "Florida St.": "Florida St",
    "North Dakota St.": "North Dakota St",
    "South Dakota St.": "South Dakota St",
    "Youngstown St.": "Youngstown St",
    "Stephen F. Austin": "SFA",
    "Queens University": "Queens NC",
    "FDU": "Fairleigh Dickinson",
    "LIU": "Long Island",
    "California Baptist": "Cal Baptist",
    "UC Irvine": "UC Irvine",
    "UC San Diego": "UC San Diego",
    "UC Santa Barbara": "UC Santa Barbara",
    "UConn": "Connecticut",
    "UCF": "UCF",
    "USC": "Southern California",
    "USC Upstate": "USC Upstate",
    "UNC Asheville": "UNC Asheville",
    "UMBC": "UMBC",
    "Mississippi State": "Mississippi St",
    "Charleston Southern": "Chas Southern",
    "Charleston": "Col Charleston",
    "Grand Canyon": "Gr Canyon",
    "Green Bay": "Green Bay",
    "High Point": "High Point",
    "Prairie View A&M": "Prairie View",
    "Western Kentucky": "W Kentucky",
    "Northern Iowa": "N Iowa",
    "Oral Roberts": "Oral Roberts",
    "Loyola Chicago": "Loyola-Chicago",
    "Eastern Kentucky": "E Kentucky",
    "Virginia Tech": "Virginia Tech",
    "Wake Forest": "Wake Forest",
    "South Florida": "South Florida",
    "Boston College": "Boston College",
    "Seton Hall": "Seton Hall",
    "San Francisco": "San Francisco",
    "Santa Clara": "Santa Clara",
    "St. Thomas": "St Thomas MN",
    "West Virginia": "West Virginia",
}

ROUND_ORDER = {"R32": 0, "S16": 1, "E8": 2, "F4": 3, "Champ": 4}

# Slot-to-round mapping for bracket simulation
SLOT_TO_ROUND = {}
# Play-in slots (W16, X11, etc.) -> don't map
# R1xx -> R32 (win first round = advance to round of 32)
# R2xx -> S16
# R3xx -> E8
# R4xx -> F4
# R5xx -> Champ (Final Four semifinal winner goes to championship)
# R6xx -> Winner
for i in range(1, 7):
    rname = {1: "R32", 2: "S16", 3: "E8", 4: "F4", 5: "Champ", 6: "Winner"}[i]
    SLOT_TO_ROUND[f"R{i}"] = rname


def fetch_kalshi_markets():
    """Fetch all KXMARMADROUND markets from Kalshi API."""
    url = ("https://api.elections.kalshi.com/trade-api/v2/events"
           "?series_ticker=KXMARMADROUND&with_nested_markets=true&limit=200")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())

    round_map = {"26RO32": "R32", "26S16": "S16", "26E8": "E8",
                 "26F4": "F4", "26T2": "Champ"}

    markets = []
    for e in data.get("events", []):
        round_key = e["event_ticker"].replace("KXMARMADROUND-", "")
        round_name = round_map.get(round_key, round_key)
        for m in e.get("markets", []):
            last_price = float(m.get("last_price_dollars", 0))
            if last_price == 0:
                continue
            markets.append({
                "kalshi_name": m.get("yes_sub_title", ""),
                "round": round_name,
                "market_ask": last_price,
                "result": m.get("result", ""),
                "volume": float(m.get("volume_fp", 0)),
            })
    return markets


def load_kalshi_from_csv(path):
    """Load cached Kalshi market data from CSV."""
    markets = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            markets.append({
                "kalshi_name": row["Team"],
                "round": row["Round"],
                "market_ask": float(row["MarketAsk"]),
                "result": row.get("Result", ""),
                "volume": float(row.get("Volume", 0)),
            })
    return markets


def build_name_to_id(data_dir):
    """Build mapping from team name to TeamID."""
    teams = pd.read_csv(data_dir / "MTeams.csv")
    name_to_id = dict(zip(teams["TeamName"], teams["TeamID"]))
    return name_to_id


def map_kalshi_to_teamid(kalshi_name, name_to_id):
    """Map a Kalshi team name to a Kaggle TeamID."""
    # Try direct mapping first
    mapped = KALSHI_NAME_MAP.get(kalshi_name, kalshi_name)
    tid = name_to_id.get(mapped)
    if tid:
        return tid
    # Try case-insensitive
    lower_map = {k.lower(): v for k, v in name_to_id.items()}
    return lower_map.get(mapped.lower())


def compute_advancement_probs(submission_path, season, data_dir, n_sims=10000, seed=42):
    """Run Monte Carlo sim and return {(TeamID, round): probability}."""
    probs = load_probabilities(submission_path, season)
    seed_to_team, slots, team_names = build_bracket(season, data_dir)

    rng = np.random.default_rng(seed)
    advancement = defaultdict(lambda: defaultdict(int))

    for _ in range(n_sims):
        results = simulate_once(slots, seed_to_team, probs, rng)
        for slot, team in results.items():
            prefix = slot[:2]
            round_name = SLOT_TO_ROUND.get(prefix)
            if round_name:
                advancement[team][round_name] += 1

    # Convert to probabilities
    adv_probs = {}
    for team, rounds in advancement.items():
        for rnd, count in rounds.items():
            adv_probs[(team, rnd)] = count / n_sims

    return adv_probs, team_names


def generate_futures_bets(
    markets,
    adv_probs,
    team_names,
    name_to_id,
    total_cost=1.01,
    kelly_frac=0.25,
    min_edge=0.02,
    max_bet_pct=0.05,
    bankroll=1000.0,
):
    """Generate bet sheet with both YES and NO positions."""
    id_to_name = {v: k for k, v in name_to_id.items()}
    # Also add team_names from bracket
    for tid, tname in team_names.items():
        if tid not in id_to_name:
            id_to_name[tid] = tname

    bets = []
    for m in markets:
        kalshi_name = m["kalshi_name"]
        rnd = m["round"]
        market_ask = m["market_ask"]
        result = m["result"]
        volume = m["volume"]

        tid = map_kalshi_to_teamid(kalshi_name, name_to_id)
        if tid is None:
            continue

        model_prob = adv_probs.get((tid, rnd), 0.0)
        team_name = id_to_name.get(tid, kalshi_name)

        # YES side: buy at market_ask, win $1 if team advances
        yes_cost = market_ask
        yes_edge = model_prob / yes_cost - 1 if yes_cost > 0 else -1
        yes_kf = kelly_fraction(model_prob, yes_cost) * kelly_frac

        # NO side: buy at (total_cost - market_ask), win $1 if team doesn't advance
        no_cost = total_cost - market_ask
        no_prob = 1.0 - model_prob
        no_edge = no_prob / no_cost - 1 if no_cost > 0 else -1
        no_kf = kelly_fraction(no_prob, no_cost) * kelly_frac

        for side, edge, kf, prob, cost in [
            ("YES", yes_edge, yes_kf, model_prob, yes_cost),
            ("NO", no_edge, no_kf, no_prob, no_cost),
        ]:
            if edge < min_edge:
                continue

            bet_pct = min(kf, max_bet_pct)
            wager = round(bankroll * bet_pct, 2)
            if wager < 1:
                continue
            contracts = int(wager / cost) if cost > 0 else 0
            if contracts < 1:
                continue
            actual_wager = round(contracts * cost, 2)
            profit = round(contracts * (1.0 - cost), 2)
            ev = round(actual_wager * edge, 2)

            bets.append({
                "Team": team_name,
                "TeamID": tid,
                "Round": rnd,
                "Side": side,
                "Model Prob": round(prob, 4),
                "Market Ask": cost,
                "Edge": round(edge, 4),
                "Kelly %": round(kf * 100, 2),
                "Contracts": contracts,
                "Wager": actual_wager,
                "Profit (win)": profit,
                "EV": ev,
                "Result": result,
                "Volume": volume,
            })

    df = pd.DataFrame(bets)
    if len(df) == 0:
        print("No positive-edge bets found.")
        return df

    # Sort by round order then EV descending
    df["_round_order"] = df["Round"].map(ROUND_ORDER)
    df = df.sort_values(["_round_order", "EV"], ascending=[True, False])
    df = df.drop(columns=["_round_order"]).reset_index(drop=True)
    return df


def print_bet_sheet(df, bankroll):
    """Pretty-print the futures bet sheet."""
    if len(df) == 0:
        return

    current_round = None
    for _, row in df.iterrows():
        rnd = row["Round"]
        if rnd != current_round:
            current_round = rnd
            print(f"\n{'='*90}")
            print(f"  {rnd}")
            print(f"{'='*90}")

        side_tag = f"[{row['Side']}]"
        result_tag = ""
        if row["Result"] == "yes":
            result_tag = " [WON]" if row["Side"] == "YES" else " [LOST]"
        elif row["Result"] == "no":
            result_tag = " [LOST]" if row["Side"] == "YES" else " [WON]"

        print(f"  {side_tag:5s} {row['Team']:25s}  "
              f"Model:{row['Model Prob']:6.1%}  "
              f"Ask:{row['Market Ask']:.0%}  "
              f"Edge:{row['Edge']:+6.1%}  "
              f"${row['Wager']:7.2f} -> EV ${row['EV']:6.2f}"
              f"{result_tag}")

    total_wagered = df["Wager"].sum()
    total_ev = df["EV"].sum()
    yes_bets = df[df["Side"] == "YES"]
    no_bets = df[df["Side"] == "NO"]

    print(f"\n{'─'*90}")
    print(f"  YES bets: {len(yes_bets):3d}  wagered: ${yes_bets['Wager'].sum():,.2f}  EV: ${yes_bets['EV'].sum():,.2f}")
    print(f"  NO  bets: {len(no_bets):3d}  wagered: ${no_bets['Wager'].sum():,.2f}  EV: ${no_bets['EV'].sum():,.2f}")
    print(f"  TOTAL:    {len(df):3d}  wagered: ${total_wagered:,.2f}  EV: ${total_ev:,.2f}")
    print(f"  EV/wagered: {total_ev/total_wagered*100:.1f}%" if total_wagered > 0 else "")

    # Score settled bets
    settled = df[df["Result"].isin(["yes", "no"])]
    if len(settled) > 0:
        wins = 0
        losses = 0
        pnl = 0
        for _, row in settled.iterrows():
            if (row["Side"] == "YES" and row["Result"] == "yes") or \
               (row["Side"] == "NO" and row["Result"] == "no"):
                wins += 1
                pnl += row["Profit (win)"]
            else:
                losses += 1
                pnl -= row["Wager"]
        print(f"\n  SETTLED: {wins}W-{losses}L  P&L: ${pnl:+,.2f}")


if __name__ == "__main__":
    print("Use 'python run.py futures ...' instead. Run 'python run.py --help' for details.")
