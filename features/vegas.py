"""Vegas odds and ATS features from Scottfree Analytics + The Odds API.

Historical data (2008-2025): Scottfree Analytics NCAAB CSV
  Source: https://www.scottfreellc.com/shop/p/college-historical-odds-data
  Location: data/external/scottfree/ncaab.csv

Current season (2026): The Odds API historical snapshots
  Source: https://the-odds-api.com
  Location: data/external/odds_api/ncaab_2026_odds.csv

This feature source aggregates game-level closing lines into team-season
summaries capturing market-implied strength, ATS performance, and trends.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from features.base import ExternalFeatureSource


# Scottfree team name -> Kaggle MTeamSpellings name (lowercase)
_SF_NAME_OVERRIDES = {
    "alabama_am": "alabama a&m",
    "american_u": "american",
    "arkansas_lr": "ualr",
    "bethune_cookman": "bethune-cookman",
    "cal_irvine": "uc irvine",
    "cal_poly_s.l.o.": "cal poly",
    "cal_santa_barbara": "uc santa barbara",
    "charl_southern": "charleston so",
    "charlotte_u": "charlotte",
    "coast_carolina": "coastal car",
    "e._washington": "e washington",
    "east_tenn_st": "etsu",
    "elon_college": "elon",
    "fair_dickinson": "fairleigh dickinson",
    "florida_am": "florida a&m",
    "illinois_chicago": "il chicago",
    "md-e_shore": "md. eastern shore",
    "miami_florida": "miami fl",
    "miami_ohio": "miami oh",
    "mount_st_mary's": "mount st. mary's",
    "n._dakota_state": "n dakota st",
    "n_carolina_a&t": "north carolina a&t",
    "n_carolina_at": "north carolina a&t",
    "n_carolina_cent": "north carolina central",
    "n_dakota_state": "n dakota st",
    "north_arizona": "northern arizona",
    "prairie_view_am": "prairie view",
    "saint_marys_ca": "st mary's ca",
    "so_illinois": "s illinois",
    "st._thomas_-_minn": "st thomas mn",
    "st_johns": "st john's",
    "st_josephs": "st joseph's pa",
    "st_peters": "st peter's",
    "st_thomas___minn": "st thomas mn",
    "tarleton": "tarleton st",
    "tenn_martin": "ut martin",
    "tex_san_antonio": "utsa",
    "texas_a&m_corpus_christi": "a&m-corpus christi",
    "texas_am": "texas a&m",
    "texas_am_commerce": "east texas a&m",
    "texas_am_corpus_christi": "a&m-corpus christi",
    "tx-arlington": "ut arlington",
    "tx_arlington": "ut arlington",
    "ul_lafayette": "louisiana",
    "ul_monroe": "la-monroe",
    "ut-rio_grande": "utrgv",
    "ut_rio_grande": "utrgv",
    "william_and_mary": "william & mary",
    "wisc_green_bay": "green bay",
    "wisc_milwaukee": "milwaukee",
}

# The Odds API team name prefix -> Kaggle MTeamSpellings name (lowercase)
# Only for names where stripping the mascot suffix doesn't match
_API_NAME_OVERRIDES = {
    "CSU Fullerton Titans": "cs fullerton",
    "East Tennessee St Buccaneers": "etsu",
    "GW Revolutionaries": "g washington",
    "Loyola (Chi) Ramblers": "loyola-chicago",
    "Miami Hurricanes": "miami fl",
    "Miss Valley St Delta Devils": "mississippi valley state",
    "Queens University Royals": "queens nc",
    "Sacramento St Hornets": "sacramento state",
    "San José St Spartans": "san jose st",
    "North Dakota St Bison": "n dakota st",
    "South Dakota St Jackrabbits": "south dakota st",
    "UL Monroe Warhawks": "la-monroe",
    "UT Rio Grande Valley Vaqueros": "utrgv",
    "Florida Int'l Golden Panthers": "florida intl",
    "South Carolina St Bulldogs": "south carolina st.",
    "Northwestern St Demons": "northwestern st.",
}


def _build_name_to_id(data_dir: Path) -> dict:
    """Build lowercase team name -> TeamID mapping."""
    name_to_id = {}
    spellings = pd.read_csv(data_dir / "MTeamSpellings.csv", encoding="latin-1")
    for _, row in spellings.iterrows():
        name_to_id[str(row["TeamNameSpelling"]).lower().strip()] = int(row["TeamID"])
    teams = pd.read_csv(data_dir / "MTeams.csv")
    for _, row in teams.iterrows():
        name_to_id[row["TeamName"].lower().strip()] = int(row["TeamID"])
    return name_to_id


def _sf_season_to_kaggle(season_str: str) -> int:
    """Convert '2007-08' -> 2008 (Kaggle uses end year)."""
    return int(season_str.split("-")[0]) + 1


def _moneyline_to_prob(ml: float) -> float:
    """Convert American money line to implied probability."""
    if pd.isna(ml) or ml == 0:
        return np.nan
    if ml < 0:
        return abs(ml) / (abs(ml) + 100)
    else:
        return 100 / (ml + 100)


def _load_scottfree(data_dir: Path, name_to_id: dict) -> pd.DataFrame:
    """Load Scottfree historical data (2008-2025) into normalized game rows."""
    csv_path = data_dir / "external" / "scottfree" / "ncaab.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path, low_memory=False)

    def resolve(sf_name):
        if sf_name in _SF_NAME_OVERRIDES:
            return name_to_id.get(_SF_NAME_OVERRIDES[sf_name].lower())
        return name_to_id.get(sf_name.replace("_", " ").lower())

    df["Season"] = df["season"].apply(_sf_season_to_kaggle)

    home_rows = pd.DataFrame({
        "Season": df["Season"],
        "team_name": df["home_team"],
        "score": df["home_score"].astype(float),
        "opp_score": df["away_score"].astype(float),
        "point_spread": df["home_point_spread"].astype(float),
        "money_line": df["home_money_line"].astype(float),
        "over_under": df["over_under"].astype(float),
        "date": pd.to_datetime(df["date"]),
    })

    away_rows = pd.DataFrame({
        "Season": df["Season"],
        "team_name": df["away_team"],
        "score": df["away_score"].astype(float),
        "opp_score": df["home_score"].astype(float),
        "point_spread": df["away_point_spread"].astype(float),
        "money_line": df["away_money_line"].astype(float),
        "over_under": df["over_under"].astype(float),
        "date": pd.to_datetime(df["date"]),
    })

    games = pd.concat([home_rows, away_rows], ignore_index=True)
    games["TeamID"] = games["team_name"].apply(resolve)
    games["won_game"] = (games["score"] > games["opp_score"]).astype(int)
    games["cover_margin"] = games["point_spread"] + (games["score"] - games["opp_score"])
    games["total_points"] = games["score"] + games["opp_score"]
    return games


def _load_odds_api(data_dir: Path, name_to_id: dict) -> pd.DataFrame:
    """Load The Odds API 2026 data and join with Kaggle scores.

    Score joining uses DayZero from MSeasons.csv for precise date-to-DayNum
    conversion, with team-pair chronological matching as fallback for any
    remaining date ambiguities.
    """
    # Prefer v2 (consensus closing lines where available)
    csv_path = data_dir / "external" / "odds_api" / "ncaab_2026_odds_v2.csv"
    if not csv_path.exists():
        csv_path = data_dir / "external" / "odds_api" / "ncaab_2026_odds.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    odds = pd.read_csv(csv_path)

    # Resolve API team names to TeamIDs
    def resolve_api(api_name):
        if api_name in _API_NAME_OVERRIDES:
            return name_to_id.get(_API_NAME_OVERRIDES[api_name].lower())
        words = api_name.lower().split()
        for i in range(len(words), 0, -1):
            attempt = " ".join(words[:i])
            if attempt in name_to_id:
                return name_to_id[attempt]
        return None

    # Get precise DayZero from MSeasons.csv
    seasons_df = pd.read_csv(data_dir / "MSeasons.csv")
    day_zero = pd.Timestamp(seasons_df[seasons_df["Season"] == 2026]["DayZero"].values[0])

    odds["_home_tid"] = odds["home_team"].apply(resolve_api)
    odds["_away_tid"] = odds["away_team"].apply(resolve_api)

    # Build Kaggle score lookup by team pair, sorted chronologically
    # Key: (min_tid, max_tid) -> list of (DayNum, {tid: (score, opp_score)})
    rs_path = data_dir / "MRegularSeasonDetailedResults.csv"
    pair_games = {}
    if rs_path.exists():
        rs = pd.read_csv(rs_path)
        rs_2026 = rs[rs["Season"] == 2026].copy()
        for _, r in rs_2026.iterrows():
            w_id, l_id = int(r["WTeamID"]), int(r["LTeamID"])
            w_score, l_score = float(r["WScore"]), float(r["LScore"])
            day = int(r["DayNum"])
            pair = (min(w_id, l_id), max(w_id, l_id))
            entry = (day, {
                w_id: (w_score, l_score),
                l_id: (l_score, w_score),
            })
            pair_games.setdefault(pair, []).append(entry)
        for pair in pair_games:
            pair_games[pair].sort(key=lambda x: x[0])

    # Also build a flat DayNum lookup for fast exact matching
    dn_lookup = {}
    for pair, games in pair_games.items():
        for day, scores in games:
            dn_lookup[(day, *pair)] = scores

    # Compute approximate DayNum for each odds row
    # Full ISO timestamps: UTC -> US/Eastern -> date -> DayNum (precise)
    # Bare date strings: parse directly as date -> DayNum, then try ±1 as fallback
    has_ts = odds["date"].str.contains("T", na=False)
    odds["_parsed_utc"] = pd.to_datetime(odds["date"], format="mixed", utc=True)

    # For full timestamps, ET conversion gives the correct game date
    odds["_DayNum"] = np.nan
    if has_ts.any():
        et_dates = odds.loc[has_ts, "_parsed_utc"].dt.tz_convert("US/Eastern").dt.date
        odds.loc[has_ts, "_DayNum"] = (pd.to_datetime(et_dates) - day_zero).dt.days.values

    # For bare dates, use the date string directly (it's the game date in some tz)
    if (~has_ts).any():
        bare_dates = pd.to_datetime(odds.loc[~has_ts, "date"].str[:10])
        odds.loc[~has_ts, "_DayNum"] = (bare_dates - day_zero).dt.days.values

    # Sort by date for chronological pair matching
    odds = odds.sort_values("_parsed_utc").reset_index(drop=True)

    # Track consumption index per pair for chronological fallback
    pair_idx = {pair: 0 for pair in pair_games}

    all_rows = []
    n_exact = n_nearby = n_chrono = n_miss = 0

    for _, row in odds.iterrows():
        htid, atid = row["_home_tid"], row["_away_tid"]
        if pd.isna(htid) or pd.isna(atid):
            continue
        htid, atid = int(htid), int(atid)
        dn = int(row["_DayNum"]) if pd.notna(row["_DayNum"]) else None
        pair = (min(htid, atid), max(htid, atid))

        scores = {}

        # Strategy 1: exact DayNum match
        if dn is not None:
            scores = dn_lookup.get((dn, *pair), {})
            if scores:
                n_exact += 1

        # Strategy 2: try ±1 day (timezone edge cases)
        if not scores and dn is not None:
            for offset in [-1, 1]:
                scores = dn_lookup.get((dn + offset, *pair), {})
                if scores:
                    n_nearby += 1
                    break

        # Strategy 3: chronological pair matching
        if not scores and pair in pair_games:
            idx = pair_idx[pair]
            candidates = pair_games[pair]
            if idx < len(candidates):
                _, scores = candidates[idx]
                pair_idx[pair] = idx + 1
                n_chrono += 1

        if not scores:
            n_miss += 1

        h_score = scores[htid][0] if htid in scores else np.nan
        a_score = scores[atid][0] if atid in scores else np.nan

        # Use parsed date for ordering
        game_date = row["_parsed_utc"].tz_convert("US/Eastern").date() \
            if pd.notna(row["_parsed_utc"]) else None
        game_date = pd.Timestamp(game_date) if game_date else pd.NaT

        all_rows.append({
            "Season": 2026, "TeamID": htid, "team_name": row["home_team"],
            "point_spread": float(row["home_point_spread"]) if pd.notna(row["home_point_spread"]) else np.nan,
            "money_line": float(row["home_money_line"]) if pd.notna(row["home_money_line"]) else np.nan,
            "over_under": float(row["over_under"]) if pd.notna(row["over_under"]) else np.nan,
            "date": game_date, "score": h_score, "opp_score": a_score,
        })
        all_rows.append({
            "Season": 2026, "TeamID": atid, "team_name": row["away_team"],
            "point_spread": float(row["away_point_spread"]) if pd.notna(row["away_point_spread"]) else np.nan,
            "money_line": float(row["away_money_line"]) if pd.notna(row["away_money_line"]) else np.nan,
            "over_under": float(row["over_under"]) if pd.notna(row["over_under"]) else np.nan,
            "date": game_date, "score": a_score, "opp_score": h_score,
        })

    total = n_exact + n_nearby + n_chrono + n_miss
    print(f"    Score join: {n_exact} exact + {n_nearby} ±1day + {n_chrono} chrono + {n_miss} miss "
          f"= {total} games ({(n_exact+n_nearby+n_chrono)/max(total,1)*100:.1f}% matched)")

    games = pd.DataFrame(all_rows)
    games["won_game"] = (games["score"] > games["opp_score"]).astype(float)
    games["cover_margin"] = games["point_spread"] + (games["score"] - games["opp_score"])
    games["total_points"] = games["score"] + games["opp_score"]
    return games


def _aggregate_to_features(all_games: pd.DataFrame) -> pd.DataFrame:
    """Aggregate game-level data to team-season features."""
    all_games = all_games.dropna(subset=["TeamID"]).copy()
    all_games["TeamID"] = all_games["TeamID"].astype(int)

    all_games["implied_prob"] = all_games["money_line"].apply(_moneyline_to_prob)
    all_games["was_favorite"] = (all_games["point_spread"] < 0).astype(int)
    all_games["was_underdog"] = (all_games["point_spread"] > 0).astype(int)
    all_games["covered"] = (all_games["cover_margin"] > 0).astype(float)
    all_games["ou_margin"] = all_games["total_points"] - all_games["over_under"]

    all_games = all_games.sort_values(["Season", "TeamID", "date"])

    result_rows = []
    for (season, tid), grp in all_games.groupby(["Season", "TeamID"]):
        n = len(grp)
        if n < 5:
            continue

        late = grp.tail(10)

        # For ATS features, only use games where we have scores
        has_scores = grp.dropna(subset=["score", "opp_score"])
        has_scores_late = late.dropna(subset=["score", "opp_score"])

        row = {
            "Season": season,
            "TeamID": tid,

            # ── Market power rating ──
            "vg_avg_spread": grp["point_spread"].mean(),
            "vg_avg_implied_prob": grp["implied_prob"].mean(),
            "vg_pct_favored": grp["was_favorite"].mean(),

            # ── ATS performance ──
            "vg_ats_win_rate": has_scores["covered"].mean() if len(has_scores) >= 5 else np.nan,
            "vg_avg_cover_margin": has_scores["cover_margin"].mean() if len(has_scores) >= 5 else np.nan,
            "vg_ats_as_fav": has_scores.loc[has_scores["was_favorite"] == 1, "covered"].mean()
                if has_scores["was_favorite"].sum() > 0 else np.nan,
            "vg_ats_as_dog": has_scores.loc[has_scores["was_underdog"] == 1, "covered"].mean()
                if has_scores["was_underdog"].sum() > 0 else np.nan,

            # ── Actual vs expected ──
            "vg_win_vs_implied": (has_scores["won_game"].mean() - has_scores["implied_prob"].mean())
                if len(has_scores) >= 5 else np.nan,
            "vg_margin_vs_spread": ((has_scores["score"] - has_scores["opp_score"]).mean()
                - (-has_scores["point_spread"]).mean()) if len(has_scores) >= 5 else np.nan,

            # ── O/U tendencies ──
            "vg_avg_total": grp["over_under"].mean(),
            "vg_over_rate": (has_scores["ou_margin"] > 0).mean() if len(has_scores) >= 5 else np.nan,

            # ── Late-season momentum ──
            "vg_late_avg_spread": late["point_spread"].mean(),
            "vg_late_implied_prob": late["implied_prob"].mean(),
            "vg_late_ats_win_rate": has_scores_late["covered"].mean()
                if len(has_scores_late) >= 3 else np.nan,
            "vg_late_cover_margin": has_scores_late["cover_margin"].mean()
                if len(has_scores_late) >= 3 else np.nan,

            # ── Trends ──
            "vg_spread_trend": late["point_spread"].mean() - grp["point_spread"].mean(),
            "vg_implied_trend": late["implied_prob"].mean() - grp["implied_prob"].mean(),
            "vg_cover_trend": (has_scores_late["cover_margin"].mean() - has_scores["cover_margin"].mean())
                if len(has_scores_late) >= 3 and len(has_scores) >= 5 else np.nan,
        }
        result_rows.append(row)

    return pd.DataFrame(result_rows)


class VegasOddsFeatures(ExternalFeatureSource):
    """Market-implied team strength and ATS performance from historical closing lines.

    Features capture three types of signal:
    1. Market power rating: closing spread/moneyline reflect sharp money consensus
    2. ATS performance: teams that consistently beat the spread are undervalued
    3. Market trends: recent cover streaks and momentum vs the line
    """

    def name(self) -> str:
        return "vegas"

    def fetch(self, data_dir: Path) -> None:
        src = data_dir / "external" / "scottfree" / "ncaab.csv"
        if not src.exists():
            raise FileNotFoundError(
                f"Scottfree NCAAB data not found at {src}. "
                "Download from: https://www.scottfreellc.com/shop/p/college-historical-odds-data "
                "and extract to data/external/scottfree/"
            )

    def build(self, data_dir: Path, gender: str = "M") -> pd.DataFrame:
        print("  Building Vegas odds/ATS features...")
        if gender != "M":
            return pd.DataFrame(columns=["Season", "TeamID"])

        name_to_id = _build_name_to_id(data_dir)

        # Load both data sources
        sf_games = _load_scottfree(data_dir, name_to_id)
        api_games = _load_odds_api(data_dir, name_to_id)

        parts = []
        if len(sf_games) > 0:
            sf_unmapped = sf_games["TeamID"].isna().sum()
            print(f"    Scottfree: {len(sf_games)} game rows ({sf_unmapped} unmapped)")
            parts.append(sf_games)
        if len(api_games) > 0:
            api_unmapped = api_games["TeamID"].isna().sum()
            print(f"    Odds API:  {len(api_games)} game rows ({api_unmapped} unmapped)")
            parts.append(api_games)

        if not parts:
            raise FileNotFoundError("No odds data found. Need Scottfree or Odds API data.")

        # Ensure consistent columns across sources
        cols = ["Season", "TeamID", "date", "point_spread", "money_line",
                "over_under", "score", "opp_score", "won_game", "cover_margin",
                "total_points"]
        all_games = pd.concat([p[cols] for p in parts], ignore_index=True)

        result = _aggregate_to_features(all_games)
        print(f"    Built {len(result)} team-season rows across {result['Season'].nunique()} seasons")
        return result
