"""Advanced regular season features: close games, scoring variance, momentum, tempo."""

from pathlib import Path
import pandas as pd
import numpy as np
from features.base import FeatureSource


def _unpivot_games(data_dir, gender="M"):
    """Create one row per team per game with margin and game metadata."""
    df = pd.read_csv(data_dir / f"{gender}RegularSeasonDetailedResults.csv")
    winners = pd.DataFrame({
        "Season": df["Season"],
        "DayNum": df["DayNum"],
        "TeamID": df["WTeamID"],
        "Score": df["WScore"],
        "OppScore": df["LScore"],
        "FGA": df["WFGA"],
        "FGM": df["WFGM"],
        "OR": df["WOR"],
        "TO": df["WTO"],
        "FTA": df["WFTA"],
        "OppFGA": df["LFGA"],
        "OppOR": df["LOR"],
        "OppTO": df["LTO"],
        "OppFTA": df["LFTA"],
        "Win": 1,
    })
    losers = pd.DataFrame({
        "Season": df["Season"],
        "DayNum": df["DayNum"],
        "TeamID": df["LTeamID"],
        "Score": df["LScore"],
        "OppScore": df["WScore"],
        "FGA": df["LFGA"],
        "FGM": df["LFGM"],
        "OR": df["LOR"],
        "TO": df["LTO"],
        "FTA": df["LFTA"],
        "OppFGA": df["WFGA"],
        "OppOR": df["WOR"],
        "OppTO": df["WTO"],
        "OppFTA": df["WFTA"],
        "Win": 0,
    })
    all_games = pd.concat([winners, losers], ignore_index=True)
    all_games["Margin"] = all_games["Score"] - all_games["OppScore"]
    return all_games


class CloseGameFeatures(FeatureSource):
    """Win percentage and record in games decided by 5 or fewer points."""

    def name(self) -> str:
        return "close_games"

    def build(self, data_dir: Path, gender: str = "M") -> pd.DataFrame:
        print("  Building close game features...")
        all_games = _unpivot_games(data_dir, gender)

        close = all_games[all_games["Margin"].abs() <= 5]
        g = close.groupby(["Season", "TeamID"])

        close_agg = pd.DataFrame({
            "cg_close_wins": g["Win"].sum(),
            "cg_close_games": g["Win"].count(),
        }).reset_index()
        close_agg["cg_close_win_pct"] = (
            close_agg["cg_close_wins"] / close_agg["cg_close_games"]
        )

        # Also compute fraction of games that were close (indicates team plays tight games)
        total_games = all_games.groupby(["Season", "TeamID"])["Win"].count().reset_index()
        total_games.rename(columns={"Win": "total_games"}, inplace=True)

        close_agg = close_agg.merge(total_games, on=["Season", "TeamID"], how="left")
        close_agg["cg_close_pct"] = close_agg["cg_close_games"] / close_agg["total_games"]

        return close_agg[["Season", "TeamID", "cg_close_win_pct", "cg_close_games", "cg_close_pct"]]


class ScoringVarianceFeatures(FeatureSource):
    """Standard deviation of scoring margin — captures consistency vs volatility."""

    def name(self) -> str:
        return "scoring_variance"

    def build(self, data_dir: Path, gender: str = "M") -> pd.DataFrame:
        print("  Building scoring variance features...")
        all_games = _unpivot_games(data_dir, gender)

        g = all_games.groupby(["Season", "TeamID"])
        result = pd.DataFrame({
            "sv_margin_std": g["Margin"].std(),
            "sv_score_std": g["Score"].std(),
            "sv_opp_score_std": g["OppScore"].std(),
        }).reset_index()

        return result


class MomentumFeatures(FeatureSource):
    """Late-season performance vs full-season — captures teams peaking or slumping."""

    LATE_SEASON_CUTOFF_PERCENTILE = 75  # top 25% of DayNum = last ~quarter of season

    def name(self) -> str:
        return "momentum"

    def build(self, data_dir: Path, gender: str = "M") -> pd.DataFrame:
        print("  Building momentum features...")
        df = pd.read_csv(data_dir / f"{gender}RegularSeasonDetailedResults.csv")
        all_games = _unpivot_games(data_dir, gender)

        # Compute per-season DayNum cutoff for "late season"
        cutoffs = df.groupby("Season")["DayNum"].quantile(
            self.LATE_SEASON_CUTOFF_PERCENTILE / 100
        ).reset_index()
        cutoffs.rename(columns={"DayNum": "cutoff"}, inplace=True)

        all_games = all_games.merge(cutoffs, on="Season", how="left")

        # Full season stats
        full = all_games.groupby(["Season", "TeamID"]).agg(
            full_win_pct=("Win", "mean"),
            full_margin=("Margin", "mean"),
        ).reset_index()

        # Late season stats
        late = all_games[all_games["DayNum"] >= all_games["cutoff"]]
        late_agg = late.groupby(["Season", "TeamID"]).agg(
            late_win_pct=("Win", "mean"),
            late_margin=("Margin", "mean"),
        ).reset_index()

        merged = full.merge(late_agg, on=["Season", "TeamID"], how="left")

        result = pd.DataFrame({
            "Season": merged["Season"],
            "TeamID": merged["TeamID"],
            "mom_win_pct_delta": merged["late_win_pct"] - merged["full_win_pct"],
            "mom_margin_delta": merged["late_margin"] - merged["full_margin"],
            "mom_late_win_pct": merged["late_win_pct"],
            "mom_late_margin": merged["late_margin"],
        })

        return result


class TempoFeatures(FeatureSource):
    """Possessions per game — high-tempo games have more variance."""

    def name(self) -> str:
        return "tempo"

    def build(self, data_dir: Path, gender: str = "M") -> pd.DataFrame:
        print("  Building tempo features...")
        all_games = _unpivot_games(data_dir, gender)

        # Estimate possessions per game
        all_games["Poss"] = (
            all_games["FGA"] - all_games["OR"]
            + all_games["TO"] + 0.475 * all_games["FTA"]
        )

        g = all_games.groupby(["Season", "TeamID"])
        result = pd.DataFrame({
            "tempo_avg_poss": g["Poss"].mean(),
            "tempo_poss_std": g["Poss"].std(),
        }).reset_index()

        return result
