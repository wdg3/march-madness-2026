from pathlib import Path
import pandas as pd
import numpy as np
from features.base import FeatureSource


class RegularSeasonFeatures(FeatureSource):
    def name(self) -> str:
        return "rs"

    def build(self, data_dir: Path, gender: str = "M") -> pd.DataFrame:
        print("  Building regular season features...")
        df = pd.read_csv(data_dir / f"{gender}RegularSeasonDetailedResults.csv")

        # Unpivot: create one row per team per game
        # Winner perspective
        winners = pd.DataFrame({
            "Season": df["Season"],
            "TeamID": df["WTeamID"],
            "Win": 1,
            "Score": df["WScore"],
            "OppScore": df["LScore"],
            "FGM": df["WFGM"], "FGA": df["WFGA"],
            "FGM3": df["WFGM3"], "FGA3": df["WFGA3"],
            "FTM": df["WFTM"], "FTA": df["WFTA"],
            "OR": df["WOR"], "DR": df["WDR"],
            "Ast": df["WAst"], "TO": df["WTO"],
            "Stl": df["WStl"], "Blk": df["WBlk"],
            "OppFGM": df["LFGM"], "OppFGA": df["LFGA"],
            "OppOR": df["LOR"], "OppDR": df["LDR"],
            "OppFTM": df["LFTM"], "OppFTA": df["LFTA"],
            "OppTO": df["LTO"],
        })

        # Loser perspective
        losers = pd.DataFrame({
            "Season": df["Season"],
            "TeamID": df["LTeamID"],
            "Win": 0,
            "Score": df["LScore"],
            "OppScore": df["WScore"],
            "FGM": df["LFGM"], "FGA": df["LFGA"],
            "FGM3": df["LFGM3"], "FGA3": df["LFGA3"],
            "FTM": df["LFTM"], "FTA": df["LFTA"],
            "OR": df["LOR"], "DR": df["LDR"],
            "Ast": df["LAst"], "TO": df["LTO"],
            "Stl": df["LStl"], "Blk": df["LBlk"],
            "OppFGM": df["WFGM"], "OppFGA": df["WFGA"],
            "OppOR": df["WOR"], "OppDR": df["WDR"],
            "OppTO": df["WTO"],
            "OppFTM": df["WFTM"], "OppFTA": df["WFTA"],
        })

        all_games = pd.concat([winners, losers], ignore_index=True)

        # Aggregate per (Season, TeamID)
        g = all_games.groupby(["Season", "TeamID"])
        agg = g.agg(
            games=("Win", "count"),
            wins=("Win", "sum"),
            total_score=("Score", "sum"),
            total_opp_score=("OppScore", "sum"),
            total_fgm=("FGM", "sum"), total_fga=("FGA", "sum"),
            total_fgm3=("FGM3", "sum"), total_fga3=("FGA3", "sum"),
            total_ftm=("FTM", "sum"), total_fta=("FTA", "sum"),
            total_or=("OR", "sum"), total_dr=("DR", "sum"),
            total_ast=("Ast", "sum"), total_to=("TO", "sum"),
            total_stl=("Stl", "sum"), total_blk=("Blk", "sum"),
            total_opp_fgm=("OppFGM", "sum"), total_opp_fga=("OppFGA", "sum"),
            total_opp_or=("OppOR", "sum"), total_opp_dr=("OppDR", "sum"),
            total_opp_ftm=("OppFTM", "sum"), total_opp_fta=("OppFTA", "sum"),
            total_opp_to=("OppTO", "sum"),
        ).reset_index()

        n = agg["games"]
        result = pd.DataFrame({
            "Season": agg["Season"],
            "TeamID": agg["TeamID"],
            "rs_win_pct": agg["wins"] / n,
            "rs_avg_score": agg["total_score"] / n,
            "rs_avg_allowed": agg["total_opp_score"] / n,
            "rs_avg_margin": (agg["total_score"] - agg["total_opp_score"]) / n,
            "rs_fg_pct": agg["total_fgm"] / agg["total_fga"],
            "rs_fg3_pct": agg["total_fgm3"] / agg["total_fga3"].replace(0, np.nan),
            "rs_ft_pct": agg["total_ftm"] / agg["total_fta"].replace(0, np.nan),
            "rs_fg_pct_allowed": agg["total_opp_fgm"] / agg["total_opp_fga"],
            "rs_or_avg": agg["total_or"] / n,
            "rs_dr_avg": agg["total_dr"] / n,
            "rs_reb_margin": (agg["total_or"] + agg["total_dr"] - agg["total_opp_or"] - agg["total_opp_dr"]) / n,
            "rs_ast_avg": agg["total_ast"] / n,
            "rs_to_avg": agg["total_to"] / n,
            "rs_ast_to_ratio": agg["total_ast"] / agg["total_to"].replace(0, np.nan),
            "rs_stl_avg": agg["total_stl"] / n,
            "rs_blk_avg": agg["total_blk"] / n,
        })

        # Efficiency metrics (KenPom-style)
        poss = agg["total_fga"] - agg["total_or"] + agg["total_to"] + 0.475 * agg["total_fta"]
        opp_poss = agg["total_opp_fga"] - agg["total_opp_or"] + agg["total_opp_to"] + 0.475 * agg["total_opp_fta"]
        result["rs_off_eff"] = agg["total_score"] / poss * 100
        result["rs_def_eff"] = agg["total_opp_score"] / opp_poss * 100
        result["rs_eff_margin"] = result["rs_off_eff"] - result["rs_def_eff"]

        return result
