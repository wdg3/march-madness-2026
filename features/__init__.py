from features.massey import MasseyFeatures
from features.seeds import SeedFeatures
from features.conference import ConferenceFeatures
from features.regular_season import RegularSeasonFeatures
from features.tourney_history import TourneyHistoryFeatures
from features.massey_meta import RankingDisagreementFeatures, SeedRankDeltaFeatures
from features.regular_season_advanced import (
    CloseGameFeatures,
    ScoringVarianceFeatures,
    MomentumFeatures,
    TempoFeatures,
)
from features.coach import CoachFeatures
from features.conf_tourney import ConfTourneyFeatures
from features.location import LocationFeatures
# Note: travel is a matchup-level feature handled in pipeline.py, not here
from features.trajectory import RegularSeasonTrajectoryFeatures
from features.massey_trajectory import MasseyTrajectoryFeatures
from features.vegas import VegasOddsFeatures
from features.roster import RosterContinuityFeatures
from features.kenpom import KenPomFeatures, APPollFeatures, PublicPicksFeatures

REGISTRY = {
    "massey": MasseyFeatures,
    "seeds": SeedFeatures,
    "conference": ConferenceFeatures,
    "regular_season": RegularSeasonFeatures,
    "tourney_history": TourneyHistoryFeatures,
    "rank_disagree": RankingDisagreementFeatures,
    "seed_rank_delta": SeedRankDeltaFeatures,
    "close_games": CloseGameFeatures,
    "scoring_variance": ScoringVarianceFeatures,
    "momentum": MomentumFeatures,
    "tempo": TempoFeatures,
    "coach": CoachFeatures,
    "conf_tourney": ConfTourneyFeatures,
    "location": LocationFeatures,
    "rs_trajectory": RegularSeasonTrajectoryFeatures,
    "massey_trajectory": MasseyTrajectoryFeatures,
    # "travel" is a matchup-level feature, not team-level — handled in pipeline.py
    # External sources (not enabled by default — require data fetching)
    "vegas": VegasOddsFeatures,
    "roster": RosterContinuityFeatures,
    "kenpom": KenPomFeatures,
    "ap_poll": APPollFeatures,
    "public_picks": PublicPicksFeatures,
}
