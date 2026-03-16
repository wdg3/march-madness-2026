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
from features.travel import TravelFeatures
from features.external_stubs import (
    VegasOddsFeatures,
    RosterContinuityFeatures,
)
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
    "travel": TravelFeatures,
    # External sources (not enabled by default — require data fetching)
    "vegas": VegasOddsFeatures,
    "roster": RosterContinuityFeatures,
    "kenpom": KenPomFeatures,
    "ap_poll": APPollFeatures,
    "public_picks": PublicPicksFeatures,
}
