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
}
