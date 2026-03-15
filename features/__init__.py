from features.massey import MasseyFeatures
from features.seeds import SeedFeatures
from features.conference import ConferenceFeatures
from features.regular_season import RegularSeasonFeatures
from features.tourney_history import TourneyHistoryFeatures

REGISTRY = {
    "massey": MasseyFeatures,
    "seeds": SeedFeatures,
    "conference": ConferenceFeatures,
    "regular_season": RegularSeasonFeatures,
    "tourney_history": TourneyHistoryFeatures,
}
