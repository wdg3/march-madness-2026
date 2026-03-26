"""Player neural net features — team embeddings + matchup predictions.

Team-level features (from self-attention encoder):
    pnn_emb_0..15: 16-dim learned team representation from player profiles.
    Pipeline creates _A, _B, _delta columns automatically.

Matchup-level features (from cross-attention + prediction head):
    pnn_pred: P(Team A wins) from the full player matchup model.
    Added at matchup construction time, like travel features.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from features.base import FeatureSource


class PlayerNNTeamFeatures(FeatureSource):
    """Team-level embeddings from the trained player matchup neural net.

    Unlike ExternalFeatureSource, this doesn't fetch data — it relies on
    player_impact having already fetched the barttorvik data and the model
    having been trained via `models.player_train.train_player_model()`.
    """

    def name(self) -> str:
        return "player_nn"

    def build(self, data_dir: Path, gender: str = "M") -> pd.DataFrame:
        print("  Building player NN team embeddings...")
        if gender != "M":
            return pd.DataFrame(columns=["Season", "TeamID"])

        ckpt_path = data_dir / "external" / "player_impact" / "player_nn.pt"
        if not ckpt_path.exists():
            print("    No trained model found — skipping player_nn features.")
            return pd.DataFrame(columns=["Season", "TeamID"])

        from models.player_train import PlayerNNExtractor

        extractor = PlayerNNExtractor(data_dir)

        # Generate embeddings for all available seasons
        frames = []
        for (tid, year) in extractor.roster_lookup:
            pass  # just need the years
        seasons = sorted({yr for (_, yr) in extractor.roster_lookup})

        for season in seasons:
            df = extractor.team_embeddings(season)
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame(columns=["Season", "TeamID"])

        result = pd.concat(frames, ignore_index=True)
        print(f"    {len(result)} team-seasons, {result.shape[1]-2} embedding dims")
        return result


def add_player_nn_to_matchups(
    matchups: pd.DataFrame,
    data_dir: Path,
) -> pd.DataFrame:
    """Add player NN matchup predictions to matchup rows.

    Runs the full model (with cross-attention) for each matchup to
    generate P(Team A wins) from individual player interactions.
    """
    ckpt_path = data_dir / "external" / "player_impact" / "player_nn.pt"
    if not ckpt_path.exists():
        return matchups

    from models.player_train import PlayerNNExtractor

    print("    Adding player NN matchup predictions...")
    extractor = PlayerNNExtractor(data_dir)

    team_a = matchups["TeamID_A"].values.astype(int)
    team_b = matchups["TeamID_B"].values.astype(int)
    seasons = matchups["Season"].values.astype(int)

    preds = extractor.matchup_predictions(team_a, team_b, seasons)
    matchups = matchups.copy()
    matchups["pnn_pred"] = preds

    n_valid = np.isfinite(preds).sum()
    print(f"    pnn_pred: {n_valid}/{len(preds)} valid, "
          f"range=[{np.nanmin(preds):.3f}, {np.nanmax(preds):.3f}], "
          f"mean={np.nanmean(preds):.3f}")
    return matchups
