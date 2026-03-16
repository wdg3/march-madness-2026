# March Madness 2026

Machine learning pipeline to predict the 2026 NCAA Men's March Madness tournament for the [Kaggle competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2026).

## Approach

Uses [AutoGluon](https://auto.gluon.ai/) with `best_quality` preset (multi-layer stacking, 8-fold bagging) to train an ensemble of gradient boosting models (LightGBM, XGBoost, CatBoost), neural networks, random forests, and more. Optimized for log loss.

Each matchup is modeled as a pairwise comparison: `[TeamA_features | TeamB_features] → P(TeamA wins)`. Predictions are symmetry-enforced so `P(A>B) + P(B>A) = 1`.

## Features

Five pluggable feature sources, each producing per-team-per-season features:

| Source | Description | Features |
|--------|-------------|----------|
| **Massey Ordinals** | 194 ranking systems (AP, Sagarin, KenPom, etc.) | ~194 |
| **Regular Season** | Efficiency stats from box scores (off/def efficiency, shooting %, rebounding, turnovers) | 20 |
| **Seeds** | Tournament seed parsed to integer (1-16) | 1 |
| **Conference** | Conference affiliation (label-encoded) | 1 |
| **Tournament History** | Appearances and wins in prior 5 seasons | 2 |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the [competition data](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data) and extract into `data/`.

## Usage

```bash
python main.py
```

Output: `output/submission.csv` in Kaggle submission format.

### Bracket Simulation

Once you have a submission CSV, run the Monte Carlo bracket simulator to fill out a bracket:

```bash
python simulate.py [--n-sims 10000] [--submission output/submission.csv] [--season 2026]
```

This simulates the full tournament thousands of times using the model's pairwise probabilities, then picks the team that advances from each bracket slot most often. Unlike naive "always pick the favorite" approaches, this properly accounts for path probability — a team's chance of reaching the Final Four depends on the probability of winning *every game along the way*, not just individual matchups.

Output: `output/bracket.csv` and a formatted bracket printed to the console showing the pick and top contenders for each slot.

## Adding New Feature Sources

1. Create a new file in `features/` implementing `FeatureSource`:

```python
from features.base import FeatureSource

class MyFeatures(FeatureSource):
    def name(self):
        return "my"

    def build(self, data_dir):
        # Return DataFrame with columns [Season, TeamID, my_feat1, my_feat2, ...]
        ...
```

2. Register it in `features/__init__.py`
3. Add the name to `ENABLED_FEATURES` in `config.py`

## Configuration

All tunable parameters are in `config.py`: train/validation/prediction seasons, AutoGluon presets, time limits, bagging folds, and enabled feature sources.
