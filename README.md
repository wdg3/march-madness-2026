# March Madness 2026

ML pipeline to predict the 2026 NCAA March Madness tournament (men's and women's) for the [Kaggle competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2026).

## Approach

Uses [AutoGluon](https://auto.gluon.ai/) with `best_quality` preset (multi-layer stacking, 8-fold bagging) to train an ensemble of gradient boosting and tree models (CatBoost, LightGBM, RandomForest, ExtraTrees). Optimized for log loss. Trained on men's data from 2010-2024, validated on 2025.

Each matchup is modeled as a pairwise comparison: `[TeamA_features | TeamB_features] -> P(TeamA wins)`. Predictions are symmetry-enforced so `P(A>B) + P(B>A) = 1`. The same model is applied to women's tournament predictions (transfer learning — features that don't exist for women become NaN, which AutoGluon handles natively).

## Features

18 pluggable feature sources:

| Source | Description | Features |
|--------|-------------|----------|
| **Massey Ordinals** | 194 ranking systems (AP, Sagarin, KenPom, etc.) | ~194 |
| **KenPom/BartTorvik** | Adjusted efficiency, tempo, BARTHAG, four factors, talent | 20 |
| **AP Poll** | Poll trajectory — weeks ranked, preseason/final rank, volatility | 8 |
| **Public Picks** | ESPN bracket pick percentages per round (Vegas proxy) | 6 |
| **Regular Season** | Efficiency stats from box scores (off/def efficiency, shooting %, rebounds) | 20 |
| **Ranking Disagreement** | Std/range/mean/median across Massey systems | 5 |
| **Seed-Rank Delta** | Gap between seed and average ranking | 2 |
| **Close Games** | Win % in games decided by 5 or fewer points | 3 |
| **Scoring Variance** | Std dev of margin, score, opponent score | 3 |
| **Momentum** | Late-season performance vs full-season average | 4 |
| **Tempo** | Possessions per game (mean, std) | 2 |
| **Seeds** | Tournament seed (1-16) | 1 |
| **Conference** | Conference affiliation (label-encoded) | 1 |
| **Tournament History** | Appearances and wins in prior 5 seasons | 2 |
| **Coach** | Career tournament wins, appearances, win rate, experience | 4 |
| **Conference Tourney** | Conference tournament performance | 4 |
| **Location** | Away/neutral game performance splits | 7 |
| **Travel** | Distance from each team's school to game venue (matchup-level) | 3 |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the [competition data](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data) and extract into `data/`.

For KenPom/AP Poll/public picks features, also download the [external dataset](https://www.kaggle.com/datasets/nishaanamin/march-madness-data) and extract into `data/external/kaggle_mm/`.

## Usage

All commands go through `run.py`:

### 1. Train the model

```bash
python run.py train [--time-limit 7200]
```

Trains on men's historical tournament data (2010-2024), validates on 2025. Saves the model to `AutogluonModels/`. Default time limit is 2 hours (7200s).

### 2. Generate men's submission

```bash
python run.py predict
```

Generates `output/submission.csv` with men's predictions only.

### 3. Simulate bracket

```bash
python run.py bracket [--n-sims 10000] [--submission output/submission.csv]
```

Runs Monte Carlo bracket simulation using the model's pairwise probabilities. Simulates the full tournament thousands of times, then picks the team that advances from each slot most often. Unlike naive "always pick the favorite" approaches, this properly accounts for path probability.

Outputs a formatted bracket to the console and saves `output/bracket.csv`.

### 4. Generate full Kaggle submission (men's + women's)

```bash
python run.py submit
```

Generates `output/submission.csv` with both men's and women's predictions, matching the format required by `SampleSubmissionStage2.csv`. Women's predictions use the same model trained on men's data — features that aren't available for women (Massey rankings, KenPom, AP Poll, coach stats) are filled with NaN.

## Configuration

All tunable parameters are in `config.py`: train/validation/prediction seasons, AutoGluon presets, time limits, bagging folds, and enabled feature sources.

## Adding New Feature Sources

1. Create a new file in `features/` implementing `FeatureSource`:

```python
from features.base import FeatureSource

class MyFeatures(FeatureSource):
    def name(self):
        return "my"

    def build(self, data_dir, gender="M"):
        # Return DataFrame with columns [Season, TeamID, my_feat1, my_feat2, ...]
        ...
```

2. Register it in `features/__init__.py`
3. Add the name to `ENABLED_FEATURES` in `config.py`

For external data sources, extend `ExternalFeatureSource` instead — this adds a `fetch()` method that downloads data to `data/external/<name>/` with automatic caching.
