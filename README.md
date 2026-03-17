# March Madness 2026

ML pipeline to predict the 2026 NCAA March Madness tournament (men's and women's) for the [Kaggle competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2026).

## Approach

Uses [AutoGluon](https://auto.gluon.ai/) with `best_quality` preset (multi-layer stacking, 8-fold bagging) to train an ensemble of gradient boosting, tree, and neural network models (CatBoost, LightGBM, RandomForest, ExtraTrees, NN_TORCH, FastAI). Optimized for **Brier score** (better calibration on lopsided matchups than log loss). GPU-accelerated for neural nets, CPU for tree models. Trained on men's data from 2010-2024, validated on 2025.

Each matchup is modeled as a pairwise comparison: `[TeamA_features | TeamB_features] -> P(TeamA wins)`. Predictions are symmetry-enforced so `P(A>B) + P(B>A) = 1`. The same model is applied to women's tournament predictions (transfer learning -- features that don't exist for women become NaN, which AutoGluon handles natively).

## Features

20 pluggable feature sources:

| Source | Description | Features |
|--------|-------------|----------|
| **Massey Ordinals** | 194 ranking systems (AP, Sagarin, KenPom, etc.) | ~194 |
| **Massey Trajectory** | Per-system rank trends, convergence, aggregate momentum | 15 |
| **KenPom/BartTorvik** | Adjusted efficiency, tempo, BARTHAG, four factors, talent | 20 |
| **AP Poll** | Poll trajectory -- weeks ranked, preseason/final rank, volatility | 8 |
| **Public Picks** | ESPN bracket pick percentages per round (Vegas proxy) | 6 |
| **Regular Season** | Efficiency stats from box scores (off/def efficiency, shooting %, rebounds) | 20 |
| **RS Trajectory** | Windowed early/late splits, linear trends, volatility for season stats | 35 |
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

Requires Python 3.10+ and a CUDA-capable GPU (for neural net models).

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data

1. Download the [competition data](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/data) and extract into `data/`:

```bash
kaggle competitions download -c march-machine-learning-mania-2026
unzip march-machine-learning-mania-2026.zip -d data/
```

2. For KenPom/AP Poll/public picks features, download the [external dataset](https://www.kaggle.com/datasets/nishaanamin/march-madness-data) and extract into `data/external/kaggle_mm/`:

```bash
kaggle datasets download -d nishaanamin/march-madness-data
unzip march-madness-data.zip -d data/external/kaggle_mm/
```

## Usage

All commands go through `run.py`. Every command requires a `--tag` to support model versioning (models save to `AutogluonModels/<tag>/`, outputs to `output/submission_<tag>.csv`).

### Train

```bash
python run.py train --tag v1
python run.py train --tag v1 --time-limit 10800   # 3 hours
```

Trains on men's historical tournament data (2010-2024), validates on 2025. Default time limit is 2 hours (7200s). Longer training times generally improve results as AutoGluon explores more model configurations.

### Predict (men's only)

```bash
python run.py predict --tag v1
```

Generates `output/submission_v1.csv` with men's predictions.

### Submit (men's + women's for Kaggle)

```bash
python run.py submit --tag v1
```

Generates `output/submission_v1.csv` with both men's and women's predictions, matching the format required by `SampleSubmissionStage2.csv`.

### Bracket simulation

```bash
python run.py bracket --tag v1
python run.py bracket --tag v1 --n-sims 50000
```

Runs Monte Carlo bracket simulation (default 10,000 sims) using the model's pairwise probabilities. Picks the team that advances from each slot most often. Outputs a formatted bracket to the console and saves `output/bracket.csv`.

### Backtest

Score a model's bracket picks against actual tournament results using ESPN scoring rules (10/20/40/80/160/320 points per round):

```bash
# Using a pre-built submission file
python backtest.py --season 2025 --submission output/submission_v1.csv

# Using a trained model directly
python backtest.py --season 2025 --tag v1
```

### Betting (Kelly Criterion)

Generate optimal bet sizing from model predictions and prediction market odds:

```bash
python betting.py --odds odds.csv --tag v1
python betting.py --odds odds.csv --tag v1 --kelly 0.5 --fee 0.02
python betting.py --odds odds.csv --submission output/submission_v1.csv --bankroll 500
```

**Odds CSV format** (prediction market style -- prices as implied probability %):
```csv
Season,TeamA,TeamB,PriceA,PriceB
2026,1181,1369,95,8
2026,1385,1314,60,42
```

**Parameters:**
- `--kelly` -- Fraction of full Kelly to use (default: 0.25 = quarter Kelly)
- `--min-edge` -- Minimum edge to place a bet (default: 0.02 = 2%)
- `--max-bet` -- Maximum bet as fraction of bankroll (default: 0.05 = 5%)
- `--bankroll` -- Starting bankroll in dollars (default: $1000)
- `--fee` -- Platform fee per contract in dollars (default: 0, e.g. 0.02 for 2c)
- `--output` -- Save bet sheet to CSV

## Configuration

All tunable parameters are in `config.py`: train/validation/prediction seasons, AutoGluon presets, time limits, bagging folds, and enabled feature sources. Toggle features on/off by editing the `ENABLED_FEATURES` list.

## Adding new feature sources

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

For external data sources, extend `ExternalFeatureSource` instead -- this adds a `fetch()` method that downloads data to `data/external/<name>/` with automatic caching.

## Example workflow

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train (2-hour run)
python run.py train --tag v1

# Generate Kaggle submission and bracket
python run.py submit --tag v1
python run.py bracket --tag v1

# Backtest against 2025 actuals
python backtest.py --season 2025 --submission output/submission_v1.csv

# Place bets on First Four games
python betting.py --odds odds_first_four.csv --tag v1 --kelly 0.5 --fee 0.02
```
