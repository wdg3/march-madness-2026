# March Madness 2026

ML pipeline to predict the 2026 NCAA March Madness tournament (men's and women's) for the [Kaggle competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2026).

## Approach

Uses [AutoGluon](https://auto.gluon.ai/) with custom regularized hyperparameters, 2-level stacking, and 10-fold bagging to train an ensemble of gradient boosting, tree, and neural network models (LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees, NN_TORCH, FastAI). Optimized for **Brier score** (better calibration on lopsided matchups than log loss). GPU-accelerated for neural nets, CPU for tree models. Trained on men's tournament data from 2010-2024 (14 seasons, excluding 2020/COVID), validated on 2025.

Each matchup is modeled as a pairwise comparison: `[TeamA_features | TeamB_features | delta_features] -> P(TeamA wins)`. Delta features (A - B for every stat) give the model direct access to relative differences. Predictions are symmetry-enforced so `P(A>B) + P(B>A) = 1`. The same model is applied to women's tournament predictions (transfer learning -- features that don't exist for women become NaN, which AutoGluon handles natively).

Model hyperparameters are explicitly regularized for the small tournament dataset (~950 games, ~1,900 training rows): capped tree depth, L1/L2 regularization, feature subsampling, and high minimum leaf samples to prevent overfitting.

## Features

22 pluggable feature sources:

| Source | Description | Features |
|--------|-------------|----------|
| **Massey Ordinals** | 194 ranking systems (AP, Sagarin, KenPom, etc.) | ~194 |
| **Massey Trajectory** | Per-system rank trends, convergence, aggregate momentum | 15 |
| **KenPom/BartTorvik** | Adjusted efficiency, tempo, BARTHAG from pre-tournament snapshots | 8 |
| **AP Poll** | Poll trajectory -- weeks ranked, preseason/final rank, volatility | 8 |
| **Public Picks** | ESPN bracket pick percentages per round (Vegas proxy) | 6 |
| **Vegas Odds** | Market-implied strength, ATS performance, cover margin trends | 18 |
| **Roster** | Returning minutes %, new player share, class seniority | 4 |
| **Regular Season** | Efficiency stats from box scores (off/def efficiency, shooting %, rebounds) | 19 |
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

3. For Vegas/ATS features, place the [Scottfree NCAAB historical odds](https://www.scottfreellc.com/shop/p/college-historical-odds-data) at `data/external/scottfree/ncaab.csv`. For the current season, run `python run.py data fetch --source odds --api-key YOUR_KEY`.

4. For roster continuity and KenPom/BartTorvik features, the pipeline auto-fetches from barttorvik.com on first run (cached in `data/external/roster/` and `data/external/kenpom/`). Or fetch explicitly: `python run.py data fetch --source kenpom roster`.

## Usage

**Everything goes through `run.py`.** Every command requires a `--tag` for model versioning. Models save to `AutogluonModels/<tag>/`, outputs to `output/<tag>/`.

### Quick start

```bash
python run.py full --tag v1               # Train + predict + bracket (one shot)
```

### Individual commands

```bash
# Training
python run.py train --tag v1              # Train (default 2hr time limit)
python run.py train --tag v1 --time-limit 10800  # 3 hours

# Predictions
python run.py predict --tag v1            # Men's submission only
python run.py submit --tag v1             # Men's + women's for Kaggle

# Bracket simulation
python run.py bracket --tag v1            # 10K sims (default)
python run.py bracket --tag v1 --n-sims 100000

# Betting
python run.py bet --tag v1 --odds odds.csv
python run.py bet --tag v1 --odds odds.csv --kelly 0.5 --fee 0.02 --bankroll 500
python run.py futures --tag v1                    # Kalshi futures (live API)
python run.py futures --tag v1 --from-csv cached.csv  # From cached data

# Analysis
python run.py backtest --tag v1 --season 2025
python run.py analyze matchups --tag v1 --seeds 8v9
python run.py analyze team --tag v1 --team Duke
python run.py analyze confidence --tag v1

# Data management
python run.py data status                         # Show data freshness
python run.py data fetch                          # Refresh all external data
python run.py data fetch --source kenpom roster   # Refresh specific sources
python run.py data fetch --source odds --api-key KEY  # Fetch Odds API data
```

### Output structure

All outputs are organized by tag:

```
output/<tag>/
    submission.csv      Kaggle-format predictions
    bracket.csv         Monte Carlo bracket picks
    bets.csv            Head-to-head bet sheet
    futures_bets.csv    Kalshi futures bet sheet
```

### Configuration

Model configuration is stored in YAML files in `configs/`:

```yaml
# configs/default.yaml (or configs/<tag>.yaml for per-model config)
features:
  - massey
  - seeds
  - vegas
  # ... all 22 features

training:
  presets: best_quality
  time_limit: 7200
  num_bag_folds: 10
  num_stack_levels: 2
  train_seasons_start: 2010
  validation_season: 2025
```

Create `configs/<tag>.yaml` to override defaults for a specific model. If no tag-specific config exists, `configs/default.yaml` is used. If that doesn't exist, hardcoded defaults in `config.py` apply.

### Provenance tracking

Every trained model saves a `manifest.json` alongside it:

```json
{
  "tag": "v1",
  "trained_at": "2026-03-21T02:30:00+00:00",
  "git_sha": "abc1234",
  "features": ["massey", "seeds", ...],
  "config": { ... },
  "train_rows": 1200,
  "val_rows": 134,
  "val_brier": -0.182,
  "data_files": { "MNCAATourneyCompactResults.csv": 1742567890123456789, ... }
}
```

When generating predictions, the pipeline warns if the current feature configuration doesn't match what the model was trained with.

### Caching

Feature matrices are cached to `.cache/` with content-aware cache keys. The cache key includes modification times of all source data files, so changing any underlying CSV automatically invalidates the cache. No need to manually bust the cache.

### Bet sheet parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--kelly` | 0.25 | Fraction of full Kelly to use (quarter Kelly) |
| `--min-edge` | 0.02 | Minimum edge to place a bet (2%) |
| `--max-bet` | 0.05 | Maximum bet as fraction of bankroll (5%) |
| `--bankroll` | 1000 | Starting bankroll in dollars |
| `--fee` | 0.00 | Platform fee per contract (for head-to-head bets) |
| `--total-cost` | 1.01 | YES + NO total cost (for futures, 101c = 1% vig) |

### Odds CSV format (for `bet` command)

Prediction market style -- prices as implied probability %:

```csv
Season,TeamA,TeamB,PriceA,PriceB
2026,1181,1369,95,8
2026,1385,1314,60,42
```

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
3. Add the name to `configs/default.yaml` (or `config.py` ENABLED_FEATURES)

For external data sources, extend `ExternalFeatureSource` instead -- this adds a `fetch()` method that downloads data to `data/external/<name>/` with automatic caching.

## Project structure

```
run.py                  Unified CLI (train/predict/bracket/bet/futures/backtest/analyze/data/full)
config.py               Constants, feature toggles, YAML config loading
pipeline.py             Feature matrix construction with content-aware caching
training.py             AutoGluon training with Brier score, regularization, provenance
submission.py           Kaggle submission CSV generation with symmetry enforcement
simulate.py             Monte Carlo bracket simulation
backtest.py             ESPN-scored bracket backtesting
betting.py              Kelly Criterion head-to-head bet sheet
futures.py              Kalshi futures bet sheet (YES/NO positions)
analyze.py              Matchup analysis, team profiles, confidence distributions
fetch_odds.py           Odds API data fetcher with ESPN schedule cross-reference
kelly.py                Shared Kelly Criterion calculations

configs/
  default.yaml          Default model configuration

features/
  base.py               FeatureSource and ExternalFeatureSource base classes
  __init__.py            Feature registry
  seeds.py              Tournament seeds
  conference.py         Conference affiliation
  massey.py             Massey Ordinals (194 ranking systems)
  massey_meta.py        Ranking disagreement and seed-rank delta
  massey_trajectory.py  Per-system rank trends and convergence
  regular_season.py     Box score efficiency stats
  trajectory.py         Regular season windowed splits and trends
  regular_season_advanced.py  Close games, scoring variance, momentum, tempo
  tourney_history.py    Prior tournament appearances and wins
  coach.py              Coach tournament record
  conf_tourney.py       Conference tournament performance
  location.py           Away/neutral game splits
  travel.py             Distance to game venue (matchup-level)
  kenpom.py             BartTorvik adjusted efficiency + AP Poll + public picks
  vegas.py              Vegas odds and ATS features (Scottfree + Odds API)
  roster.py             Roster continuity from BartTorvik player data
```

## Data sources

| Source | Location | Coverage | Notes |
|--------|----------|----------|-------|
| Kaggle competition data | `data/` | All seasons | Official competition datasets |
| BartTorvik time machine | `data/external/kenpom/` | 2010-2026 | Pre-tournament snapshots (Selection Sunday), auto-fetched |
| Kaggle external dataset | `data/external/kaggle_mm/` | 2010-2025 | AP Poll, public picks |
| Scottfree NCAAB odds | `data/external/scottfree/` | 2008-2025 | Historical closing lines (paid dataset) |
| The Odds API | `data/external/odds_api/` | 2026 | Current season consensus closing lines |
| BartTorvik roster data | `data/external/roster/` | 2008-2026 | Player stats for returning minutes %, auto-fetched |
| Kalshi markets | Live API | 2026 | Tournament advancement futures |
