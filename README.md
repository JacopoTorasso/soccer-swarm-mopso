<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/MOPSO--CD-pymoo-orange" alt="MOPSO-CD">
  <img src="https://img.shields.io/badge/XGBoost-3.x-blue?logo=xgboost" alt="XGBoost">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Tests-42%20passing-brightgreen" alt="Tests">
</p>

# Soccer Swarm MOPSO

A **multi-agent swarm prediction system** for football match outcomes using **Multi-Objective Particle Swarm Optimization (MOPSO-CD)**. Four independent prediction agents are combined through Pareto-optimal ensemble configurations that balance accuracy, profitability, and risk.

## How It Works

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Poisson    │  │     ELO     │  │   XGBoost   │  │ Odds-Implied│
│   Agent      │  │   Agent     │  │   Agent     │  │   Agent     │
│              │  │              │  │              │  │              │
│ Scoreline    │  │ Rating-based │  │ ML features  │  │ Bookmaker   │
│ probability  │  │ goals        │  │ gradient     │  │ overround   │
│ matrix       │  │ regression   │  │ boosting     │  │ removal     │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │                 │
       └────────┬────────┴────────┬────────┘                 │
                │                 │                           │
                ▼                 ▼                           │
        ┌───────────────────────────────────┐                │
        │         MOPSO-CD Optimizer        │◄───────────────┘
        │                                   │
        │  Decision Variables (7D):         │
        │  [w1, w2, w3, w4,                │
        │   thresh_1x2, thresh_ou,          │
        │   thresh_btts]                    │
        │                                   │
        │  3 Competing Objectives:          │
        │  ● Minimize Log-Loss (accuracy)   │
        │  ● Maximize ROI (profitability)   │
        │  ● Minimize Max Drawdown (risk)   │
        └───────────────┬───────────────────┘
                        │
                        ▼
              ┌───────────────────┐
              │   Pareto Front    │
              │                   │
              │  Conservative ──► Low risk, steady returns
              │  Balanced     ──► Best trade-off
              │  Aggressive   ──► High ROI, higher risk
              └───────────────────┘
```

## Features

- **4 Prediction Agents** — Poisson regression, ELO ratings, XGBoost, and odds-implied probabilities
- **3 Market Types** — Match result (1X2), Over/Under 2.5 goals, Both Teams To Score
- **MOPSO-CD Optimization** — Finds Pareto-optimal ensemble weights via pymoo
- **Walk-Forward Backtesting** — Monthly optimization windows with no data leakage
- **Top 5 European Leagues** — Serie A, Premier League, La Liga, Bundesliga, Ligue 1
- **API-Football Integration** — Free tier compatible (100 req/day) with SQLite caching
- **3 Risk Profiles** — Conservative, balanced, and aggressive Pareto front selection

## Quick Start

### Installation

```bash
git clone https://github.com/JacopoTorasso/soccer-swarm-mopso.git
cd soccer-swarm-mopso
pip install -e .
```

### Usage

```bash
# 1. Set your API-Football key
export API_FOOTBALL_KEY=your_key_here

# 2. Fetch historical data
python -m soccer_swarm fetch

# 3. Run MOPSO optimization + backtest
python -m soccer_swarm optimize --pop-size 100 --n-gen 300

# 4. Generate predictions for upcoming matches
python -m soccer_swarm predict --profile balanced

# 5. Inspect the Pareto front
python -m soccer_swarm pareto --profile conservative
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `fetch` | Download fixtures, standings, and odds from API-Football |
| `train` | Train individual prediction agents on historical data |
| `optimize` | Run MOPSO-CD optimization with walk-forward validation |
| `backtest` | Run walk-forward backtest (alias for optimize) |
| `predict` | Generate ensemble predictions for upcoming matches |
| `pareto` | Inspect and select from the Pareto front |

## Project Structure

```
soccer_swarm/
├── __main__.py              # CLI entry point (argparse)
├── config.py                # Constants, league IDs, API config
├── data/
│   ├── db.py                # SQLite schema (9 tables)
│   ├── client.py            # API client with cache + rate limiting
│   └── features.py          # Feature engineering for XGBoost
├── agents/
│   ├── base.py              # PredictionAgent ABC + MarketPrediction
│   ├── poisson.py           # Poisson scoreline matrix
│   ├── elo.py               # ELO ratings + goals regression
│   ├── xgboost_agent.py     # Gradient boosting (3 classifiers)
│   └── odds_implied.py      # Bookmaker odds normalization
├── optimizer/
│   ├── problem.py           # pymoo Problem (3 objectives, 7 vars)
│   └── mopso.py             # MOPSO-CD runner + Pareto selection
├── backtest/
│   ├── metrics.py           # Log-loss, ROI, max drawdown
│   └── engine.py            # Walk-forward monthly validation
└── output/
    └── formatter.py         # CSV/JSON output
```

## The Agents

### Poisson Agent
Estimates attack/defense strength per team using historical goals. Builds a scoreline probability matrix via `scipy.stats.poisson` and derives all three markets (1X2, O/U, BTTS) from it.

### ELO Agent
Maintains ELO ratings (K=32, home advantage=65) and maps rating differences to expected goals via linear regression. Uses the resulting lambda parameters to build a Poisson matrix for market probabilities.

### XGBoost Agent
Trains three separate `XGBClassifier` models (1X2 multi-class, O/U binary, BTTS binary) on 13 engineered features including form, rank, recent goals average, and goal differential.

### Odds-Implied Agent
Converts bookmaker odds to fair probabilities by removing the overround (normalizing implied probabilities to sum to 1.0). Requires no training — purely transforms market data.

## MOPSO-CD Optimization

The optimizer searches a 7-dimensional space:
- **4 agent weights** — how much to trust each agent (normalized to sum to 1)
- **3 betting thresholds** — minimum edge required to place a bet per market

Three competing objectives are minimized simultaneously:
1. **Log-loss** — prediction accuracy across all markets
2. **Negative ROI** — profitability (minimizing negative = maximizing positive)
3. **Max drawdown** — worst peak-to-trough loss

The result is a **Pareto front** of non-dominated solutions, from which you select based on your risk preference.

## Walk-Forward Backtesting

The backtesting engine prevents data leakage through proper temporal validation:

1. Group fixtures by calendar month
2. For each test month (starting from month 3):
   - **Train** agents on all prior months only
   - **Optimize** MOPSO weights on training data
   - **Evaluate** on the test month (truly out-of-sample)
3. Aggregate accuracy, ROI, and max drawdown across all windows

## Tech Stack

| Library | Purpose |
|---------|---------|
| [pymoo](https://pymoo.org/) | MOPSO-CD multi-objective optimization |
| [XGBoost](https://xgboost.readthedocs.io/) | Gradient boosting classifiers |
| [SciPy](https://scipy.org/) | Poisson distribution, statistical functions |
| [NumPy](https://numpy.org/) | Array operations, linear algebra |
| [scikit-learn](https://scikit-learn.org/) | Preprocessing utilities |
| [requests](https://requests.readthedocs.io/) | API-Football HTTP client |
| SQLite | Local data caching and persistence |

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```


## API-Football Free Tier

This project is designed to work within the [API-Football](https://www.api-football.com/) free tier constraints:
- **100 requests/day** — managed by built-in rate limiter and daily budget
- **Endpoint-specific caching** — fixtures (24h), statistics (7d), standings (12h), odds (1h)
- **All endpoints available** — no feature restrictions on free tier

## License

MIT
