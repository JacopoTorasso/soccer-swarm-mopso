import numpy as np
import pytest
from unittest.mock import MagicMock
from soccer_swarm.backtest.engine import BacktestEngine
from soccer_swarm.agents.base import MarketPrediction

def _make_mock_agent(p_home=0.5, p_draw=0.25, p_away=0.25):
    agent = MagicMock()
    agent.predict.return_value = MarketPrediction(
        match_1x2=(p_home, p_draw, p_away),
        over_under_25=(0.55, 0.45),
        btts=(0.50, 0.50),
    )
    agent.train.return_value = None
    return agent

def _make_fixtures(n=30):
    fixtures = []
    for i in range(n):
        month = 9 + i // 10
        day = (i % 28) + 1
        fixtures.append({
            "id": i, "league_id": 1,
            "home_team_id": 1, "away_team_id": 2,
            "home_goals": 2 if i % 3 == 0 else 1,
            "away_goals": 1 if i % 3 != 2 else 2,
            "status": "FT",
            "date": f"2025-{month:02d}-{day:02d}T15:00:00+00:00",
        })
    return fixtures

def test_backtest_engine_produces_results():
    agents = [_make_mock_agent() for _ in range(4)]
    engine = BacktestEngine(agents=agents, window_days=30, mopso_pop=10, mopso_gen=5)
    fixtures = _make_fixtures(30)
    results = engine.run(fixtures, standings={})
    assert "global" in results
    assert "accuracy" in results["global"]
    assert "roi" in results["global"]
    assert results["global"]["total_bets"] >= 0

def test_backtest_handles_few_fixtures():
    agents = [_make_mock_agent() for _ in range(4)]
    engine = BacktestEngine(agents=agents, window_days=30, mopso_pop=10, mopso_gen=5)
    results = engine.run(_make_fixtures(5), standings={})
    assert results["global"]["total_bets"] == 0  # not enough data
