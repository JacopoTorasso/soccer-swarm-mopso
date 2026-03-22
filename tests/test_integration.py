import numpy as np
import pytest
from soccer_swarm.agents.poisson import PoissonAgent
from soccer_swarm.agents.elo import EloAgent
from soccer_swarm.agents.odds_implied import OddsImpliedAgent
from soccer_swarm.backtest.engine import BacktestEngine


def _make_realistic_fixtures(n=60):
    """Generate semi-realistic fixtures for 2 teams."""
    rng = np.random.default_rng(42)
    fixtures = []
    for i in range(n):
        home_goals = int(rng.poisson(1.5))
        away_goals = int(rng.poisson(1.1))
        month = 9 + i // 20
        day = (i % 28) + 1
        fixtures.append({
            "id": i, "league_id": 1,
            "home_team_id": 1 if i % 2 == 0 else 2,
            "away_team_id": 2 if i % 2 == 0 else 1,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "status": "FT",
            "date": f"2025-{month:02d}-{day:02d}T15:00:00+00:00",
        })
    return fixtures


def test_full_pipeline_runs_without_error():
    fixtures = _make_realistic_fixtures(60)

    agents = [PoissonAgent(), EloAgent(), OddsImpliedAgent()]
    engine = BacktestEngine(agents=agents, mopso_pop=10, mopso_gen=5)
    results = engine.run(fixtures, standings={})

    assert "global" in results
    assert isinstance(results["global"]["accuracy"], float)
    assert isinstance(results["global"]["roi"], float)
    assert results["global"]["total_fixtures"] > 0


def test_agents_produce_valid_predictions():
    fixtures = _make_realistic_fixtures(60)

    poisson = PoissonAgent()
    poisson.train(fixtures, [])

    elo = EloAgent()
    elo.train(fixtures, [])

    test_fixture = {"home_team_id": 1, "away_team_id": 2, "league_id": 1}

    for agent in [poisson, elo]:
        pred = agent.predict(test_fixture)
        assert pred is not None
        assert abs(sum(pred.match_1x2) - 1.0) < 1e-6
        assert abs(sum(pred.over_under_25) - 1.0) < 1e-6
        assert abs(sum(pred.btts) - 1.0) < 1e-6
        assert all(0 <= p <= 1 for p in pred.match_1x2)
