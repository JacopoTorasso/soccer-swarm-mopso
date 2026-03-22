import pytest
from soccer_swarm.agents.poisson import PoissonAgent
from soccer_swarm.agents.base import MarketPrediction

def _make_fixtures():
    fixtures = []
    for i in range(10):
        fixtures.append({
            "id": i, "league_id": 1,
            "home_team_id": 1, "away_team_id": 2,
            "home_goals": 2, "away_goals": 1, "status": "FT",
        })
    for i in range(10, 20):
        fixtures.append({
            "id": i, "league_id": 1,
            "home_team_id": 2, "away_team_id": 1,
            "home_goals": 1, "away_goals": 1, "status": "FT",
        })
    return fixtures

def test_poisson_train_sets_strengths():
    agent = PoissonAgent()
    agent.train(_make_fixtures(), [])
    assert agent.trained
    assert 1 in agent.attack_strengths
    assert 2 in agent.attack_strengths

def test_poisson_predict_returns_market_prediction():
    agent = PoissonAgent()
    agent.train(_make_fixtures(), [])
    fixture = {"home_team_id": 1, "away_team_id": 2, "league_id": 1}
    pred = agent.predict(fixture)
    assert isinstance(pred, MarketPrediction)
    assert abs(sum(pred.match_1x2) - 1.0) < 1e-6
    assert abs(sum(pred.over_under_25) - 1.0) < 1e-6
    assert abs(sum(pred.btts) - 1.0) < 1e-6

def test_poisson_home_team_favored_when_stronger():
    agent = PoissonAgent()
    agent.train(_make_fixtures(), [])
    fixture = {"home_team_id": 1, "away_team_id": 2, "league_id": 1}
    pred = agent.predict(fixture)
    assert pred.match_1x2[0] > pred.match_1x2[2]
