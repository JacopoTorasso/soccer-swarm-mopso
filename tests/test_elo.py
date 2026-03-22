import pytest
from soccer_swarm.agents.elo import EloAgent
from soccer_swarm.agents.base import MarketPrediction

def _make_fixtures():
    fixtures = []
    for i in range(10):
        fixtures.append({
            "id": i, "league_id": 1,
            "home_team_id": 1, "away_team_id": 2,
            "home_goals": 2, "away_goals": 0,
            "status": "FT", "date": f"2025-09-{i+1:02d}T15:00:00+00:00",
        })
    return fixtures

def test_elo_train_updates_ratings():
    agent = EloAgent()
    agent.train(_make_fixtures(), [])
    assert agent.ratings[1] > 1500
    assert agent.ratings[2] < 1500

def test_elo_predict_favors_stronger_team():
    agent = EloAgent()
    agent.train(_make_fixtures(), [])
    pred = agent.predict({"home_team_id": 1, "away_team_id": 2, "league_id": 1})
    assert isinstance(pred, MarketPrediction)
    assert pred.match_1x2[0] > pred.match_1x2[2]

def test_elo_ratings_are_symmetric():
    agent = EloAgent()
    agent.train(_make_fixtures(), [])
    assert abs(agent.ratings[1] + agent.ratings[2] - 3000) < 1e-6

def test_elo_probabilities_sum_to_one():
    agent = EloAgent()
    agent.train(_make_fixtures(), [])
    pred = agent.predict({"home_team_id": 1, "away_team_id": 2, "league_id": 1})
    assert abs(sum(pred.match_1x2) - 1.0) < 1e-6
    assert abs(sum(pred.over_under_25) - 1.0) < 1e-6
    assert abs(sum(pred.btts) - 1.0) < 1e-6
