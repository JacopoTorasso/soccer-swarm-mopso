import pytest
from soccer_swarm.agents.xgboost_agent import XGBoostAgent
from soccer_swarm.agents.base import MarketPrediction

def _make_training_data():
    """Generate 50 fixtures with standings for training."""
    fixtures = []
    for i in range(50):
        home_goals = 2 if i % 3 != 2 else 0
        away_goals = 1 if i % 3 != 0 else 2
        fixtures.append({
            "id": i, "league_id": 1,
            "home_team_id": 1, "away_team_id": 2,
            "home_goals": home_goals, "away_goals": away_goals,
            "status": "FT",
            "date": f"2025-09-{(i % 28) + 1:02d}T15:00:00+00:00",
        })
    standings = {
        1: {"rank": 3, "points": 25, "played": 12, "won": 7, "drawn": 4, "lost": 1,
            "goals_for": 20, "goals_against": 10, "form": "WWDWL"},
        2: {"rank": 8, "points": 15, "played": 12, "won": 4, "drawn": 3, "lost": 5,
            "goals_for": 14, "goals_against": 18, "form": "LWDWL"},
    }
    return fixtures, standings

def test_xgboost_train_and_predict():
    agent = XGBoostAgent()
    fixtures, standings = _make_training_data()
    agent.train(fixtures, [], standings=standings)
    pred = agent.predict({
        "home_team_id": 1, "away_team_id": 2, "league_id": 1,
        "date": "2025-12-01T15:00:00+00:00",
    }, history=fixtures, standings=standings)
    assert isinstance(pred, MarketPrediction)
    assert abs(sum(pred.match_1x2) - 1.0) < 1e-6

def test_xgboost_save_and_load(tmp_path):
    agent = XGBoostAgent()
    fixtures, standings = _make_training_data()
    agent.train(fixtures, [], standings=standings)

    save_dir = str(tmp_path)
    agent.save(save_dir)

    agent2 = XGBoostAgent()
    agent2.load(save_dir)
    pred = agent2.predict({
        "home_team_id": 1, "away_team_id": 2, "league_id": 1,
        "date": "2025-12-01T15:00:00+00:00",
    }, history=fixtures, standings=standings)
    assert isinstance(pred, MarketPrediction)
