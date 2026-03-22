import pytest
from soccer_swarm.agents.odds_implied import OddsImpliedAgent
from soccer_swarm.agents.base import MarketPrediction


def test_odds_to_probabilities_removes_overround():
    agent = OddsImpliedAgent()
    fixture = {
        "odds": {
            "1x2": {"home": 2.10, "draw": 3.40, "away": 3.50},
            "ou25": {"over": 1.85, "under": 2.00},
            "btts": {"yes": 1.90, "no": 1.95},
        }
    }
    pred = agent.predict(fixture)
    assert isinstance(pred, MarketPrediction)
    assert abs(sum(pred.match_1x2) - 1.0) < 1e-6
    assert abs(sum(pred.over_under_25) - 1.0) < 1e-6
    assert abs(sum(pred.btts) - 1.0) < 1e-6


def test_returns_none_when_no_odds():
    agent = OddsImpliedAgent()
    pred = agent.predict({"odds": {}})
    assert pred is None


def test_favorite_has_highest_probability():
    agent = OddsImpliedAgent()
    fixture = {
        "odds": {
            "1x2": {"home": 1.50, "draw": 4.00, "away": 6.00},
            "ou25": {"over": 1.85, "under": 2.00},
            "btts": {"yes": 1.90, "no": 1.95},
        }
    }
    pred = agent.predict(fixture)
    assert pred.match_1x2[0] > pred.match_1x2[1]
    assert pred.match_1x2[0] > pred.match_1x2[2]
