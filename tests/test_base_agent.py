import pytest
from soccer_swarm.agents.base import MarketPrediction

def test_market_prediction_probabilities_are_accessible():
    pred = MarketPrediction(
        match_1x2=(0.45, 0.25, 0.30),
        over_under_25=(0.55, 0.45),
        btts=(0.48, 0.52),
    )
    assert pred.match_1x2 == (0.45, 0.25, 0.30)
    assert pred.over_under_25 == (0.55, 0.45)
    assert pred.btts == (0.48, 0.52)

def test_market_prediction_1x2_sums_to_one():
    pred = MarketPrediction(
        match_1x2=(0.45, 0.25, 0.30),
        over_under_25=(0.55, 0.45),
        btts=(0.48, 0.52),
    )
    assert abs(sum(pred.match_1x2) - 1.0) < 1e-9

def test_market_prediction_as_array():
    pred = MarketPrediction(
        match_1x2=(0.45, 0.25, 0.30),
        over_under_25=(0.55, 0.45),
        btts=(0.48, 0.52),
    )
    arr = pred.as_array()
    assert len(arr) == 7
    assert arr[0] == 0.45
