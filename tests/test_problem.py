import numpy as np
import pytest
from soccer_swarm.optimizer.problem import SwarmProblem


def _make_agent_predictions():
    """4 agents, 10 fixtures, 7 values per prediction."""
    rng = np.random.default_rng(42)
    preds = rng.random((4, 10, 7))
    for a in range(4):
        for f in range(10):
            preds[a, f, 0:3] /= preds[a, f, 0:3].sum()
            preds[a, f, 3:5] /= preds[a, f, 3:5].sum()
            preds[a, f, 5:7] /= preds[a, f, 5:7].sum()
    return preds


def _make_actuals():
    """10 fixtures: 1x2 result index, ou result, btts result."""
    return {
        "result_1x2": np.array([0, 2, 1, 0, 0, 2, 1, 0, 0, 2]),
        "result_ou": np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 1]),
        "result_btts": np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1]),
        "implied_odds_1x2": np.random.default_rng(42).random((10, 3)) + 1.2,
    }


def test_problem_has_correct_dimensions():
    problem = SwarmProblem(
        agent_predictions=_make_agent_predictions(),
        actuals=_make_actuals(),
    )
    assert problem.n_var == 7
    assert problem.n_obj == 3


def test_problem_evaluate_returns_three_objectives():
    problem = SwarmProblem(
        agent_predictions=_make_agent_predictions(),
        actuals=_make_actuals(),
    )
    X = np.array([[0.25, 0.25, 0.25, 0.25, 0.05, 0.05, 0.05]])
    out = {}
    problem._evaluate(X, out)
    assert "F" in out
    assert out["F"].shape == (1, 3)
