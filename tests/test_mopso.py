import numpy as np
import pytest
from soccer_swarm.optimizer.mopso import run_mopso, select_from_pareto


def _make_test_data():
    rng = np.random.default_rng(42)
    preds = rng.random((4, 20, 7))
    for a in range(4):
        for f in range(20):
            preds[a, f, 0:3] /= preds[a, f, 0:3].sum()
            preds[a, f, 3:5] /= preds[a, f, 3:5].sum()
            preds[a, f, 5:7] /= preds[a, f, 5:7].sum()
    actuals = {
        "result_1x2": rng.integers(0, 3, size=20),
        "result_ou": rng.integers(0, 2, size=20),
        "result_btts": rng.integers(0, 2, size=20),
        "implied_odds_1x2": rng.random((20, 3)) + 1.5,
    }
    return preds, actuals


def test_run_mopso_returns_pareto_front():
    preds, actuals = _make_test_data()
    result = run_mopso(preds, actuals, pop_size=20, n_gen=10)
    assert result.F is not None
    assert result.X is not None
    assert result.F.shape[1] == 3


def test_select_from_pareto_balanced():
    F = np.array([[0.5, 0.5, 0.5], [0.3, 0.8, 0.2], [0.8, 0.2, 0.5]])
    X = np.array([[0.25, 0.25, 0.25, 0.25, 0.05, 0.05, 0.05]] * 3)
    idx = select_from_pareto(F, X, profile="balanced")
    assert 0 <= idx < 3
