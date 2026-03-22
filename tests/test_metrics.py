import numpy as np
import pytest
from soccer_swarm.backtest.metrics import compute_log_loss, compute_roi, compute_max_drawdown

def test_log_loss_perfect_prediction():
    probs = np.array([0.99, 0.99, 0.99])
    actuals = np.array([1, 1, 1])
    ll = compute_log_loss(probs, actuals)
    assert ll < 0.05

def test_log_loss_bad_prediction():
    probs = np.array([0.01, 0.01, 0.01])
    actuals = np.array([1, 1, 1])
    ll = compute_log_loss(probs, actuals)
    assert ll > 3.0

def test_roi_all_wins():
    stakes = np.array([1.0, 1.0, 1.0])
    odds = np.array([2.0, 2.0, 2.0])
    wins = np.array([1, 1, 1])
    roi = compute_roi(stakes, odds, wins)
    assert abs(roi - 1.0) < 1e-6

def test_roi_all_losses():
    stakes = np.array([1.0, 1.0, 1.0])
    odds = np.array([2.0, 2.0, 2.0])
    wins = np.array([0, 0, 0])
    roi = compute_roi(stakes, odds, wins)
    assert abs(roi - (-1.0)) < 1e-6

def test_max_drawdown():
    pnl = np.array([1.0, 1.0, -1.0, -1.0, -1.0, 1.0])
    dd = compute_max_drawdown(pnl)
    assert abs(dd - 3.0) < 1e-6

def test_max_drawdown_no_loss():
    pnl = np.array([1.0, 1.0, 1.0])
    dd = compute_max_drawdown(pnl)
    assert dd == 0.0
