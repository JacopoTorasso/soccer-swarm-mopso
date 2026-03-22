import numpy as np


def compute_log_loss(predicted_probs: np.ndarray, actuals: np.ndarray, eps: float = 1e-15) -> float:
    clipped = np.clip(predicted_probs, eps, 1 - eps)
    losses = -(actuals * np.log(clipped) + (1 - actuals) * np.log(1 - clipped))
    return float(np.mean(losses))


def compute_roi(stakes: np.ndarray, odds: np.ndarray, wins: np.ndarray) -> float:
    total_staked = np.sum(stakes)
    if total_staked == 0:
        return 0.0
    total_return = np.sum(stakes * odds * wins)
    return float((total_return - total_staked) / total_staked)


def compute_max_drawdown(pnl: np.ndarray) -> float:
    if len(pnl) == 0:
        return 0.0
    cumulative = np.cumsum(pnl)
    peak = np.maximum.accumulate(cumulative)
    drawdowns = peak - cumulative
    return float(np.max(drawdowns))
