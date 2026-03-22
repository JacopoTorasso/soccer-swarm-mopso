import logging

import numpy as np
from pymoo.algorithms.moo.mopso_cd import MOPSO_CD
from pymoo.optimize import minimize

from soccer_swarm.optimizer.problem import SwarmProblem

logger = logging.getLogger(__name__)

PROFILES = {
    "conservative": np.array([0.1, 0.3, 0.6]),
    "balanced": np.array([0.33, 0.34, 0.33]),
    "aggressive": np.array([0.2, 0.7, 0.1]),
}


def run_mopso(
    agent_predictions: np.ndarray,
    actuals: dict,
    pop_size: int = 100,
    n_gen: int = 300,
    seed: int = 42,
):
    problem = SwarmProblem(agent_predictions, actuals)
    algorithm = MOPSO_CD(
        pop_size=pop_size,
        w=0.6,
        c1=1.5,
        c2=1.5,
        max_velocity_rate=0.5,
        archive_size=200,
    )
    logger.info("Running MOPSO: pop=%d gen=%d", pop_size, n_gen)
    result = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=False)
    logger.info("MOPSO complete: %d Pareto solutions", len(result.F) if result.F is not None else 0)
    return result


def select_from_pareto(
    F: np.ndarray,
    X: np.ndarray,
    profile: str = "balanced",
    weights: np.ndarray | None = None,
) -> int:
    if weights is None:
        weights = PROFILES.get(profile, PROFILES["balanced"])

    F_min = F.min(axis=0)
    F_max = F.max(axis=0)
    F_range = F_max - F_min
    F_range[F_range == 0] = 1.0
    F_norm = (F - F_min) / F_range

    scores = F_norm @ weights
    return int(np.argmin(scores))


def save_pareto(path: str, X: np.ndarray, F: np.ndarray) -> None:
    np.savez(path, X=X, F=F)
    logger.info("Pareto front saved to %s (%d solutions)", path, len(F))


def load_pareto(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["X"], data["F"]
