import logging
from datetime import datetime, timedelta

import numpy as np

from soccer_swarm.agents.base import PredictionAgent
from soccer_swarm.backtest.metrics import compute_log_loss, compute_max_drawdown, compute_roi
from soccer_swarm.optimizer.mopso import run_mopso, select_from_pareto

logger = logging.getLogger(__name__)

MIN_FIXTURES_FOR_TRAINING = 20


class BacktestEngine:
    def __init__(
        self,
        agents: list[PredictionAgent],
        window_days: int = 30,
        mopso_pop: int = 100,
        mopso_gen: int = 300,
    ):
        self.agents = agents
        self.window_days = window_days
        self.mopso_pop = mopso_pop
        self.mopso_gen = mopso_gen

    def run(
        self,
        fixtures: list[dict],
        standings: dict[int, dict],
        odds_data: dict[int, dict] | None = None,
    ) -> dict:
        sorted_fixtures = sorted(fixtures, key=lambda f: f["date"])
        completed = [f for f in sorted_fixtures if f["status"] in ("FT", "AET", "PEN")]

        if len(completed) < MIN_FIXTURES_FOR_TRAINING + 10:
            logger.warning("Not enough fixtures for backtest: %d", len(completed))
            return {"global": {"accuracy": 0, "roi": 0, "max_drawdown": 0, "total_bets": 0}}

        n_agents = len(self.agents)
        n_fixtures = len(completed)

        # Build actuals
        result_1x2 = np.array([
            0 if f["home_goals"] > f["away_goals"]
            else (1 if f["home_goals"] == f["away_goals"] else 2)
            for f in completed
        ])
        result_ou = np.array([1 if f["home_goals"] + f["away_goals"] > 2 else 0 for f in completed])
        result_btts = np.array([1 if f["home_goals"] > 0 and f["away_goals"] > 0 else 0 for f in completed])

        # Use uniform implied odds if no odds data
        implied_odds_1x2 = np.full((n_fixtures, 3), 3.0)
        if odds_data:
            for f_idx, f in enumerate(completed):
                fid = f["id"]
                if fid in odds_data and "1x2" in odds_data[fid]:
                    o = odds_data[fid]["1x2"]
                    implied_odds_1x2[f_idx] = [o.get("home", 3.0), o.get("draw", 3.0), o.get("away", 3.0)]

        actuals = {
            "result_1x2": result_1x2,
            "result_ou": result_ou,
            "result_btts": result_btts,
            "implied_odds_1x2": implied_odds_1x2,
        }

        # Build monthly windows for walk-forward validation
        dates = [datetime.fromisoformat(f["date"]) for f in completed]

        # Group fixtures by month
        month_groups: dict[str, list[int]] = {}
        for idx, d in enumerate(dates):
            key = f"{d.year}-{d.month:02d}"
            month_groups.setdefault(key, []).append(idx)

        months = sorted(month_groups.keys())
        if len(months) < 3:
            return {"global": {"accuracy": 0, "roi": 0, "max_drawdown": 0, "total_bets": 0}}

        all_correct = 0
        all_total = 0
        all_pnl = []
        all_bets = 0
        last_pareto_X = None
        last_pareto_F = None

        # Walk-forward: optimize on months[:i], test on month[i]
        for i in range(2, len(months)):
            # Training set: all months before current
            train_indices = []
            for m in months[:i]:
                train_indices.extend(month_groups[m])
            test_indices = month_groups[months[i]]

            if len(train_indices) < MIN_FIXTURES_FOR_TRAINING or len(test_indices) == 0:
                continue

            # Retrain agents on training window only (no data leakage)
            train_fixtures = [completed[idx] for idx in train_indices]
            for agent in self.agents:
                try:
                    agent.train(train_fixtures, [])
                except TypeError:
                    agent.train(train_fixtures, [], standings=standings)

            # Compute predictions for train + test using only train-fitted agents
            window_indices = train_indices + test_indices
            agent_preds_window = np.zeros((n_agents, len(window_indices), 7))
            for a_idx, agent in enumerate(self.agents):
                for w_idx, f_idx in enumerate(window_indices):
                    f = completed[f_idx]
                    try:
                        pred = agent.predict(f)
                    except TypeError:
                        pred = agent.predict(f, history=train_fixtures, standings=standings)
                    if pred is not None:
                        agent_preds_window[a_idx, w_idx] = pred.as_array()
                    else:
                        agent_preds_window[a_idx, w_idx] = np.array([1/3, 1/3, 1/3, 0.5, 0.5, 0.5, 0.5])

            n_train = len(train_indices)
            train_preds = agent_preds_window[:, :n_train, :]
            train_actuals_w = {k: v[train_indices] for k, v in actuals.items()}
            test_preds = agent_preds_window[:, n_train:, :]
            test_actuals_w = {k: v[test_indices] for k, v in actuals.items()}

            # Run MOPSO on training window
            result = run_mopso(train_preds, train_actuals_w, pop_size=self.mopso_pop, n_gen=self.mopso_gen)

            if result.F is None or len(result.F) == 0:
                continue

            last_pareto_X = result.X
            last_pareto_F = result.F

            best_idx = select_from_pareto(result.F, result.X, profile="balanced")
            best_x = result.X[best_idx]

            weights_raw = best_x[:n_agents]
            weights = weights_raw / weights_raw.sum()
            thresh_1x2 = best_x[n_agents]

            # Evaluate on test window
            ensemble_test = np.zeros_like(test_preds[0])
            for a_i, w in enumerate(weights):
                ensemble_test += w * test_preds[a_i]

            test_result_1x2 = test_actuals_w["result_1x2"]
            implied_test = test_actuals_w["implied_odds_1x2"]
            implied_probs = 1.0 / implied_test
            implied_probs = implied_probs / implied_probs.sum(axis=1, keepdims=True)

            for j in range(len(test_result_1x2)):
                pred_1x2 = ensemble_test[j, :3]
                predicted_class = int(np.argmax(pred_1x2))
                actual_class = int(test_result_1x2[j])
                all_total += 1
                if predicted_class == actual_class:
                    all_correct += 1
                edge = pred_1x2[predicted_class] - implied_probs[j, predicted_class]
                if edge > thresh_1x2:
                    all_bets += 1
                    odds_val = implied_test[j, predicted_class]
                    if predicted_class == actual_class:
                        all_pnl.append(odds_val - 1.0)
                    else:
                        all_pnl.append(-1.0)

            logger.info("Window %s: %d fixtures, %d bets", months[i], len(test_indices), all_bets)

        accuracy = all_correct / all_total if all_total > 0 else 0
        pnl_arr = np.array(all_pnl) if all_pnl else np.array([0.0])
        roi_val = float(np.sum(pnl_arr) / all_bets) if all_bets > 0 else 0
        dd = compute_max_drawdown(pnl_arr) if len(pnl_arr) > 0 else 0

        # Save last Pareto front
        if last_pareto_X is not None:
            from soccer_swarm.optimizer.mopso import save_pareto
            import os
            os.makedirs("models", exist_ok=True)
            save_pareto("models/pareto_front.npz", last_pareto_X, last_pareto_F)

        return {
            "global": {
                "accuracy": round(accuracy, 4),
                "roi": round(roi_val, 4),
                "max_drawdown": round(float(dd), 4),
                "total_bets": all_bets,
                "total_fixtures": all_total,
            },
        }
