import numpy as np
from pymoo.core.problem import Problem

from soccer_swarm.backtest.metrics import compute_log_loss, compute_max_drawdown, compute_roi


class SwarmProblem(Problem):
    def __init__(self, agent_predictions: np.ndarray, actuals: dict):
        """
        agent_predictions: shape (n_agents, n_fixtures, 7)
        actuals: dict with result_1x2, result_ou, result_btts, implied_odds_1x2
                 optional: implied_odds_ou (n,2), implied_odds_btts (n,2)
        """
        self.agent_preds = agent_predictions
        self.actuals = actuals
        self.has_ou_odds = "implied_odds_ou" in actuals
        self.has_btts_odds = "implied_odds_btts" in actuals
        n_agents = agent_predictions.shape[0]

        super().__init__(
            n_var=n_agents + 3,  # weights + 3 thresholds
            n_obj=3,
            xl=np.array([0.01] * n_agents + [0.01, 0.01, 0.01]),
            xu=np.array([1.0] * n_agents + [0.20, 0.20, 0.20]),
        )

    def _evaluate(self, X, out, *args, **kwargs):
        n_agents = self.agent_preds.shape[0]
        F = []
        n_fix = self.agent_preds.shape[1]

        for x in X:
            weights_raw = x[:n_agents]
            weights = weights_raw / weights_raw.sum()
            thresh_1x2, thresh_ou, thresh_btts = x[n_agents], x[n_agents + 1], x[n_agents + 2]

            # Ensemble predictions: weighted average
            ensemble = np.zeros_like(self.agent_preds[0])
            for i, w in enumerate(weights):
                ensemble += w * self.agent_preds[i]

            # --- f1: log-loss across ALL markets ---
            result_1x2 = self.actuals["result_1x2"]
            result_ou = self.actuals["result_ou"]
            result_btts = self.actuals["result_btts"]

            # 1X2 log-loss: probability assigned to correct outcome
            probs_1x2 = np.array([ensemble[j, r] for j, r in enumerate(result_1x2)])
            ll_1x2 = compute_log_loss(probs_1x2, np.ones(len(result_1x2)))

            # O/U log-loss: ensemble[:, 3] = P(over)
            probs_ou = ensemble[:, 3]
            ll_ou = compute_log_loss(probs_ou, result_ou.astype(float))

            # BTTS log-loss: ensemble[:, 5] = P(btts_yes)
            probs_btts = ensemble[:, 5]
            ll_btts = compute_log_loss(probs_btts, result_btts.astype(float))

            f1 = (ll_1x2 + ll_ou + ll_btts) / 3.0

            # --- f2: -ROI across all markets ---
            all_pnl = []

            # 1X2 bets
            implied = self.actuals["implied_odds_1x2"]
            implied_probs = 1.0 / implied
            implied_probs = implied_probs / implied_probs.sum(axis=1, keepdims=True)
            edges_1x2 = ensemble[:, :3] - implied_probs
            for j in range(n_fix):
                best_idx = int(np.argmax(edges_1x2[j]))
                edge = edges_1x2[j, best_idx]
                if edge > thresh_1x2:
                    won = 1.0 if best_idx == result_1x2[j] else 0.0
                    all_pnl.append(won * implied[j, best_idx] - 1.0)

            # O/U bets (only when real odds are available)
            if self.has_ou_odds:
                implied_ou = self.actuals["implied_odds_ou"]
                for j in range(n_fix):
                    p_over = ensemble[j, 3]
                    imp_over = 1.0 / implied_ou[j, 0]
                    imp_under = 1.0 / implied_ou[j, 1]
                    total_imp = imp_over + imp_under
                    fair_over = imp_over / total_imp
                    edge = p_over - fair_over
                    if abs(edge) > thresh_ou:
                        bet_over = edge > 0
                        actual_over = result_ou[j] == 1
                        odds_val = implied_ou[j, 0] if bet_over else implied_ou[j, 1]
                        if bet_over == actual_over:
                            all_pnl.append(odds_val - 1.0)
                        else:
                            all_pnl.append(-1.0)

            # BTTS bets (only when real odds are available)
            if self.has_btts_odds:
                implied_btts = self.actuals["implied_odds_btts"]
                for j in range(n_fix):
                    p_yes = ensemble[j, 5]
                    imp_yes = 1.0 / implied_btts[j, 0]
                    imp_no = 1.0 / implied_btts[j, 1]
                    total_imp = imp_yes + imp_no
                    fair_yes = imp_yes / total_imp
                    edge = p_yes - fair_yes
                    if abs(edge) > thresh_btts:
                        bet_yes = edge > 0
                        actual_yes = result_btts[j] == 1
                        odds_val = implied_btts[j, 0] if bet_yes else implied_btts[j, 1]
                        if bet_yes == actual_yes:
                            all_pnl.append(odds_val - 1.0)
                        else:
                            all_pnl.append(-1.0)

            pnl_arr = np.array(all_pnl) if all_pnl else np.array([0.0])
            total_bets = len(all_pnl)
            f2 = -(float(np.sum(pnl_arr)) / total_bets) if total_bets > 0 else 0.0

            # --- f3: max drawdown ---
            f3 = compute_max_drawdown(pnl_arr) if total_bets > 0 else 0.0

            F.append([f1, f2, f3])

        out["F"] = np.array(F)
