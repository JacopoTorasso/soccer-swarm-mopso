import json
from collections import defaultdict

import numpy as np
from scipy.stats import poisson

from soccer_swarm.agents.base import MarketPrediction, PredictionAgent
from soccer_swarm.config import (
    COMPLETED_STATUSES,
    ELO_DEFAULT_RATING,
    ELO_HOME_ADVANTAGE,
    ELO_K_FACTOR,
    ELO_K_FACTOR_HIGH,
)

MAX_GOALS = 6
DEFAULT_BASE_RATE = 1.35
DEFAULT_BETA = 0.15


class EloAgent(PredictionAgent):
    def __init__(self):
        self.ratings: dict[int, float] = {}
        self.base_rate: dict[int, float] = {}
        self.beta: dict[int, float] = {}
        self.trained = False

    def train(self, fixtures: list[dict], stats: list[dict]) -> None:
        completed = sorted(
            [f for f in fixtures if f["status"] in COMPLETED_STATUSES],
            key=lambda f: f["date"],
        )
        elo_diffs: dict[int, list[float]] = defaultdict(list)
        goals_home: dict[int, list[int]] = defaultdict(list)
        goals_away: dict[int, list[int]] = defaultdict(list)

        for f in completed:
            home_id = f["home_team_id"]
            away_id = f["away_team_id"]
            league_id = f["league_id"]

            r_home = self.ratings.get(home_id, ELO_DEFAULT_RATING)
            r_away = self.ratings.get(away_id, ELO_DEFAULT_RATING)

            diff = r_home + ELO_HOME_ADVANTAGE - r_away
            elo_diffs[league_id].append(diff)
            goals_home[league_id].append(f["home_goals"])
            goals_away[league_id].append(f["away_goals"])

            e_home = 1 / (1 + 10 ** (-diff / 400))
            if f["home_goals"] > f["away_goals"]:
                s_home = 1.0
            elif f["home_goals"] == f["away_goals"]:
                s_home = 0.5
            else:
                s_home = 0.0

            k = ELO_K_FACTOR
            self.ratings[home_id] = r_home + k * (s_home - e_home)
            self.ratings[away_id] = r_away + k * ((1 - s_home) - (1 - e_home))

        for league_id in elo_diffs:
            diffs = np.array(elo_diffs[league_id])
            gh = np.array(goals_home[league_id], dtype=float)
            if len(diffs) > 5:
                x = diffs / 400
                self.base_rate[league_id] = float(np.mean(gh))
                if np.std(x) > 0:
                    self.beta[league_id] = float(np.corrcoef(x, gh)[0, 1] * np.std(gh) / np.std(x))
                else:
                    self.beta[league_id] = DEFAULT_BETA
            else:
                self.base_rate[league_id] = DEFAULT_BASE_RATE
                self.beta[league_id] = DEFAULT_BETA

        self.trained = True

    def predict(self, fixture: dict) -> MarketPrediction | None:
        if not self.trained:
            return None
        home_id = fixture["home_team_id"]
        away_id = fixture["away_team_id"]
        league_id = fixture["league_id"]

        r_home = self.ratings.get(home_id, ELO_DEFAULT_RATING)
        r_away = self.ratings.get(away_id, ELO_DEFAULT_RATING)
        diff = r_home + ELO_HOME_ADVANTAGE - r_away

        p_home_win_raw = 1 / (1 + 10 ** (-diff / 400))
        p_draw = max(0.15, 0.28 - 0.001 * abs(diff))
        remaining = 1.0 - p_draw
        p_home = remaining * p_home_win_raw
        p_away = remaining * (1 - p_home_win_raw)

        br = self.base_rate.get(league_id, DEFAULT_BASE_RATE)
        bt = self.beta.get(league_id, DEFAULT_BETA)
        lambda_home = max(0.3, br + bt * diff / 400)
        lambda_away = max(0.3, br - bt * diff / 400)

        home_probs = poisson.pmf(range(MAX_GOALS + 1), lambda_home)
        away_probs = poisson.pmf(range(MAX_GOALS + 1), lambda_away)
        matrix = np.outer(home_probs, away_probs)

        goal_sums = np.arange(MAX_GOALS + 1)[:, None] + np.arange(MAX_GOALS + 1)[None, :]
        p_under = float(np.sum(matrix[goal_sums <= 2]))
        p_over = 1.0 - p_under

        p_btts_no = float(np.sum(matrix[0, :]) + np.sum(matrix[:, 0]) - matrix[0, 0])
        p_btts_yes = 1.0 - p_btts_no

        return MarketPrediction(
            match_1x2=(p_home, p_draw, p_away),
            over_under_25=(p_over, p_under),
            btts=(p_btts_yes, p_btts_no),
        )

    def save(self, path: str) -> None:
        data = {
            "ratings": {str(k): v for k, v in self.ratings.items()},
            "base_rate": {str(k): v for k, v in self.base_rate.items()},
            "beta": {str(k): v for k, v in self.beta.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.ratings = {int(k): v for k, v in data["ratings"].items()}
        self.base_rate = {int(k): v for k, v in data["base_rate"].items()}
        self.beta = {int(k): v for k, v in data["beta"].items()}
        self.trained = True
