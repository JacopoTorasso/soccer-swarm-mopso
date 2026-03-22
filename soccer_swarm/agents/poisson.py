import json
from collections import defaultdict

import numpy as np
from scipy.stats import poisson

from soccer_swarm.agents.base import MarketPrediction, PredictionAgent
from soccer_swarm.config import COMPLETED_STATUSES

MAX_GOALS = 6


class PoissonAgent(PredictionAgent):
    def __init__(self):
        self.attack_strengths: dict[int, float] = {}
        self.defense_strengths: dict[int, float] = {}
        self.league_avg: dict[int, float] = {}
        self.home_advantage: dict[int, float] = {}
        self.trained = False

    def train(self, fixtures: list[dict], stats: list[dict]) -> None:
        completed = [f for f in fixtures if f["status"] in COMPLETED_STATUSES]
        if not completed:
            return

        by_league: dict[int, list[dict]] = defaultdict(list)
        for f in completed:
            by_league[f["league_id"]].append(f)

        goals_scored: dict[int, list[int]] = defaultdict(list)
        goals_conceded: dict[int, list[int]] = defaultdict(list)

        for league_id, league_fixtures in by_league.items():
            total_home = sum(f["home_goals"] for f in league_fixtures)
            total_away = sum(f["away_goals"] for f in league_fixtures)
            n = len(league_fixtures)
            avg_home = total_home / n if n > 0 else 1.3
            avg_away = total_away / n if n > 0 else 1.1
            self.league_avg[league_id] = (total_home + total_away) / (2 * n) if n > 0 else 1.2
            self.home_advantage[league_id] = avg_home / avg_away if avg_away > 0 else 1.2

            for f in league_fixtures:
                goals_scored[f["home_team_id"]].append(f["home_goals"])
                goals_conceded[f["home_team_id"]].append(f["away_goals"])
                goals_scored[f["away_team_id"]].append(f["away_goals"])
                goals_conceded[f["away_team_id"]].append(f["home_goals"])

        league_avg_all = np.mean([v for v in self.league_avg.values()]) if self.league_avg else 1.2
        for team_id in goals_scored:
            avg_scored = np.mean(goals_scored[team_id]) if goals_scored[team_id] else league_avg_all
            avg_conceded = np.mean(goals_conceded[team_id]) if goals_conceded[team_id] else league_avg_all
            self.attack_strengths[team_id] = avg_scored / league_avg_all if league_avg_all > 0 else 1.0
            self.defense_strengths[team_id] = avg_conceded / league_avg_all if league_avg_all > 0 else 1.0

        self.trained = True

    def predict(self, fixture: dict) -> MarketPrediction | None:
        if not self.trained:
            return None
        home_id = fixture["home_team_id"]
        away_id = fixture["away_team_id"]
        league_id = fixture["league_id"]

        atk_home = self.attack_strengths.get(home_id, 1.0)
        def_home = self.defense_strengths.get(home_id, 1.0)
        atk_away = self.attack_strengths.get(away_id, 1.0)
        def_away = self.defense_strengths.get(away_id, 1.0)
        avg = self.league_avg.get(league_id, 1.2)
        ha = self.home_advantage.get(league_id, 1.2)

        lambda_home = atk_home * def_away * avg * ha
        lambda_away = atk_away * def_home * avg / ha

        lambda_home = max(0.1, min(lambda_home, 6.0))
        lambda_away = max(0.1, min(lambda_away, 6.0))

        home_probs = poisson.pmf(range(MAX_GOALS + 1), lambda_home)
        away_probs = poisson.pmf(range(MAX_GOALS + 1), lambda_away)
        matrix = np.outer(home_probs, away_probs)

        # 1X2: matrix[i,j] = P(home=i, away=j)
        # Home win: i > j -> below diagonal (row index > col index)
        # Away win: i < j -> above diagonal (col index > row index)
        p_home = float(np.sum(np.tril(matrix, -1)))   # lower triangle: home goals > away goals
        p_away = float(np.sum(np.triu(matrix, 1)))    # upper triangle: away goals > home goals
        p_draw = float(np.trace(matrix))               # diagonal: home goals = away goals
        total = p_home + p_draw + p_away
        p_home, p_draw, p_away = p_home / total, p_draw / total, p_away / total

        # O/U 2.5: Under = total goals <= 2
        goal_sums = np.arange(MAX_GOALS + 1)[:, None] + np.arange(MAX_GOALS + 1)[None, :]
        p_under = float(np.sum(matrix[goal_sums <= 2]))
        p_over = 1.0 - p_under

        # BTTS: No = at least one team scores 0
        p_btts_no = float(np.sum(matrix[0, :]) + np.sum(matrix[:, 0]) - matrix[0, 0])
        p_btts_yes = 1.0 - p_btts_no

        return MarketPrediction(
            match_1x2=(p_home, p_draw, p_away),
            over_under_25=(p_over, p_under),
            btts=(p_btts_yes, p_btts_no),
        )

    def save(self, path: str) -> None:
        data = {
            "attack_strengths": {str(k): v for k, v in self.attack_strengths.items()},
            "defense_strengths": {str(k): v for k, v in self.defense_strengths.items()},
            "league_avg": {str(k): v for k, v in self.league_avg.items()},
            "home_advantage": {str(k): v for k, v in self.home_advantage.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)
        self.trained = True

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.attack_strengths = {int(k): v for k, v in data["attack_strengths"].items()}
        self.defense_strengths = {int(k): v for k, v in data["defense_strengths"].items()}
        self.league_avg = {int(k): v for k, v in data["league_avg"].items()}
        self.home_advantage = {int(k): v for k, v in data["home_advantage"].items()}
        self.trained = True
