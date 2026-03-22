import os

import numpy as np
import xgboost as xgb

from soccer_swarm.agents.base import MarketPrediction, PredictionAgent
from soccer_swarm.config import COMPLETED_STATUSES
from soccer_swarm.data.features import build_features


class XGBoostAgent(PredictionAgent):
    def __init__(self):
        self.model_1x2: xgb.XGBClassifier | None = None
        self.model_ou: xgb.XGBClassifier | None = None
        self.model_btts: xgb.XGBClassifier | None = None
        self.trained = False

    def train(
        self,
        fixtures: list[dict],
        stats: list[dict],
        *,
        standings: dict[int, dict] | None = None,
    ) -> None:
        if standings is None:
            standings = {}

        completed = [f for f in fixtures if f["status"] in COMPLETED_STATUSES]
        if len(completed) < 20:
            return

        X, y_1x2, y_ou, y_btts = [], [], [], []

        for f in completed:
            features = build_features(f, completed, standings)
            X.append(list(features.values()))

            # 1X2 label
            if f["home_goals"] > f["away_goals"]:
                y_1x2.append(0)  # home
            elif f["home_goals"] == f["away_goals"]:
                y_1x2.append(1)  # draw
            else:
                y_1x2.append(2)  # away

            # O/U 2.5
            y_ou.append(1 if f["home_goals"] + f["away_goals"] > 2 else 0)

            # BTTS
            y_btts.append(1 if f["home_goals"] > 0 and f["away_goals"] > 0 else 0)

        X = np.array(X)
        y_1x2 = np.array(y_1x2)
        y_ou = np.array(y_ou)
        y_btts = np.array(y_btts)

        params = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "eval_metric": "logloss",
            "verbosity": 0,
        }

        self.model_1x2 = xgb.XGBClassifier(objective="multi:softprob", num_class=3, **params)
        self.model_1x2.fit(X, y_1x2)

        self.model_ou = xgb.XGBClassifier(objective="binary:logistic", **params)
        self.model_ou.fit(X, y_ou)

        self.model_btts = xgb.XGBClassifier(objective="binary:logistic", **params)
        self.model_btts.fit(X, y_btts)

        self.trained = True

    def predict(
        self,
        fixture: dict,
        *,
        history: list[dict] | None = None,
        standings: dict[int, dict] | None = None,
    ) -> MarketPrediction | None:
        if not self.trained or history is None:
            return None
        if standings is None:
            standings = {}

        features = build_features(fixture, history, standings)
        X = np.array([list(features.values())])

        proba_1x2 = self.model_1x2.predict_proba(X)[0]
        proba_ou = self.model_ou.predict_proba(X)[0]
        proba_btts = self.model_btts.predict_proba(X)[0]

        return MarketPrediction(
            match_1x2=(float(proba_1x2[0]), float(proba_1x2[1]), float(proba_1x2[2])),
            over_under_25=(float(proba_ou[1]), float(proba_ou[0])),  # [1]=over, [0]=under
            btts=(float(proba_btts[1]), float(proba_btts[0])),       # [1]=yes, [0]=no
        )

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        if self.model_1x2:
            self.model_1x2.save_model(os.path.join(path, "xgboost_1x2.json"))
        if self.model_ou:
            self.model_ou.save_model(os.path.join(path, "xgboost_ou.json"))
        if self.model_btts:
            self.model_btts.save_model(os.path.join(path, "xgboost_btts.json"))

    def load(self, path: str) -> None:
        self.model_1x2 = xgb.XGBClassifier()
        self.model_1x2.load_model(os.path.join(path, "xgboost_1x2.json"))
        self.model_ou = xgb.XGBClassifier()
        self.model_ou.load_model(os.path.join(path, "xgboost_ou.json"))
        self.model_btts = xgb.XGBClassifier()
        self.model_btts.load_model(os.path.join(path, "xgboost_btts.json"))
        self.trained = True
