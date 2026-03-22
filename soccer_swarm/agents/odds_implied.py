from soccer_swarm.agents.base import MarketPrediction, PredictionAgent


class OddsImpliedAgent(PredictionAgent):

    def train(self, fixtures: list[dict], stats: list[dict]) -> None:
        pass  # No training needed

    def predict(self, fixture: dict) -> MarketPrediction | None:
        odds = fixture.get("odds", {})
        if not odds or "1x2" not in odds:
            return None

        o1x2 = odds["1x2"]
        if not all(o1x2.get(k) for k in ("home", "draw", "away")):
            return None

        # 1X2: remove overround
        imp_h = 1 / o1x2["home"]
        imp_d = 1 / o1x2["draw"]
        imp_a = 1 / o1x2["away"]
        total = imp_h + imp_d + imp_a
        p_home, p_draw, p_away = imp_h / total, imp_d / total, imp_a / total

        # O/U 2.5
        ou = odds.get("ou25", {})
        if ou.get("over") and ou.get("under"):
            imp_over = 1 / ou["over"]
            imp_under = 1 / ou["under"]
            total_ou = imp_over + imp_under
            p_over, p_under = imp_over / total_ou, imp_under / total_ou
        else:
            p_over, p_under = 0.5, 0.5

        # BTTS
        btts = odds.get("btts", {})
        if btts.get("yes") and btts.get("no"):
            imp_yes = 1 / btts["yes"]
            imp_no = 1 / btts["no"]
            total_btts = imp_yes + imp_no
            p_yes, p_no = imp_yes / total_btts, imp_no / total_btts
        else:
            p_yes, p_no = 0.5, 0.5

        return MarketPrediction(
            match_1x2=(p_home, p_draw, p_away),
            over_under_25=(p_over, p_under),
            btts=(p_yes, p_no),
        )

    def save(self, path: str) -> None:
        pass  # Nothing to persist

    def load(self, path: str) -> None:
        pass
