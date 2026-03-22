import numpy as np

FORM_POINTS = {"W": 3, "D": 1, "L": 0}


def _form_to_points(form: str) -> int:
    return sum(FORM_POINTS.get(c, 0) for c in form)


def _recent_goals(team_id: int, fixtures: list[dict], n: int = 5) -> tuple[float, float]:
    """Average goals scored and conceded in last n matches."""
    relevant = sorted(
        [f for f in fixtures if f["home_team_id"] == team_id or f["away_team_id"] == team_id],
        key=lambda f: f["date"],
        reverse=True,
    )[:n]
    if not relevant:
        return 0.0, 0.0
    scored, conceded = [], []
    for f in relevant:
        if f["home_team_id"] == team_id:
            scored.append(f["home_goals"])
            conceded.append(f["away_goals"])
        else:
            scored.append(f["away_goals"])
            conceded.append(f["home_goals"])
    return float(np.mean(scored)), float(np.mean(conceded))


def build_features(
    fixture: dict,
    history: list[dict],
    standings: dict[int, dict],
) -> dict[str, float]:
    home_id = fixture["home_team_id"]
    away_id = fixture["away_team_id"]

    home_st = standings.get(home_id, {})
    away_st = standings.get(away_id, {})

    home_scored_avg, home_conceded_avg = _recent_goals(home_id, history)
    away_scored_avg, away_conceded_avg = _recent_goals(away_id, history)

    return {
        "home_rank": home_st.get("rank", 10),
        "away_rank": away_st.get("rank", 10),
        "home_points": home_st.get("points", 0),
        "away_points": away_st.get("points", 0),
        "home_form_points": _form_to_points(home_st.get("form", "")),
        "away_form_points": _form_to_points(away_st.get("form", "")),
        "home_goals_scored_avg": home_scored_avg,
        "home_goals_conceded_avg": home_conceded_avg,
        "away_goals_scored_avg": away_scored_avg,
        "away_goals_conceded_avg": away_conceded_avg,
        "home_gd": home_st.get("goals_for", 0) - home_st.get("goals_against", 0),
        "away_gd": away_st.get("goals_for", 0) - away_st.get("goals_against", 0),
        "rank_diff": home_st.get("rank", 10) - away_st.get("rank", 10),
    }
