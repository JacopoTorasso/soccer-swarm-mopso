import pytest
from soccer_swarm.data.features import build_features

def _make_fixture_history():
    """Team 1 vs Team 2, 10 home matches for team 1."""
    fixtures = []
    for i in range(10):
        fixtures.append({
            "id": i, "league_id": 1,
            "home_team_id": 1, "away_team_id": 2,
            "home_goals": 2, "away_goals": 1,
            "status": "FT",
            "date": f"2025-09-{i+1:02d}T15:00:00+00:00",
        })
    return fixtures

def _make_standings():
    return {
        1: {"rank": 3, "points": 25, "played": 12, "won": 7, "drawn": 4, "lost": 1,
            "goals_for": 20, "goals_against": 10, "form": "WWDWL"},
        2: {"rank": 8, "points": 15, "played": 12, "won": 4, "drawn": 3, "lost": 5,
            "goals_for": 14, "goals_against": 18, "form": "LWDWL"},
    }

def test_build_features_returns_correct_keys():
    fixture = {"home_team_id": 1, "away_team_id": 2, "league_id": 1,
               "date": "2025-12-01T15:00:00+00:00"}
    features = build_features(fixture, _make_fixture_history(), _make_standings())
    assert "home_rank" in features
    assert "away_rank" in features
    assert "home_form_points" in features
    assert "home_goals_scored_avg" in features

def test_build_features_form_points_calculation():
    fixture = {"home_team_id": 1, "away_team_id": 2, "league_id": 1,
               "date": "2025-12-01T15:00:00+00:00"}
    features = build_features(fixture, _make_fixture_history(), _make_standings())
    # "WWDWL" = 3+3+1+3+0 = 10
    assert features["home_form_points"] == 10
