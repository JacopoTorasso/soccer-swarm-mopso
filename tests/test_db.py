import sqlite3
import pytest
from soccer_swarm.data.db import create_tables, get_connection

@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")

def test_create_tables_creates_all_expected_tables(db_path):
    conn = get_connection(db_path)
    create_tables(conn)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row[0] for row in cursor.fetchall()]
    assert "leagues" in tables
    assert "teams" in tables
    assert "fixtures" in tables
    assert "fixture_stats" in tables
    assert "standings" in tables
    assert "odds" in tables
    assert "api_log" in tables
    assert "elo_ratings" in tables
    assert "poisson_params" in tables
    conn.close()

def test_create_tables_is_idempotent(db_path):
    conn = get_connection(db_path)
    create_tables(conn)
    create_tables(conn)
    conn.close()

def test_fixtures_api_id_is_unique(db_path):
    conn = get_connection(db_path)
    create_tables(conn)
    conn.execute("INSERT INTO leagues VALUES (1, 'SerieA', 'Italy', 2025, 135)")
    conn.execute("INSERT INTO teams VALUES (1, 'Roma', 1, 497)")
    conn.execute("INSERT INTO teams VALUES (2, 'Lazio', 1, 487)")
    conn.execute(
        "INSERT INTO fixtures (id, league_id, home_team_id, away_team_id, date, status, home_goals, away_goals, api_id) "
        "VALUES (1, 1, 1, 2, '2026-01-15T15:00:00+00:00', 'FT', 2, 1, 99001)"
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO fixtures (id, league_id, home_team_id, away_team_id, date, status, home_goals, away_goals, api_id) "
            "VALUES (2, 1, 2, 1, '2026-03-15T15:00:00+00:00', 'NS', NULL, NULL, 99001)"
        )
    conn.close()
