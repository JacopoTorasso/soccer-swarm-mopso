import sqlite3

def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn

def create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS leagues (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            country TEXT NOT NULL,
            season INTEGER NOT NULL,
            api_id INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS teams (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            league_id INTEGER REFERENCES leagues(id),
            api_id INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS fixtures (
            id INTEGER PRIMARY KEY,
            league_id INTEGER REFERENCES leagues(id),
            home_team_id INTEGER REFERENCES teams(id),
            away_team_id INTEGER REFERENCES teams(id),
            date TEXT NOT NULL,
            status TEXT NOT NULL,
            home_goals INTEGER,
            away_goals INTEGER,
            api_id INTEGER UNIQUE NOT NULL
        );
        CREATE TABLE IF NOT EXISTS fixture_stats (
            fixture_id INTEGER REFERENCES fixtures(id),
            team_id INTEGER REFERENCES teams(id),
            shots INTEGER,
            shots_on_target INTEGER,
            possession REAL,
            corners INTEGER,
            fouls INTEGER,
            yellow_cards INTEGER,
            red_cards INTEGER,
            PRIMARY KEY (fixture_id, team_id)
        );
        CREATE TABLE IF NOT EXISTS standings (
            league_id INTEGER REFERENCES leagues(id),
            team_id INTEGER REFERENCES teams(id),
            season INTEGER NOT NULL,
            rank INTEGER,
            points INTEGER,
            played INTEGER,
            won INTEGER,
            drawn INTEGER,
            lost INTEGER,
            goals_for INTEGER,
            goals_against INTEGER,
            goal_diff INTEGER,
            form TEXT,
            PRIMARY KEY (league_id, team_id, season)
        );
        CREATE TABLE IF NOT EXISTS odds (
            fixture_id INTEGER REFERENCES fixtures(id),
            market TEXT NOT NULL,
            bookmaker TEXT NOT NULL,
            value_home REAL,
            value_draw REAL,
            value_away REAL,
            PRIMARY KEY (fixture_id, market, bookmaker)
        );
        CREATE TABLE IF NOT EXISTS api_log (
            endpoint TEXT NOT NULL,
            params_hash TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            response_json TEXT NOT NULL,
            PRIMARY KEY (endpoint, params_hash)
        );
        CREATE TABLE IF NOT EXISTS elo_ratings (
            team_id INTEGER PRIMARY KEY REFERENCES teams(id),
            rating REAL NOT NULL DEFAULT 1500.0,
            last_updated TEXT
        );
        CREATE TABLE IF NOT EXISTS poisson_params (
            team_id INTEGER NOT NULL,
            season INTEGER NOT NULL,
            league_id INTEGER NOT NULL,
            attack_strength REAL NOT NULL,
            defense_strength REAL NOT NULL,
            PRIMARY KEY (team_id, season, league_id)
        );
    """)
    conn.commit()
