"""Import match data from football-data.co.uk CSV files."""
import csv
import hashlib
import io
import logging
import sqlite3
from datetime import datetime

from datetime import timedelta
from itertools import product

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://www.football-data.co.uk"

# football-data.co.uk division codes → our league IDs
DIV_TO_LEAGUE = {
    "I1": 135,   # Serie A
    "E0": 39,    # Premier League
    "SP1": 140,  # La Liga
    "D1": 78,    # Bundesliga
    "F1": 61,    # Ligue 1
}

LEAGUE_TO_DIV = {v: k for k, v in DIV_TO_LEAGUE.items()}

LEAGUE_NAMES = {
    135: "SerieA",
    39: "PremierLeague",
    140: "LaLiga",
    78: "Bundesliga",
    61: "Ligue1",
}


def _team_id(team_name: str, league_id: int) -> int:
    """Generate a stable numeric ID for a CSV-sourced team."""
    key = f"csv:{league_id}:{team_name}"
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) + 500000


def _fixture_id(home: str, away: str, date: str, league_id: int) -> int:
    """Generate a stable numeric fixture ID for CSV-sourced matches."""
    key = f"csv:{league_id}:{date}:{home}:{away}"
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) + 5000000


def _parse_date(date_str: str) -> str:
    """Parse DD/MM/YYYY to ISO format."""
    try:
        dt = datetime.strptime(date_str, "%d/%m/%Y")
        return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    except ValueError:
        return date_str


def _download_csv(url: str) -> list[dict]:
    """Download and parse a CSV from football-data.co.uk."""
    logger.info("Downloading %s", url)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    # Decode with utf-8-sig to handle BOM
    text = resp.content.decode("utf-8-sig").strip()
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for row in reader:
        # Skip empty rows
        if not row.get("Div") and not row.get("HomeTeam"):
            continue
        rows.append(row)
    return rows


def import_season(conn: sqlite3.Connection, season_code: str, leagues: list[int] | None = None):
    """
    Import a full season from football-data.co.uk.
    season_code: e.g. '2526' for 2025/26
    """
    target_leagues = leagues or list(DIV_TO_LEAGUE.values())
    total = 0

    for league_id in target_leagues:
        div = LEAGUE_TO_DIV.get(league_id)
        if not div:
            continue
        url = f"{BASE_URL}/mmz4281/{season_code}/{div}.csv"
        try:
            rows = _download_csv(url)
        except Exception as e:
            logger.warning("Failed to download %s: %s", url, e)
            continue

        count = _import_rows(conn, rows, league_id)
        league_name = LEAGUE_NAMES.get(league_id, str(league_id))
        logger.info("Imported %d fixtures for %s season %s", count, league_name, season_code)
        print(f"  {league_name}: {count} fixtures ({len(rows)} in CSV)")
        total += count

    return total


def import_upcoming(conn: sqlite3.Connection, leagues: list[int] | None = None):
    """Import upcoming fixtures from fixtures.csv."""
    url = f"{BASE_URL}/fixtures.csv"
    try:
        rows = _download_csv(url)
    except Exception as e:
        logger.error("Failed to download fixtures: %s", e)
        return 0

    target_leagues = set(leagues or DIV_TO_LEAGUE.values())
    filtered = [r for r in rows if DIV_TO_LEAGUE.get(r.get("Div", "")) in target_leagues]

    total = 0
    by_league: dict[int, int] = {}
    for row in filtered:
        league_id = DIV_TO_LEAGUE[row["Div"]]
        count = _import_rows(conn, [row], league_id, upcoming=True)
        by_league[league_id] = by_league.get(league_id, 0) + count
        total += count

    for lid, cnt in sorted(by_league.items()):
        print(f"  {LEAGUE_NAMES.get(lid, str(lid))}: {cnt} upcoming fixtures")

    return total


def _import_rows(conn: sqlite3.Connection, rows: list[dict], league_id: int, upcoming: bool = False) -> int:
    """Import CSV rows into the database."""
    count = 0
    season = 2025  # 2025/26 season

    # Ensure league exists
    conn.execute(
        "INSERT OR IGNORE INTO leagues VALUES (?, ?, ?, ?, ?)",
        (league_id, LEAGUE_NAMES.get(league_id, ""), "", season, league_id),
    )

    for row in rows:
        home_name = row.get("HomeTeam", "").strip()
        away_name = row.get("AwayTeam", "").strip()
        date_str = row.get("Date", "").strip()

        if not home_name or not away_name or not date_str:
            continue

        home_id = _team_id(home_name, league_id)
        away_id = _team_id(away_name, league_id)
        fix_id = _fixture_id(home_name, away_name, date_str, league_id)
        date_iso = _parse_date(date_str)

        # Determine status and goals
        fthg = row.get("FTHG", "").strip()
        ftag = row.get("FTAG", "").strip()

        if fthg and ftag and not upcoming:
            status = "FT"
            home_goals = int(fthg)
            away_goals = int(ftag)
        else:
            status = "NS"
            home_goals = None
            away_goals = None

        # Insert team records
        conn.execute(
            "INSERT OR IGNORE INTO teams VALUES (?, ?, ?, ?)",
            (home_id, home_name, league_id, home_id),
        )
        conn.execute(
            "INSERT OR IGNORE INTO teams VALUES (?, ?, ?, ?)",
            (away_id, away_name, league_id, away_id),
        )

        # Insert or update fixture
        existing = conn.execute(
            "SELECT id, status FROM fixtures WHERE api_id = ?", (fix_id,)
        ).fetchone()

        if existing:
            if status == "FT" and existing["status"] == "NS":
                # Update NS → FT with scores
                conn.execute(
                    "UPDATE fixtures SET status = ?, home_goals = ?, away_goals = ? WHERE id = ?",
                    (status, home_goals, away_goals, existing["id"]),
                )
                count += 1
            continue

        conn.execute(
            "INSERT OR IGNORE INTO fixtures (league_id, home_team_id, away_team_id, date, status, home_goals, away_goals, api_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (league_id, home_id, away_id, date_iso, status, home_goals, away_goals, fix_id),
        )
        count += 1

        # Insert odds if available (Bet365 as primary)
        b365h = row.get("B365H", "").strip()
        b365d = row.get("B365D", "").strip()
        b365a = row.get("B365A", "").strip()

        if b365h and b365d and b365a:
            db_fix_id = conn.execute(
                "SELECT id FROM fixtures WHERE api_id = ?", (fix_id,)
            ).fetchone()
            if db_fix_id:
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO odds VALUES (?, ?, ?, ?, ?, ?)",
                        (db_fix_id["id"], "1x2", "Bet365",
                         float(b365h), float(b365d), float(b365a)),
                    )
                except (ValueError, KeyError):
                    pass

                # O/U 2.5 odds
                ou_over = row.get("B365>2.5", "").strip()
                ou_under = row.get("B365<2.5", "").strip()
                if ou_over and ou_under:
                    try:
                        conn.execute(
                            "INSERT OR IGNORE INTO odds VALUES (?, ?, ?, ?, ?, ?)",
                            (db_fix_id["id"], "ou25", "Bet365",
                             float(ou_over), None, float(ou_under)),
                        )
                    except (ValueError, KeyError):
                        pass

    conn.commit()
    return count


def generate_remaining_fixtures(conn: sqlite3.Connection, leagues: list[int] | None = None):
    """
    Generate NS fixtures for remaining matches in the 2025/26 season.
    Each team plays every other team once at home and once away.
    We find which home/away pairs haven't been played yet and create them.
    """
    target_leagues = leagues or list(DIV_TO_LEAGUE.values())
    total = 0

    for league_id in target_leagues:
        # Get all CSV-sourced teams for this league
        rows = conn.execute(
            "SELECT DISTINCT id, name FROM teams WHERE league_id = ? AND id >= 500000",
            (league_id,),
        ).fetchall()
        teams = [(r["id"], r["name"]) for r in rows]

        if not teams:
            continue

        # Get all existing home/away pairs for this league (CSV-sourced)
        existing = conn.execute(
            "SELECT home_team_id, away_team_id FROM fixtures "
            "WHERE league_id = ? AND home_team_id >= 500000",
            (league_id,),
        ).fetchall()
        existing_pairs = {(r["home_team_id"], r["away_team_id"]) for r in existing}

        # Find latest match date to schedule after it
        last_date_row = conn.execute(
            "SELECT MAX(date) as d FROM fixtures WHERE league_id = ? AND home_team_id >= 500000",
            (league_id,),
        ).fetchone()
        last_date = datetime.fromisoformat(last_date_row["d"].replace("+00:00", ""))
        next_date = last_date + timedelta(days=1)

        # Generate missing pairs
        count = 0
        for home_id, home_name in teams:
            for away_id, away_name in teams:
                if home_id == away_id:
                    continue
                if (home_id, away_id) in existing_pairs:
                    continue

                # Create a fixture with status NS
                fix_api_id = _fixture_id(home_name, away_name, "remaining", league_id)
                date_iso = next_date.strftime("%Y-%m-%dT15:00:00+00:00")

                conn.execute(
                    "INSERT OR IGNORE INTO fixtures (league_id, home_team_id, away_team_id, date, status, home_goals, away_goals, api_id) "
                    "VALUES (?, ?, ?, ?, 'NS', NULL, NULL, ?)",
                    (league_id, home_id, away_id, date_iso, fix_api_id),
                )
                count += 1
                # Spread fixtures across days (10 per day)
                if count % 10 == 0:
                    next_date += timedelta(days=1)

        conn.commit()
        league_name = LEAGUE_NAMES.get(league_id, str(league_id))
        print(f"  {league_name}: {count} remaining fixtures generated")
        total += count

    return total
