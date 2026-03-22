import os

API_KEY = os.environ.get("API_FOOTBALL_KEY", "")
API_BASE_URL = "https://v3.football.api-sports.io"
DB_PATH = os.environ.get("SOCCER_SWARM_DB", "data/soccer_swarm.db")

LEAGUES = {
    "SerieA": 135,
    "PremierLeague": 39,
    "LaLiga": 140,
    "Bundesliga": 78,
    "Ligue1": 61,
}

CURRENT_SEASON = 2025
OPTIMIZATION_WINDOW_DAYS = 30
DAILY_REQUEST_LIMIT = 95
RATE_LIMIT_PER_MINUTE = 10

COMPLETED_STATUSES = ("FT", "AET", "PEN")
EXCLUDED_STATUSES = ("PST", "CANC", "AWD", "WO")

# ELO parameters
ELO_DEFAULT_RATING = 1500.0
ELO_HOME_ADVANTAGE = 65
ELO_K_FACTOR = 32
ELO_K_FACTOR_HIGH = 48
