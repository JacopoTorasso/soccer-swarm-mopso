import hashlib
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone

import requests

from soccer_swarm.config import API_BASE_URL, DAILY_REQUEST_LIMIT, RATE_LIMIT_PER_MINUTE

logger = logging.getLogger(__name__)


class ApiClient:
    DEFAULT_TTL = {
        "fixtures": 86400,
        "fixtures/statistics": 604800,
        "standings": 43200,
        "odds": 3600,
    }

    def __init__(self, api_key: str, conn: sqlite3.Connection, cache_ttl_seconds: int = 3600):
        self.api_key = api_key
        self.conn = conn
        self.cache_ttl_seconds = cache_ttl_seconds
        self._daily_count = 0
        self._last_request_time = 0.0
        self._minute_timestamps: list[float] = []

    def get(self, endpoint: str, params: dict) -> dict:
        params_hash = hashlib.md5(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()
        cached = self._check_cache(endpoint, params_hash)
        if cached is not None:
            return cached
        if self._daily_count >= DAILY_REQUEST_LIMIT:
            raise RuntimeError(f"Daily API budget exceeded ({DAILY_REQUEST_LIMIT} requests)")
        self._enforce_rate_limit()
        url = f"{API_BASE_URL}/{endpoint}"
        headers = {"x-apisports-key": self.api_key}
        logger.info("API request: %s params=%s", endpoint, params)
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        self._daily_count += 1
        logger.info("API response: %s status=%d daily_count=%d", endpoint, response.status_code, self._daily_count)
        self._write_cache(endpoint, params_hash, data)
        return data

    def _check_cache(self, endpoint: str, params_hash: str) -> dict | None:
        row = self.conn.execute(
            "SELECT response_json, timestamp FROM api_log WHERE endpoint = ? AND params_hash = ?",
            (endpoint, params_hash),
        ).fetchone()
        if row is None:
            return None
        cached_time = datetime.fromisoformat(row["timestamp"])
        age = (datetime.now(timezone.utc) - cached_time).total_seconds()
        default_ttl = self.DEFAULT_TTL.get(endpoint, self.cache_ttl_seconds)
        ttl = min(default_ttl, self.cache_ttl_seconds)
        if age > ttl:
            return None
        logger.debug("Cache hit: %s hash=%s age=%.0fs", endpoint, params_hash, age)
        return json.loads(row["response_json"])

    def _write_cache(self, endpoint: str, params_hash: str, data: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT OR REPLACE INTO api_log (endpoint, params_hash, timestamp, response_json) VALUES (?, ?, ?, ?)",
            (endpoint, params_hash, now, json.dumps(data)),
        )
        self.conn.commit()

    def _enforce_rate_limit(self) -> None:
        now = time.monotonic()
        self._minute_timestamps = [t for t in self._minute_timestamps if now - t < 60]
        if len(self._minute_timestamps) >= RATE_LIMIT_PER_MINUTE:
            sleep_time = 60 - (now - self._minute_timestamps[0])
            if sleep_time > 0:
                logger.info("Rate limit: sleeping %.1fs", sleep_time)
                time.sleep(sleep_time)
        self._minute_timestamps.append(time.monotonic())
