import json
import time
from unittest.mock import patch, MagicMock
import pytest
from soccer_swarm.data.client import ApiClient

@pytest.fixture
def client(db_conn):
    return ApiClient(api_key="test_key", conn=db_conn)

def test_request_is_cached_on_second_call(client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": [{"id": 1}]}
    mock_response.raise_for_status = MagicMock()
    with patch("requests.get", return_value=mock_response) as mock_get:
        result1 = client.get("fixtures", {"league": 135, "season": 2025})
        result2 = client.get("fixtures", {"league": 135, "season": 2025})
        assert mock_get.call_count == 1
        assert result1 == result2

def test_daily_budget_raises_when_exceeded(client):
    client._daily_count = 95
    with pytest.raises(RuntimeError, match="Daily API budget exceeded"):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": []}
        mock_response.raise_for_status = MagicMock()
        with patch("requests.get", return_value=mock_response):
            client.get("fixtures", {"league": 135, "season": 2025, "_bust": "unique"})

def test_cache_ttl_expires(client):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": [{"id": 1}]}
    mock_response.raise_for_status = MagicMock()
    client.cache_ttl_seconds = 0
    with patch("requests.get", return_value=mock_response) as mock_get:
        client.get("fixtures", {"league": 135, "season": 2025})
        time.sleep(0.01)
        client.get("fixtures", {"league": 135, "season": 2025})
        assert mock_get.call_count == 2
