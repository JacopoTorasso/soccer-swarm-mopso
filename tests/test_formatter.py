import json
import csv
import io
import pytest
from soccer_swarm.output.formatter import format_backtest_csv, format_backtest_json, format_predictions_json

def test_format_backtest_csv():
    rows = [
        {"fixture_id": 1, "date": "2025-09-01", "home_team": "Roma",
         "away_team": "Lazio", "league": "SerieA", "market": "1x2",
         "pred_1": 0.55, "pred_2": 0.25, "pred_3": 0.20,
         "actual": "1", "correct": 1, "edge": 0.07, "bet_placed": 1},
    ]
    output = format_backtest_csv(rows)
    reader = csv.DictReader(io.StringIO(output))
    row_list = list(reader)
    assert len(row_list) == 1
    assert row_list[0]["home_team"] == "Roma"

def test_format_backtest_json():
    summary = {"global": {"accuracy": 0.55, "roi": 0.08}, "by_league": {}}
    output = format_backtest_json(summary)
    parsed = json.loads(output)
    assert parsed["global"]["accuracy"] == 0.55

def test_format_predictions_json():
    predictions = [
        {"fixture_id": 1, "home_team": "Roma", "away_team": "Lazio",
         "match_1x2": (0.55, 0.25, 0.20), "over_under_25": (0.6, 0.4),
         "btts": (0.5, 0.5), "recommended_bets": ["1X2: Home"]},
    ]
    output = format_predictions_json(predictions)
    parsed = json.loads(output)
    assert len(parsed) == 1
    assert parsed[0]["home_team"] == "Roma"
