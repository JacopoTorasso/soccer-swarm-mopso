import subprocess
import sys
import os
import pytest

def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "soccer_swarm", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "fetch" in result.stdout
    assert "train" in result.stdout
    assert "predict" in result.stdout
    assert "backtest" in result.stdout
    assert "optimize" in result.stdout
    assert "pareto" in result.stdout

def test_cli_fetch_requires_api_key():
    result = subprocess.run(
        [sys.executable, "-m", "soccer_swarm", "fetch"],
        capture_output=True, text=True,
        env={**os.environ, "API_FOOTBALL_KEY": ""},
    )
    # Should fail gracefully with no API key
    assert result.returncode != 0 or "API_FOOTBALL_KEY" in result.stderr
