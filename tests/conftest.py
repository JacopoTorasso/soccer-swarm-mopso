import sqlite3
import pytest
from soccer_swarm.data.db import get_connection, create_tables

@pytest.fixture
def db_conn(tmp_path):
    db_path = str(tmp_path / "test.db")
    conn = get_connection(db_path)
    create_tables(conn)
    yield conn
    conn.close()
