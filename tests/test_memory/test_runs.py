"""Tests for runs table (memory/database.py)."""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def temp_db(tmp_path):
    """Create a temp database for testing."""
    db_path = tmp_path / "test_runs.db"
    with patch("autfin.memory.database.DB_PATH", db_path):
        from autfin.memory.database import init_database
        init_database()
        yield db_path


class TestRunsTable:
    def test_record_run_start(self, temp_db):
        with patch("autfin.memory.database.DB_PATH", temp_db):
            from autfin.memory.database import record_run_start
            record_run_start("run-001", query="Buy RELIANCE", mode="recommendation")
            conn = sqlite3.connect(temp_db)
            row = conn.execute("SELECT run_id, status, query, mode FROM runs WHERE run_id='run-001'").fetchone()
            conn.close()
            assert row is not None
            assert row[0] == "run-001"
            assert row[1] == "running"
            assert row[2] == "Buy RELIANCE"
            assert row[3] == "recommendation"

    def test_record_run_end(self, temp_db):
        with patch("autfin.memory.database.DB_PATH", temp_db):
            from autfin.memory.database import record_run_end, record_run_start
            record_run_start("run-002", query="Sell INFY")
            record_run_end(
                "run-002",
                status="completed",
                total_tokens=1500,
                total_cost_inr=0.0,
                decisions_made=2,
                trades_executed=1,
                duration_ms=5200.0,
            )
            conn = sqlite3.connect(temp_db)
            row = conn.execute(
                "SELECT status, total_tokens, decisions_made, trades_executed, duration_ms "
                "FROM runs WHERE run_id='run-002'"
            ).fetchone()
            conn.close()
            assert row[0] == "completed"
            assert row[1] == 1500
            assert row[2] == 2
            assert row[3] == 1
            assert row[4] == 5200.0

    def test_record_run_error(self, temp_db):
        with patch("autfin.memory.database.DB_PATH", temp_db):
            from autfin.memory.database import record_run_end, record_run_start
            record_run_start("run-003", query="Scan market")
            record_run_end("run-003", status="failed", error_message="LLM timeout")
            conn = sqlite3.connect(temp_db)
            row = conn.execute("SELECT status, error_message FROM runs WHERE run_id='run-003'").fetchone()
            conn.close()
            assert row[0] == "failed"
            assert row[1] == "LLM timeout"

    def test_record_run_with_metadata(self, temp_db):
        with patch("autfin.memory.database.DB_PATH", temp_db):
            from autfin.memory.database import record_run_end, record_run_start
            record_run_start("run-004", mode="scan")
            record_run_end("run-004", metadata={"symbols_scanned": 500})
            conn = sqlite3.connect(temp_db)
            row = conn.execute("SELECT metadata_json FROM runs WHERE run_id='run-004'").fetchone()
            conn.close()
            assert row is not None
            assert "500" in row[0]

    def test_record_run_nodes_visited(self, temp_db):
        with patch("autfin.memory.database.DB_PATH", temp_db):
            from autfin.memory.database import record_run_end, record_run_start
            record_run_start("run-005")
            record_run_end(
                "run-005",
                nodes_visited=["supervisor", "fundamentals", "risk_manager", "portfolio_manager"],
            )
            conn = sqlite3.connect(temp_db)
            row = conn.execute("SELECT nodes_visited FROM runs WHERE run_id='run-005'").fetchone()
            conn.close()
            assert "fundamentals" in row[0]

    def test_record_run_overwrite(self, temp_db):
        with patch("autfin.memory.database.DB_PATH", temp_db):
            from autfin.memory.database import record_run_end, record_run_start
            record_run_start("run-006")
            record_run_end("run-006", status="completed")
            record_run_end("run-006", status="failed", error_message="overwritten")
            conn = sqlite3.connect(temp_db)
            row = conn.execute("SELECT status, error_message FROM runs WHERE run_id='run-006'").fetchone()
            conn.close()
            assert row[0] == "failed"
            assert row[1] == "overwritten"
