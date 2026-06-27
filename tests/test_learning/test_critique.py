"""Tests for self-critique loop (learning/critique.py)."""

import sqlite3
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
from autfin.learning.critique import (
    build_critique_prompt,
    get_decisions_needing_review,
    parse_critique_response,
    run_weekly_critique,
    store_critique,
)


@pytest.fixture
def critique_db(tmp_path):
    """Create a temp database with learning_journal and episodic_memory tables."""
    db_path = tmp_path / "test_critique.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE learning_journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            conviction REAL,
            thesis_summary TEXT,
            outcome TEXT,
            pnl_pct REAL,
            created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE episodic_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id INTEGER,
            symbol TEXT,
            outcome TEXT,
            pnl_pct REAL,
            lesson TEXT,
            context TEXT,
            created_at TEXT
        )
    """)
    # Insert test decisions: one old win, one old loss, one recent
    old_date = (datetime.now() - timedelta(days=15)).isoformat()
    recent_date = (datetime.now() - timedelta(days=3)).isoformat()
    very_old = (datetime.now() - timedelta(days=40)).isoformat()

    conn.execute(
        "INSERT INTO learning_journal (symbol, conviction, thesis_summary, outcome, pnl_pct, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("RELIANCE", 0.8, "Strong fundamentals", "completed", 5.2, old_date),
    )
    conn.execute(
        "INSERT INTO learning_journal (symbol, conviction, thesis_summary, outcome, pnl_pct, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("INFY", 0.6, "AI growth story", "loss", -3.1, old_date),
    )
    conn.execute(
        "INSERT INTO learning_journal (symbol, conviction, thesis_summary, outcome, pnl_pct, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("TCS", 0.9, "Cloud leader", "completed", 2.0, recent_date),
    )
    conn.execute(
        "INSERT INTO learning_journal (symbol, conviction, thesis_summary, outcome, pnl_pct, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("WIPRO", 0.5, "Cheap valuation", "loss", -1.5, very_old),
    )
    conn.commit()
    yield db_path, conn
    conn.close()


class TestGetDecisionsNeedingReview:
    def test_finds_eligible_decisions(self, critique_db):
        db_path, conn = critique_db
        decisions = get_decisions_needing_review(conn)
        # Should find the two 15-day-old decisions, not the 3-day or 40-day
        assert len(decisions) == 2
        symbols = {d["symbol"] for d in decisions}
        assert "RELIANCE" in symbols
        assert "INFY" in symbols

    def test_excludes_recent(self, critique_db):
        db_path, conn = critique_db
        decisions = get_decisions_needing_review(conn, min_age_days=5)
        symbols = {d["symbol"] for d in decisions}
        assert "TCS" not in symbols  # Only 3 days old

    def test_excludes_very_old(self, critique_db):
        db_path, conn = critique_db
        decisions = get_decisions_needing_review(conn, max_age_days=30)
        symbols = {d["symbol"] for d in decisions}
        assert "WIPRO" not in symbols  # 40 days old

    def test_empty_db(self, tmp_path):
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE learning_journal (
                id INTEGER PRIMARY KEY, symbol TEXT, conviction REAL,
                thesis_summary TEXT, outcome TEXT, pnl_pct REAL, created_at TEXT
            )
        """)
        decisions = get_decisions_needing_review(conn)
        assert decisions == []
        conn.close()


class TestBuildCritiquePrompt:
    def test_win_prompt(self):
        decision = {
            "symbol": "RELIANCE",
            "conviction": 0.8,
            "thesis": "Strong fundamentals",
            "outcome": "completed",
            "pnl_pct": 5.2,
        }
        prompt = build_critique_prompt(decision)
        assert "RELIANCE" in prompt
        assert "WIN" in prompt
        assert "0.8" in prompt
        assert "5.2%" in prompt

    def test_loss_prompt(self):
        decision = {
            "symbol": "INFY",
            "conviction": 0.6,
            "thesis": "AI growth",
            "outcome": "loss",
            "pnl_pct": -3.1,
        }
        prompt = build_critique_prompt(decision)
        assert "LOSS" in prompt
        assert "-3.1%" in prompt

    def test_json_in_prompt(self):
        decision = {"symbol": "TCS", "conviction": 0.9, "thesis": "Cloud", "outcome": "completed", "pnl_pct": 2.0}
        prompt = build_critique_prompt(decision)
        assert "calibration_score" in prompt
        assert "lesson" in prompt


class TestParseCritiqueResponse:
    def test_valid_json(self):
        response = '''Here's my review:
        {"calibration_score": 0.8, "overweighted_factors": ["momentum"], "underweighted_factors": ["valuation"], "lesson": "Focus on valuation more", "would_repeat": true}
        '''
        result = parse_critique_response(response)
        assert result is not None
        assert result["calibration_score"] == 0.8
        assert "momentum" in result["overweighted_factors"]
        assert result["would_repeat"] is True

    def test_invalid_json(self):
        result = parse_critique_response("This is not JSON at all")
        assert result is None

    def test_none_response(self):
        result = parse_critique_response(None)
        assert result is None

    def test_empty_response(self):
        result = parse_critique_response("")
        assert result is None


class TestStoreCritique:
    def test_stores_critique(self, critique_db):
        db_path, conn = critique_db
        critique = {
            "calibration_score": 0.7,
            "overweighted_factors": ["news_sentiment"],
            "underweighted_factors": ["sector_rotation"],
            "lesson": "Watch sector rotation signals",
            "would_repeat": True,
        }
        store_critique(conn, decision_id=1, critique=critique)

        # Check episodic memory was updated
        row = conn.execute("SELECT lesson FROM episodic_memory WHERE decision_id=1").fetchone()
        assert row is not None
        assert "sector rotation" in row[0]

        # Check learning journal was updated
        row = conn.execute("SELECT thesis_summary FROM learning_journal WHERE id=1").fetchone()
        assert "CALIBRATED" in row[0]


class TestRunWeeklyCritique:
    def test_no_decisions(self, critique_db):
        db_path, conn = critique_db
        # Use window that excludes all test data (15 days old)
        # max_age_days=10 excludes anything older than 10 days
        from autfin.learning.critique import get_decisions_needing_review
        decisions = get_decisions_needing_review(conn, max_age_days=10)
        assert len(decisions) == 0  # All test data is 15 days old, excluded by max_age_days=10

    def test_with_llm(self, critique_db):
        db_path, conn = critique_db

        def mock_llm(prompt):
            return '{"calibration_score": 0.6, "overweighted_factors": [], "underweighted_factors": [], "lesson": "Test lesson", "would_repeat": true}'

        result = run_weekly_critique(conn, llm_generate_fn=mock_llm, max_decisions=2)
        assert result["reviewed"] == 2
        assert len(result["lessons"]) == 2

    def test_without_llm(self, critique_db):
        db_path, conn = critique_db
        result = run_weekly_critique(conn, llm_generate_fn=None)
        assert result["reviewed"] == 0
        assert result["decisions_available"] == 2

    def test_max_decisions_limit(self, critique_db):
        db_path, conn = critique_db

        def mock_llm(prompt):
            return '{"calibration_score": 0.5, "overweighted_factors": [], "underweighted_factors": [], "lesson": "ok", "would_repeat": false}'

        result = run_weekly_critique(conn, llm_generate_fn=mock_llm, max_decisions=1)
        assert result["reviewed"] == 1
