"""Comprehensive tests for dashboard data fetchers."""

import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.autfin.dashboard.data import (
    get_latest_snapshot,
    get_snapshot_history,
    save_snapshot,
    get_trade_history,
    get_trade_summary,
    get_signal_history,
    get_signal_summary,
    get_learning_entries,
    get_balance_history,
    get_dashboard_overview,
    get_kite_status,
    format_inr,
    format_pct,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_with_data(tmp_path, monkeypatch):
    """Create a temp database with sample data for testing."""
    db_path = tmp_path / "test_dashboard.db"
    monkeypatch.setattr(
        "src.autfin.dashboard.data.DB_PATH", db_path
    )
    monkeypatch.setattr(
        "src.autfin.memory.database.DB_PATH", db_path
    )
    monkeypatch.setattr(
        "src.autfin.memory.performance.DB_PATH", db_path
    )

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("PRAGMA journal_mode=WAL")

    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL DEFAULT 'NSE',
            action TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price REAL NOT NULL,
            total_value REAL NOT NULL,
            order_id TEXT,
            order_status TEXT,
            decision_id TEXT,
            rationale_summary TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            available_cash REAL NOT NULL,
            utilized_margin REAL NOT NULL,
            total_holdings_value REAL NOT NULL,
            total_pnl REAL NOT NULL,
            total_pnl_pct REAL NOT NULL,
            holdings_json TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            strength REAL NOT NULL,
            fundamentals_score REAL,
            technical_score REAL,
            sentiment_score REAL,
            macro_score REAL,
            risk_score REAL,
            rationale TEXT,
            was_acted_on BOOLEAN DEFAULT FALSE,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS balance_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            available_cash REAL NOT NULL,
            utilized_margin REAL NOT NULL,
            source TEXT DEFAULT 'kite_sdk'
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS learning_journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL DEFAULT 'BUY',
            conviction REAL NOT NULL DEFAULT 0.5,
            pnl_pct REAL,
            thesis_summary TEXT,
            outcome TEXT DEFAULT 'open',
            sector TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    # Insert sample data
    now = datetime.now(timezone.utc)
    recent = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    older = (now - timedelta(days=60)).strftime("%Y-%m-%d")
    recent_ts = now.strftime("%Y-%m-%dT%H:%M:%S")

    # Portfolio snapshots
    cursor.execute(
        "INSERT INTO portfolio_snapshots (date, available_cash, utilized_margin, total_holdings_value, total_pnl, total_pnl_pct, holdings_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (recent, 50000.0, 10000.0, 75000.0, 5000.0, 0.05, json.dumps([{"symbol": "RELIANCE", "qty": 10}])),
    )
    cursor.execute(
        "INSERT INTO portfolio_snapshots (date, available_cash, utilized_margin, total_holdings_value, total_pnl, total_pnl_pct, holdings_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (older, 60000.0, 5000.0, 65000.0, -2000.0, -0.02, json.dumps([])),
    )

    # Trades
    cursor.execute(
        "INSERT INTO trades (timestamp, symbol, action, quantity, price, total_value, order_status) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (recent_ts, "RELIANCE", "BUY", 10, 2500.0, 25000.0, "COMPLETE"),
    )
    cursor.execute(
        "INSERT INTO trades (timestamp, symbol, action, quantity, price, total_value, order_status) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (recent_ts, "TCS", "SELL", 5, 3500.0, 17500.0, "COMPLETE"),
    )

    # Signals
    cursor.execute(
        "INSERT INTO signals (timestamp, symbol, signal_type, strength, fundamentals_score, technical_score, sentiment_score, was_acted_on) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (recent_ts, "RELIANCE", "BUY", 0.85, 0.9, 0.8, 0.7, True),
    )
    cursor.execute(
        "INSERT INTO signals (timestamp, symbol, signal_type, strength, was_acted_on) VALUES (?, ?, ?, ?, ?)",
        (recent_ts, "INFY", "HOLD", 0.5, False),
    )

    # Learning journal
    cursor.execute(
        "INSERT INTO learning_journal (symbol, action, conviction, pnl_pct, thesis_summary, outcome, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("RELIANCE", "BUY", 0.75, 0.15, "Good entry timing on support level", "completed", recent),
    )
    cursor.execute(
        "INSERT INTO learning_journal (symbol, action, conviction, pnl_pct, thesis_summary, outcome, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("TCS", "SELL", 0.35, -0.08, "Entered too late, missed the move", "loss", older),
    )

    # Balance history
    cursor.execute(
        "INSERT INTO balance_history (timestamp, available_cash, utilized_margin, source) VALUES (?, ?, ?, ?)",
        (recent_ts, 50000.0, 10000.0, "kite_sdk"),
    )

    conn.commit()
    conn.close()

    return db_path


# ---------------------------------------------------------------------------
# format_inr tests
# ---------------------------------------------------------------------------

class TestFormatINR:
    """Tests for Indian Rupee formatting."""

    def test_zero(self):
        assert format_inr(0) == "₹0"

    def test_small_amount(self):
        assert format_inr(500.50) == "₹500.50"

    def test_thousands(self):
        assert "₹" in format_inr(5000)

    def test_lakhs(self):
        result = format_inr(150000)
        assert "L" in result

    def test_crores(self):
        result = format_inr(10000000)
        assert "Cr" in result

    def test_negative(self):
        result = format_inr(-5000)
        assert result.startswith("-")

    def test_negative_lakhs(self):
        result = format_inr(-150000)
        assert "-" in result and "L" in result

    def test_one_lakh(self):
        result = format_inr(100000)
        assert "L" in result


# ---------------------------------------------------------------------------
# format_pct tests
# ---------------------------------------------------------------------------

class TestFormatPct:
    """Tests for percentage formatting."""

    def test_positive(self):
        assert format_pct(0.05) == "+5.0%"

    def test_negative(self):
        assert format_pct(-0.03) == "-3.0%"

    def test_zero(self):
        assert format_pct(0.0) == "+0.0%"

    def test_large_positive(self):
        assert format_pct(0.25) == "+25.0%"


# ---------------------------------------------------------------------------
# Portfolio snapshot tests
# ---------------------------------------------------------------------------

class TestSnapshots:
    """Tests for portfolio snapshot operations."""

    def test_save_and_get_snapshot(self, db_with_data):
        row_id = save_snapshot(
            available_cash=40000.0,
            utilized_margin=8000.0,
            total_holdings_value=60000.0,
            total_pnl=3000.0,
            total_pnl_pct=0.03,
            holdings_json="[]",
        )
        assert row_id >= 0

        snapshot = get_latest_snapshot()
        assert snapshot is not None
        assert snapshot["available_cash"] == 40000.0

    def test_get_latest_snapshot_returns_most_recent(self, db_with_data):
        snapshot = get_latest_snapshot()
        assert snapshot is not None
        # Most recent has 50000 cash
        assert snapshot["available_cash"] == 50000.0

    def test_get_latest_snapshot_empty_db(self, tmp_path, monkeypatch):
        db_path = tmp_path / "empty.db"
        monkeypatch.setattr("src.autfin.dashboard.data.DB_PATH", db_path)
        monkeypatch.setattr("src.autfin.memory.database.DB_PATH", db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                available_cash REAL NOT NULL,
                utilized_margin REAL NOT NULL,
                total_holdings_value REAL NOT NULL,
                total_pnl REAL NOT NULL,
                total_pnl_pct REAL NOT NULL,
                holdings_json TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
        conn.close()

        assert get_latest_snapshot() is None

    def test_get_snapshot_history(self, db_with_data):
        history = get_snapshot_history(days=365)
        assert len(history) == 2
        # Should be ordered by date ascending
        assert history[0]["date"] < history[1]["date"]

    def test_get_snapshot_history_recent(self, db_with_data):
        history = get_snapshot_history(days=7)
        # Only the recent one should be within 7 days
        assert len(history) == 1


# ---------------------------------------------------------------------------
# Trade history tests
# ---------------------------------------------------------------------------

class TestTradeHistory:
    """Tests for trade history queries."""

    def test_get_trade_history_all(self, db_with_data):
        trades = get_trade_history(days=90)
        assert len(trades) == 2

    def test_get_trade_history_by_symbol(self, db_with_data):
        trades = get_trade_history(symbol="RELIANCE", days=90)
        assert len(trades) == 1
        assert trades[0]["symbol"] == "RELIANCE"

    def test_get_trade_history_empty(self, db_with_data):
        trades = get_trade_history(symbol="NOPE", days=90)
        assert len(trades) == 0

    def test_get_trade_summary(self, db_with_data):
        summary = get_trade_summary()
        assert summary["total_trades"] == 2
        assert summary["total_buys"] == 1
        assert summary["total_sells"] == 1
        assert summary["symbols_traded"] == 2
        assert summary["total_volume"] == 42500.0


# ---------------------------------------------------------------------------
# Signal history tests
# ---------------------------------------------------------------------------

class TestSignalHistory:
    """Tests for signal history queries."""

    def test_get_signal_history_all(self, db_with_data):
        signals = get_signal_history(days=90)
        assert len(signals) == 2

    def test_get_signal_history_by_symbol(self, db_with_data):
        signals = get_signal_history(symbol="RELIANCE", days=90)
        assert len(signals) == 1

    def test_get_signal_summary(self, db_with_data):
        summary = get_signal_summary()
        assert summary["total_signals"] == 2
        assert summary["buy_signals"] == 1
        assert summary["hold_signals"] == 1
        assert summary["acted_count"] == 1
        assert summary["pending_count"] == 1


# ---------------------------------------------------------------------------
# Learning entries tests
# ---------------------------------------------------------------------------

class TestLearningEntries:
    """Tests for learning journal queries."""

    def test_get_learning_entries_all(self, db_with_data):
        entries = get_learning_entries(days=365)
        assert len(entries) == 2

    def test_get_learning_entries_by_symbol(self, db_with_data):
        entries = get_learning_entries(symbol="RELIANCE", days=365)
        assert len(entries) == 1
        assert entries[0]["symbol"] == "RELIANCE"

    def test_get_learning_entries_recent(self, db_with_data):
        entries = get_learning_entries(days=7)
        # Only 1 is recent
        assert len(entries) == 1


# ---------------------------------------------------------------------------
# Balance history tests
# ---------------------------------------------------------------------------

class TestBalanceHistory:
    """Tests for balance history queries."""

    def test_get_balance_history(self, db_with_data):
        history = get_balance_history(days=90)
        assert len(history) == 1
        assert history[0]["available_cash"] == 50000.0


# ---------------------------------------------------------------------------
# Dashboard overview test
# ---------------------------------------------------------------------------

class TestDashboardOverview:
    """Tests for the aggregate dashboard overview."""

    def test_get_dashboard_overview(self, db_with_data):
        overview = get_dashboard_overview()
        assert "snapshot" in overview
        assert "trade_summary" in overview
        assert "signal_summary" in overview
        assert "performance" in overview
        assert "drawdown" in overview
        assert "equity_curve" in overview
        assert "recent_trades" in overview
        assert "recent_signals" in overview

        assert overview["trade_summary"]["total_trades"] == 2
        assert overview["signal_summary"]["total_signals"] == 2
        assert len(overview["equity_curve"]) == 2  # 2 snapshots

    def test_overview_empty_db(self, tmp_path, monkeypatch):
        db_path = tmp_path / "empty_overview.db"
        monkeypatch.setattr("src.autfin.dashboard.data.DB_PATH", db_path)
        monkeypatch.setattr("src.autfin.memory.database.DB_PATH", db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        for table in ["trades", "portfolio_snapshots", "signals", "balance_history", "learning_journal"]:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    date TEXT,
                    action TEXT,
                    quantity INTEGER,
                    price REAL,
                    total_value REAL,
                    order_id TEXT,
                    order_status TEXT,
                    decision_id TEXT,
                    trade_date TEXT,
                    created_at TEXT,
                    pnl_pct REAL,
                    outcome TEXT,
                    thesis_summary TEXT,
                    signal_type TEXT,
                    strength REAL,
                    fundamentals_score REAL,
                    technical_score REAL,
                    sentiment_score REAL,
                    was_acted_on BOOLEAN,
                    available_cash REAL,
                    utilized_margin REAL,
                    total_holdings_value REAL,
                    total_pnl REAL,
                    total_pnl_pct REAL,
                    holdings_json TEXT,
                    source TEXT
                )
            """)
        conn.commit()
        conn.close()

        overview = get_dashboard_overview()
        assert overview["snapshot"] is None
        assert overview["trade_summary"]["total_trades"] == 0


# ---------------------------------------------------------------------------
# Kite status test
# ---------------------------------------------------------------------------

class TestKiteStatus:
    """Tests for Kite SDK connectivity check."""

    def test_kite_status_when_unavailable(self):
        # The function does `from ..tools.kite_tools import KiteTools` inside a try/except
        # Patch the KiteTools class itself to raise ImportError on instantiation
        mock_kite_tools = MagicMock()
        mock_kite_tools.side_effect = ImportError("no kite module")
        with patch("src.autfin.tools.kite_tools.KiteTools", mock_kite_tools):
            result = get_kite_status()
            assert result["connected"] is False
            assert result["error"] is not None
