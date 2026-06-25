"""Tests for dashboard Streamlit app rendering functions."""

import json
import sqlite3
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock, PropertyMock
import pytest


# ---------------------------------------------------------------------------
# Mock streamlit module before importing app
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_streamlit():
    """Create a mock streamlit module for all tests."""
    mock_st = MagicMock()
    mock_st.set_page_config = MagicMock()
    mock_st.title = MagicMock()
    mock_st.header = MagicMock()
    mock_st.subheader = MagicMock()
    mock_st.caption = MagicMock()
    mock_st.metric = MagicMock()
    mock_st.divider = MagicMock()
    mock_st.columns = MagicMock(side_effect=lambda n: [MagicMock() for _ in range(n)])
    mock_st.radio = MagicMock(return_value="Overview")
    mock_st.text_input = MagicMock(return_value="")
    mock_st.selectbox = MagicMock(return_value=90)
    mock_st.expander = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
    mock_st.dataframe = MagicMock()
    mock_st.line_chart = MagicMock()
    mock_st.info = MagicMock()
    mock_st.success = MagicMock()
    mock_st.warning = MagicMock()
    mock_st.write = MagicMock()

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        # Also ensure the app module sees STREAMLIT_AVAILABLE = True
        with patch("src.autfin.dashboard.app.STREAMLIT_AVAILABLE", True):
            yield mock_st


@pytest.fixture
def db_with_data(tmp_path, monkeypatch):
    """Create temp database with sample data."""
    db_path = tmp_path / "test_app.db"
    monkeypatch.setattr("src.autfin.dashboard.data.DB_PATH", db_path)
    monkeypatch.setattr("src.autfin.memory.database.DB_PATH", db_path)
    monkeypatch.setattr("src.autfin.memory.performance.DB_PATH", db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")

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

    now = datetime.now(timezone.utc)
    recent = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    recent_ts = now.strftime("%Y-%m-%dT%H:%M:%S")

    cursor.execute(
        "INSERT INTO portfolio_snapshots (date, available_cash, utilized_margin, total_holdings_value, total_pnl, total_pnl_pct, holdings_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (recent, 50000.0, 10000.0, 75000.0, 5000.0, 0.05, json.dumps([{"symbol": "RELIANCE", "qty": 10}])),
    )
    cursor.execute(
        "INSERT INTO trades (timestamp, symbol, action, quantity, price, total_value, order_status) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (recent_ts, "RELIANCE", "BUY", 10, 2500.0, 25000.0, "COMPLETE"),
    )
    cursor.execute(
        "INSERT INTO signals (timestamp, symbol, signal_type, strength, fundamentals_score, technical_score, sentiment_score, was_acted_on) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (recent_ts, "RELIANCE", "BUY", 0.85, 0.9, 0.8, 0.7, True),
    )
    cursor.execute(
        "INSERT INTO learning_journal (symbol, action, conviction, pnl_pct, thesis_summary, outcome, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("RELIANCE", "BUY", 0.75, 0.15, "Good entry timing on support level", "completed", recent),
    )

    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# configure_page tests
# ---------------------------------------------------------------------------

class TestConfigurePage:
    """Tests for page configuration."""

    def test_sets_page_config(self, mock_streamlit):
        from src.autfin.dashboard.app import configure_page
        configure_page()
        mock_streamlit.set_page_config.assert_called_once()
        mock_streamlit.title.assert_called_once_with("AutFin — Portfolio Dashboard")


# ---------------------------------------------------------------------------
# render_sidebar tests
# ---------------------------------------------------------------------------

class TestRenderSidebar:
    """Tests for sidebar rendering."""

    def test_sidebar_renders(self, mock_streamlit, db_with_data):
        from src.autfin.dashboard.app import render_sidebar
        page, auto_refresh, refresh_interval = render_sidebar()
        assert page == "Overview"
        # In Streamlit, st.header() inside `with st.sidebar:` renders to sidebar
        # but it's still the same st.header call — the mock captures it as mock_st.header
        mock_streamlit.header.assert_called_with("AutFin")
        mock_streamlit.radio.assert_called_once()

    def test_sidebar_with_snapshot(self, mock_streamlit, db_with_data):
        from src.autfin.dashboard.app import render_sidebar
        page, auto_refresh, refresh_interval = render_sidebar()
        # Sidebar metric calls go through st.metric (context manager redirects)
        assert mock_streamlit.metric.call_count >= 2


# ---------------------------------------------------------------------------
# render_overview tests
# ---------------------------------------------------------------------------

class TestRenderOverview:
    """Tests for portfolio overview panel."""

    def test_overview_renders(self, mock_streamlit, db_with_data):
        from src.autfin.dashboard.app import render_overview
        render_overview()
        mock_streamlit.header.assert_called_with("Portfolio Overview")
        # Should have metrics
        assert mock_streamlit.metric.call_count >= 5

    def test_overview_empty_db(self, mock_streamlit, tmp_path, monkeypatch):
        db_path = tmp_path / "empty_overview.db"
        monkeypatch.setattr("src.autfin.dashboard.data.DB_PATH", db_path)
        monkeypatch.setattr("src.autfin.memory.database.DB_PATH", db_path)
        monkeypatch.setattr("src.autfin.memory.performance.DB_PATH", db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        for table in ["trades", "portfolio_snapshots", "signals", "balance_history", "learning_journal"]:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT, symbol TEXT, date TEXT, action TEXT,
                    quantity INTEGER, price REAL, total_value REAL,
                    order_id TEXT, order_status TEXT, decision_id TEXT,
                    trade_date TEXT, created_at TEXT, pnl_pct REAL, outcome TEXT,
                    thesis_summary TEXT, signal_type TEXT, strength REAL,
                    fundamentals_score REAL, technical_score REAL,
                    sentiment_score REAL, was_acted_on BOOLEAN,
                    available_cash REAL, utilized_margin REAL,
                    total_holdings_value REAL, total_pnl REAL,
                    total_pnl_pct REAL, holdings_json TEXT, source TEXT
                )
            """)
        conn.commit()
        conn.close()

        from src.autfin.dashboard.app import render_overview
        render_overview()
        # Should not crash on empty data
        mock_streamlit.header.assert_called_with("Portfolio Overview")


# ---------------------------------------------------------------------------
# render_trades tests
# ---------------------------------------------------------------------------

class TestRenderTrades:
    """Tests for trade history panel."""

    def test_trades_renders(self, mock_streamlit, db_with_data):
        from src.autfin.dashboard.app import render_trades
        render_trades()
        mock_streamlit.header.assert_called_with("Trade History")

    def test_trades_with_filter(self, mock_streamlit, db_with_data):
        from src.autfin.dashboard.app import render_trades
        mock_streamlit.text_input.return_value = "RELIANCE"
        render_trades()
        # Should show filtered results
        mock_streamlit.dataframe.assert_called()


# ---------------------------------------------------------------------------
# render_signals tests
# ---------------------------------------------------------------------------

class TestRenderSignals:
    """Tests for signal log panel."""

    def test_signals_renders(self, mock_streamlit, db_with_data):
        from src.autfin.dashboard.app import render_signals
        render_signals()
        mock_streamlit.header.assert_called_with("Signal Log")


# ---------------------------------------------------------------------------
# render_performance tests
# ---------------------------------------------------------------------------

class TestRenderPerformance:
    """Tests for performance metrics panel."""

    def test_performance_renders(self, mock_streamlit, db_with_data):
        from src.autfin.dashboard.app import render_performance
        render_performance()
        mock_streamlit.header.assert_called_with("Performance Metrics")

    def test_performance_empty_db(self, mock_streamlit, tmp_path, monkeypatch):
        db_path = tmp_path / "empty_perf.db"
        monkeypatch.setattr("src.autfin.dashboard.data.DB_PATH", db_path)
        monkeypatch.setattr("src.autfin.memory.database.DB_PATH", db_path)
        monkeypatch.setattr("src.autfin.memory.performance.DB_PATH", db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        for table in ["trades", "portfolio_snapshots", "signals", "balance_history", "learning_journal"]:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT, symbol TEXT, date TEXT, action TEXT,
                    quantity INTEGER, price REAL, total_value REAL,
                    order_id TEXT, order_status TEXT, decision_id TEXT,
                    trade_date TEXT, created_at TEXT, pnl_pct REAL, outcome TEXT,
                    thesis_summary TEXT, signal_type TEXT, strength REAL,
                    fundamentals_score REAL, technical_score REAL,
                    sentiment_score REAL, was_acted_on BOOLEAN,
                    available_cash REAL, utilized_margin REAL,
                    total_holdings_value REAL, total_pnl REAL,
                    total_pnl_pct REAL, holdings_json TEXT, source TEXT
                )
            """)
        conn.commit()
        conn.close()

        from src.autfin.dashboard.app import render_performance
        render_performance()
        mock_streamlit.header.assert_called_with("Performance Metrics")


# ---------------------------------------------------------------------------
# render_learnings tests
# ---------------------------------------------------------------------------

class TestRenderLearnings:
    """Tests for learning journal panel."""

    def test_learnings_renders(self, mock_streamlit, db_with_data):
        from src.autfin.dashboard.app import render_learnings
        render_learnings()
        mock_streamlit.header.assert_called_with("Learning Journal")

    def test_learnings_empty(self, mock_streamlit, tmp_path, monkeypatch):
        db_path = tmp_path / "empty_learnings.db"
        monkeypatch.setattr("src.autfin.dashboard.data.DB_PATH", db_path)
        monkeypatch.setattr("src.autfin.memory.database.DB_PATH", db_path)
        monkeypatch.setattr("src.autfin.memory.performance.DB_PATH", db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
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
        conn.commit()
        conn.close()

        from src.autfin.dashboard.app import render_learnings
        render_learnings()
        mock_streamlit.info.assert_called()


# ---------------------------------------------------------------------------
# render_equity_curve tests
# ---------------------------------------------------------------------------

class TestRenderEquityCurve:
    """Tests for equity curve panel."""

    def test_equity_curve_with_data(self, mock_streamlit, db_with_data):
        from src.autfin.dashboard.app import render_equity_curve
        render_equity_curve()
        mock_streamlit.subheader.assert_called_with("Equity Curve")
        mock_streamlit.line_chart.assert_called()

    def test_equity_curve_empty(self, mock_streamlit, tmp_path, monkeypatch):
        db_path = tmp_path / "empty_eq.db"
        monkeypatch.setattr("src.autfin.dashboard.data.DB_PATH", db_path)
        monkeypatch.setattr("src.autfin.memory.database.DB_PATH", db_path)
        monkeypatch.setattr("src.autfin.memory.performance.DB_PATH", db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL, available_cash REAL NOT NULL,
                utilized_margin REAL NOT NULL, total_holdings_value REAL NOT NULL,
                total_pnl REAL NOT NULL, total_pnl_pct REAL NOT NULL,
                holdings_json TEXT NOT NULL, created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
        conn.close()

        from src.autfin.dashboard.app import render_equity_curve
        render_equity_curve()
        mock_streamlit.info.assert_called()


# ---------------------------------------------------------------------------
# main function tests
# ---------------------------------------------------------------------------

class TestMain:
    """Tests for the main entry point."""

    def test_main_runs(self, mock_streamlit, db_with_data):
        from src.autfin.dashboard.app import main
        main()
        mock_streamlit.set_page_config.assert_called_once()

    def test_main_when_streamlit_not_available(self, monkeypatch):
        monkeypatch.setattr("src.autfin.dashboard.app.STREAMLIT_AVAILABLE", False)
        from src.autfin.dashboard.app import main
        with pytest.raises(SystemExit):
            main()
