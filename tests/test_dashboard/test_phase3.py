"""Tests for dashboard Phase 3 additions (3.1 stock pitches, 3.2 fan-out progress, 3.3 learning insights)."""

import json
import sqlite3
import sys
import types
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
import pytest


# ---------------------------------------------------------------------------
# Mock streamlit module before importing app
# ---------------------------------------------------------------------------

def _ensure_app_imported(streamlit_module):
    """Force-import src.autfin.dashboard.app with `streamlit_module` bound in
    sys.modules so `app.st` resolves to the same mock the tests assert on.
    Idempotent only via re-import under the current streamlit stub.
    """
    import importlib
    import src.autfin.dashboard as _pkg

    sys.modules.pop("src.autfin.dashboard.app", None)
    if hasattr(_pkg, "app"):
        try:
            delattr(_pkg, "app")
        except Exception:
            pass

    saved = sys.modules.get("streamlit")
    sys.modules["streamlit"] = streamlit_module
    try:
        importlib.import_module("src.autfin.dashboard.app")
    except Exception:
        # Fall back to a stub module so test patching still resolves cleanly.
        dummy = types.ModuleType("src.autfin.dashboard.app")
        dummy.STREAMLIT_AVAILABLE = False
        for name in (
            "render_overview", "render_trades", "render_signals", "render_performance",
            "render_learnings", "render_paper_trading", "render_system",
            "render_equity_curve", "render_sidebar", "render_stock_pitches",
            "render_fanout_progress", "render_learning_insights",
            "configure_page", "main",
        ):
            setattr(dummy, name, lambda *a, **k: None)
        sys.modules["src.autfin.dashboard.app"] = dummy
        _pkg.app = dummy
    finally:
        if saved is not None:
            sys.modules["streamlit"] = saved


def _selectbox_side_effect(label, options, **kwargs):
    """Return sensible defaults for each selectbox call by matching label keywords."""
    label_lower = str(label).lower()
    if "context" in label_lower:
        return "All"
    if "period" in label_lower or "days" in label_lower:
        return 90
    if "window" in label_lower or "hours" in label_lower:
        return 24
    if "min" in label_lower:
        return 5
    return options[0] if options else ""


def _make_mock_streamlit():
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
    mock_st.selectbox = MagicMock(side_effect=_selectbox_side_effect)
    mock_st.select_slider = MagicMock(return_value=60)
    mock_st.slider = MagicMock(return_value=5)
    mock_st.checkbox = MagicMock(return_value=False)
    mock_st.button = MagicMock(return_value=False)
    mock_st.rerun = MagicMock()
    mock_st.expander = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
    mock_st.dataframe = MagicMock()
    mock_st.line_chart = MagicMock()
    mock_st.bar_chart = MagicMock()
    mock_st.info = MagicMock()
    mock_st.success = MagicMock()
    mock_st.warning = MagicMock()
    mock_st.error = MagicMock()
    mock_st.write = MagicMock()
    mock_st.markdown = MagicMock()
    return mock_st


@pytest.fixture(autouse=True)
def mock_streamlit():
    """Create a mock streamlit module for all tests."""
    mock_st = _make_mock_streamlit()
    _ensure_app_imported(mock_st)

    with patch.dict("sys.modules", {"streamlit": mock_st}):
        with patch("src.autfin.dashboard.app.STREAMLIT_AVAILABLE", True):
            yield mock_st


def _setup_pitches_table(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS stock_pitches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            context TEXT NOT NULL DEFAULT 'manual',
            investment_thesis TEXT,
            catalysts_json TEXT,
            risks_json TEXT,
            valuation_summary TEXT,
            competitive_moat TEXT,
            price_target_base REAL,
            price_target_upside REAL,
            price_target_downside REAL,
            price_target_rationale TEXT,
            raw_json TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()


def _setup_fanout_table(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fanout_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            context TEXT NOT NULL,
            symbol TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            started_at TEXT,
            finished_at TEXT,
            error TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()


def _setup_portfolio_tables(db_path):
    """Create minimal tables needed by sidebar/main (portfolio_snapshots, etc.)."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            available_cash REAL DEFAULT 0,
            utilized_margin REAL DEFAULT 0,
            total_holdings_value REAL DEFAULT 0,
            total_pnl REAL DEFAULT 0,
            total_pnl_pct REAL DEFAULT 0,
            holdings_json TEXT
        )
    """)
    conn.commit()
    conn.close()


@pytest.fixture
def db_with_pitches(tmp_path, monkeypatch):
    db_path = tmp_path / "test_pitches.db"
    monkeypatch.setattr("src.autfin.dashboard.data.DB_PATH", db_path)
    monkeypatch.setattr("src.autfin.memory.database.DB_PATH", db_path)
    monkeypatch.setattr("src.autfin.memory.performance.DB_PATH", db_path)
    _setup_pitches_table(db_path)
    _setup_portfolio_tables(db_path)
    return db_path


@pytest.fixture
def db_with_fanout(tmp_path, monkeypatch):
    db_path = tmp_path / "test_fanout.db"
    monkeypatch.setattr("src.autfin.dashboard.data.DB_PATH", db_path)
    monkeypatch.setattr("src.autfin.memory.database.DB_PATH", db_path)
    monkeypatch.setattr("src.autfin.memory.performance.DB_PATH", db_path)
    _setup_fanout_table(db_path)
    _setup_portfolio_tables(db_path)
    return db_path


@pytest.fixture
def db_with_learning(tmp_path, monkeypatch):
    db_path = tmp_path / "test_learning.db"
    monkeypatch.setattr("src.autfin.dashboard.data.DB_PATH", db_path)
    monkeypatch.setattr("src.autfin.memory.database.DB_PATH", db_path)
    monkeypatch.setattr("src.autfin.memory.performance.DB_PATH", db_path)
    _setup_portfolio_tables(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
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
    return db_path


# ===========================================================================
# 3.1 — Stock Pitch data layer
# ===========================================================================

class TestSaveStockPitch:
    """Tests for save_stock_pitch data function."""

    def test_save_persists_basic_fields(self, db_with_pitches):
        from src.autfin.dashboard.data import save_stock_pitch, get_stock_pitches

        pitch = {
            "investment_thesis": "Strong moat, sector leader",
            "catalysts": ["Capacity expansion", "New SEBI policy"],
            "risks": ["Commodity risk", "Currency risk"],
            "valuation_summary": "Trading at 22x earnings, fair value",
            "competitive_moat": "Brand + distribution",
            "price_target": {
                "base": 3200.0,
                "upside": 3500.0,
                "downside": 2800.0,
                "rationale": "DCF supports base; bull scenario for upside",
            },
        }
        row_id = save_stock_pitch("RELIANCE", "morning_routine", pitch)
        assert row_id > 0

        rows = get_stock_pitches(symbol="RELIANCE")
        assert len(rows) == 1
        r = rows[0]
        assert r["symbol"] == "RELIANCE"
        assert r["context"] == "morning_routine"
        assert r["investment_thesis"] == "Strong moat, sector leader"
        assert r["catalysts"] == ["Capacity expansion", "New SEBI policy"]
        assert r["risks"] == ["Commodity risk", "Currency risk"]
        assert r["price_target_base"] == 3200.0
        assert r["price_target_upside"] == 3500.0
        assert r["price_target_downside"] == 2800.0

    def test_save_strips_pitch_wrapper(self, db_with_pitches):
        """build_stock_pitch returns {status:..., pitch:{...}}; save should unwrap."""
        from src.autfin.dashboard.data import save_stock_pitch, get_stock_pitches

        wrapped = {
            "status": "success",
            "symbol": "TCS",
            "pitch": {
                "investment_thesis": "IT services leader",
                "catalysts": ["AI deal wins"],
                "risks": ["Wage inflation"],
                "valuation_summary": "Trades at 26x",
                "competitive_moat": "Scale and clients",
                "price_target": {"base": 4200, "upside": 4500, "downside": 3800, "rationale": "Growth"},
            },
        }
        save_stock_pitch("TCS", "manual", wrapped)
        rows = get_stock_pitches(symbol="TCS")
        assert len(rows) == 1
        assert rows[0]["investment_thesis"] == "IT services leader"
        assert rows[0]["price_target_base"] == 4200.0

    def test_save_without_price_target(self, db_with_pitches):
        """LLM may sometimes omit price_target."""
        from src.autfin.dashboard.data import save_stock_pitch, get_stock_pitches

        pitch = {
            "investment_thesis": "Defensive pick",
            "catalysts": [],
            "risks": [],
            "valuation_summary": "Stable",
            "competitive_moat": "Cash flows",
        }
        row_id = save_stock_pitch("HDFCBANK", "quarterly_review", pitch)
        assert row_id > 0

        rows = get_stock_pitches(symbol="HDFCBANK")
        assert rows[0]["price_target_base"] is None
        assert rows[0]["price_target_upside"] is None
        assert rows[0]["price_target_downside"] is None

    def test_save_handles_string_pitch(self, db_with_pitches):
        """Defensive: pitch could be a raw string (JSON parse failure)."""
        from src.autfin.dashboard.data import save_stock_pitch, get_stock_pitches

        save_stock_pitch("INFY", "manual", "raw string response from LLM")
        rows = get_stock_pitches(symbol="INFY")
        assert len(rows) == 1
        assert "raw_response" in (rows[0].get("raw") or {})

    def test_save_handles_empty_payload(self, db_with_pitches):
        """Empty dict still persists a row (degraded but useful)."""
        from src.autfin.dashboard.data import save_stock_pitch, get_stock_pitches

        row_id = save_stock_pitch("WIPRO", "manual", {})
        assert row_id > 0
        rows = get_stock_pitches(symbol="WIPRO")
        assert len(rows) == 1


class TestGetStockPitches:
    """Tests for get_stock_pitches query/filter functions."""

    def test_filter_by_symbol(self, db_with_pitches):
        from src.autfin.dashboard.data import save_stock_pitch, get_stock_pitches
        save_stock_pitch("RELIANCE", "manual", {"investment_thesis": "x"})
        save_stock_pitch("TCS", "manual", {"investment_thesis": "y"})

        rows = get_stock_pitches(symbol="RELIANCE")
        assert len(rows) == 1
        assert rows[0]["symbol"] == "RELIANCE"

    def test_filter_by_context(self, db_with_pitches):
        from src.autfin.dashboard.data import save_stock_pitch, get_stock_pitches
        save_stock_pitch("RELIANCE", "morning_routine", {"investment_thesis": "m"})
        save_stock_pitch("TCS", "price_alert", {"investment_thesis": "p"})

        rows = get_stock_pitches(context="morning_routine")
        assert len(rows) == 1
        assert rows[0]["symbol"] == "RELIANCE"

    def test_filter_by_days_excludes_old(self, db_with_pitches):
        from src.autfin.dashboard.data import save_stock_pitch, get_stock_pitches
        old_ts = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
        new_ts = datetime.now(timezone.utc).isoformat()

        save_stock_pitch("OLD", "manual", {"investment_thesis": "x"}, created_at=old_ts)
        save_stock_pitch("NEW", "manual", {"investment_thesis": "y"}, created_at=new_ts)

        rows = get_stock_pitches(days=90)
        assert all(r["symbol"] == "NEW" for r in rows)

    def test_ordering_newest_first(self, db_with_pitches):
        from src.autfin.dashboard.data import save_stock_pitch, get_stock_pitches
        ts_old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        ts_new = datetime.now(timezone.utc).isoformat()

        save_stock_pitch("A", "manual", {"x": 1}, created_at=ts_old)
        save_stock_pitch("B", "manual", {"x": 2}, created_at=ts_new)

        rows = get_stock_pitches(days=30)
        assert rows[0]["symbol"] == "B"
        assert rows[1]["symbol"] == "A"

    def test_limit_respected(self, db_with_pitches):
        from src.autfin.dashboard.data import save_stock_pitch, get_stock_pitches
        for i in range(5):
            save_stock_pitch(f"SYM{i}", "manual", {"investment_thesis": f"t{i}"})
        rows = get_stock_pitches(limit=3)
        assert len(rows) == 3

    def test_handles_malformed_json_in_db(self, db_with_pitches):
        """If catalysts_json is corrupt, should not crash — returns empty list."""
        from src.autfin.dashboard.data import get_stock_pitches
        import sqlite3 as _sq
        conn = _sq.connect(__import__("pathlib").Path("DUMMY") if False else db_with_pitches)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO stock_pitches (symbol, context, catalysts_json, risks_json, raw_json) "
            "VALUES (?, ?, ?, ?, ?)",
            ("BROKEN", "manual", "{not valid json", "{also bad", "raw text"),
        )
        conn.commit()
        conn.close()

        rows = get_stock_pitches(symbol="BROKEN")
        assert len(rows) == 1
        assert rows[0]["catalysts"] == []
        assert rows[0]["risks"] == []
        assert rows[0]["raw"] == []  # parse-failed raw_json becomes empty list field


class TestGetLatestPitchesBySymbol:
    """Tests for get_latest_pitches_by_symbol."""

    def test_returns_one_per_symbol(self, db_with_pitches):
        from src.autfin.dashboard.data import save_stock_pitch, get_latest_pitches_by_symbol
        save_stock_pitch("RELIANCE", "manual", {"x": 1})
        save_stock_pitch("RELIANCE", "morning_routine", {"x": 2})
        save_stock_pitch("TCS", "manual", {"x": 3})

        rows = get_latest_pitches_by_symbol()
        symbols = {r["symbol"] for r in rows}
        assert symbols == {"RELIANCE", "TCS"}
        rel = [r for r in rows if r["symbol"] == "RELIANCE"][0]
        assert rel["pitch_count"] == 2

    def test_filter_by_symbol(self, db_with_pitches):
        from src.autfin.dashboard.data import save_stock_pitch, get_latest_pitches_by_symbol
        save_stock_pitch("A", "manual", {"x": 1})
        save_stock_pitch("B", "manual", {"x": 2})

        rows = get_latest_pitches_by_symbol(symbol="A")
        assert len(rows) == 1
        assert rows[0]["symbol"] == "A"

    def test_empty_db(self, db_with_pitches):
        from src.autfin.dashboard.data import get_latest_pitches_by_symbol
        assert get_latest_pitches_by_symbol() == []


# ===========================================================================
# 3.2 — Fan-Out Progress data layer
# ===========================================================================

class TestFanOutProgress:
    """Tests for record/start/update fan-out progress tracking."""

    def test_record_fanout_start_creates_rows(self, db_with_fanout):
        from src.autfin.dashboard.data import (
            record_fanout_start,
            get_fanout_progress,
        )
        count = record_fanout_start("sess-A", "morning_routine", ["RELIANCE", "TCS", "INFY"])
        assert count == 3

        rows = get_fanout_progress(session_id="sess-A")
        assert len(rows) == 3
        for r in rows:
            assert r["session_id"] == "sess-A"
            assert r["context"] == "morning_routine"
            assert r["status"] == "running"
            assert r["started_at"] is not None

    def test_update_fanout_status_to_completed(self, db_with_fanout):
        from src.autfin.dashboard.data import (
            record_fanout_start,
            update_fanout_status,
            get_fanout_progress,
        )
        record_fanout_start("sess-B", "price_alert", ["RELIANCE"])
        ok = update_fanout_status("sess-B", "RELIANCE", "completed")
        assert ok

        rows = get_fanout_progress(session_id="sess-B")
        assert rows[0]["status"] == "completed"
        assert rows[0]["finished_at"] is not None
        assert rows[0]["error"] is None

    def test_update_fanout_status_to_failed_records_error(self, db_with_fanout):
        from src.autfin.dashboard.data import (
            record_fanout_start,
            update_fanout_status,
            get_fanout_progress,
        )
        record_fanout_start("sess-C", "manual", ["BAD"])
        update_fanout_status("sess-C", "BAD", "failed", error="got exception X")

        rows = get_fanout_progress(session_id="sess-C")
        assert rows[0]["status"] == "failed"
        assert rows[0]["finished_at"] is not None
        assert "exception X" in rows[0]["error"]

    def test_update_without_finish_does_not_set_finished_at(self, db_with_fanout):
        from src.autfin.dashboard.data import (
            record_fanout_start,
            update_fanout_status,
            get_fanout_progress,
        )
        record_fanout_start("sess-D", "morning_routine", ["RELIANCE"])
        update_fanout_status("sess-D", "RELIANCE", "running")
        rows = get_fanout_progress(session_id="sess-D")
        assert rows[0]["finished_at"] is None

    def test_get_fanout_progress_filters_by_session(self, db_with_fanout):
        from src.autfin.dashboard.data import (
            record_fanout_start,
        )
        record_fanout_start("sess-X", "morning_routine", ["AAA"])
        record_fanout_start("sess-Y", "morning_routine", ["BBB"])

        from src.autfin.dashboard.data import get_fanout_progress
        xs = get_fanout_progress(session_id="sess-X")
        ys = get_fanout_progress(session_id="sess-Y")
        assert len(xs) == 1
        assert len(ys) == 1
        assert xs[0]["symbol"] == "AAA"
        assert ys[0]["symbol"] == "BBB"

    def test_get_fanout_progress_empty_session(self, db_with_fanout):
        from src.autfin.dashboard.data import get_fanout_progress
        rows = get_fanout_progress(session_id="nope")
        assert rows == []


class TestFanOutSummary:
    """Tests for get_fanout_summary aggregator."""

    def test_summary_counts_by_status(self, db_with_fanout):
        from src.autfin.dashboard.data import (
            record_fanout_start,
            update_fanout_status,
            get_fanout_summary,
        )
        record_fanout_start("sess-S", "morning_routine", ["A", "B", "C", "D"])
        update_fanout_status("sess-S", "A", "completed")
        update_fanout_status("sess-S", "B", "completed")
        update_fanout_status("sess-S", "C", "failed", error="x")
        # D remains running

        s = get_fanout_summary(hours=24)
        assert s["total"] == 4
        assert s["running"] == 1
        assert s["completed"] == 2
        assert s["failed"] == 1
        assert s["pending"] == 0
        assert s["completion_pct"] == 50.0

    def test_summary_pct_zero_when_empty(self, db_with_fanout):
        from src.autfin.dashboard.data import get_fanout_summary
        s = get_fanout_summary(hours=24)
        assert s["total"] == 0
        assert s["completion_pct"] == 0.0

    def test_summary_window_hours_echoed(self, db_with_fanout):
        from src.autfin.dashboard.data import get_fanout_summary
        assert get_fanout_summary(hours=12)["window_hours"] == 12


# ===========================================================================
# 3.3 — Learning Insights renderers
# ===========================================================================

class TestRenderLearningInsights:
    """Tests for the Learning Insights panel (Phase 3.3)."""

    def test_renders_with_calibration_data(self, mock_streamlit, db_with_learning):
        """Inserts calibration-eligible completed trades and renders."""
        import sqlite3
        conn = sqlite3.connect(db_with_learning)
        cur = conn.cursor()
        # high_conv wins
        cur.execute(
            "INSERT INTO learning_journal (symbol, action, conviction, pnl_pct, outcome, sector, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("A", "BUY", 0.85, 0.12, "completed", "IT", datetime.now(timezone.utc).isoformat()),
        )
        cur.execute(
            "INSERT INTO learning_journal (symbol, action, conviction, pnl_pct, outcome, sector, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("B", "BUY", 0.80, 0.18, "completed", "Banks", datetime.now(timezone.utc).isoformat()),
        )
        cur.execute(
            "INSERT INTO learning_journal (symbol, action, conviction, pnl_pct, outcome, sector, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("C", "BUY", 0.50, -0.05, "completed", "IT", datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()

        from src.autfin.dashboard.app import render_learning_insights
        render_learning_insights()
        mock_streamlit.header.assert_any_call("Learning Insights")
        # Data renders via dataframe/bar_chart; empty renders via info
        assert mock_streamlit.metric.called or mock_streamlit.info.called or mock_streamlit.dataframe.called

    def test_renders_empty_calibration_gracefully(self, mock_streamlit, db_with_learning):
        from src.autfin.dashboard.app import render_learning_insights
        render_learning_insights()
        # Should emit "No calibration data yet" info
        info_calls = [c.args[0] for c in mock_streamlit.info.call_args_list if c.args]
        assert any("No calibration" in str(c) or "No sector" in str(c) or "No patterns" in str(c)
                   for c in info_calls)

    def test_renders_empty_db_fully(self, mock_streamlit, tmp_path, monkeypatch):
        db_path = tmp_path / "empty_insights.db"
        monkeypatch.setattr("src.autfin.dashboard.data.DB_PATH", db_path)
        monkeypatch.setattr("src.autfin.memory.database.DB_PATH", db_path)
        monkeypatch.setattr("src.autfin.memory.performance.DB_PATH", db_path)
        # No tables at all
        from src.autfin.dashboard.app import render_learning_insights
        render_learning_insights()
        mock_streamlit.header.assert_called_with("Learning Insights")


# ===========================================================================
# Panel render tests for new pages
# ===========================================================================

class TestRenderStockPitches:
    """Tests for the Stock Pitches panel."""

    def test_renders_empty(self, mock_streamlit, db_with_pitches):
        from src.autfin.dashboard.app import render_stock_pitches
        render_stock_pitches()
        mock_streamlit.header.assert_called_with("Stock Pitches")

    def test_renders_with_pitch(self, mock_streamlit, db_with_pitches):
        from src.autfin.dashboard.data import save_stock_pitch, get_stock_pitches
        from src.autfin.memory.database import get_connection
        save_stock_pitch(
            "RELIANCE", "morning_routine",
            {
                "investment_thesis": "Strong moat, sector leader",
                "catalysts": ["Capacity expansion"],
                "risks": ["Commodity risk"],
                "valuation_summary": "Fair value at 22x earnings",
                "competitive_moat": "Brand and distribution",
                "price_target": {"base": 3200, "upside": 3500, "downside": 2800, "rationale": "DCF"},
            },
        )
        # Verify the pitch was saved by reading directly from the same DB
        direct = get_stock_pitches(symbol="RELIANCE")
        assert len(direct) == 1, f"Direct read failed; DB_PATH={db_with_pitches}"

        from src.autfin.dashboard.app import render_stock_pitches
        render_stock_pitches()
        mock_streamlit.header.assert_called_with("Stock Pitches")
        assert mock_streamlit.metric.called


class TestRenderFanOutProgress:
    """Tests for the Fan-Out Progress panel."""

    def test_renders_empty(self, mock_streamlit, db_with_fanout):
        from src.autfin.dashboard.app import render_fanout_progress
        render_fanout_progress()
        mock_streamlit.header.assert_called_with("Fan-Out Progress")
        # 5 KPI metrics always emitted
        assert mock_streamlit.metric.call_count >= 5

    def test_renders_with_rows(self, mock_streamlit, db_with_fanout):
        from src.autfin.dashboard.data import (
            record_fanout_start,
            update_fanout_status,
        )
        record_fanout_start("sess-F", "morning_routine", ["RELIANCE", "TCS"])
        update_fanout_status("sess-F", "RELIANCE", "completed")
        update_fanout_status("sess-F", "TCS", "failed", error="timeout")

        from src.autfin.dashboard.app import render_fanout_progress
        render_fanout_progress()
        mock_streamlit.header.assert_called_with("Fan-Out Progress")
        mock_streamlit.dataframe.assert_called()


class TestSidebarAddsNewPages:
    """Tests that the sidebar nav now includes the new pages."""

    def test_nav_includes_new_pages(self, mock_streamlit, db_with_pitches):
        mock_streamlit.radio.return_value = "Stock Pitches"
        from src.autfin.dashboard.app import render_sidebar
        page, _, _ = render_sidebar()
        # First arg of any radio call should contain the new entries
        radio_call_args = mock_streamlit.radio.call_args
        if radio_call_args:
            options = radio_call_args[0][1] if len(radio_call_args[0]) > 1 else []
            assert "Stock Pitches" in options
            assert "Fan-Out Progress" in options
            assert "Learning Insights" in options


class TestMainRoutesNewPages:
    """Tests that main() routes to the new pages."""

    def test_routes_to_stock_pitches(self, mock_streamlit, db_with_pitches):
        mock_streamlit.radio.return_value = "Stock Pitches"
        with patch("src.autfin.dashboard.app.render_stock_pitches") as mock_render:
            from src.autfin.dashboard.app import main
            main()
            mock_render.assert_called_once()

    def test_routes_to_fanout(self, mock_streamlit, db_with_fanout):
        mock_streamlit.radio.return_value = "Fan-Out Progress"
        with patch("src.autfin.dashboard.app.render_fanout_progress") as mock_render:
            from src.autfin.dashboard.app import main
            main()
            mock_render.assert_called_once()

    def test_routes_to_learning_insights(self, mock_streamlit, db_with_learning):
        mock_streamlit.radio.return_value = "Learning Insights"
        with patch("src.autfin.dashboard.app.render_learning_insights") as mock_render:
            from src.autfin.dashboard.app import main
            main()
            mock_render.assert_called_once()


# ===========================================================================
# Integration: build_stock_pitch persistence wiring
# ===========================================================================

class TestBuildStockPitchPersistence:
    """Tests that build_stock_pitch persistes via save_stock_pitch on success."""

    @pytest.mark.asyncio
    async def test_persists_pitch_with_status_success(self, monkeypatch, db_with_pitches):
        from src.autfin.tools import supervisor_tools
        from src.autfin.llm import client as llm_client_mod

        # Patch LLM client to return a JSON pitch (patch at source — get_llm_client is lazily imported)
        fake_pitch = {
            "investment_thesis": "Test thesis",
            "catalysts": ["A"],
            "risks": ["B"],
            "valuation_summary": "Fair",
            "competitive_moat": "Brand",
            "price_target": {"base": 100, "upside": 120, "downside": 90, "rationale": "x"},
        }

        class FakeClient:
            async def generate(self, system, messages, json_mode=False, max_tokens=None):
                import json
                return json.dumps(fake_pitch)

        monkeypatch.setattr(llm_client_mod, "get_llm_client", lambda: FakeClient())

        result = await supervisor_tools.build_stock_pitch("RELIANCE", {})
        assert result["status"] == "success"

        from src.autfin.dashboard.data import get_stock_pitches
        rows = get_stock_pitches(symbol="RELIANCE")
        assert len(rows) == 1
        assert rows[0]["investment_thesis"] == "Test thesis"

    @pytest.mark.asyncio
    async def test_does_not_persist_on_llm_unavailable(self, monkeypatch, db_with_pitches):
        from src.autfin.tools import supervisor_tools
        from src.autfin.llm import client as llm_client_mod
        monkeypatch.setattr(llm_client_mod, "get_llm_client", lambda: None)

        result = await supervisor_tools.build_stock_pitch("TCS", {})
        assert result["status"] == "error"

        from src.autfin.dashboard.data import get_stock_pitches
        rows = get_stock_pitches(symbol="TCS")
        assert rows == []

    @pytest.mark.asyncio
    async def test_context_propagates_to_persistence(self, monkeypatch, db_with_pitches):
        from src.autfin.tools import supervisor_tools
        from src.autfin.llm import client as llm_client_mod
        fake = {"investment_thesis": "x", "catalysts": [], "risks": [], "valuation_summary": "", "competitive_moat": "", "price_target": {}}

        class FakeClient:
            async def generate(self, system, messages, json_mode=False, max_tokens=None):
                import json
                return json.dumps(fake)

        monkeypatch.setattr(llm_client_mod, "get_llm_client", lambda: FakeClient())
        await supervisor_tools.build_stock_pitch("INFY", {"context": "price_alert"})

        from src.autfin.dashboard.data import get_stock_pitches
        rows = get_stock_pitches(symbol="INFY")
        assert rows[0]["context"] == "price_alert"
