"""
Tests for performance tracking module (v3.0 schema).

Covers learning_journal CRUD, performance metrics computation,
outcome backfill, and MemoryTools integration.
"""

import os
import sqlite3
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# Use a temp database for all tests
_test_db_dir = tempfile.mkdtemp()
_test_db_path = Path(_test_db_dir) / "test_performance.db"


@pytest.fixture(autouse=True)
def _patch_db_path(monkeypatch):
    """Redirect database to a temp file for every test."""
    monkeypatch.setattr(
        "src.autfin.memory.database.DB_PATH", _test_db_path
    )
    if _test_db_path.exists():
        _test_db_path.unlink()
    from src.autfin.memory.database import init_database
    init_database()
    yield
    if _test_db_path.exists():
        _test_db_path.unlink()


# ---------------------------------------------------------------------------
# Helper to insert test data
# ---------------------------------------------------------------------------

def _insert_learning(symbol, action="BUY", conviction=0.5, pnl_pct=None,
                     thesis_summary=None, outcome="open", sector=None, created_at=None):
    """Insert a learning journal row directly (v3.0 columns)."""
    conn = sqlite3.connect(_test_db_path)
    conn.execute(
        """
        INSERT INTO learning_journal
            (symbol, action, conviction, pnl_pct, thesis_summary, outcome, sector, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (symbol, action, conviction, pnl_pct, thesis_summary, outcome, sector, created_at),
    )
    conn.commit()
    conn.close()


def _insert_snapshot(date, cash=100000, margin=0, holdings_val=50000, pnl=5000, pnl_pct=5.0):
    """Insert a portfolio snapshot row directly."""
    conn = sqlite3.connect(_test_db_path)
    conn.execute(
        """
        INSERT INTO portfolio_snapshots (date, available_cash, utilized_margin, total_holdings_value, total_pnl, total_pnl_pct, holdings_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (date, cash, margin, holdings_val, pnl, pnl_pct, "[]"),
    )
    conn.commit()
    conn.close()


# ===========================================================================
# Test class 1: learning_journal CRUD
# ===========================================================================

class TestLearningJournalCRUD:
    """Test record_learning, get_learnings_for_symbol, update_learning_outcome, etc."""

    def test_record_learning_returns_row_id(self):
        from src.autfin.memory.performance import record_learning
        row_id = record_learning("RELIANCE", pnl_pct=0.15, outcome="completed", thesis_summary="Good entry")
        assert row_id == 1

    def test_record_learning_stores_correctly(self):
        from src.autfin.memory.performance import record_learning, get_learnings_for_symbol
        record_learning("INFY", pnl_pct=-0.05, outcome="loss", thesis_summary="Bad timing")
        entries = get_learnings_for_symbol("INFY")
        assert len(entries) == 1
        entry = entries[0]
        assert entry["symbol"] == "INFY"
        assert entry["pnl_pct"] == -0.05
        assert entry["outcome"] == "loss"
        assert entry["thesis_summary"] == "Bad timing"

    def test_record_learning_uppercases_symbol(self):
        from src.autfin.memory.performance import record_learning, get_learnings_for_symbol
        record_learning("tcs", pnl_pct=0.02, outcome="completed")
        entries = get_learnings_for_symbol("TCS")
        assert len(entries) == 1
        assert entries[0]["symbol"] == "TCS"

    def test_get_learning_missing_returns_none(self):
        from src.autfin.memory.performance import get_learnings_for_symbol
        assert get_learnings_for_symbol("NONEXISTENT") == []

    def test_get_learnings_for_symbol(self):
        from src.autfin.memory.performance import record_learning, get_learnings_for_symbol
        record_learning("RELIANCE", pnl_pct=0.10, outcome="completed")
        record_learning("RELIANCE", pnl_pct=-0.05, outcome="loss")
        record_learning("INFY", pnl_pct=0.20, outcome="completed")

        results = get_learnings_for_symbol("RELIANCE")
        assert len(results) == 2
        # Should be ordered by created_at DESC
        assert results[0]["created_at"] >= results[1]["created_at"]

    def test_get_learnings_for_symbol_limit(self):
        from src.autfin.memory.performance import record_learning, get_learnings_for_symbol
        for i in range(5):
            record_learning("RELIANCE", pnl_pct=0.10, outcome="completed")
        results = get_learnings_for_symbol("RELIANCE", limit=3)
        assert len(results) == 3

    def test_get_recent_learnings(self):
        from src.autfin.memory.performance import record_learning, get_recent_learnings
        old_date = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%d %H:%M:%S")
        _insert_learning("RELIANCE", pnl_pct=0.10, outcome="completed", created_at=old_date)
        row_id = record_learning("INFY", pnl_pct=0.20, outcome="completed")

        recent = get_recent_learnings(days=30)
        assert len(recent) == 1
        assert recent[0]["symbol"] == "INFY"

    def test_update_learning_outcome(self):
        from src.autfin.memory.performance import record_learning, update_learning_outcome, get_learnings_for_symbol
        row_id = record_learning("RELIANCE")
        updated = update_learning_outcome(row_id, 0.05, "completed", "Great trade!")
        assert updated is True
        entries = get_learnings_for_symbol("RELIANCE")
        assert len(entries) == 1
        entry = entries[0]
        assert entry["pnl_pct"] == 0.05
        assert entry["outcome"] == "completed"
        assert entry["thesis_summary"] == "Great trade!"

    def test_update_learning_without_lesson(self):
        from src.autfin.memory.performance import record_learning, update_learning_outcome, get_learnings_for_symbol
        row_id = record_learning("TCS", pnl_pct=0.01, outcome="completed")
        update_learning_outcome(row_id, 0.05, "completed")
        entries = get_learnings_for_symbol("TCS")
        assert len(entries) == 1
        entry = entries[0]
        assert entry["pnl_pct"] == 0.05
        # thesis_summary should remain unchanged (None from original insert)
        assert entry["thesis_summary"] is None

    def test_update_learning_nonexistent_returns_false(self):
        from src.autfin.memory.performance import update_learning_outcome
        assert update_learning_outcome(99999, 0.0, "loss") is False


# ===========================================================================
# Test class 2: Performance metrics
# ===========================================================================

class TestPerformanceMetrics:
    """Test get_win_rate, get_profit_factor, get_symbol_performance, etc."""

    def test_win_rate_empty(self):
        from src.autfin.memory.performance import get_win_rate
        result = get_win_rate()
        assert result["total_trades"] == 0
        assert result["win_rate"] == 0.0

    def test_win_rate_with_data(self):
        from src.autfin.memory.performance import record_learning, get_win_rate
        # 3 wins, 2 losses
        record_learning("RELIANCE", pnl_pct=0.10, outcome="completed")
        record_learning("INFY", pnl_pct=0.05, outcome="completed")
        record_learning("TCS", pnl_pct=0.02, outcome="completed")
        record_learning("RELIANCE", pnl_pct=-0.03, outcome="loss")
        record_learning("INFY", pnl_pct=-0.01, outcome="loss")

        result = get_win_rate()
        assert result["total_trades"] == 5
        assert result["wins"] == 3
        assert result["losses"] == 2
        assert result["win_rate"] == 0.6
        assert result["total_pnl"] == 0.13
        assert result["avg_win"] == pytest.approx(0.05667, rel=1e-3)
        assert result["avg_loss"] == -0.02

    def test_win_rate_with_days_filter(self):
        from src.autfin.memory.performance import record_learning, get_win_rate
        old = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%d %H:%M:%S")
        _insert_learning("RELIANCE", pnl_pct=0.10, outcome="completed", created_at=old)
        record_learning("INFY", pnl_pct=0.05, outcome="completed")

        result = get_win_rate(days=30)
        assert result["total_trades"] == 1

    def test_profit_factor_empty(self):
        from src.autfin.memory.performance import get_profit_factor
        result = get_profit_factor()
        assert result["profit_factor"] == 0.0

    def test_profit_factor_with_data(self):
        from src.autfin.memory.performance import record_learning, get_profit_factor
        record_learning("RELIANCE", pnl_pct=0.30, outcome="completed")
        record_learning("INFY", pnl_pct=-0.10, outcome="loss")

        result = get_profit_factor()
        assert result["gross_profit"] == 0.30
        assert result["gross_loss"] == 0.10
        assert result["profit_factor"] == pytest.approx(3.0)

    def test_profit_factor_all_wins(self):
        from src.autfin.memory.performance import record_learning, get_profit_factor
        record_learning("RELIANCE", pnl_pct=0.30, outcome="completed")
        result = get_profit_factor()
        assert result["profit_factor"] == float("inf")

    def test_symbol_performance(self):
        from src.autfin.memory.performance import record_learning, get_symbol_performance
        record_learning("RELIANCE", pnl_pct=0.10, outcome="completed")
        record_learning("RELIANCE", pnl_pct=-0.02, outcome="loss")
        record_learning("INFY", pnl_pct=0.05, outcome="completed")

        results = get_symbol_performance()
        assert len(results) == 2
        rel = next(r for r in results if r["symbol"] == "RELIANCE")
        assert rel["total_trades"] == 2
        assert rel["win_rate"] == 0.5
        assert rel["total_pnl"] == 0.08

    def test_symbol_performance_filtered(self):
        from src.autfin.memory.performance import record_learning, get_symbol_performance
        record_learning("RELIANCE", pnl_pct=0.10, outcome="completed")
        record_learning("INFY", pnl_pct=0.05, outcome="completed")

        results = get_symbol_performance(symbol="RELIANCE")
        assert len(results) == 1
        assert results[0]["symbol"] == "RELIANCE"

    def test_monthly_summary_empty(self):
        from src.autfin.memory.performance import get_monthly_summary
        result = get_monthly_summary(2026, 1)
        assert result["total_trades"] == 0
        assert result["year"] == 2026
        assert result["month"] == 1

    def test_monthly_summary_with_data(self):
        from src.autfin.memory.performance import record_learning, get_monthly_summary
        record_learning("RELIANCE", pnl_pct=0.10, outcome="completed")
        record_learning("INFY", pnl_pct=-0.05, outcome="loss")
        # Both records created now (June 2026)
        month = datetime.now(timezone.utc).month
        year = datetime.now(timezone.utc).year

        result = get_monthly_summary(year, month)
        assert result["total_trades"] == 2
        assert result["wins"] == 1
        assert result["losses"] == 1
        assert result["win_rate"] == 0.5
        assert result["total_pnl"] == 0.05

    def test_get_cumulative_pnl_empty(self):
        from src.autfin.memory.performance import get_cumulative_pnl
        assert get_cumulative_pnl() == []

    def test_get_cumulative_pnl_with_data(self):
        from src.autfin.memory.performance import get_cumulative_pnl
        _insert_snapshot("2026-06-01", pnl=1000, pnl_pct=1.0)
        _insert_snapshot("2026-06-02", pnl=1500, pnl_pct=1.5)

        result = get_cumulative_pnl()
        assert len(result) == 2
        assert result[0]["total_pnl"] == 1000
        assert result[1]["total_pnl"] == 1500

    def test_get_drawdown_stats_empty(self):
        from src.autfin.memory.performance import get_drawdown_stats
        result = get_drawdown_stats()
        assert result["max_drawdown"] == 0.0

    def test_get_drawdown_stats_with_data(self):
        from src.autfin.memory.performance import get_drawdown_stats
        _insert_snapshot("2026-06-01", pnl=1000, pnl_pct=1.0)
        _insert_snapshot("2026-06-02", pnl=1500, pnl_pct=1.5)  # peak
        _insert_snapshot("2026-06-03", pnl=800, pnl_pct=0.8)   # drawdown
        _insert_snapshot("2026-06-04", pnl=1200, pnl_pct=1.2)  # recovery

        result = get_drawdown_stats()
        assert result["max_drawdown"] == 700.0  # 1500 - 800
        assert result["peak_date"] == "2026-06-02"
        assert result["trough_date"] == "2026-06-03"

    def test_get_all_performance_metrics(self):
        from src.autfin.memory.performance import record_learning, get_all_performance_metrics
        record_learning("RELIANCE", pnl_pct=0.10, outcome="completed")
        record_learning("INFY", pnl_pct=-0.05, outcome="loss")

        result = get_all_performance_metrics()
        assert "win_rate" in result
        assert "profit_factor" in result
        assert "symbol_performance" in result
        assert "drawdown" in result
        assert result["win_rate"]["total_trades"] == 2


# ===========================================================================
# Test class 3: Outcome backfill
# ===========================================================================

class TestOutcomeBackfill:
    """Test record_trade_outcome function."""

    def test_buy_outcome_profit(self):
        from src.autfin.memory.performance import record_trade_outcome, get_learnings_for_symbol
        result = record_trade_outcome("RELIANCE", 2400.0, 2600.0, 10, "BUY")
        assert result["pnl_pct"] == 2000.0  # (2600-2400)*10
        assert result["outcome"] == "completed"
        assert result["symbol"] == "RELIANCE"

        entries = get_learnings_for_symbol("RELIANCE")
        assert len(entries) == 1
        assert entries[0]["pnl_pct"] == 2000.0

    def test_buy_outcome_loss(self):
        from src.autfin.memory.performance import record_trade_outcome
        result = record_trade_outcome("INFY", 1500.0, 1400.0, 5, "BUY")
        assert result["pnl_pct"] == -500.0  # (1400-1500)*5
        assert result["outcome"] == "loss"

    def test_sell_outcome_profit(self):
        from src.autfin.memory.performance import record_trade_outcome
        result = record_trade_outcome("TCS", 3000.0, 2800.0, 5, "SELL")
        assert result["pnl_pct"] == 1000.0  # (3000-2800)*5
        assert result["outcome"] == "completed"

    def test_sell_outcome_loss(self):
        from src.autfin.memory.performance import record_trade_outcome
        result = record_trade_outcome("HDFC", 1600.0, 1700.0, 3, "SELL")
        assert result["pnl_pct"] == -300.0  # (1600-1700)*3
        assert result["outcome"] == "loss"

    def test_outcome_with_lesson(self):
        from src.autfin.memory.performance import record_trade_outcome, get_learnings_for_symbol
        record_trade_outcome("RELIANCE", 2400.0, 2600.0, 10, "BUY", thesis_summary="Good entry timing")
        entries = get_learnings_for_symbol("RELIANCE")
        assert len(entries) == 1
        assert entries[0]["thesis_summary"] == "Good entry timing"


# ===========================================================================
# Test class 4: MemoryTools integration
# ===========================================================================

class TestMemoryToolsPerformance:
    """Test MemoryTools performance tracking methods."""

    def _make_tools(self):
        from src.autfin.tools.memory_tools import MemoryTools
        return MemoryTools(store=MagicMock(), vector_store=None)

    def test_get_performance_metrics(self):
        tools = self._make_tools()
        result = tools.get_performance_metrics()
        assert "win_rate" in result
        assert "profit_factor" in result

    def test_get_win_rate(self):
        tools = self._make_tools()
        result = tools.get_win_rate()
        assert "total_trades" in result
        assert "win_rate" in result

    def test_get_strategy_comparison(self):
        tools = self._make_tools()
        result = tools.get_strategy_comparison()
        assert isinstance(result, list)

    def test_record_trade_outcome(self):
        tools = self._make_tools()
        result = tools.record_trade_outcome(
            symbol="RELIANCE",
            entry_price=2400.0,
            exit_price=2600.0,
            quantity=10,
            action="BUY",
            thesis_summary="Good trade",
        )
        assert result["pnl_pct"] == 2000.0
        assert result["outcome"] == "completed"

    def test_get_monthly_summary(self):
        tools = self._make_tools()
        result = tools.get_monthly_summary(2026, 6)
        assert result["year"] == 2026
        assert result["month"] == 6

    def test_get_drawdown_stats(self):
        tools = self._make_tools()
        result = tools.get_drawdown_stats()
        assert "max_drawdown" in result
        assert "current_drawdown" in result


# ===========================================================================
# Test class 5: build_research_context integration
# ===========================================================================

class TestBuildResearchContextPerformance:
    """Test that build_research_context includes performance metrics."""

    @pytest.mark.asyncio
    async def test_context_includes_performance(self):
        from src.autfin.memory.performance import record_learning
        from src.autfin.memory.retrieval import build_research_context

        record_learning("RELIANCE", pnl_pct=0.10, outcome="completed")
        record_learning("RELIANCE", pnl_pct=-0.02, outcome="loss")

        ctx = await build_research_context("RELIANCE")

        assert "performance" in ctx
        perf = ctx["performance"]
        assert perf["total_trades"] == 2
        assert perf["wins"] == 1
        assert perf["win_rate"] == 0.5
        assert perf["total_pnl"] == 0.08

    @pytest.mark.asyncio
    async def test_context_performance_empty_when_no_data(self):
        from src.autfin.memory.retrieval import build_research_context

        ctx = await build_research_context("TCS")
        perf = ctx["performance"]
        assert perf["total_trades"] == 0
        assert perf["win_rate"] == 0.0


# ===========================================================================
# Test class 6: Edge cases
# ===========================================================================

class TestPerformanceEdgeCases:
    """Test edge cases and error handling."""

    def test_win_rate_all_pending_no_outcomes(self):
        """Trades with NULL outcome should not be counted toward win_rate."""
        from src.autfin.memory.performance import record_learning, get_win_rate
        record_learning("RELIANCE", outcome=None)  # NULL outcome = pending
        record_learning("INFY", pnl_pct=0.05, outcome="completed")

        result = get_win_rate()
        assert result["total_trades"] == 1  # only the one with non-NULL outcome

    def test_symbol_performance_empty_when_no_data(self):
        from src.autfin.memory.performance import get_symbol_performance
        assert get_symbol_performance() == []

    def test_profit_factor_zero_when_no_pnl(self):
        from src.autfin.memory.performance import record_learning, get_profit_factor
        record_learning("RELIANCE")  # pnl_pct defaults to None
        result = get_profit_factor()
        assert result["profit_factor"] == 0.0
