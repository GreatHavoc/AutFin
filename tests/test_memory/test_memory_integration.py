"""
Tests for AutFin memory integration (Task 21).

Covers:
- retrieval.py: store/read functions, build_research_context, get_latest_research
- memory_tools.py: MemoryTools class
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone, timedelta


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_store():
    """Create a mock AutFinStore with in-memory dict backend."""
    from src.autfin.memory.store import AutFinStore
    return AutFinStore(mode="in-memory")


@pytest.fixture
def mock_vector_store():
    """Create a mock VectorStore."""
    vs = MagicMock()
    vs.get_collection = MagicMock(return_value=None)
    vs.client = MagicMock()
    return vs


# ===========================================================================
# Tests for retrieval.py — Store Write Operations
# ===========================================================================

class TestStoreWriteOperations:
    """Tests for store_* functions in retrieval.py."""

    @pytest.mark.asyncio
    async def test_store_research_report(self, mock_store):
        from src.autfin.memory.retrieval import store_research_report

        report = {"technical": {"rsi": 65}, "fundamentals": {"pe": 22.5}}
        await store_research_report("RELIANCE", report, store=mock_store)

        # Verify it was stored
        from src.autfin.memory.retrieval import get_research_report
        result = await get_research_report("RELIANCE", store=mock_store)
        assert result is not None
        assert result["technical"]["rsi"] == 65

    @pytest.mark.asyncio
    async def test_store_research_report_uppercase(self, mock_store):
        from src.autfin.memory.retrieval import store_research_report, get_research_report

        await store_research_report("reliance", {"score": 80}, store=mock_store)
        result = await get_research_report("RELIANCE", store=mock_store)
        assert result is not None

    @pytest.mark.asyncio
    async def test_store_research_by_type(self, mock_store):
        from src.autfin.memory.retrieval import store_research, get_research_report

        await store_research(
            ["RELIANCE", "TCS"],
            "fundamentals",
            {"RELIANCE": {"pe": 22.5}, "TCS": {"pe": 30.1}},
            store=mock_store,
        )

        # Check RELIANCE has fundamentals
        result = await get_research_report("RELIANCE", store=mock_store)
        assert result is not None
        assert "fundamentals" in result
        assert result["fundamentals"]["pe"] == 22.5

    @pytest.mark.asyncio
    async def test_store_research_merges_types(self, mock_store):
        from src.autfin.memory.retrieval import store_research, get_research_report

        # Store fundamentals first
        await store_research(
            ["RELIANCE"], "fundamentals",
            {"RELIANCE": {"pe": 22.5}},
            store=mock_store,
        )
        # Store technical next
        await store_research(
            ["RELIANCE"], "technical",
            {"RELIANCE": {"rsi": 65}},
            store=mock_store,
        )

        result = await get_research_report("RELIANCE", store=mock_store)
        assert "fundamentals" in result
        assert "technical" in result
        assert result["fundamentals"]["pe"] == 22.5
        assert result["technical"]["rsi"] == 65

    @pytest.mark.asyncio
    async def test_store_trade_decision(self, mock_store):
        from src.autfin.memory.retrieval import store_trade_decision, get_decision

        decision = {"symbol": "RELIANCE", "action": "BUY", "conviction": 0.8}
        await store_trade_decision("dec-001", decision, store=mock_store)

        result = await get_decision("dec-001", store=mock_store)
        assert result is not None
        assert result["decision"]["symbol"] == "RELIANCE"

    @pytest.mark.asyncio
    async def test_store_trade_history(self, mock_store):
        from src.autfin.memory.retrieval import store_trade_history, get_trade_history

        trade = {"action": "BUY", "quantity": 10, "price": 2500.0}
        await store_trade_history("RELIANCE", trade, store=mock_store)

        history = await get_trade_history("RELIANCE", store=mock_store)
        assert len(history) == 1
        assert history[0]["action"] == "BUY"
        assert "recorded_at" in history[0]

    @pytest.mark.asyncio
    async def test_store_trade_history_appends(self, mock_store):
        from src.autfin.memory.retrieval import store_trade_history, get_trade_history

        await store_trade_history("RELIANCE", {"action": "BUY"}, store=mock_store)
        await store_trade_history("RELIANCE", {"action": "SELL"}, store=mock_store)

        history = await get_trade_history("RELIANCE", store=mock_store)
        assert len(history) == 2
        assert history[0]["action"] == "BUY"
        assert history[1]["action"] == "SELL"

    @pytest.mark.asyncio
    async def test_store_capital_state(self, mock_store):
        from src.autfin.memory.retrieval import store_capital_state, get_capital_state

        await store_capital_state(
            {"available_cash": 50000.0, "holdings_count": 3},
            store=mock_store,
        )

        result = await get_capital_state(store=mock_store)
        assert result is not None
        assert result["available_cash"] == 50000.0
        assert "snapshot_at" in result

    @pytest.mark.asyncio
    async def test_store_portfolio_state(self, mock_store):
        from src.autfin.memory.retrieval import store_portfolio_state, get_portfolio_state

        await store_portfolio_state(
            {"holdings": [{"symbol": "RELIANCE"}], "total_pnl": 5000.0},
            store=mock_store,
        )

        result = await get_portfolio_state(store=mock_store)
        assert result is not None
        assert result["total_pnl"] == 5000.0


# ===========================================================================
# Tests for retrieval.py — Store Read Operations
# ===========================================================================

class TestStoreReadOperations:
    """Tests for get_* functions in retrieval.py."""

    @pytest.mark.asyncio
    async def test_get_research_report_missing(self, mock_store):
        from src.autfin.memory.retrieval import get_research_report
        result = await get_research_report("NONEXISTENT", store=mock_store)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_trade_history_empty(self, mock_store):
        from src.autfin.memory.retrieval import get_trade_history
        history = await get_trade_history("NONEXISTENT", store=mock_store)
        assert history == []

    @pytest.mark.asyncio
    async def test_get_capital_state_missing(self, mock_store):
        from src.autfin.memory.retrieval import get_capital_state
        result = await get_capital_state(store=mock_store)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_decision_missing(self, mock_store):
        from src.autfin.memory.retrieval import get_decision
        result = await get_decision("nonexistent", store=mock_store)
        assert result is None


# ===========================================================================
# Tests for retrieval.py — get_latest_research
# ===========================================================================

class TestGetLatestResearch:
    """Tests for freshness checking via get_latest_research."""

    @pytest.mark.asyncio
    async def test_returns_fresh_research(self, mock_store):
        from src.autfin.memory.retrieval import store_research, get_latest_research

        await store_research(
            ["RELIANCE"], "fundamentals",
            {"RELIANCE": {"pe": 22.5}},
            store=mock_store,
        )

        cached = await get_latest_research(
            ["RELIANCE"], "fundamentals", max_age_days=7, store=mock_store
        )
        assert "RELIANCE" in cached
        assert cached["RELIANCE"]["pe"] == 22.5

    @pytest.mark.asyncio
    async def test_excludes_stale_research(self, mock_store):
        from src.autfin.memory.retrieval import store_research, get_latest_research

        # Manually store with an old timestamp
        store = mock_store
        await store.aput(
            namespace="research_reports",
            key="RELIANCE",
            value={
                "stock": "RELIANCE",
                "fundamentals": {"pe": 22.5},
                "fundamentals_stored_at": (
                    datetime.now(timezone.utc) - timedelta(days=10)
                ).isoformat(),
                "stored_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        cached = await get_latest_research(
            ["RELIANCE"], "fundamentals", max_age_days=7, store=store
        )
        assert "RELIANCE" not in cached

    @pytest.mark.asyncio
    async def test_excludes_missing_type(self, mock_store):
        from src.autfin.memory.retrieval import store_research, get_latest_research

        await store_research(
            ["RELIANCE"], "technical",
            {"RELIANCE": {"rsi": 65}},
            store=mock_store,
        )

        cached = await get_latest_research(
            ["RELIANCE"], "fundamentals", max_age_days=7, store=mock_store
        )
        assert "RELIANCE" not in cached

    @pytest.mark.asyncio
    async def test_multiple_symbols(self, mock_store):
        from src.autfin.memory.retrieval import store_research, get_latest_research

        await store_research(
            ["RELIANCE", "TCS", "INFY"], "sentiment",
            {
                "RELIANCE": {"score": 0.7},
                "TCS": {"score": 0.5},
                "INFY": {"score": 0.3},
            },
            store=mock_store,
        )

        cached = await get_latest_research(
            ["RELIANCE", "TCS", "INFY"], "sentiment",
            max_age_days=7, store=mock_store,
        )
        assert len(cached) == 3


# ===========================================================================
# Tests for retrieval.py — build_research_context
# ===========================================================================

class TestBuildResearchContext:
    """Tests for composite context building."""

    @pytest.mark.asyncio
    async def test_returns_empty_context(self, mock_store):
        from src.autfin.memory.retrieval import build_research_context

        ctx = await build_research_context("RELIANCE", store=mock_store)
        assert ctx["stock"] == "RELIANCE"
        assert ctx["past_research"] is None
        assert ctx["trade_history"] == []
        assert ctx["similar_trades_learnings"] == []
        assert ctx["sector_context"] == []
        assert ctx["recent_90d_pnl"] == 0.0
        assert "No prior memory" in ctx["context_summary"]

    @pytest.mark.asyncio
    async def test_includes_past_research(self, mock_store):
        from src.autfin.memory.retrieval import store_research_report, build_research_context

        await store_research_report(
            "RELIANCE", {"technical": {"rsi": 65}}, store=mock_store
        )

        ctx = await build_research_context("RELIANCE", store=mock_store)
        assert ctx["past_research"] is not None
        assert "Past research available" in ctx["context_summary"]

    @pytest.mark.asyncio
    async def test_includes_trade_history(self, mock_store):
        from src.autfin.memory.retrieval import store_trade_history, build_research_context

        await store_trade_history(
            "RELIANCE", {"action": "BUY", "pnl": 5000}, store=mock_store
        )

        ctx = await build_research_context("RELIANCE", store=mock_store)
        assert len(ctx["trade_history"]) == 1
        assert "1 past trade(s)" in ctx["context_summary"]

    @pytest.mark.asyncio
    async def test_computes_recent_pnl(self, mock_store):
        from src.autfin.memory.retrieval import store_trade_history, build_research_context

        # Store a trade with pnl
        await store_trade_history(
            "RELIANCE",
            {"action": "BUY", "pnl_pct": 5000.0},
            store=mock_store,
        )

        ctx = await build_research_context("RELIANCE", store=mock_store)
        assert ctx["recent_90d_pnl"] == 5000.0
        assert "profit" in ctx["context_summary"]

    @pytest.mark.asyncio
    async def test_computes_negative_pnl(self, mock_store):
        from src.autfin.memory.retrieval import store_trade_history, build_research_context

        await store_trade_history(
            "RELIANCE",
            {"action": "SELL", "pnl_pct": -3000.0},
            store=mock_store,
        )

        ctx = await build_research_context("RELIANCE", store=mock_store)
        assert ctx["recent_90d_pnl"] == -3000.0
        assert "loss" in ctx["context_summary"]


# ===========================================================================
# Tests for retrieval.py — _compute_recent_pnl
# ===========================================================================

class TestComputeRecentPnl:
    """Tests for the P&L computation helper."""

    def test_empty_trades(self):
        from src.autfin.memory.retrieval import _compute_recent_pnl
        assert _compute_recent_pnl([]) == 0.0

    def test_no_pnl_fields(self):
        from src.autfin.memory.retrieval import _compute_recent_pnl
        trades = [{"action": "BUY", "recorded_at": datetime.now(timezone.utc).isoformat()}]
        assert _compute_recent_pnl(trades) == 0.0

    def test_with_pnl(self):
        from src.autfin.memory.retrieval import _compute_recent_pnl
        trades = [
            {"pnl_pct": 5000.0, "recorded_at": datetime.now(timezone.utc).isoformat()},
            {"pnl_pct": -2000.0, "recorded_at": datetime.now(timezone.utc).isoformat()},
        ]
        assert _compute_recent_pnl(trades) == 3000.0

    def test_old_trades_excluded(self):
        from src.autfin.memory.retrieval import _compute_recent_pnl
        old_date = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        trades = [{"pnl_pct": 5000.0, "recorded_at": old_date}]
        assert _compute_recent_pnl(trades, days=90) == 0.0


# ===========================================================================
# Tests for memory_tools.py — MemoryTools class
# ===========================================================================

class TestMemoryTools:
    """Tests for the MemoryTools wrapper class."""

    @pytest.mark.asyncio
    async def test_get_cached_research(self):
        from src.autfin.tools.memory_tools import MemoryTools
        from src.autfin.memory.retrieval import store_research

        mt = MemoryTools()
        await store_research(
            ["RELIANCE"], "fundamentals",
            {"RELIANCE": {"pe": 22.5}},
            store=mt.store,
        )

        cached = await mt.get_cached_research(["RELIANCE"], "fundamentals")
        assert "RELIANCE" in cached

    @pytest.mark.asyncio
    async def test_save_and_get_research(self):
        from src.autfin.tools.memory_tools import MemoryTools

        mt = MemoryTools()
        await mt.save_research_report("TCS", {"technical": {"rsi": 70}})
        result = await mt.get_research("TCS")
        assert result is not None
        assert result["technical"]["rsi"] == 70

    @pytest.mark.asyncio
    async def test_save_and_get_trades(self):
        from src.autfin.tools.memory_tools import MemoryTools

        mt = MemoryTools()
        await mt.save_trade("INFY", {"action": "BUY", "qty": 5})
        trades = await mt.get_trades("INFY")
        assert len(trades) == 1
        assert trades[0]["action"] == "BUY"

    @pytest.mark.asyncio
    async def test_save_and_get_decision(self):
        from src.autfin.tools.memory_tools import MemoryTools

        mt = MemoryTools()
        await mt.save_decision("d-001", {"symbol": "RELIANCE", "action": "BUY"})
        result = await mt.get_decision("d-001")
        assert result is not None

    @pytest.mark.asyncio
    async def test_save_and_get_capital(self):
        from src.autfin.tools.memory_tools import MemoryTools

        mt = MemoryTools()
        await mt.save_capital({"available_cash": 75000.0})
        result = await mt.get_capital()
        assert result["available_cash"] == 75000.0

    @pytest.mark.asyncio
    async def test_save_and_get_portfolio(self):
        from src.autfin.tools.memory_tools import MemoryTools

        mt = MemoryTools()
        await mt.save_portfolio({"holdings": [], "total_pnl": 1000.0})
        result = await mt.get_portfolio()
        assert result["total_pnl"] == 1000.0

    @pytest.mark.asyncio
    async def test_build_context(self):
        from src.autfin.tools.memory_tools import MemoryTools

        mt = MemoryTools()
        await mt.save_research_report("RELIANCE", {"score": 80})
        ctx = await mt.build_context("RELIANCE")
        assert ctx["stock"] == "RELIANCE"
        assert ctx["past_research"] is not None

    def test_vector_store_operations_return_none_without_vs(self):
        from src.autfin.tools.memory_tools import MemoryTools

        mt = MemoryTools()
        assert mt.save_strategy_learning("test", {}) is None
        assert mt.save_company_knowledge("test", {}) is None
        assert mt.save_market_knowledge("test", {}) is None
