"""
Tests for auto-execute small trades (Task 25).

End-to-end integration tests verifying the full auto-small execution pipeline:
  - executor_agent auto-small mode with mocked KiteTools
  - trade_gate pass-through for small trades in auto-small mode
"""
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trade(**overrides) -> dict[str, Any]:
    """A valid BUY LIMIT trade. Override any field per-test."""
    base = {
        "symbol": "RELIANCE",
        "action": "BUY",
        "qty": 2,
        "price": 2500.0,
        "order_type": "LIMIT",
        "position_size": 5000.0,
        "conviction": 0.8,
        "rationale": "Bullish breakout above resistance",
        "risk_approved": True,
        "decision_id": "dec-auto-001",
    }
    base.update(overrides)
    return base


def _base_state(**overrides) -> dict[str, Any]:
    """Minimal AgentState dict."""
    state: dict[str, Any] = {
        "thread_id": "test-auto-001",
        "messages": [],
        "available_cash": 30000.0,
        "holdings": [],
        "positions": [],
        "research_debates": {},
        "risk_assessments": {},
        "trade_queue": [],
        "execution_reports": [],
        "errors": [],
        "metadata": {},
    }
    state.update(overrides)
    return state


def _holding(symbol: str, qty: int, avg_price: float = 100.0) -> dict[str, Any]:
    return {"tradingsymbol": symbol, "quantity": qty, "average_price": avg_price}


def _config(configurable: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"configurable": configurable or {}}


def _patch_kite(monkeypatch, place_order_result: dict[str, Any] | None = None):
    """Patch KiteTools for auto-small tests."""
    from src.autfin.tools.kite_tools import KiteTools

    if place_order_result is None:
        place_order_result = {"status": "success", "order_id": "AUTO999"}

    instance = MagicMock()
    place_method = AsyncMock(return_value=place_order_result)
    instance.place_order = place_method

    monkeypatch.setattr(KiteTools, "__new__", lambda cls, *a, **kw: instance)
    return instance, place_method


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Clear environment so .env-loaded defaults don't leak between tests."""
    for key in [
        "EXECUTION_MODE",
        "AUTO_EXECUTE_MAX_AMOUNT",
        "KITE_USER_ID",
        "KITE_PASSWORD",
        "KITE_TOTP_SECRET",
        "MIN_TRADE_AMOUNT",
    ]:
        monkeypatch.delenv(key, raising=False)
    yield


# ═══════════════════════════════════════════════════════════════════
# 1. Executor auto-small — full pipeline tests
# ═══════════════════════════════════════════════════════════════════

class TestAutoSmallExecutionPipeline:
    """End-to-end executor tests in auto-small mode."""

    @pytest.mark.asyncio
    async def test_buy_under_threshold_executes(self, monkeypatch):
        """BUY trade under auto_execute_max_amount → status=executed."""
        monkeypatch.setenv("EXECUTION_MODE", "auto-small")
        monkeypatch.setenv("KITE_USER_ID", "test123")
        monkeypatch.setenv("KITE_PASSWORD", "testpass")
        monkeypatch.setenv("KITE_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
        monkeypatch.setenv("AUTO_EXECUTE_MAX_AMOUNT", "10000.0")
        instance, place_method = _patch_kite(
            monkeypatch, place_order_result={"status": "success", "order_id": "BUY001"}
        )

        from src.autfin.agents.executor import executor_agent

        state = _base_state(
            trade_queue=[_trade(qty=2, price=2500.0)],  # ₹5,000 < ₹10,000
            available_cash=30000.0,
        )
        result = await executor_agent(state, _config())

        assert result["execution_reports"][0]["status"] == "executed"
        assert result["execution_reports"][0]["order_id"] == "BUY001"
        assert result["execution_reports"][0]["mode"] == "auto-small"
        place_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_sell_under_threshold_executes(self, monkeypatch):
        """SELL trade under threshold → status=executed with sufficient holdings."""
        monkeypatch.setenv("EXECUTION_MODE", "auto-small")
        monkeypatch.setenv("KITE_USER_ID", "test123")
        monkeypatch.setenv("KITE_PASSWORD", "testpass")
        monkeypatch.setenv("KITE_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
        monkeypatch.setenv("AUTO_EXECUTE_MAX_AMOUNT", "50000.0")
        instance, place_method = _patch_kite(
            monkeypatch, place_order_result={"status": "success", "order_id": "SELL001"}
        )

        from src.autfin.agents.executor import executor_agent

        state = _base_state(
            trade_queue=[_trade(symbol="TCS", action="SELL", qty=5, price=3500.0)],  # ₹17,500
            holdings=[_holding("TCS", 10)],
        )
        result = await executor_agent(state, _config())

        assert result["execution_reports"][0]["status"] == "executed"
        assert result["execution_reports"][0]["order_id"] == "SELL001"
        place_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_trade_at_threshold_executes(self, monkeypatch):
        """Trade value exactly at threshold → should execute (not >)."""
        monkeypatch.setenv("EXECUTION_MODE", "auto-small")
        monkeypatch.setenv("KITE_USER_ID", "test123")
        monkeypatch.setenv("KITE_PASSWORD", "testpass")
        monkeypatch.setenv("KITE_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
        monkeypatch.setenv("AUTO_EXECUTE_MAX_AMOUNT", "5000.0")
        instance, place_method = _patch_kite(
            monkeypatch, place_order_result={"status": "success", "order_id": "EXACT01"}
        )

        from src.autfin.agents.executor import executor_agent

        # ₹5,000 == threshold, should NOT trigger downgrade
        state = _base_state(
            trade_queue=[_trade(qty=5, price=1000.0)],
            available_cash=10000.0,
        )
        result = await executor_agent(state, _config())

        assert result["execution_reports"][0]["status"] == "executed"
        place_method.assert_called_once()

    @pytest.mark.asyncio
    async def test_trade_above_threshold_downgrades(self, monkeypatch):
        """Trade value above threshold → downgrades to recommendation."""
        monkeypatch.setenv("EXECUTION_MODE", "auto-small")
        monkeypatch.setenv("KITE_USER_ID", "test123")
        monkeypatch.setenv("KITE_PASSWORD", "testpass")
        monkeypatch.setenv("KITE_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
        monkeypatch.setenv("AUTO_EXECUTE_MAX_AMOUNT", "4000.0")
        instance, place_method = _patch_kite(monkeypatch)

        from src.autfin.agents.executor import executor_agent

        # ₹5,000 > ₹4,000 threshold
        state = _base_state(
            trade_queue=[_trade(qty=2, price=2500.0)],
            available_cash=30000.0,
        )
        result = await executor_agent(state, _config())

        assert result["execution_reports"][0]["status"] == "recommended"
        assert "exceeds" in result["execution_reports"][0]["message"].lower()
        place_method.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_trades_mixed_sizes(self, monkeypatch):
        """Multiple trades: some under threshold execute, some above downgrade."""
        monkeypatch.setenv("EXECUTION_MODE", "auto-small")
        monkeypatch.setenv("KITE_USER_ID", "test123")
        monkeypatch.setenv("KITE_PASSWORD", "testpass")
        monkeypatch.setenv("KITE_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
        monkeypatch.setenv("AUTO_EXECUTE_MAX_AMOUNT", "6000.0")
        instance, place_method = _patch_kite(
            monkeypatch, place_order_result={"status": "success", "order_id": "MIX001"}
        )

        from src.autfin.agents.executor import executor_agent

        state = _base_state(
            trade_queue=[
                _trade(qty=2, price=2500.0),         # ₹5,000 → executes
                _trade(symbol="TCS", qty=3, price=3000.0),  # ₹9,000 → downgrades
            ],
            available_cash=50000.0,
            holdings=[_holding("TCS", 10)],
        )
        result = await executor_agent(state, _config())

        reports = result["execution_reports"]
        assert len(reports) == 2
        assert reports[0]["status"] == "executed"
        assert reports[1]["status"] == "recommended"
        # Only the small trade called KiteTools
        assert place_method.call_count == 1

    @pytest.mark.asyncio
    async def test_market_order_auto_executes(self, monkeypatch):
        """MARKET order in auto-small mode → price=0 passed to KiteTools."""
        monkeypatch.setenv("EXECUTION_MODE", "auto-small")
        monkeypatch.setenv("KITE_USER_ID", "test123")
        monkeypatch.setenv("KITE_PASSWORD", "testpass")
        monkeypatch.setenv("KITE_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
        monkeypatch.setenv("AUTO_EXECUTE_MAX_AMOUNT", "10000.0")
        instance, place_method = _patch_kite(
            monkeypatch, place_order_result={"status": "success", "order_id": "MKT001"}
        )

        from src.autfin.agents.executor import executor_agent

        state = _base_state(
            trade_queue=[_trade(order_type="MARKET", qty=2, price=2500.0)],
            available_cash=30000.0,
        )
        result = await executor_agent(state, _config())

        assert result["execution_reports"][0]["status"] == "executed"
        # MARKET order should pass price=0
        call_kwargs = place_method.call_args[1]
        assert call_kwargs["price"] == 0.0

    @pytest.mark.asyncio
    async def test_insufficient_cash_downgrades_buy(self, monkeypatch):
        """BUY with insufficient cash → downgrade to recommendation."""
        monkeypatch.setenv("EXECUTION_MODE", "auto-small")
        monkeypatch.setenv("KITE_USER_ID", "test123")
        monkeypatch.setenv("KITE_PASSWORD", "testpass")
        monkeypatch.setenv("KITE_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
        monkeypatch.setenv("AUTO_EXECUTE_MAX_AMOUNT", "100000.0")
        instance, place_method = _patch_kite(monkeypatch)

        from src.autfin.agents.executor import executor_agent

        state = _base_state(
            trade_queue=[_trade(qty=50, price=1000.0)],  # ₹50,000 but only ₹30,000 cash
            available_cash=30000.0,
        )
        result = await executor_agent(state, _config())

        assert result["execution_reports"][0]["status"] == "recommended"
        assert "funds" in result["execution_reports"][0]["message"].lower()
        place_method.assert_not_called()

    @pytest.mark.asyncio
    async def test_insufficient_holdings_downgrades_sell(self, monkeypatch):
        """SELL with insufficient holdings → downgrade to recommendation."""
        monkeypatch.setenv("EXECUTION_MODE", "auto-small")
        monkeypatch.setenv("KITE_USER_ID", "test123")
        monkeypatch.setenv("KITE_PASSWORD", "testpass")
        monkeypatch.setenv("KITE_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
        monkeypatch.setenv("AUTO_EXECUTE_MAX_AMOUNT", "100000.0")
        instance, place_method = _patch_kite(monkeypatch)

        from src.autfin.agents.executor import executor_agent

        state = _base_state(
            trade_queue=[_trade(symbol="INFY", action="SELL", qty=10, price=1500.0)],
            holdings=[_holding("INFY", 3)],  # Only 3, need 10
        )
        result = await executor_agent(state, _config())

        assert result["execution_reports"][0]["status"] == "recommended"
        assert "holdings" in result["execution_reports"][0]["message"].lower()
        place_method.assert_not_called()

    @pytest.mark.asyncio
    async def test_kite_error_sets_error_status(self, monkeypatch):
        """KiteTools returns failure → status=error."""
        monkeypatch.setenv("EXECUTION_MODE", "auto-small")
        monkeypatch.setenv("KITE_USER_ID", "test123")
        monkeypatch.setenv("KITE_PASSWORD", "testpass")
        monkeypatch.setenv("KITE_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
        monkeypatch.setenv("AUTO_EXECUTE_MAX_AMOUNT", "10000.0")
        instance, place_method = _patch_kite(
            monkeypatch, place_order_result={"status": "error", "message": "Insufficient margin"}
        )

        from src.autfin.agents.executor import executor_agent

        state = _base_state(
            trade_queue=[_trade(qty=2, price=2500.0)],
            available_cash=30000.0,
        )
        result = await executor_agent(state, _config())

        assert result["execution_reports"][0]["status"] == "error"
        assert "insufficient margin" in result["execution_reports"][0]["message"].lower()
        assert len(result["errors"]) >= 1

    @pytest.mark.asyncio
    async def test_kite_exception_graceful(self, monkeypatch):
        """KiteTools raises exception → status=error, no crash."""
        monkeypatch.setenv("EXECUTION_MODE", "auto-small")
        monkeypatch.setenv("KITE_USER_ID", "test123")
        monkeypatch.setenv("KITE_PASSWORD", "testpass")
        monkeypatch.setenv("KITE_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
        monkeypatch.setenv("AUTO_EXECUTE_MAX_AMOUNT", "10000.0")
        instance, place_method = _patch_kite(monkeypatch)
        place_method.side_effect = Exception("Network timeout")

        from src.autfin.agents.executor import executor_agent

        state = _base_state(
            trade_queue=[_trade(qty=2, price=2500.0)],
            available_cash=30000.0,
        )
        result = await executor_agent(state, _config())

        assert result["execution_reports"][0]["status"] == "error"
        assert "network timeout" in result["execution_reports"][0]["message"].lower()

    @pytest.mark.asyncio
    async def test_self_hosted_false_downgrades(self, monkeypatch):
        """kite_auto_login_configured=False → auto-small cannot place real orders."""
        monkeypatch.setenv("EXECUTION_MODE", "auto-small")
        monkeypatch.setenv("KITE_USER_ID", "")
        monkeypatch.setenv("KITE_PASSWORD", "")
        monkeypatch.setenv("KITE_TOTP_SECRET", "")
        monkeypatch.setenv("AUTO_EXECUTE_MAX_AMOUNT", "100000.0")
        instance, place_method = _patch_kite(monkeypatch)

        from src.autfin.agents.executor import executor_agent

        state = _base_state(trade_queue=[_trade()])
        result = await executor_agent(state, _config())

        assert result["execution_reports"][0]["status"] == "recommended"
        assert "auto-login" in result["execution_reports"][0]["message"].lower() or "configured" in result["execution_reports"][0]["message"].lower()
        place_method.assert_not_called()


# ═══════════════════════════════════════════════════════════════════
# 2. Trade gate auto-small integration
# ═══════════════════════════════════════════════════════════════════

class TestTradeGateAutoSmallIntegration:
    """Verify trade_gate correctly passes small trades through in auto-small mode."""

    @pytest.mark.asyncio
    @patch("src.autfin.graph.hitl.AutFinSettings")
    async def test_small_trade_passes_through(self, mock_settings_cls):
        """Small trade in auto-small mode → no interrupt, trade_queue unchanged."""
        mock_settings = MagicMock()
        mock_settings.auto_execute_max_amount = 10000.0
        mock_settings.execution_mode = "auto-small"
        mock_settings_cls.return_value = mock_settings

        from src.autfin.graph.hitl import trade_gate_node

        trade = _trade(qty=2, price=2500.0)
        state = _base_state(trade_queue=[trade])
        result = await trade_gate_node(state, _config())

        # Should NOT trigger interrupt — trade is small
        assert result.get("pending_human_approval") is None
        assert result.get("human_decision") is None
        # trade_queue not returned means it's unchanged (gate passes through)
        assert "trade_queue" not in result
        assert any("within auto-execute limit" in m.content.lower() for m in result["messages"])

    @pytest.mark.asyncio
    @patch("src.autfin.graph.hitl.AutFinSettings")
    @patch("src.autfin.graph.hitl.interrupt")
    async def test_large_trade_triggers_interrupt(self, mock_interrupt, mock_settings_cls):
        """Large trade in auto-small mode → interrupt() called."""
        mock_settings = MagicMock()
        mock_settings.auto_execute_max_amount = 5000.0
        mock_settings.execution_mode = "auto-small"
        mock_settings_cls.return_value = mock_settings
        mock_interrupt.return_value = {"approved": True}

        from src.autfin.graph.hitl import trade_gate_node

        state = _base_state(
            trade_queue=[_trade(qty=5, price=2500.0)],  # ₹12,500 > ₹5,000
        )
        result = await trade_gate_node(state, _config())

        mock_interrupt.assert_called_once()
        assert result["human_decision"] == "approved"
