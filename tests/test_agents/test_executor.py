"""
Tests for the Execution Agent (Task 19).

Validates executor_agent against the EXACT contract in executor.py (291 lines).

Key contract values (MUST match tests):
  - recommendation mode: status="recommended", message from _recommendation_message()
  - paper mode: status="paper_executed", simulated_order_id="PAPER-..."
  - auto-small success: status="executed", order_id from KiteTools
  - auto-small downgrade: status="recommended", downgrade merged into message
  - error (kite/unknown): status="error"
  - invalid trade: NO execution report; only errors.append()
  - empty queue: execution_reports=[], summary message appended
  - execution_reports REPLACED (not appended to existing)
  - messages always ends with summary AIMessage
  - KiteTools: variety='regular', exchange NSE then BSE fallback, product='CNC', tag='AutFin'
  - No 'needs_approval' field in reports
  - Holding lookup matches tradingsymbol OR symbol OR trading_symbol keys

Pattern mirrors tests/test_agents/test_portfolio_manager.py:
  - dict-based AgentState
  - pytest.mark.asyncio
  - _base_state(**overrides) helper
  - monkeypatch.setenv(...) to override settings
"""
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


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
        "CASH_BUFFER_PCT",
        "MAX_SINGLE_POSITION_PCT",
        "MAX_SECTOR_EXPOSURE_PCT",
        "MAX_DAILY_TRADES",
        "DEEP_MODEL",
        "FAST_MODEL",
        "FALLBACK_MODEL",
    ]:
        monkeypatch.delenv(key, raising=False)
    yield


# ─────────────────────────────────────────────────────────────────
# Helpers (mirror test_portfolio_manager.py pattern)
# ─────────────────────────────────────────────────────────────────

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
        "decision_id": "dec-001",
    }
    base.update(overrides)
    return base


def _base_state(**overrides) -> dict[str, Any]:
    """Minimal AgentState dict for executor_agent."""
    state: dict[str, Any] = {
        "thread_id": "test-thread",
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


# ─────────────────────────────────────────────────────────────────
# KiteTools patch helper
# ─────────────────────────────────────────────────────────────────

def _patch_kite(monkeypatch, place_order_result: dict[str, Any] | None = None):
    """
    Patch src.autfin.tools.kite_tools.KiteTools so that:
      - KiteTools is importable inside executor_agent
      - place_order returns place_order_result (default success)

    Returns (instance, place_method_mock).
    """
    from src.autfin.tools.kite_tools import KiteTools

    if place_order_result is None:
        place_order_result = {"status": "success", "order_id": "NSE12345"}

    instance = MagicMock()
    place_method = AsyncMock(return_value=place_order_result)
    instance.place_order = place_method

    monkeypatch.setattr(KiteTools, "__new__", lambda cls, *a, **kw: instance)
    return instance, place_method


# ═══════════════════════════════════════════════════════════════════
# Tests — Empty queue
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_empty_trade_queue_returns_noop(monkeypatch):
    """No trades → empty execution_reports, summary message, no errors."""
    from src.autfin.agents.executor import executor_agent

    state = _base_state(trade_queue=[])
    result = await executor_agent(state, _config())

    assert result["execution_reports"] == []
    assert result["errors"] == []
    # Always has summary message; first message is the "nothing to do" message
    assert len(result["messages"]) >= 1
    assert "empty" in result["messages"][0].content.lower() or "nothing to do" in result["messages"][0].content.lower()


# ═══════════════════════════════════════════════════════════════════
# Tests — Invalid trade handling (never produces execution report)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_invalid_trade_missing_keys_produces_error_no_report(monkeypatch):
    """Missing required keys → errors.append, NO execution_report."""
    monkeypatch.setenv("EXECUTION_MODE", "recommendation")
    from src.autfin.agents.executor import executor_agent

    bad_trade = {"rationale": "no symbol/qty/price"}  # missing everything required
    state = _base_state(trade_queue=[bad_trade])
    result = await executor_agent(state, _config())

    assert result["execution_reports"] == []  # Invalid trades produce NO report
    assert len(result["errors"]) == 1
    assert "Missing required keys" in result["errors"][0]


@pytest.mark.asyncio
async def test_invalid_trade_bad_action_produces_error_no_report(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "recommendation")
    from src.autfin.agents.executor import executor_agent

    state = _base_state(trade_queue=[_trade(action="HOLD")])
    result = await executor_agent(state, _config())

    assert result["execution_reports"] == []
    assert any("Invalid action" in e for e in result["errors"])


@pytest.mark.asyncio
async def test_invalid_trade_non_numeric_qty_produces_error(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "recommendation")
    from src.autfin.agents.executor import executor_agent

    state = _base_state(trade_queue=[_trade(qty="two")])
    result = await executor_agent(state, _config())

    assert result["execution_reports"] == []
    assert any("not a number" in e for e in result["errors"])


@pytest.mark.asyncio
async def test_invalid_trade_zero_qty_produces_error(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "recommendation")
    from src.autfin.agents.executor import executor_agent

    state = _base_state(trade_queue=[_trade(qty=0)])
    result = await executor_agent(state, _config())

    assert result["execution_reports"] == []
    assert any("Invalid qty" in e for e in result["errors"])


@pytest.mark.asyncio
async def test_invalid_trade_non_numeric_price_produces_error(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "recommendation")
    from src.autfin.agents.executor import executor_agent

    state = _base_state(trade_queue=[_trade(order_type="LIMIT", price="bc")])
    result = await executor_agent(state, _config())

    assert result["execution_reports"] == []
    assert any("not a number" in e for e in result["errors"])


@pytest.mark.asyncio
async def test_invalid_trade_zero_price_for_limit_produces_error(monkeypatch):
    """LIMIT order with price=0 is invalid."""
    monkeypatch.setenv("EXECUTION_MODE", "recommendation")
    from src.autfin.agents.executor import executor_agent

    state = _base_state(trade_queue=[_trade(order_type="LIMIT", price=0)])
    result = await executor_agent(state, _config())

    assert result["execution_reports"] == []
    assert any("Invalid price" in e for e in result["errors"])


@pytest.mark.asyncio
async def test_invalid_trade_not_dict_produces_error(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "recommendation")
    from src.autfin.agents.executor import executor_agent

    state = _base_state(trade_queue=["not a dict"])
    result = await executor_agent(state, _config())

    assert result["execution_reports"] == []
    assert any("not a dict" in e for e in result["errors"])


@pytest.mark.asyncio
async def test_mixed_valid_and_invalid_trades(monkeypatch):
    """One valid + one invalid → valid produces report, invalid produces only error."""
    monkeypatch.setenv("EXECUTION_MODE", "recommendation")
    from src.autfin.agents.executor import executor_agent

    state = _base_state(
        trade_queue=[
            _trade(symbol="GOOD"),
            _trade(symbol="BAD", action="SIDEWAYS"),
        ]
    )
    result = await executor_agent(state, _config())

    # Only 1 valid trade produces a report; invalid produces no report
    assert len(result["execution_reports"]) == 1
    assert result["execution_reports"][0]["symbol"] == "GOOD"
    assert result["execution_reports"][0]["status"] == "recommended"
    # Invalid trade generates error
    assert any("BAD" in e or "SIDEWAYS" in e for e in result["errors"])


# ═══════════════════════════════════════════════════════════════════
# Tests — Recommendation mode
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_recommendation_mode_buy_limit(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "recommendation")
    from src.autfin.agents.executor import executor_agent

    state = _base_state(trade_queue=[_trade()])
    result = await executor_agent(state, _config())

    rep = result["execution_reports"][0]
    assert rep["status"] == "recommended"  # NOT "recommendation"
    assert rep["mode"] == "recommendation"
    assert rep["symbol"] == "RELIANCE"
    assert rep["action"] == "BUY"
    assert rep["qty"] == 2
    assert rep["price"] == 2500.0
    assert rep["order_type"] == "LIMIT"
    assert "Recommend" in rep["message"] or "recommend" in rep["message"]


@pytest.mark.asyncio
async def test_recommendation_mode_market_order(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "recommendation")
    from src.autfin.agents.executor import executor_agent

    state = _base_state(trade_queue=[_trade(order_type="MARKET", price=0.0)])
    result = await executor_agent(state, _config())

    rep = result["execution_reports"][0]
    assert rep["status"] == "recommended"
    assert rep["order_type"] == "MARKET"
    assert rep["price"] == 0.0


@pytest.mark.asyncio
async def test_recommendation_mode_sell(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "recommendation")
    from src.autfin.agents.executor import executor_agent

    state = _base_state(
        trade_queue=[_trade(action="SELL", symbol="TCS")],
        holdings=[_holding("TCS", 5)],
    )
    result = await executor_agent(state, _config())

    rep = result["execution_reports"][0]
    assert rep["status"] == "recommended"
    assert rep["action"] == "SELL"
    assert rep["symbol"] == "TCS"


@pytest.mark.asyncio
async def test_recommendation_mode_multiple_trades(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "recommendation")
    from src.autfin.agents.executor import executor_agent

    state = _base_state(
        trade_queue=[
            _trade(symbol="RELIANCE"),
            _trade(symbol="TCS", action="SELL"),
        ],
        holdings=[_holding("TCS", 5)],
    )
    result = await executor_agent(state, _config())

    assert len(result["execution_reports"]) == 2
    symbols = {r["symbol"] for r in result["execution_reports"]}
    assert symbols == {"RELIANCE", "TCS"}


@pytest.mark.asyncio
async def test_recommendation_mode_never_calls_kite(monkeypatch):
    """Recommendation must be HITL-only — never auto-place."""
    monkeypatch.setenv("EXECUTION_MODE", "recommendation")
    instance, place_method = _patch_kite(
        monkeypatch, place_order_result={"status": "success", "order_id": "OVERSHOOT"}
    )

    from src.autfin.agents.executor import executor_agent
    state = _base_state(trade_queue=[_trade()])
    result = await executor_agent(state, _config())

    place_method.assert_not_called()
    assert result["execution_reports"][0]["status"] == "recommended"


# ═══════════════════════════════════════════════════════════════════
# Tests — Paper mode
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_paper_mode_buy_generates_paper_order_id(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "paper")
    from src.autfin.agents.executor import executor_agent

    state = _base_state(trade_queue=[_trade()])
    result = await executor_agent(state, _config())

    rep = result["execution_reports"][0]
    assert rep["status"] == "paper_executed"  # NOT "paper"
    assert rep["mode"] == "paper"
    assert rep["simulated_order_id"].startswith("PAPER-")
    assert rep["order_id"] is None  # order_id not set for paper


@pytest.mark.asyncio
async def test_paper_mode_sell_generates_paper_order_id(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "paper")
    from src.autfin.agents.executor import executor_agent

    state = _base_state(
        trade_queue=[_trade(action="SELL", symbol="TCS")],
        holdings=[_holding("TCS", 5)],
    )
    result = await executor_agent(state, _config())

    rep = result["execution_reports"][0]
    assert rep["status"] == "paper_executed"
    assert rep["action"] == "SELL"
    assert rep["simulated_order_id"].startswith("PAPER-")


@pytest.mark.asyncio
async def test_paper_mode_never_calls_kite_place_order(monkeypatch):
    """Paper mode must NOT touch the real order API."""
    monkeypatch.setenv("EXECUTION_MODE", "paper")
    instance, place_method = _patch_kite(
        monkeypatch,
        place_order_result={"status": "success", "order_id": "SHOULD-NOT-BE-USED"},
    )
    from src.autfin.agents.executor import executor_agent

    state = _base_state(trade_queue=[_trade()])
    result = await executor_agent(state, _config())

    place_method.assert_not_called()
    rep = result["execution_reports"][0]
    assert rep["simulated_order_id"].startswith("PAPER-")
    assert rep["order_id"] is None


@pytest.mark.asyncio
async def test_paper_mode_works_with_zero_cash(monkeypatch):
    """Paper mode simulates everything — cash state shouldn't matter."""
    monkeypatch.setenv("EXECUTION_MODE", "paper")
    from src.autfin.agents.executor import executor_agent

    state = _base_state(available_cash=0.0, trade_queue=[_trade()])
    result = await executor_agent(state, _config())

    rep = result["execution_reports"][0]
    assert rep["status"] == "paper_executed"
    assert rep["simulated_order_id"].startswith("PAPER-")


# ═══════════════════════════════════════════════════════════════════
# Tests — Auto-small mode downgrades
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_auto_small_downgrades_when_not_self_hosted(monkeypatch):
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

    place_method.assert_not_called()
    rep = result["execution_reports"][0]
    assert rep["status"] == "recommended"
    # Downgrade reason is in the message, not a separate field
    assert "auto-login" in rep["message"].lower() or "configured" in rep["message"].lower()


@pytest.mark.asyncio
async def test_auto_small_downgrades_when_above_max_amount(monkeypatch):
    """Trade value > auto_execute_max_amount → downgrade."""
    monkeypatch.setenv("EXECUTION_MODE", "auto-small")
    monkeypatch.setenv("KITE_USER_ID", "test123")
    monkeypatch.setenv("KITE_PASSWORD", "testpass")
    monkeypatch.setenv("KITE_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
    monkeypatch.setenv("AUTO_EXECUTE_MAX_AMOUNT", "1000.0")  # trade is 5000
    instance, place_method = _patch_kite(monkeypatch)

    from src.autfin.agents.executor import executor_agent

    state = _base_state(trade_queue=[_trade(position_size=5000.0)])
    result = await executor_agent(state, _config())

    place_method.assert_not_called()
    rep = result["execution_reports"][0]
    assert rep["status"] == "recommended"
    assert "exceeds" in rep["message"].lower() or "limit" in rep["message"].lower()


@pytest.mark.asyncio
async def test_auto_small_downgrades_buy_when_insufficient_cash(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "auto-small")
    monkeypatch.setenv("KITE_USER_ID", "test123")
    monkeypatch.setenv("KITE_PASSWORD", "testpass")
    monkeypatch.setenv("KITE_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
    monkeypatch.setenv("AUTO_EXECUTE_MAX_AMOUNT", "100000.0")
    instance, place_method = _patch_kite(monkeypatch)

    from src.autfin.agents.executor import executor_agent

    # trade = 50 * 1000 = 50,000 but cash is only 30,000
    state = _base_state(
        trade_queue=[_trade(qty=50, price=1000.0, position_size=50_000.0)]
    )
    result = await executor_agent(state, _config())

    place_method.assert_not_called()
    rep = result["execution_reports"][0]
    assert rep["status"] == "recommended"
    assert "funds" in rep["message"].lower() or "cash" in rep["message"].lower()


@pytest.mark.asyncio
async def test_auto_small_downgrades_sell_when_insufficient_holdings(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "auto-small")
    monkeypatch.setenv("KITE_USER_ID", "test123")
    monkeypatch.setenv("KITE_PASSWORD", "testpass")
    monkeypatch.setenv("KITE_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
    monkeypatch.setenv("AUTO_EXECUTE_MAX_AMOUNT", "10000.0")
    instance, place_method = _patch_kite(monkeypatch)

    from src.autfin.agents.executor import executor_agent

    # Want to SELL 10 TCS, but only hold 2
    state = _base_state(
        trade_queue=[_trade(symbol="TCS", action="SELL", qty=10, price=3500.0, position_size=35_000.0)],
        holdings=[_holding("TCS", 2)],
    )
    result = await executor_agent(state, _config())

    place_method.assert_not_called()
    rep = result["execution_reports"][0]
    assert rep["status"] == "recommended"
    assert "holdings" in rep["message"].lower() or "shares" in rep["message"].lower()


# ═══════════════════════════════════════════════════════════════════
# Tests — Auto-small mode actually executes
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_auto_small_executes_when_qualified(monkeypatch):
    """All checks pass → real KiteTools.place_order called."""
    monkeypatch.setenv("EXECUTION_MODE", "auto-small")
    monkeypatch.setenv("KITE_USER_ID", "test123")
    monkeypatch.setenv("KITE_PASSWORD", "testpass")
    monkeypatch.setenv("KITE_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
    monkeypatch.setenv("AUTO_EXECUTE_MAX_AMOUNT", "10000.0")
    instance, place_method = _patch_kite(
        monkeypatch, place_order_result={"status": "success", "order_id": "NSE999"}
    )

    from src.autfin.agents.executor import executor_agent

    # trade = 2 * 2500 = 5000 < 10000 max, cash=30000 > 5000
    state = _base_state(trade_queue=[_trade()])
    result = await executor_agent(state, _config())

    rep = result["execution_reports"][0]
    assert rep["status"] == "executed"  # NOT "success"
    assert rep["mode"] == "auto-small"
    assert rep["order_id"] == "NSE999"

    assert place_method.await_count >= 1
    first_kwargs = place_method.await_args.kwargs
    assert first_kwargs["tradingsymbol"] == "RELIANCE"
    assert first_kwargs["transaction_type"] == "BUY"
    assert first_kwargs["quantity"] == 2
    assert first_kwargs["order_type"] == "LIMIT"
    assert first_kwargs["price"] == 2500.0
    assert first_kwargs["product"] == "CNC"
    assert first_kwargs["variety"] == "regular"
    assert first_kwargs["tag"] == "AutFin"


@pytest.mark.asyncio
async def test_auto_small_market_order_passes_zero_price(monkeypatch):
    """MARKET order → KiteTools gets price=0.0."""
    monkeypatch.setenv("EXECUTION_MODE", "auto-small")
    monkeypatch.setenv("KITE_USER_ID", "test123")
    monkeypatch.setenv("KITE_PASSWORD", "testpass")
    monkeypatch.setenv("KITE_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
    monkeypatch.setenv("AUTO_EXECUTE_MAX_AMOUNT", "10000.0")
    instance, place_method = _patch_kite(
        monkeypatch, place_order_result={"status": "success", "order_id": "NSE999"}
    )

    from src.autfin.agents.executor import executor_agent

    state = _base_state(trade_queue=[_trade(order_type="MARKET", price=0.0)])
    result = await executor_agent(state, _config())

    rep = result["execution_reports"][0]
    assert rep["status"] == "executed"
    first_kwargs = place_method.await_args.kwargs
    assert first_kwargs["order_type"] == "MARKET"
    assert first_kwargs["price"] == 0.0


@pytest.mark.asyncio
async def test_auto_small_sell_uses_sell_transaction(monkeypatch):
    """SELL → transaction_type='SELL'."""
    monkeypatch.setenv("EXECUTION_MODE", "auto-small")
    monkeypatch.setenv("KITE_USER_ID", "test123")
    monkeypatch.setenv("KITE_PASSWORD", "testpass")
    monkeypatch.setenv("KITE_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
    monkeypatch.setenv("AUTO_EXECUTE_MAX_AMOUNT", "100000.0")
    instance, place_method = _patch_kite(
        monkeypatch, place_order_result={"status": "success", "order_id": "NSE555"}
    )

    from src.autfin.agents.executor import executor_agent

    state = _base_state(
        trade_queue=[_trade(symbol="TCS", action="SELL", qty=2, price=3500.0, position_size=7000.0)],
        holdings=[_holding("TCS", 10)],
    )
    result = await executor_agent(state, _config())

    first_kwargs = place_method.await_args.kwargs
    assert first_kwargs["transaction_type"] == "SELL"
    assert first_kwargs["tradingsymbol"] == "TCS"


@pytest.mark.asyncio
async def test_auto_small_propagates_kite_error(monkeypatch):
    """KiteTools returns error status → report status reflects it."""
    monkeypatch.setenv("EXECUTION_MODE", "auto-small")
    monkeypatch.setenv("KITE_USER_ID", "test123")
    monkeypatch.setenv("KITE_PASSWORD", "testpass")
    monkeypatch.setenv("KITE_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
    monkeypatch.setenv("AUTO_EXECUTE_MAX_AMOUNT", "10000.0")
    instance, place_method = _patch_kite(
        monkeypatch,
        place_order_result={"status": "error", "message": "Insufficient margin"},
    )

    from src.autfin.agents.executor import executor_agent

    state = _base_state(trade_queue=[_trade()])
    result = await executor_agent(state, _config())

    rep = result["execution_reports"][0]
    assert rep["status"] == "error"
    assert "Insufficient margin" in rep["message"]
    assert any("failed" in e.lower() or "error" in e.lower() for e in result["errors"])


@pytest.mark.asyncio
async def test_auto_small_handles_kite_exception_gracefully(monkeypatch):
    """If KiteTools.place_order raises → error report, no crash."""
    monkeypatch.setenv("EXECUTION_MODE", "auto-small")
    monkeypatch.setenv("KITE_USER_ID", "test123")
    monkeypatch.setenv("KITE_PASSWORD", "testpass")
    monkeypatch.setenv("KITE_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
    monkeypatch.setenv("AUTO_EXECUTE_MAX_AMOUNT", "10000.0")

    from src.autfin.tools.kite_tools import KiteTools

    boom = AsyncMock(side_effect=RuntimeError("network down"))
    instance = MagicMock()
    instance.place_order = boom
    monkeypatch.setattr(KiteTools, "__new__", lambda cls, *a, **kw: instance)

    from src.autfin.agents.executor import executor_agent

    state = _base_state(trade_queue=[_trade()])
    result = await executor_agent(state, _config())

    rep = result["execution_reports"][0]
    assert rep["status"] == "error"
    assert "network down" in rep["message"]
    assert len(result["errors"]) >= 1


@pytest.mark.asyncio
async def test_auto_small_buy_rejected_when_cash_zero(monkeypatch):
    """No cash → cannot BUY even in auto-small."""
    monkeypatch.setenv("EXECUTION_MODE", "auto-small")
    monkeypatch.setenv("KITE_USER_ID", "test123")
    monkeypatch.setenv("KITE_PASSWORD", "testpass")
    monkeypatch.setenv("KITE_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
    monkeypatch.setenv("AUTO_EXECUTE_MAX_AMOUNT", "10000.0")
    instance, place_method = _patch_kite(monkeypatch)

    from src.autfin.agents.executor import executor_agent

    state = _base_state(available_cash=0.0, trade_queue=[_trade()])
    result = await executor_agent(state, _config())

    place_method.assert_not_called()
    rep = result["execution_reports"][0]
    assert rep["status"] == "recommended"


@pytest.mark.asyncio
async def test_auto_small_buy_accepted_when_enough_cash(monkeypatch):
    """Enough cash → auto-small executes BUY."""
    monkeypatch.setenv("EXECUTION_MODE", "auto-small")
    monkeypatch.setenv("KITE_USER_ID", "test123")
    monkeypatch.setenv("KITE_PASSWORD", "testpass")
    monkeypatch.setenv("KITE_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
    monkeypatch.setenv("AUTO_EXECUTE_MAX_AMOUNT", "10000.0")
    instance, place_method = _patch_kite(
        monkeypatch, place_order_result={"status": "success", "order_id": "NSE777"}
    )

    from src.autfin.agents.executor import executor_agent

    # trade=5000, cash=30000 > 5000 → should execute
    state = _base_state(available_cash=30000.0, trade_queue=[_trade()])
    result = await executor_agent(state, _config())

    rep = result["execution_reports"][0]
    assert rep["status"] == "executed"
    assert rep["order_id"] == "NSE777"
    assert place_method.await_count >= 1


# ═══════════════════════════════════════════════════════════════════
# Tests — Unknown execution mode
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_unknown_execution_mode_produces_error_report(monkeypatch):
    """Garbage EXECUTION_MODE → error report + error entry."""
    monkeypatch.setenv("EXECUTION_MODE", "yolo-mode")
    from src.autfin.agents.executor import executor_agent

    state = _base_state(trade_queue=[_trade()])
    result = await executor_agent(state, _config())

    rep = result["execution_reports"][0]
    assert rep["status"] == "error"
    assert "Unknown execution mode" in rep["message"]
    assert any("yolo-mode" in e for e in result["errors"])


# ═══════════════════════════════════════════════════════════════════
# Tests — Holding lookup (3 key variants)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_auto_small_holding_lookup_tradingsymbol_key(monkeypatch):
    """Holdings use 'tradingsymbol' key (Kite default)."""
    monkeypatch.setenv("EXECUTION_MODE", "auto-small")
    monkeypatch.setenv("KITE_USER_ID", "test123")
    monkeypatch.setenv("KITE_PASSWORD", "testpass")
    monkeypatch.setenv("KITE_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
    monkeypatch.setenv("AUTO_EXECUTE_MAX_AMOUNT", "100000.0")
    instance, place_method = _patch_kite(
        monkeypatch, place_order_result={"status": "success", "order_id": "NSE111"}
    )

    from src.autfin.agents.executor import executor_agent

    state = _base_state(
        trade_queue=[_trade(symbol="TCS", action="SELL", qty=2, price=3500.0, position_size=7000.0)],
        holdings=[{"tradingsymbol": "TCS", "quantity": 10, "average_price": 3000.0}],
    )
    result = await executor_agent(state, _config())

    assert result["execution_reports"][0]["status"] == "executed"
    assert place_method.await_count >= 1


@pytest.mark.asyncio
async def test_auto_small_holding_lookup_symbol_key(monkeypatch):
    """Holdings use 'symbol' key (alternative format)."""
    monkeypatch.setenv("EXECUTION_MODE", "auto-small")
    monkeypatch.setenv("KITE_USER_ID", "test123")
    monkeypatch.setenv("KITE_PASSWORD", "testpass")
    monkeypatch.setenv("KITE_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
    monkeypatch.setenv("AUTO_EXECUTE_MAX_AMOUNT", "100000.0")
    instance, place_method = _patch_kite(
        monkeypatch, place_order_result={"status": "success", "order_id": "NSE222"}
    )

    from src.autfin.agents.executor import executor_agent

    state = _base_state(
        trade_queue=[_trade(symbol="TCS", action="SELL", qty=2, price=3500.0, position_size=7000.0)],
        holdings=[{"symbol": "TCS", "quantity": 10, "average_price": 3000.0}],
    )
    result = await executor_agent(state, _config())

    assert result["execution_reports"][0]["status"] == "executed"


@pytest.mark.asyncio
async def test_auto_small_holding_lookup_trading_symbol_key(monkeypatch):
    """Holdings use 'trading_symbol' key (yet another variant)."""
    monkeypatch.setenv("EXECUTION_MODE", "auto-small")
    monkeypatch.setenv("KITE_USER_ID", "test123")
    monkeypatch.setenv("KITE_PASSWORD", "testpass")
    monkeypatch.setenv("KITE_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
    monkeypatch.setenv("AUTO_EXECUTE_MAX_AMOUNT", "100000.0")
    instance, place_method = _patch_kite(
        monkeypatch, place_order_result={"status": "success", "order_id": "NSE333"}
    )

    from src.autfin.agents.executor import executor_agent

    state = _base_state(
        trade_queue=[_trade(symbol="TCS", action="SELL", qty=2, price=3500.0, position_size=7000.0)],
        holdings=[{"trading_symbol": "TCS", "quantity": 10, "average_price": 3000.0}],
    )
    result = await executor_agent(state, _config())

    assert result["execution_reports"][0]["status"] == "executed"


# ═══════════════════════════════════════════════════════════════════
# Tests — Report shape consistency
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_execution_reports_have_required_keys(monkeypatch):
    """Every report must include the expected keys."""
    monkeypatch.setenv("EXECUTION_MODE", "recommendation")
    from src.autfin.agents.executor import executor_agent

    state = _base_state(trade_queue=[_trade()])
    result = await executor_agent(state, _config())

    rep = result["execution_reports"][0]
    required = {"decision_id", "symbol", "action", "qty", "price", "order_type",
                 "trade_value", "mode", "status", "order_id", "simulated_order_id",
                 "message", "timestamp"}
    assert required <= set(rep.keys())


@pytest.mark.asyncio
async def test_execution_reports_replaced_not_appended(monkeypatch):
    """executor_agent REPLACES execution_reports — does not append to existing."""
    monkeypatch.setenv("EXECUTION_MODE", "recommendation")
    from src.autfin.agents.executor import executor_agent

    existing = [{"status": "success", "decision_id": "prior-1", "symbol": "X"}]
    state = _base_state(
        trade_queue=[_trade()],
        execution_reports=list(existing),
    )
    result = await executor_agent(state, _config())

    # Only 1 report — the new one. The old prior-1 is GONE.
    assert len(result["execution_reports"]) == 1
    assert result["execution_reports"][0]["decision_id"] == "dec-001"


@pytest.mark.asyncio
async def test_messages_always_ends_with_summary(monkeypatch):
    """messages list always ends with the summary AIMessage."""
    monkeypatch.setenv("EXECUTION_MODE", "recommendation")
    from src.autfin.agents.executor import executor_agent

    state = _base_state(trade_queue=[_trade()])
    result = await executor_agent(state, _config())

    last_msg = result["messages"][-1].content
    assert "processed 1 trade(s)" in last_msg
    assert "1 report(s)" in last_msg


@pytest.mark.asyncio
async def test_errors_from_state_preserved(monkeypatch):
    """Pre-existing errors in state should be preserved."""
    monkeypatch.setenv("EXECUTION_MODE", "recommendation")
    from src.autfin.agents.executor import executor_agent

    state = _base_state(
        trade_queue=[_trade()],
        errors=["pre-existing error from another agent"],
    )
    result = await executor_agent(state, _config())

    assert "pre-existing error from another agent" in result["errors"]
