"""
Tests for the Portfolio Manager Agent.

All tests use mocked state and do not require real API keys.
"""

import pytest

from src.autfin.agents.portfolio_manager import portfolio_manager_agent


def _base_state(**overrides) -> dict:
    """Build a minimal AgentState-like dict with required keys."""
    state = {
        "messages": [],
        "available_cash": 100_000.0,
        "utilized_margin": 0.0,
        "holdings": [],
        "positions": [],
        "research_debates": {},
        "risk_assessments": {},
        "macro_signals": {},
        "thread_id": "test-thread",
        "run_mode": "daily_morning",
        "errors": [],
    }
    state.update(overrides)
    return state


def _debate(symbol: str, recommendation: str, conviction: float, price: float | None = None) -> dict:
    """Build a per-symbol debate record."""
    record = {
        "recommendation": recommendation,
        "conviction": conviction,
    }
    if price is not None:
        record["price"] = price
    return record


def _risk(symbol: str, approved: bool = True, score: float = 25.0) -> dict:
    """Build a per-symbol risk assessment record."""
    return {
        "approved": approved,
        "risk_score": score,
        "risk_level": "LOW",
        "checks": [],
        "veto_reason": None,
    }


@pytest.mark.asyncio
async def test_no_risk_approved_signals_returns_empty_queue():
    """If risk assessment rejects a BUY signal, no trade is queued."""
    state = _base_state(
        research_debates={"RELIANCE": _debate("RELIANCE", "BUY", 0.85, 2500.0)},
        risk_assessments={"RELIANCE": _risk("RELIANCE", approved=False, score=95.0)},
    )
    result = await portfolio_manager_agent(state)

    assert result["trade_queue"] == []
    assert any("0 BUYs" in str(m.content) for m in result["messages"])


@pytest.mark.asyncio
async def test_single_buy_with_enough_cash():
    """A single approved BUY with sufficient cash gets the correct position size."""
    state = _base_state(
        available_cash=100_000.0,
        research_debates={"RELIANCE": _debate("RELIANCE", "BUY", 0.85, 2500.0)},
        risk_assessments={"RELIANCE": _risk("RELIANCE")},
    )
    result = await portfolio_manager_agent(state)

    trades = result["trade_queue"]
    assert len(trades) == 1
    trade = trades[0]
    assert trade["symbol"] == "RELIANCE"
    assert trade["action"] == "BUY"
    assert trade["risk_approved"] is True

    # deployable cash = 100_000 * 0.9 = 90_000
    # max position = 90_000 * 0.3 = 27_000
    # qty = 27_000 // 2500 = 10; position_size = 25_000
    assert trade["qty"] == 10
    assert trade["price"] == 2500.0
    assert trade["position_size"] == 25_000.0


@pytest.mark.asyncio
async def test_multiple_buys_respect_max_single_position():
    """Multiple BUYs are sized so none exceeds max_single_position_pct."""
    state = _base_state(
        available_cash=100_000.0,
        research_debates={
            "RELIANCE": _debate("RELIANCE", "BUY", 0.9, 2500.0),
            "TCS": _debate("TCS", "BUY", 0.8, 3200.0),
            "INFY": _debate("INFY", "BUY", 0.7, 1500.0),
        },
        risk_assessments={
            "RELIANCE": _risk("RELIANCE"),
            "TCS": _risk("TCS"),
            "INFY": _risk("INFY"),
        },
    )
    result = await portfolio_manager_agent(state)

    buys = [t for t in result["trade_queue"] if t["action"] == "BUY"]
    assert len(buys) == 3

    deployable = 90_000.0
    max_position = deployable * 0.30
    for trade in buys:
        assert trade["position_size"] <= max_position + 0.01
        assert trade["position_size"] >= 1000.0

    total_bought = sum(t["position_size"] for t in buys)
    assert total_bought <= deployable + 0.01


@pytest.mark.asyncio
async def test_cash_buffer_respected():
    """The agent does not allocate the cash buffer."""
    state = _base_state(
        available_cash=100_000.0,
        research_debates={"RELIANCE": _debate("RELIANCE", "BUY", 0.95, 2500.0)},
        risk_assessments={"RELIANCE": _risk("RELIANCE")},
    )
    result = await portfolio_manager_agent(state)

    buys = [t for t in result["trade_queue"] if t["action"] == "BUY"]
    assert len(buys) == 1
    # Total deployed must be <= 90_000 (i.e. 100_000 - 10% buffer).
    assert buys[0]["position_size"] <= 90_000.0


@pytest.mark.asyncio
async def test_sector_exposure_limit_blocks_same_sector_buys():
    """An additional BUY in an already-full sector is blocked."""
    state = _base_state(
        available_cash=100_000.0,
        holdings=[
            {
                "tradingsymbol": "INFY",
                "qty": 10,
                "ltp": 5000.0,
                "avg_price": 4500.0,
            }
        ],
        research_debates={
            "TCS": _debate("TCS", "BUY", 0.85, 3200.0),
        },
        risk_assessments={"TCS": _risk("TCS")},
    )
    result = await portfolio_manager_agent(state)

    # INFY holding value = 50_000; portfolio = 150_000; max IT sector = 45_000.
    # Existing IT exposure already exceeds the limit, so TCS must be skipped.
    buys = [t for t in result["trade_queue"] if t["action"] == "BUY"]
    assert buys == []


@pytest.mark.asyncio
async def test_max_daily_trades_respected(monkeypatch):
    """Only max_daily_trades BUY/SELL orders are queued."""
    monkeypatch.setenv("MAX_DAILY_TRADES", "2")

    state = _base_state(
        available_cash=100_000.0,
        research_debates={
            "RELIANCE": _debate("RELIANCE", "BUY", 0.9, 2500.0),
            "TCS": _debate("TCS", "BUY", 0.85, 3200.0),
            "INFY": _debate("INFY", "BUY", 0.8, 1500.0),
        },
        risk_assessments={
            "RELIANCE": _risk("RELIANCE"),
            "TCS": _risk("TCS"),
            "INFY": _risk("INFY"),
        },
    )
    result = await portfolio_manager_agent(state)

    assert len(result["trade_queue"]) == 2
    assert any("Max daily trades reached" in str(m.content) for m in result["messages"])


@pytest.mark.asyncio
async def test_sell_signal_created_regardless_of_cash():
    """A SELL signal generates an order even when available cash is zero."""
    state = _base_state(
        available_cash=0.0,
        holdings=[
            {
                "tradingsymbol": "RELIANCE",
                "qty": 10,
                "ltp": 2500.0,
                "avg_price": 2400.0,
            }
        ],
        research_debates={"RELIANCE": _debate("RELIANCE", "SELL", 0.75, 2500.0)},
        risk_assessments={"RELIANCE": _risk("RELIANCE")},
    )
    result = await portfolio_manager_agent(state)

    sells = [t for t in result["trade_queue"] if t["action"] == "SELL"]
    assert len(sells) == 1
    assert sells[0]["qty"] == 10
    assert sells[0]["position_size"] == 25_000.0


@pytest.mark.asyncio
async def test_deployable_cash_near_zero_only_sells_allowed():
    """When deployable cash is below min_trade_amount, only SELLs are queued."""
    state = _base_state(
        available_cash=500.0,
        holdings=[
            {
                "tradingsymbol": "RELIANCE",
                "qty": 5,
                "ltp": 2500.0,
                "avg_price": 2400.0,
            }
        ],
        research_debates={
            "RELIANCE": _debate("RELIANCE", "SELL", 0.75, 2500.0),
            "TCS": _debate("TCS", "BUY", 0.85, 3200.0),
        },
        risk_assessments={
            "RELIANCE": _risk("RELIANCE"),
            "TCS": _risk("TCS"),
        },
    )
    result = await portfolio_manager_agent(state)

    assert len(result["trade_queue"]) == 1
    assert result["trade_queue"][0]["action"] == "SELL"


@pytest.mark.asyncio
async def test_missing_optional_keys_handled_gracefully():
    """Missing optional state keys do not crash the agent."""
    state = {
        "messages": [],
        "available_cash": 100_000.0,
        "utilized_margin": 0.0,
        "holdings": [],
        "positions": [],
        "thread_id": "test-thread",
        "run_mode": "daily_morning",
        "errors": [],
    }
    result = await portfolio_manager_agent(state)

    assert result["trade_queue"] == []
    assert "errors" in result
    assert any("0 BUYs" in str(m.content) for m in result["messages"])


@pytest.mark.asyncio
async def test_aggregate_debate_format_uses_risk_assessment_symbols():
    """Aggregate researcher output is attached to symbols present in risk_assessments."""
    state = _base_state(
        available_cash=100_000.0,
        research_debates={
            "summary": {"recommendation": "BUY", "conviction_score": 0.8},
            "bull": {"count": 3},
            "bear": {"count": 1},
            "neutral": {"count": 0},
        },
        risk_assessments={
            "RELIANCE": _risk("RELIANCE"),
        },
        market_snapshot={
            "quotes": {
                "RELIANCE": {"ltp": 2500.0},
            }
        },
    )
    result = await portfolio_manager_agent(state)

    buys = [t for t in result["trade_queue"] if t["action"] == "BUY"]
    assert len(buys) == 1
    assert buys[0]["symbol"] == "RELIANCE"
