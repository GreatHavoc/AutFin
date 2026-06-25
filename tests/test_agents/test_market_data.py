"""
Tests for the Market Data Agent. All tests use mocked state and do not require real API keys.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.autfin.agents.market_data import market_data_agent, _cache


@pytest.fixture(autouse=True)
def _enable_kite_quotes():
    """Enable Kite quotes in capabilities for test duration."""
    from src.autfin.config import capabilities
    old = capabilities.CAPABILITIES["kite"]["quotes"]
    capabilities.CAPABILITIES["kite"]["quotes"] = True
    yield
    capabilities.CAPABILITIES["kite"]["quotes"] = old


def _base_state(**overrides) -> dict:
    state = {
        "messages": [],
        "available_cash": 100_000.0,
        "utilized_margin": 0.0,
        "holdings": [],
        "positions": [],
        "research_debates": {},
        "risk_assessments": {},
        "macro_signals": {},
        "market_snapshot": {},
        "sentiment_reports": {},
        "technical_reports": {},
        "fundamental_reports": {},
        "thread_id": "test-thread",
        "run_mode": "daily_morning",
        "errors": [],
        "total_tokens": 0,
        "total_cost": 0.0,
        "budget_limit": 1.0,
        "validation_count": 0,
    }
    state.update(overrides)
    return state


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for key in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    _cache.clear()


def _mock_kite(monkeypatch, margins=None, holdings=None, positions=None, quotes=None):
    mock_kite = MagicMock()
    mock_kite.get_margins = AsyncMock(return_value=margins or {
        "status": "success",
        "data": {"equity": {"available": {"cash": 75000.0}, "utilised": {"debits": 5000.0}}},
    })
    mock_kite.get_holdings = AsyncMock(return_value=holdings or {
        "status": "success",
        "data": [
            {"tradingsymbol": "RELIANCE", "instrument_token": 738561, "qty": 10},
            {"tradingsymbol": "TCS", "instrument_token": 340481, "qty": 5},
        ],
    })
    mock_kite.get_positions = AsyncMock(return_value=positions or {
        "status": "success",
        "data": {"day": []},
    })
    mock_kite.get_quotes = AsyncMock(return_value=quotes or {
        "status": "success",
        "data": {
            "738561": {"tradingsymbol": "RELIANCE", "last_price": 2500.0},
            "340481": {"tradingsymbol": "TCS", "last_price": 3200.0},
        },
    })
    monkeypatch.setattr(
        "src.autfin.agents.market_data.create_kite_tools", lambda: mock_kite
    )


@pytest.mark.asyncio
async def test_returns_available_cash_when_market_data_available(monkeypatch):
    _mock_kite(monkeypatch)
    state = _base_state()
    result = await market_data_agent(state)

    assert result["available_cash"] == 75000.0
    assert result["utilized_margin"] == 5000.0
    assert "agent" in result
    assert result["agent"] == "market_data"


@pytest.mark.asyncio
async def test_returns_holdings_list(monkeypatch):
    _mock_kite(monkeypatch)
    state = _base_state()
    result = await market_data_agent(state)

    assert "holdings" in result
    assert len(result["holdings"]) == 2
    assert result["holdings"][0]["tradingsymbol"] == "RELIANCE"
    assert result["holdings"][1]["tradingsymbol"] == "TCS"


@pytest.mark.asyncio
async def test_error_handling_when_no_market_snapshot(monkeypatch):
    _mock_kite(monkeypatch, quotes={
        "status": "error",
        "message": "Kite API unavailable",
        "data": {},
    })
    state = _base_state()
    result = await market_data_agent(state)

    assert "errors" in result
    assert any("Failed to fetch prices" in e for e in result["errors"])


@pytest.mark.asyncio
async def test_cash_defaults_to_zero_when_kite_margins_fail(monkeypatch):
    _mock_kite(monkeypatch, margins={
        "status": "error",
        "message": "Network timeout",
        "data": {},
    })
    state = _base_state(available_cash=999.0)
    result = await market_data_agent(state)

    assert result["available_cash"] == 0
    assert result["utilized_margin"] == 0


@pytest.mark.asyncio
async def test_returns_positions_list(monkeypatch):
    _mock_kite(monkeypatch, positions={
        "status": "success",
        "data": {
            "day": [{"tradingsymbol": "INFY", "qty": 100, "product": "MIS"}],
        },
    })
    state = _base_state()
    result = await market_data_agent(state)

    assert "positions" in result
    assert len(result["positions"]) == 1
    assert result["positions"][0]["tradingsymbol"] == "INFY"


@pytest.mark.asyncio
async def test_balances_and_prices_returned_together(monkeypatch):
    _mock_kite(monkeypatch)
    state = _base_state()
    result = await market_data_agent(state)

    assert result["available_cash"] == 75000.0
    assert "market_snapshot" in result
    assert "quotes" in result["market_snapshot"]
    assert "RELIANCE" in result["market_snapshot"]["quotes"]
    assert result["market_snapshot"]["quotes"]["RELIANCE"]["ltp"] == 2500.0
