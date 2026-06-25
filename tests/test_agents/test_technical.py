"""
Tests for the Technical Agent. All tests use mocked state and do not require real API keys.
"""
import pandas as pd
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.autfin.agents.technical import technical_agent, TechnicalAgent


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


def _make_ohlcv_dataframe():
    import numpy as np
    dates = pd.date_range(end=pd.Timestamp.today(), periods=200, freq="B")
    np.random.seed(42)
    close_arr = 2500 + np.cumsum(np.random.randn(200) * 20)
    open_arr = close_arr - np.random.randn(200) * 5
    high_arr = np.maximum(open_arr, close_arr) + np.abs(np.random.randn(200) * 10)
    low_arr = np.minimum(open_arr, close_arr) - np.abs(np.random.randn(200) * 10)
    volume_arr = np.random.randint(100000, 500000, 200)
    df = pd.DataFrame({
        "open": open_arr,
        "high": high_arr,
        "low": low_arr,
        "close": close_arr,
        "volume": volume_arr.astype(float),
    }, index=dates)
    return df


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for key in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)


def _make_llm_unavailable(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.is_available = False
    monkeypatch.setattr(
        "src.autfin.agents.technical.get_llm_client", lambda: mock_llm
    )


def _mock_agent_analyze(monkeypatch, return_data=None):
    """Mock TechnicalAgent.analyze to return test data directly, bypassing yfinance."""
    if return_data is None:
        return_data = {
            "symbol": "RELIANCE",
            "signal": {
                "signal": "HOLD",
                "confidence": 0.5,
                "reasoning": ["Mock reasoning"],
                "indicator_scores": {},
                "pattern_signals": [],
            },
        }
    monkeypatch.setattr(
        TechnicalAgent, "analyze",
        AsyncMock(return_value=return_data),
    )


@pytest.mark.asyncio
async def test_agent_processes_technical_reports_with_mock_data(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    _mock_agent_analyze(monkeypatch)
    state = _base_state(
        market_snapshot={"quotes": {"RELIANCE": {"ltp": 2500.0}}}
    )
    result = await technical_agent(state)

    assert "technical_reports" in result
    assert "RELIANCE" in result["technical_reports"]
    report = result["technical_reports"]["RELIANCE"]
    if "error" not in report:
        assert "signal" in report


@pytest.mark.asyncio
async def test_error_handling_when_no_market_snapshot(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    state = _base_state(market_snapshot={})
    result = await technical_agent(state)

    assert "errors" in result
    assert any("No market data" in e for e in result["errors"])


@pytest.mark.asyncio
async def test_error_handling_when_yfinance_unavailable(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    # Mock analyze to return an error, simulating yfinance being unavailable.
    # The agent code skips adding reports with "error" key to technical_reports,
    # instead appending to errors list.
    _mock_agent_analyze(monkeypatch, return_data={
        "symbol": "TCS",
        "error": "yfinance_ohlcv_unavailable",
        "message": "Cannot analyze TCS — OHLCV data unavailable from yfinance",
        "status": "error",
    })
    state = _base_state(
        market_snapshot={"quotes": {"TCS": {"ltp": 3200.0}}}
    )
    result = await technical_agent(state)

    assert "technical_reports" in result
    assert "errors" in result
    assert any("yfinance_ohlcv_unavailable" in e for e in result["errors"])


@pytest.mark.asyncio
async def test_agent_returns_correct_keys(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    _mock_agent_analyze(monkeypatch)
    state = _base_state(
        market_snapshot={"quotes": {"RELIANCE": {"ltp": 2500.0}}}
    )
    result = await technical_agent(state)

    assert "technical_reports" in result
    assert "errors" in result
    assert "next_agent" in result


@pytest.mark.asyncio
async def test_with_mock_ohlcv_via_monkeypatch(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    _mock_agent_analyze(monkeypatch, return_data={
        "symbol": "INFY",
        "signal": {
            "signal": "BUY",
            "confidence": 0.8,
            "reasoning": ["Bullish engulfing pattern", "RSI oversold"],
            "indicator_scores": {"trend": 2, "momentum": 1},
            "pattern_signals": [{"type": "bullish", "pattern": "Hammer"}],
        },
    })
    state = _base_state(
        market_snapshot={"quotes": {"INFY": {"ltp": 1500.0}}}
    )
    result = await technical_agent(state)

    assert "technical_reports" in result
    assert "INFY" in result["technical_reports"]
    report = result["technical_reports"]["INFY"]
    if "error" not in report:
        assert "signal" in report


@pytest.mark.asyncio
async def test_empty_quotes_returns_error(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    state = _base_state(
        market_snapshot={"quotes": {}}
    )
    result = await technical_agent(state)

    assert "errors" in result
    assert any("No market data" in e for e in result["errors"])
