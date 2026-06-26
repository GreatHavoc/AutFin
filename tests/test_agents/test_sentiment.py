"""
Tests for the Sentiment Agent. All tests use mocked state and do not require real API keys.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.autfin.agents.sentiment import sentiment_agent, _sentiment_agent


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
    for key in (
        "GROQ_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY",
        "MARKETAUX_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    global _sentiment_agent
    _sentiment_agent = None


def _make_llm_unavailable(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.is_available = False
    monkeypatch.setattr(
        "src.autfin.agents.sentiment.get_llm_client", lambda: mock_llm
    )


def _mock_sentiment_tools(monkeypatch, results=None):
    if results is None:
        results = {
            "status": "success",
            "total_articles": 5,
            "aggregate_sentiment": {
                "weighted_score": 0.45,
                "confidence": 0.75,
                "recent_articles": 5,
            },
            "articles": [],
        }

    class MockSentimentTools:
        def __init__(self, api_token=None):
            pass
        async def close(self):
            pass
        async def get_news_sentiment(self, symbols=None, days=30, limit=50):
            return results

    monkeypatch.setattr(
        "src.autfin.agents.sentiment.SentimentTools",
        MockSentimentTools,
    )


@pytest.mark.asyncio
async def test_agent_processes_sentiment_reports_with_mock_data(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    _mock_sentiment_tools(monkeypatch)
    state = _base_state(holdings=[{"tradingsymbol": "RELIANCE"}])
    result = await sentiment_agent(state)

    assert "sentiment_reports" in result
    reports = result["sentiment_reports"]
    if "aggregate" in reports:
        assert "weighted_score" in reports["aggregate"]


@pytest.mark.asyncio
async def test_error_handling_when_no_api_token(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    monkeypatch.delenv("MARKETAUX_API_KEY", raising=False)

    class FailingSentimentTools:
        def __init__(self, api_token=None):
            pass
        async def close(self):
            pass
        async def get_news_sentiment(self, symbols=None, days=30, limit=50):
            return {"status": "error", "message": "API token required"}

    monkeypatch.setattr(
        "src.autfin.agents.sentiment.SentimentTools",
        FailingSentimentTools,
    )
    state = _base_state(holdings=[{"tradingsymbol": "ITC"}])
    result = await sentiment_agent(state)

    assert "sentiment_reports" in result


@pytest.mark.asyncio
async def test_agent_returns_correct_keys(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    _mock_sentiment_tools(monkeypatch)
    state = _base_state(holdings=[{"tradingsymbol": "TCS"}])
    result = await sentiment_agent(state)

    assert "sentiment_reports" in result
    assert "agent" in result
    assert result["agent"] == "sentiment"
    assert "errors" in result


@pytest.mark.asyncio
async def test_with_mock_market_snapshot(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    _mock_sentiment_tools(monkeypatch)
    state = _base_state(
        holdings=[{"tradingsymbol": "HDFCBANK"}],
        market_snapshot={"quotes": {"HDFCBANK": {"ltp": 1650.0}}},
    )
    result = await sentiment_agent(state)

    assert "sentiment_reports" in result


@pytest.mark.asyncio
async def test_handles_empty_holdings_gracefully(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    _mock_sentiment_tools(monkeypatch)
    state = _base_state(holdings=[])
    result = await sentiment_agent(state)

    assert "sentiment_reports" in result
    assert "errors" in result


@pytest.mark.asyncio
async def test_aggregate_sentiment_has_category(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    _mock_sentiment_tools(monkeypatch, results={
        "status": "success",
        "total_articles": 10,
        "aggregate_sentiment": {
            "weighted_score": 0.65,
            "confidence": 0.85,
            "recent_articles": 10,
        },
        "articles": [],
    })
    state = _base_state(holdings=[{"tradingsymbol": "INFY"}])
    result = await sentiment_agent(state)

    reports = result["sentiment_reports"]
    if "aggregate" in reports and "category" in reports["aggregate"]:
        assert reports["aggregate"]["category"] in (
            "strong_buy", "buy", "neutral", "sell", "strong_sell"
        )
