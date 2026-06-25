"""
Tests for the Researcher Agent. All tests use mocked state and do not require real API keys.
"""
import pytest
from unittest.mock import MagicMock

from src.autfin.agents.researcher import researcher_agent


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


def _make_llm_unavailable(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.is_available = False
    monkeypatch.setattr(
        "src.autfin.agents.researcher.get_llm_client", lambda: mock_llm
    )


def _build_research_state(*, with_technical=True, with_fundamentals=True,
                          with_sentiment=True, with_macro=True) -> dict:
    state = {}
    if with_technical:
        state["technical_reports"] = {
            "reports": {
                "RELIANCE": {
                    "signal": {
                        "signal": "BUY",
                        "confidence": 0.75,
                        "indicator_scores": {"trend": 2, "momentum": 1},
                        "reasoning": ["Golden Cross detected"],
                        "pattern_signals": [{"type": "bullish", "pattern": "Hammer"}],
                    },
                    "timestamp": "2026-01-01T09:00:00",
                },
            },
        }
    if with_fundamentals:
        state["fundamental_reports"] = {
            "reports": {
                "RELIANCE": {
                    "health_score": 72.5,
                    "health_breakdown": {
                        "profitability": {"score": 22.0, "max": 30},
                    },
                    "ratios": {"pe_ratio": 18.5, "roe": 22.5},
                    "timestamp": "2026-01-01T09:00:00",
                },
            },
        }
    if with_sentiment:
        state["sentiment_reports"] = {
            "aggregate": {
                "weighted_score": 0.35,
                "category": "buy",
                "confidence": 0.70,
                "symbol_count": 1,
                "sentiment_categories": {"buy": 1},
            },
            "timestamp": "2026-01-01T09:00:00",
        }
    if with_macro:
        state["macro_signals"] = {
            "indicators": {"nifty_50": {"value": 23000}},
            "impact_analysis": {
                "impact_score": "medium_positive",
                "recommendations": ["Maintain moderate exposure"],
                "correlations": {},
            },
            "timestamp": "2026-01-01T09:00:00",
        }
    return state


@pytest.mark.asyncio
async def test_agent_synthesizes_research_from_multiple_sources(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    research = _build_research_state()
    state = _base_state(**research)
    result = await researcher_agent(state)

    assert "research_debates" in result
    debates = result["research_debates"]
    assert debates is not None
    assert "summary" in debates
    assert "recommendation" in debates["summary"]
    assert "conviction_score" in debates["summary"]


@pytest.mark.asyncio
async def test_missing_research_debates_handling_does_not_crash(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    state = _base_state(research_debates={})
    result = await researcher_agent(state)

    assert "research_debates" in result
    assert result["research_debates"] is None
    assert any("No research data" in e for e in result["errors"])


@pytest.mark.asyncio
async def test_agent_returns_correct_keys(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    research = _build_research_state()
    state = _base_state(**research)
    result = await researcher_agent(state)

    assert "research_debates" in result
    assert "researcher_notes" in result
    assert "researcher_recommendation" in result
    assert "agent" in result
    assert result["agent"] == "researcher"
    assert "errors" in result


@pytest.mark.asyncio
async def test_with_full_aggregate_research_data(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    research = _build_research_state()
    state = _base_state(**research)
    result = await researcher_agent(state)

    debates = result["research_debates"]
    assert "bull" in debates
    assert "bear" in debates
    assert "neutral" in debates
    assert "count" in debates["bull"]
    assert isinstance(debates["bull"]["count"], int)


@pytest.mark.asyncio
async def test_conviction_scoring(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    research = _build_research_state()
    state = _base_state(**research)
    result = await researcher_agent(state)

    rec = result["researcher_recommendation"]
    assert "conviction_score" in rec
    assert 0.0 <= rec["conviction_score"] <= 1.0
    assert "recommendation" in rec
    assert rec["recommendation"] in ("strong_buy", "buy", "hold", "sell", "strong_sell")


@pytest.mark.asyncio
async def test_bullish_signals_produce_bullish_sections(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    research = _build_research_state()
    state = _base_state(**research)
    result = await researcher_agent(state)

    debates = result["research_debates"]
    assert "bull" in debates
    assert isinstance(debates["bull"].get("signals"), list)


@pytest.mark.asyncio
async def test_all_sources_missing_produces_no_data_error(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    state = _base_state(
        technical_reports={},
        fundamental_reports={},
        sentiment_reports={},
        macro_signals={},
    )
    result = await researcher_agent(state)

    assert any("No research data" in e for e in result["errors"])


@pytest.mark.asyncio
async def test_partial_data_synthesizes_with_available_sources(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    research = _build_research_state(with_fundamentals=False, with_sentiment=False)
    state = _base_state(**research)
    result = await researcher_agent(state)

    debates = result["research_debates"]
    assert debates is not None
    assert "summary" in debates


@pytest.mark.asyncio
async def test_researcher_notes_generated(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    research = _build_research_state()
    state = _base_state(**research)
    result = await researcher_agent(state)

    assert isinstance(result["researcher_notes"], list)
    assert len(result["researcher_notes"]) > 0
