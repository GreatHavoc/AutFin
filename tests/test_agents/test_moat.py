"""Tests for the Moat Assessment Agent."""
import pytest
from src.autfin.agents.moat import moat_agent


def _base_state(**overrides) -> dict:
    state = {
        "messages": [],
        "available_cash": 100_000.0,
        "utilized_margin": 0.0,
        "holdings": [],
        "positions": [],
        "candidates": [],
        "fundamental_reports": {},
        "research_debates": {},
        "risk_assessments": {},
        "macro_signals": {},
        "thread_id": "test-thread",
        "run_mode": "daily_morning",
        "errors": [],
    }
    state.update(overrides)
    return state


def _fundamental_report(**ratios) -> dict:
    return {"ratios": ratios, "market_cap": ratios.get("market_cap", 0)}


def _build_report(symbol: str, **ratios) -> dict:
    return {symbol: _fundamental_report(**ratios)}


@pytest.mark.asyncio
async def test_moat_agent_scores_high_for_quality_company():
    state = _base_state(
        fundamental_reports=_build_report(
            "RELIANCE",
            roe=0.30,
            pb=8.0,
            debt_equity=0.2,
            revenue_growth=0.30,
            fcf_yield=0.10,
            market_cap=15_00_000_00_00_000,
        ),
    )
    result = await moat_agent(state, symbol="RELIANCE")
    assessment = result["moat_assessments"]["RELIANCE"]
    assert assessment["score"] >= 75
    assert assessment["rating"] == "WIDE"
    assert "sources" in assessment
    assert len(assessment["sources"]) == 5


@pytest.mark.asyncio
async def test_moat_agent_scores_low_for_weak_company():
    state = _base_state(
        fundamental_reports=_build_report(
            "WEAKCO",
            roe=0.05,
            pb=0.8,
            debt_equity=5.0,
            revenue_growth=0.02,
            fcf_yield=-0.02,
            market_cap=10_00_00_00_000,
        ),
    )
    result = await moat_agent(state, symbol="WEAKCO")
    assessment = result["moat_assessments"]["WEAKCO"]
    assert assessment["score"] < 50
    assert assessment["rating"] == "NONE"


@pytest.mark.asyncio
async def test_moat_agent_returns_wide_rating_for_high_score():
    state = _base_state(
        fundamental_reports=_build_report(
            "WIDE",
            roe=0.28,
            pb=6.0,
            debt_equity=0.15,
            revenue_growth=0.35,
            fcf_yield=0.12,
            market_cap=20_00_000_00_00_000,
        ),
    )
    result = await moat_agent(state, symbol="WIDE")
    assert result["moat_assessments"]["WIDE"]["rating"] == "WIDE"


@pytest.mark.asyncio
async def test_moat_agent_returns_narrow_rating_for_medium_score():
    state = _base_state(
        fundamental_reports=_build_report(
            "NARROW",
            roe=0.22,
            pb=3.0,
            debt_equity=0.4,
            revenue_growth=0.16,
            fcf_yield=0.09,
            market_cap=5_00_000_00_00_000,
        ),
    )
    result = await moat_agent(state, symbol="NARROW")
    assert result["moat_assessments"]["NARROW"]["rating"] == "NARROW"


@pytest.mark.asyncio
async def test_moat_agent_returns_none_rating_for_low_score():
    state = _base_state(
        fundamental_reports=_build_report(
            "NONE",
            roe=0.02,
            pb=0.5,
            debt_equity=8.0,
            revenue_growth=-0.05,
            fcf_yield=-0.10,
            market_cap=50_00_00_00_000,
        ),
    )
    result = await moat_agent(state, symbol="NONE")
    assert result["moat_assessments"]["NONE"]["rating"] == "NONE"


@pytest.mark.asyncio
async def test_moat_agent_handles_missing_data():
    state = _base_state(
        fundamental_reports=_build_report(
            "MISSING",
        ),
    )
    result = await moat_agent(state, symbol="MISSING")
    assessment = result["moat_assessments"]["MISSING"]
    assert assessment["score"] >= 0
    assert assessment["score"] <= 100
    assert assessment["rating"] in ("WIDE", "NARROW", "NONE")


@pytest.mark.asyncio
async def test_moat_agent_auto_symbol_from_holdings():
    state = _base_state(
        holdings=[{"tradingsymbol": "RELIANCE"}],
        fundamental_reports=_build_report(
            "RELIANCE",
            roe=0.18,
            pb=3.0,
            debt_equity=0.5,
            revenue_growth=0.12,
            fcf_yield=0.05,
            market_cap=5_00_000_00_00_000,
        ),
    )
    result = await moat_agent(state)
    assert "RELIANCE" in result["moat_assessments"]
