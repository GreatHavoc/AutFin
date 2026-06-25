"""
Tests for the Fundamentals Agent. All tests use mocked state and do not require real API keys.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.autfin.agents.fundamentals import fundamentals_agent, _cache


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


def _mock_ratios():
    return {
        "pe_ratio": 18.5,
        "pb_ratio": 4.2,
        "debt_to_equity": 0.35,
        "roe": 22.5,
        "roa": 8.0,
        "net_margin": 15.2,
        "operating_margin": 22.0,
        "ev_ebitda": 12.0,
        "price_to_sales": 3.5,
        "interest_coverage": 8.5,
        "inventory_turnover": 6.0,
        "asset_turnover": 1.2,
        "revenue_growth": 18.0,
        "eps_growth": 22.0,
        "profit_growth": 20.0,
        "roic": 16.0,
        "fcf_yield": 4.5,
    }


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for key in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "FRED_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    _cache.clear()


def _make_llm_unavailable(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.is_available = False
    monkeypatch.setattr(
        "src.autfin.agents.fundamentals.get_llm_client", lambda: mock_llm
    )


@pytest.mark.asyncio
async def test_agent_processes_fundamental_reports_with_mock_data(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    monkeypatch.setattr(
        "src.autfin.agents.fundamentals._fetch_from_yfinance",
        lambda symbol: {
            "status": "success",
            "symbol": symbol,
            "data": {
                "company_info": {"sector": "IT", "industry": "Software"},
                "ratios": _mock_ratios(),
                "financials": {"revenue": 500000, "net_income": 75000},
                "growth_metrics": {"revenue_cagr_3yr": 18.0},
            },
        },
    )
    state = _base_state(holdings=[{"tradingsymbol": "TCS"}])
    result = await fundamentals_agent(state)

    assert "fundamental_reports" in result
    reports = result["fundamental_reports"]
    assert reports["symbols_analyzed"] >= 1
    assert "reports" in reports
    assert "TCS" in reports["reports"]
    tcs_report = reports["reports"]["TCS"]
    assert "health_score" in tcs_report
    assert "health_breakdown" in tcs_report
    assert "moat_level" in tcs_report


@pytest.mark.asyncio
async def test_error_handling_when_no_data_available(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    monkeypatch.setattr(
        "src.autfin.agents.fundamentals._fetch_from_yfinance",
        lambda symbol: {"status": "error", "error": "yfinance not installed"},
    )
    state = _base_state(holdings=[{"tradingsymbol": "RELIANCE"}])
    result = await fundamentals_agent(state)

    assert "errors" in result
    assert any("Failed to analyze" in e for e in result["errors"])


@pytest.mark.asyncio
async def test_agent_returns_correct_keys(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    monkeypatch.setattr(
        "src.autfin.agents.fundamentals._fetch_from_yfinance",
        lambda symbol: {
            "status": "success",
            "symbol": symbol,
            "data": {
                "company_info": {"sector": "Energy"},
                "ratios": _mock_ratios(),
                "financials": {},
                "growth_metrics": {},
            },
        },
    )
    state = _base_state(holdings=[{"tradingsymbol": "RELIANCE"}])
    result = await fundamentals_agent(state)

    assert "fundamental_reports" in result
    assert "agent" in result
    assert result["agent"] == "fundamentals"
    assert "errors" in result
    report = result["fundamental_reports"]
    assert "timestamp" in report
    assert "symbols_analyzed" in report
    assert "reports" in report
    assert "health_score" in report["reports"]["RELIANCE"]
    assert "moat_level" in report["reports"]["RELIANCE"]
    assert "health_breakdown" in report["reports"]["RELIANCE"]


@pytest.mark.asyncio
async def test_with_mock_market_snapshot_providing_symbol(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    monkeypatch.setattr(
        "src.autfin.agents.fundamentals._fetch_from_yfinance",
        lambda symbol: {
            "status": "success",
            "symbol": symbol,
            "data": {
                "company_info": {"sector": "Bank"},
                "ratios": _mock_ratios(),
                "financials": {},
                "growth_metrics": {},
            },
        },
    )
    state = _base_state(
        holdings=[{"tradingsymbol": "HDFCBANK"}],
        market_snapshot={"quotes": {"HDFCBANK": {"ltp": 1650.0}}},
    )
    result = await fundamentals_agent(state)

    assert "HDFCBANK" in result["fundamental_reports"]["reports"]
    hdfc_report = result["fundamental_reports"]["reports"]["HDFCBANK"]
    assert hdfc_report["status"] == "success"
    assert hdfc_report["health_score"] >= 0


@pytest.mark.asyncio
async def test_health_score_is_between_0_and_100(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    monkeypatch.setattr(
        "src.autfin.agents.fundamentals._fetch_from_yfinance",
        lambda symbol: {
            "status": "success",
            "symbol": symbol,
            "data": {
                "company_info": {"sector": "IT"},
                "ratios": _mock_ratios(),
                "financials": {},
                "growth_metrics": {},
            },
        },
    )
    state = _base_state(holdings=[{"tradingsymbol": "TCS"}])
    result = await fundamentals_agent(state)

    report = result["fundamental_reports"]["reports"]["TCS"]
    assert 0 <= report["health_score"] <= 100


@pytest.mark.asyncio
async def test_moat_returns_valid_level(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    monkeypatch.setattr(
        "src.autfin.agents.fundamentals._fetch_from_yfinance",
        lambda symbol: {
            "status": "success",
            "symbol": symbol,
            "data": {
                "company_info": {"sector": "FMCG"},
                "ratios": {**_mock_ratios(), "roe": 28.0, "operating_margin": 30.0},
                "financials": {},
                "growth_metrics": {},
            },
        },
    )
    state = _base_state(holdings=[{"tradingsymbol": "ITC"}])
    result = await fundamentals_agent(state)

    report = result["fundamental_reports"]["reports"]["ITC"]
    assert report["moat_level"] in ("wide", "narrow", "none")
    assert 0 <= report["moat_score"] <= 100
    assert isinstance(report["moat_rationale"], str)
