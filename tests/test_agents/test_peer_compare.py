"""Tests for the Peer Comparison Agent."""
import pytest
from src.autfin.agents.peer_compare import (
    peer_compare_agent,
    _get_peer_symbols,
    _extract_metrics,
    _rank_metrics,
)


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


# ---------------------------------------------------------------------------
# _get_peer_symbols tests (no mocking needed)
# ---------------------------------------------------------------------------

def test_get_peer_symbols_returns_peers_for_known_symbol():
    peers = _get_peer_symbols("RELIANCE")
    assert isinstance(peers, list)
    assert len(peers) >= 1
    assert "ONGC" in peers


def test_get_peer_symbols_returns_peers_for_it_stock():
    peers = _get_peer_symbols("TCS")
    assert "INFY" in peers or "HCLTECH" in peers
    assert len(peers) >= 2


def test_get_peer_symbols_returns_empty_for_unknown_symbol():
    peers = _get_peer_symbols("UNKNOWNSYMBOL")
    assert peers == []


# ---------------------------------------------------------------------------
# _extract_metrics tests
# ---------------------------------------------------------------------------

def test_extract_metrics_extracts_all_8_keys():
    info = {
        "trailingPE": 25.0,
        "priceToBook": 4.5,
        "returnOnEquity": 0.22,
        "freeCashflow": 1_000_000,
        "marketCap": 10_000_000,
        "revenueGrowth": 0.15,
        "profitMargins": 0.20,
        "debtToEquity": 30.0,
    }
    metrics = _extract_metrics(info)
    assert "pe" in metrics
    assert "pb" in metrics
    assert "roe" in metrics
    assert "fcf_yield" in metrics
    assert "revenue_growth" in metrics
    assert "net_margin" in metrics
    assert "debt_equity" in metrics
    assert "earnings_yield" in metrics
    assert metrics["pe"] == 25.0
    assert metrics["roe"] == 0.22
    assert metrics["fcf_yield"] == 0.1


def test_extract_metrics_handles_missing_fields():
    info = {}
    metrics = _extract_metrics(info)
    assert metrics["fcf_yield"] == 0.0
    assert metrics["earnings_yield"] == 0


# ---------------------------------------------------------------------------
# _rank_metrics tests
# ---------------------------------------------------------------------------

def test_rank_metrics_returns_leader_position():
    peers = {
        "TARGET": {
            "pe": 15.0, "pb": 2.0, "roe": 0.25, "fcf_yield": 0.08,
            "revenue_growth": 0.20, "net_margin": 25.0, "debt_equity": -10.0,
            "earnings_yield": 0.0667,
        },
        "PEER_A": {
            "pe": 25.0, "pb": 4.0, "roe": 0.12, "fcf_yield": 0.03,
            "revenue_growth": 0.10, "net_margin": 15.0, "debt_equity": -30.0,
            "earnings_yield": 0.04,
        },
        "PEER_B": {
            "pe": 30.0, "pb": 5.0, "roe": 0.08, "fcf_yield": 0.02,
            "revenue_growth": 0.05, "net_margin": 10.0, "debt_equity": -50.0,
            "earnings_yield": 0.0333,
        },
    }
    result = _rank_metrics(peers, "TARGET")
    assert result["position"] == "LEADER"
    assert result["overall_percentile"] >= 70
    assert "rankings" in result
    assert result["peers_count"] == 3
    assert len(result["rankings"]) >= 6


def test_rank_metrics_returns_laggard_position():
    peers = {
        "TARGET": {
            "pe": 40.0, "pb": 8.0, "roe": 0.04, "fcf_yield": 0.005,
            "revenue_growth": 0.01, "net_margin": 3.0, "debt_equity": -120.0,
            "earnings_yield": 0.025,
        },
        "PEER_A": {
            "pe": 15.0, "pb": 2.0, "roe": 0.25, "fcf_yield": 0.08,
            "revenue_growth": 0.20, "net_margin": 25.0, "debt_equity": -10.0,
            "earnings_yield": 0.0667,
        },
        "PEER_B": {
            "pe": 18.0, "pb": 3.0, "roe": 0.18, "fcf_yield": 0.05,
            "revenue_growth": 0.15, "net_margin": 20.0, "debt_equity": -15.0,
            "earnings_yield": 0.0556,
        },
    }
    result = _rank_metrics(peers, "TARGET")
    assert result["position"] == "LAGGARD"
    assert result["overall_percentile"] < 40


def test_rank_metrics_handles_missing_target():
    peers = {"OTHER": {"pe": 15.0}}
    result = _rank_metrics(peers, "MISSING")
    assert "error" in result


# ---------------------------------------------------------------------------
# peer_compare_agent tests (mock yfinance)
# ---------------------------------------------------------------------------

@pytest.fixture
def _mock_yfinance(monkeypatch):
    class MockTicker:
        def __init__(self, ticker_str):
            self._ticker = ticker_str

        @property
        def info(self):
            return _MOCK_YF_INFO.get(self._ticker.replace(".NS", ""), {})

    _MOCK_YF_INFO = {
        "RELIANCE": {
            "trailingPE": 25.0, "priceToBook": 3.0, "returnOnEquity": 0.18,
            "freeCashflow": 5_000_000_000, "marketCap": 15_000_000_000,
            "revenueGrowth": 0.12, "profitMargins": 0.15,
            "debtToEquity": 20.0, "sector": "Oil & Gas",
        },
        "ONGC": {
            "trailingPE": 8.0, "priceToBook": 0.8, "returnOnEquity": 0.12,
            "freeCashflow": 2_000_000_000, "marketCap": 5_000_000_000,
            "revenueGrowth": 0.05, "profitMargins": 0.10,
            "debtToEquity": 15.0,
        },
        "BPCL": {
            "trailingPE": 10.0, "priceToBook": 1.2, "returnOnEquity": 0.15,
            "freeCashflow": 3_000_000_000, "marketCap": 8_000_000_000,
            "revenueGrowth": 0.08, "profitMargins": 0.12,
            "debtToEquity": 25.0,
        },
    }

    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", MockTicker)
    return _MOCK_YF_INFO


@pytest.mark.asyncio
async def test_peer_compare_agent_returns_rankings(_mock_yfinance):
    state = _base_state()
    result = await peer_compare_agent(state, symbol="RELIANCE")
    comp = result["peer_comparisons"]["RELIANCE"]
    assert "rankings" in comp
    assert "position" in comp
    assert "overall_percentile" in comp
    assert comp["peers_count"] >= 2
    assert comp["position"] in ("LEADER", "MID-PACK", "LAGGARD")


@pytest.mark.asyncio
async def test_peer_compare_agent_handles_leader_position(_mock_yfinance):
    state = _base_state()
    result = await peer_compare_agent(state, symbol="RELIANCE")
    comp = result["peer_comparisons"]["RELIANCE"]
    # RELIANCE has higher ROE, revenue_growth, profitMargins than peers in mock data
    assert comp["overall_percentile"] > 50


@pytest.mark.asyncio
async def test_peer_compare_agent_handles_no_peers(_mock_yfinance, monkeypatch):
    import src.autfin.agents.peer_compare as pc
    monkeypatch.setattr(pc, "PEER_MAP", {"RELIANCE": []})
    state = _base_state()
    result = await peer_compare_agent(state, symbol="RELIANCE")
    comp = result["peer_comparisons"]["RELIANCE"]
    assert "error" in comp
    assert comp["peers_count"] <= 1
