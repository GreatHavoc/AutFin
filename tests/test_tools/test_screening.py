"""Tests for v3.0 screening pipeline."""
import pytest
from src.autfin.tools.screening import (
    liquidity_filter,
    magic_formula_rank,
    piotroski_f_score,
    indian_red_flags,
    NIFTY_500_SYMBOLS,
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for key in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# liquidity_filter tests
# ---------------------------------------------------------------------------

def test_liquidity_filter_removes_small_caps():
    stocks = [
        {"symbol": "SMALL", "avg_volume": 50_000, "market_cap": 100_00_00_000},
        {"symbol": "TINY", "avg_volume": 10_000, "market_cap": 10_00_00_000},
    ]
    result = liquidity_filter(stocks)
    assert len(result) == 0


def test_liquidity_filter_keeps_large_caps():
    stocks = [
        {"symbol": "BIG", "avg_volume": 5_00_00_000, "market_cap": 5_000_00_00_000},
        {"symbol": "MED", "avg_volume": 2_00_00_000, "market_cap": 1_000_00_00_000},
        {"symbol": "SMALL", "avg_volume": 50_000, "market_cap": 100_00_00_000},
    ]
    result = liquidity_filter(stocks)
    assert len(result) == 2
    symbols = {s["symbol"] for s in result}
    assert symbols == {"BIG", "MED"}


def test_liquidity_filter_handles_missing_fields():
    stocks = [
        {"symbol": "NO_VOL"},
        {"symbol": "OK", "avg_volume": 2_00_00_000, "market_cap": 1_000_00_00_000},
    ]
    result = liquidity_filter(stocks)
    assert len(result) == 1
    assert result[0]["symbol"] == "OK"


# ---------------------------------------------------------------------------
# magic_formula_rank tests
# ---------------------------------------------------------------------------

def test_magic_formula_ranks_by_ey_and_roic():
    stocks = [
        {
            "symbol": "A", "ebit": 1_000, "market_cap": 10_000,
            "total_debt": 0, "cash": 0, "net_fixed_assets": 5_000,
            "current_assets": 2_000, "current_liabilities": 1_000,
        },
        {
            "symbol": "B", "ebit": 500, "market_cap": 5_000,
            "total_debt": 0, "cash": 0, "net_fixed_assets": 2_000,
            "current_assets": 1_000, "current_liabilities": 500,
        },
    ]
    result = magic_formula_rank(stocks)
    assert len(result) == 2
    for s in result:
        assert "earnings_yield" in s
        assert "roic" in s
        assert "magic_rank" in s
    # Higher earnings_yield + higher ROIC → lower magic_rank → sorted first
    assert result[0]["magic_rank"] <= result[1]["magic_rank"]
    assert result[0]["earnings_yield"] >= result[1]["earnings_yield"]


def test_magic_formula_caps_at_85():
    stocks = [
        {
            "symbol": f"S{i:03d}", "ebit": 1_000, "market_cap": 10_000,
            "total_debt": 0, "cash": 0, "net_fixed_assets": 5_000,
            "current_assets": 2_000, "current_liabilities": 1_000,
        }
        for i in range(100)
    ]
    result = magic_formula_rank(stocks)
    assert len(result) == 85


def test_magic_formula_handles_zero_ev():
    stocks = [{"symbol": "ZERO", "ebit": 0, "market_cap": 0, "total_debt": 0,
                "cash": 0, "net_fixed_assets": 0, "current_assets": 0, "current_liabilities": 0}]
    result = magic_formula_rank(stocks)
    assert len(result) == 1
    assert result[0]["earnings_yield"] == 0.0
    assert result[0]["roic"] == 0.0


# ---------------------------------------------------------------------------
# piotroski_f_score tests
# ---------------------------------------------------------------------------

def test_piotroski_f_score_perfect_company_scores_9():
    current = {
        "roa": 0.15, "operating_cf": 3_000, "total_assets": 10_000,
        "current_assets": 5_000, "current_liabilities": 2_000,
        "long_term_debt": 1_000, "shares_outstanding": 1_000_000,
        "gross_profit": 4_000, "revenue": 10_000,
    }
    prior = {
        "roa": 0.10, "operating_cf": 500, "total_assets": 8_000,
        "current_assets": 4_500, "current_liabilities": 3_000,
        "long_term_debt": 1_500, "shares_outstanding": 1_000_000,
        "gross_profit": 2_500, "revenue": 7_000,
    }
    score = piotroski_f_score(current, prior)
    assert score == 9


def test_piotroski_f_score_struggling_company_scores_low():
    current = {
        "roa": -0.05, "operating_cf": -500, "total_assets": 10_000,
        "current_assets": 1_000, "current_liabilities": 3_000,
        "long_term_debt": 5_000, "shares_outstanding": 1_500_000,
        "gross_profit": 1_000, "revenue": 10_000,
    }
    prior = {
        "roa": 0.05, "operating_cf": 200, "total_assets": 9_000,
        "current_assets": 2_000, "current_liabilities": 1_500,
        "long_term_debt": 4_000, "shares_outstanding": 1_000_000,
        "gross_profit": 2_000, "revenue": 8_000,
    }
    score = piotroski_f_score(current, prior)
    assert score <= 3


def test_piotroski_f_score_handles_missing_keys():
    current = {}
    prior = {}
    score = piotroski_f_score(current, prior)
    assert score == 1  # only shares_outstanding unchanged (default 1e12 = 1e12, no dilution = 1 point)


# ---------------------------------------------------------------------------
# indian_red_flags tests
# ---------------------------------------------------------------------------

def test_indian_red_flags_detects_promoter_pledge():
    stock = {"promoter_pledge_pct": 35}
    flags = indian_red_flags(stock)
    assert len(flags) >= 1
    assert any("Promoter pledge" in f for f in flags)


def test_indian_red_flags_empty_for_clean_company():
    stock = {
        "promoter_holding_change_4q": 0.02,
        "promoter_pledge_pct": 5,
        "auditor_qualification_2y": False,
        "related_party_txn_pct": 3,
        "negative_ocf_streak": 0,
        "sebi_penalty_12m": False,
        "debt_equity": 0.5,
    }
    flags = indian_red_flags(stock)
    assert len(flags) == 0


def test_indian_red_flags_detects_auditor_qualification():
    stock = {"auditor_qualification_2y": True}
    flags = indian_red_flags(stock)
    assert any("Auditor qualification" in f for f in flags)


def test_indian_red_flags_detects_critical_pledge_and_debt():
    stock = {"promoter_pledge_pct": 65, "debt_equity": 3.0}
    flags = indian_red_flags(stock)
    assert any("CRITICAL" in f for f in flags)


# ---------------------------------------------------------------------------
# run_screening_pipeline / fetch_yfinance_bulk tests (skip if yfinance absent)
# ---------------------------------------------------------------------------

try:
    import yfinance  # noqa: F401
    _HAS_YFINANCE = True
except ImportError:
    _HAS_YFINANCE = False


@pytest.mark.asyncio
@pytest.mark.skip(reason="yfinance network calls hang in CI/no-network environments")
async def test_fetch_yfinance_bulk_returns_data_for_symbols():
    from src.autfin.tools.screening import fetch_yfinance_bulk
    result = await fetch_yfinance_bulk(["RELIANCE", "TCS"])
    assert isinstance(result, list)
    assert len(result) == 2
    for stock in result:
        assert "symbol" in stock
        assert "market_cap" in stock
        assert "avg_volume" in stock
        assert stock["symbol"] in ("RELIANCE", "TCS")


@pytest.mark.asyncio
@pytest.mark.skip(reason="yfinance network calls hang in CI/no-network environments")
async def test_run_screening_pipeline_returns_candidates():
    from src.autfin.tools.screening import run_screening_pipeline
    result = await run_screening_pipeline()
    assert "candidates" in result
    assert "candidate_details" in result
    assert "stats" in result
    assert isinstance(result["candidates"], list)
    assert isinstance(result["stats"]["universe_size"], int)
    assert result["stats"]["universe_size"] == len(NIFTY_500_SYMBOLS)


# ---------------------------------------------------------------------------
# NIFTY_500_SYMBOLS sanity checks
# ---------------------------------------------------------------------------

def test_nifty_symbols_contains_major_stocks():
    assert "RELIANCE" in NIFTY_500_SYMBOLS
    assert "TCS" in NIFTY_500_SYMBOLS


# ---------------------------------------------------------------------------
# Market Regime Detection & F-Score Threshold (Phase 2.6)
# ---------------------------------------------------------------------------

def test_get_fscore_threshold_bull():
    from src.autfin.tools.screening import get_fscore_threshold
    assert get_fscore_threshold("bull") == 5


def test_get_fscore_threshold_bear():
    from src.autfin.tools.screening import get_fscore_threshold
    assert get_fscore_threshold("bear") == 7


def test_get_fscore_threshold_neutral():
    from src.autfin.tools.screening import get_fscore_threshold
    assert get_fscore_threshold("neutral") == 6


def test_get_fscore_threshold_unknown_defaults_to_6():
    from src.autfin.tools.screening import get_fscore_threshold
    assert get_fscore_threshold("unknown") == 6


@pytest.mark.asyncio
async def test_detect_market_regime_returns_structure():
    from src.autfin.tools.screening import detect_market_regime
    result = await detect_market_regime()
    assert "regime" in result
    assert result["regime"] in ("bull", "bear", "neutral")
    assert "nifty_price" in result
    assert "sma_50" in result
    assert "sma_200" in result
    assert "confidence" in result


@pytest.mark.asyncio
async def test_detect_market_regime_handles_yfinance_failure():
    from src.autfin.tools.screening import detect_market_regime
    import unittest.mock as mock
    with mock.patch("yfinance.Ticker") as MockTicker:
        MockTicker.return_value.history.return_value = []
        result = await detect_market_regime()
        assert result["regime"] == "neutral"
    assert "HDFCBANK" in NIFTY_500_SYMBOLS
    assert len(NIFTY_500_SYMBOLS) >= 50
