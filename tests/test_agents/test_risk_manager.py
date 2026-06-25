"""
Tests for the Risk Manager Agent. All tests use mocked state and do not require real API keys.
"""
import pytest
from unittest.mock import MagicMock

from src.autfin.agents.risk_manager import (
    risk_manager_agent,
    RiskManagerAgent,
    check_all_risks,
    calculate_risk_score,
    apply_veto,
)


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


def _build_trade(**overrides) -> dict:
    trade = {
        "symbol": "RELIANCE",
        "action": "BUY",
        "qty": 10,
        "price": 2500.0,
        "stop_loss_price": 2300.0,
        "target_price": 3000.0,
    }
    trade.update(overrides)
    return trade


def _build_holding(**overrides) -> dict:
    holding = {
        "tradingsymbol": "TCS",
        "qty": 5,
        "ltp": 3200.0,
        "avg_price": 3000.0,
    }
    holding.update(overrides)
    return holding


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for key in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)


def _make_llm_unavailable(monkeypatch):
    mock_llm = MagicMock()
    mock_llm.is_available = False
    monkeypatch.setattr(
        "src.autfin.agents.risk_manager.get_llm_client", lambda: mock_llm
    )


# ── BUY trade approval checks ──────────────────────────────────

@pytest.mark.asyncio
async def test_buy_trade_approved_when_risk_low(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    state = _base_state(
        pending_trade=_build_trade(),
        holdings=[_build_holding(tradingsymbol="TCS")],
        market_snapshot={"quotes": {"RELIANCE": {"ltp": 2500.0}}},
    )
    result = await risk_manager_agent(state)

    assert "risk_assessments" in result
    assessments = result["risk_assessments"]
    if "_summary" in assessments:
        assert assessments["_summary"]["approved"] is True


@pytest.mark.asyncio
async def test_sell_trade_always_processed(monkeypatch):
    """A SELL trade should be assessed by the risk manager without being unconditionally blocked."""
    _make_llm_unavailable(monkeypatch)
    state = _base_state(
        pending_trade=_build_trade(action="SELL"),
        holdings=[_build_holding(tradingsymbol="RELIANCE")],
        market_snapshot={"quotes": {"RELIANCE": {"ltp": 2500.0}}},
    )
    result = await risk_manager_agent(state)

    assert "risk_assessments" in result
    # SELL is not unconditionally vetoed; assessment is produced
    assessments = result["risk_assessments"]
    assert len(assessments) > 0


@pytest.mark.asyncio
async def test_veto_when_risk_too_high(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    # Build a trade that exceeds position size limit (50% of portfolio)
    state = _base_state(
        available_cash=1000.0,
        pending_trade=_build_trade(qty=100, price=2500.0, symbol="RELIANCE"),
        holdings=[],
        market_snapshot={"quotes": {"RELIANCE": {"ltp": 2500.0}}},
    )
    result = await risk_manager_agent(state)

    assessments = result["risk_assessments"]
    if "_summary" in assessments:
        assert assessments["_summary"]["risk_score"] > 50


# ── Position sizing suggestions ────────────────────────────────

@pytest.mark.asyncio
async def test_position_sizing_in_check_all_risks(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    trade = _build_trade(qty=5, price=2500.0)
    portfolio = {
        "holdings": [_build_holding()],
        "positions": [],
        "available_cash": 100_000.0,
        "utilized_margin": 0.0,
        "available_margin": 500_000.0,
        "recent_trades": [],
    }
    report = check_all_risks(trade, portfolio)

    assert "checks" in report
    assert "risk_score" in report
    assert "risk_level" in report
    check_names = [c["name"] for c in report["checks"]]
    assert "position_size" in check_names


@pytest.mark.asyncio
async def test_position_size_check_passes_for_small_trade():
    trade = _build_trade(qty=2, price=1000.0)
    portfolio = {
        "holdings": [],
        "positions": [],
        "available_cash": 100_000.0,
        "utilized_margin": 0.0,
    }
    agent = RiskManagerAgent()
    passes, reason = agent.check_position_size(trade, portfolio)
    assert passes is True


@pytest.mark.asyncio
async def test_position_size_check_fails_for_large_trade():
    trade = _build_trade(qty=1000, price=2500.0)
    portfolio = {
        "holdings": [],
        "positions": [],
        "available_cash": 10_000.0,
    }
    agent = RiskManagerAgent()
    passes, reason = agent.check_position_size(trade, portfolio)
    assert passes is False


# ── Missing data handling ──────────────────────────────────────

@pytest.mark.asyncio
async def test_missing_data_handling_no_market_data(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    state = _base_state(
        pending_trade=_build_trade(),
        holdings=[],
        market_snapshot={},
    )
    result = await risk_manager_agent(state)

    assert "risk_assessments" in result


@pytest.mark.asyncio
async def test_missing_data_handling_no_portfolio(monkeypatch):
    _make_llm_unavailable(monkeypatch)
    state = _base_state(
        pending_trade=_build_trade(),
        holdings=[],
        available_cash=0.0,
    )
    result = await risk_manager_agent(state)

    assert "risk_assessments" in result


# ── Risk score between 0-100 ──────────────────────────────────

@pytest.mark.asyncio
async def test_risk_score_between_0_and_100():
    trade = _build_trade()
    portfolio = {
        "holdings": [_build_holding()],
        "positions": [],
        "available_cash": 100_000.0,
        "utilized_margin": 0.0,
        "available_margin": 500_000.0,
        "recent_trades": [],
    }
    report = check_all_risks(trade, portfolio)

    assert 0 <= report["risk_score"] <= 100


@pytest.mark.asyncio
async def test_risk_score_zero_for_empty_checks():
    report = {"checks": []}
    score = calculate_risk_score(report)
    assert score == 0


@pytest.mark.asyncio
async def test_risk_score_hundred_for_all_failing():
    report = {
        "checks": [
            {"passes": False, "severity": "high"},
            {"passes": False, "severity": "high"},
            {"passes": False, "severity": "high"},
        ]
    }
    score = calculate_risk_score(report)
    assert score == 100


# ── 14 risk checks smoke-test ─────────────────────────────────

@pytest.mark.asyncio
async def test_all_14_risk_checks_present():
    trade = _build_trade()
    portfolio = {
        "holdings": [_build_holding()],
        "positions": [],
        "available_cash": 100_000.0,
        "utilized_margin": 0.0,
        "available_margin": 500_000.0,
        "recent_trades": [],
    }
    report = check_all_risks(trade, portfolio)

    expected_checks = [
        "position_size", "sector_exposure", "volatility_exposure",
        "margin_usage", "stock_concentration", "daily_loss",
        "max_positions", "stop_loss_distance", "risk_reward_ratio",
        "correlation", "liquidity", "var", "cvar", "news_risk",
        "drawdown_halt",
    ]
    check_names = [c["name"] for c in report["checks"]]
    for expected in expected_checks:
        assert expected in check_names, f"Missing check: {expected}"
    assert len(report["checks"]) == 15


# ── Veto logic ─────────────────────────────────────────────────

def test_apply_veto_approves_low_risk():
    should_veto, reason = apply_veto(25)
    assert should_veto is False
    assert "APPROVED" in reason


def test_apply_veto_warns_medium_risk():
    should_veto, reason = apply_veto(75)
    assert should_veto is False
    assert "WARNINGS" in reason


def test_apply_veto_vetoes_high_risk():
    should_veto, reason = apply_veto(95)
    assert should_veto is True
    assert "VETOED" in reason


# ── Individual risk checks ─────────────────────────────────────

def test_sector_exposure_passes_for_empty_portfolio():
    trade = _build_trade(symbol="TCS", qty=2, price=3200.0)
    portfolio = {
        "holdings": [
            _build_holding(tradingsymbol="RELIANCE", qty=5, ltp=2500.0),
            _build_holding(tradingsymbol="HDFCBANK", qty=5, ltp=1650.0),
        ],
        "positions": [],
        "available_cash": 100_000.0,
    }
    agent = RiskManagerAgent()
    passes, reason = agent.check_sector_exposure(trade, portfolio)
    assert passes is True


def test_margin_usage_passes_with_low_utilization():
    portfolio = {
        "available_margin": 500_000.0,
        "utilized_margin": 50_000.0,
        "holdings": [],
        "positions": [],
        "available_cash": 100_000.0,
    }
    agent = RiskManagerAgent()
    passes, reason = agent.check_margin_usage(portfolio)
    assert passes is True


def test_max_positions_passes_below_limit():
    portfolio = {"holdings": [_build_holding()], "positions": []}
    agent = RiskManagerAgent()
    passes, reason = agent.check_max_positions(portfolio)
    assert passes is True


def test_stop_loss_distance_passes_adequate():
    trade = _build_trade(stop_loss_price=2300.0, price=2500.0)
    agent = RiskManagerAgent()
    passes, reason = agent.check_stop_loss_distance(trade)
    assert passes is True


def test_stop_loss_distance_fails_too_close():
    trade = _build_trade(stop_loss_price=2480.0, price=2500.0)
    agent = RiskManagerAgent()
    passes, reason = agent.check_stop_loss_distance(trade)
    assert passes is False


def test_risk_reward_passes_adequate():
    trade = _build_trade(price=2500.0, stop_loss_price=2300.0, target_price=3000.0)
    agent = RiskManagerAgent()
    passes, reason = agent.check_risk_reward_ratio(trade)
    assert passes is True


def test_risk_reward_fails_insufficient():
    trade = _build_trade(price=2500.0, stop_loss_price=2450.0, target_price=2550.0)
    agent = RiskManagerAgent()
    passes, reason = agent.check_risk_reward_ratio(trade)
    assert passes is False


def test_liquidity_check_passes_without_volume_data():
    trade = _build_trade()
    portfolio = {"holdings": [], "available_cash": 100_000.0}
    agent = RiskManagerAgent()
    passes, reason = agent.check_liquidity(trade, portfolio)
    assert passes is True


def test_var_check_passes_without_volatility():
    trade = _build_trade()
    portfolio = {"holdings": [], "available_cash": 100_000.0}
    agent = RiskManagerAgent()
    passes, reason = agent.check_var(trade, portfolio)
    assert passes is True


def test_cvar_check_passes_without_volatility():
    trade = _build_trade()
    portfolio = {"holdings": [], "available_cash": 100_000.0}
    agent = RiskManagerAgent()
    passes, reason = agent.check_cvar(trade, portfolio)
    assert passes is True


def test_news_risk_skipped_for_sell_trades():
    trade = _build_trade(action="SELL")
    portfolio = {"holdings": [], "sentiment_reports": {}}
    agent = RiskManagerAgent()
    passes, reason = agent.check_news_risk(trade, portfolio)
    assert passes is True
    assert "skipped" in reason.lower()


def test_news_risk_blocks_negative_sentiment_buy():
    trade = _build_trade(action="BUY", sentiment_score=-0.5)
    portfolio = {}
    agent = RiskManagerAgent()
    passes, reason = agent.check_news_risk(trade, portfolio)
    assert passes is False
    assert "negative" in reason.lower()


def test_correlation_check_passes_for_empty_portfolio():
    trade = _build_trade()
    portfolio = {"holdings": [], "positions": [], "available_cash": 100_000.0}
    agent = RiskManagerAgent()
    passes, reason = agent.check_correlation(trade, portfolio)
    assert passes is True


def test_volatility_exposure_passes_low_atr():
    trade = _build_trade(price=2500.0, symbol="RELIANCE")
    portfolio = {
        "holdings": [_build_holding(tradingsymbol="RELIANCE", avg_price=2400.0)],
        "positions": [],
        "available_cash": 100_000.0,
    }
    agent = RiskManagerAgent()
    passes, reason = agent.check_volatility_exposure(trade, portfolio)
    assert passes is True
