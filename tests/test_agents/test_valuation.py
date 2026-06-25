"""Tests for DCF Valuation Agent functions."""
import pytest
from src.autfin.agents.valuation import (
    compute_dcf,
    compute_wacc,
    get_sector_wacc,
    SECTOR_WACC,
    DEFAULT_WACC,
)


@pytest.mark.asyncio
async def test_reverse_dcf_returns_implied_growth():
    from src.autfin.agents.valuation import reverse_dcf
    result = await reverse_dcf(
        current_price=100,
        current_fcf=100_000_000,
        shares_outstanding=10_000_000,
        wacc=0.10,
    )
    assert "implied_growth" in result
    assert "fair_value" in result
    assert "mispricing_pct" in result
    assert isinstance(result["implied_growth"], float)
    assert -0.11 < result["implied_growth"] <= 0.31
    assert result["fair_value"] > 0
    assert "scenarios" in result
    assert "current_price" in result
    assert result["enterprise_value"] > 0


@pytest.mark.asyncio
async def test_reverse_dcf_high_growth_implies_undervaluation():
    from src.autfin.agents.valuation import reverse_dcf
    result = await reverse_dcf(
        current_price=100,
        current_fcf=500_000_000,
        shares_outstanding=10_000_000,
        wacc=0.10,
    )
    assert result["implied_growth"] > 0.03
    assert result["mispricing_pct"] > 0
    assert result["mispricing_label"] == "undervalued"


@pytest.mark.asyncio
async def test_reverse_dcf_low_growth_implies_overvaluation():
    from src.autfin.agents.valuation import reverse_dcf
    result = await reverse_dcf(
        current_price=5000,
        current_fcf=100_000_000,
        shares_outstanding=1_000_000_000,
        wacc=0.10,
    )
    assert result["implied_growth"] < 0.10
    assert result["mispricing_pct"] < 0
    assert result["mispricing_label"] == "overvalued"


# ---------------------------------------------------------------------------
# compute_wacc tests
# ---------------------------------------------------------------------------

def test_compute_wacc_returns_valid_rate():
    wacc = compute_wacc(
        market_cap=1_000_000,
        total_debt=500_000,
        cost_of_equity=0.12,
        cost_of_debt=0.07,
        tax_rate=0.25,
    )
    assert 0.07 < wacc < 0.15
    # Manual check: equity=1M, debt=500k, total=1.5M
    # E_weight = 1/1.5, D_weight = 0.5/1.5
    # WACC = (1/1.5)*0.12 + (0.5/1.5)*0.07*(0.75) = 0.08 + 0.0175 = 0.0975
    assert wacc == pytest.approx(0.0975, rel=0.01)


def test_compute_wacc_handles_zero_capital():
    wacc = compute_wacc(market_cap=0, total_debt=0)
    assert wacc == DEFAULT_WACC


# ---------------------------------------------------------------------------
# get_sector_wacc tests
# ---------------------------------------------------------------------------

def test_sector_wacc_returns_default_for_unknown_sector():
    result = get_sector_wacc("TotallyUnknownSector")
    assert result == SECTOR_WACC["default"]
    assert result["wacc"] == 0.10


def test_sector_wacc_returns_it_services_params():
    result = get_sector_wacc("IT Services")
    assert result["wacc"] == 0.095
    assert result["growth_period"] == 10
    assert result["terminal_growth"] == 0.035


def test_sector_wacc_returns_banks_params():
    result = get_sector_wacc("Banks")
    assert result["wacc"] == 0.105
    assert result["growth_period"] == 8
    assert result["terminal_growth"] == 0.03


# ---------------------------------------------------------------------------
# compute_dcf tests
# ---------------------------------------------------------------------------

def test_compute_dcf_returns_all_expected_keys():
    result = compute_dcf(
        free_cash_flow=1_000_000,
        revenue_growth=0.10,
        wacc=0.10,
        terminal_growth=0.03,
        total_debt=100_000,
        cash=50_000,
        shares_outstanding=1_000_000,
    )
    assert result["scenario"] == "base"
    assert "enterprise_value" in result
    assert "equity_value" in result
    assert "implied_price" in result
    assert "pv_projected_fcfs" in result
    assert "pv_terminal_value" in result
    assert len(result["projections"]) == 5
    assert result["implied_price"] > 0


def test_compute_dcf_higher_growth_increases_enterprise_value():
    low_growth = compute_dcf(
        free_cash_flow=1_000_000,
        revenue_growth=0.05,
        wacc=0.10,
        shares_outstanding=1_000_000,
    )
    high_growth = compute_dcf(
        free_cash_flow=1_000_000,
        revenue_growth=0.20,
        wacc=0.10,
        shares_outstanding=1_000_000,
    )
    assert high_growth["enterprise_value"] > low_growth["enterprise_value"]
    assert high_growth["implied_price"] > low_growth["implied_price"]


def test_compute_dcf_handles_debt_reducing_equity_value():
    no_debt = compute_dcf(
        free_cash_flow=1_000_000,
        revenue_growth=0.10,
        wacc=0.10,
        total_debt=0,
        cash=0,
        shares_outstanding=1_000_000,
    )
    with_debt = compute_dcf(
        free_cash_flow=1_000_000,
        revenue_growth=0.10,
        wacc=0.10,
        total_debt=500_000,
        cash=0,
        shares_outstanding=1_000_000,
    )
    assert with_debt["equity_value"] < no_debt["equity_value"]
    assert with_debt["implied_price"] < no_debt["implied_price"]


def test_compute_dcf_handles_zero_fcf():
    result = compute_dcf(
        free_cash_flow=0,
        revenue_growth=0.10,
        shares_outstanding=1_000_000,
    )
    assert result["implied_price"] >= 0
    assert "projections" in result
