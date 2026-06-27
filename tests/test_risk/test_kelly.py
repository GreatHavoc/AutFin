"""Tests for Kelly criterion."""

import pytest

from src.autfin.risk.kelly import kelly_fraction, kelly_portfolio


class TestKellyFraction:
    def test_positive_edge(self):
        result = kelly_fraction(win_rate=0.6, avg_win=0.05, avg_loss=0.03)
        assert result["kelly_raw"] > 0
        assert result["position_pct"] > 0
        assert result["position_pct"] <= 0.25

    def test_no_edge(self):
        result = kelly_fraction(win_rate=0.4, avg_win=0.03, avg_loss=0.05)
        assert result["kelly_raw"] <= 0
        assert result["position_pct"] == 0.0

    def test_regime_on_full_kelly(self):
        r_on = kelly_fraction(0.6, 0.05, 0.03, regime="risk_on")
        r_off = kelly_fraction(0.6, 0.05, 0.03, regime="risk_off")
        assert r_on["position_pct"] > r_off["position_pct"]

    def test_drawdown_reduces_size(self):
        no_dd = kelly_fraction(0.6, 0.05, 0.03, drawdown_pct=0.0)
        with_dd = kelly_fraction(0.6, 0.05, 0.03, drawdown_pct=0.03)
        assert no_dd["position_pct"] > with_dd["position_pct"]

    def test_conviction_scales(self):
        low = kelly_fraction(0.6, 0.05, 0.03, conviction=0.3)
        high = kelly_fraction(0.6, 0.05, 0.03, conviction=0.9)
        assert high["position_pct"] > low["position_pct"]

    def test_max_cap(self):
        result = kelly_fraction(0.9, 0.10, 0.01, regime="risk_on", conviction=1.0)
        assert result["position_pct"] <= 0.25

    def test_invalid_inputs(self):
        result = kelly_fraction(0.0, 0.05, 0.03)
        assert result["position_pct"] == 0.0


class TestKellyPortfolio:
    def test_basic_allocation(self):
        opps = [
            {"symbol": "RELIANCE", "win_rate": 0.6, "avg_win": 0.05, "avg_loss": 0.03, "conviction": 0.8},
            {"symbol": "INFY", "win_rate": 0.55, "avg_win": 0.04, "avg_loss": 0.03, "conviction": 0.6},
        ]
        result = kelly_portfolio(opps, total_capital=1000000)
        assert result["total_allocated"] > 0
        assert len(result["allocations"]) == 2
        # Higher conviction gets more
        assert result["allocations"][0]["symbol"] == "RELIANCE"

    def test_empty_opportunities(self):
        result = kelly_portfolio([], total_capital=1000000)
        assert result["total_allocated"] == 0
