"""Tests for vol-targeting overlay (risk/vol_targeting.py)."""

import pytest
from autfin.risk.vol_targeting import (
    compute_realized_vol,
    compute_vol_scaling,
    vol_target_positions,
)


class TestComputeRealizedVol:
    def test_basic(self):
        returns = [0.01, -0.02, 0.005, -0.01, 0.015, -0.008, 0.012, -0.005]
        vol = compute_realized_vol(returns)
        assert vol is not None
        assert vol > 0
        assert vol < 2.0  # Reasonable annualized vol

    def test_empty_returns(self):
        assert compute_realized_vol([]) is None

    def test_too_few_returns(self):
        assert compute_realized_vol([0.01, 0.02]) is None

    def test_constant_returns(self):
        # Constant returns → very low vol
        vol = compute_realized_vol([0.001] * 20)
        assert vol is not None
        assert vol < 0.01

    def test_high_vol_returns(self):
        # Alternating large returns → high vol
        returns = [0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05]
        vol = compute_realized_vol(returns)
        assert vol > 0.5  # Should be high

    def test_not_annualized(self):
        returns = [0.01, -0.02, 0.005, -0.01, 0.015]
        vol = compute_realized_vol(returns, annualize=False)
        assert vol is not None
        # Without annualization should be smaller
        vol_annual = compute_realized_vol(returns, annualize=True)
        assert vol < vol_annual


class TestComputeVolScaling:
    def test_target_equals_realized(self):
        scaling = compute_vol_scaling(0.10, target_vol=0.10)
        assert abs(scaling - 1.0) < 0.01

    def test_high_vol_reduces(self):
        scaling = compute_vol_scaling(0.20, target_vol=0.10)
        assert scaling < 1.0

    def test_low_vol_increases(self):
        scaling = compute_vol_scaling(0.05, target_vol=0.10)
        assert scaling > 1.0

    def test_zero_vol_returns_max(self):
        scaling = compute_vol_scaling(0.0, target_vol=0.10)
        assert scaling == 1.5  # max_scaling default

    def test_clamped_to_min(self):
        scaling = compute_vol_scaling(0.50, target_vol=0.10, min_scaling=0.25)
        assert scaling == 0.25

    def test_clamped_to_max(self):
        scaling = compute_vol_scaling(0.01, target_vol=0.10, max_scaling=1.5)
        assert scaling == 1.5

    def test_custom_clamps(self):
        scaling = compute_vol_scaling(0.40, target_vol=0.10, min_scaling=0.5, max_scaling=2.0)
        assert scaling == 0.5


class TestVolTargetPositions:
    def test_basic(self):
        positions = [
            {"symbol": "RELIANCE", "qty": 10},
            {"symbol": "INFY", "qty": 20},
        ]
        # Simulate moderate vol returns
        returns = [0.01, -0.02, 0.005, -0.01, 0.015, -0.008, 0.012, -0.005, 0.01, -0.015]
        result = vol_target_positions(positions, returns, target_vol=0.10)

        assert "scaling" in result
        assert "realized_vol" in result
        assert "positions" in result
        assert len(result["positions"]) == 2
        assert result["scaling"] > 0

    def test_empty_positions(self):
        result = vol_target_positions([], [])
        assert result["scaling"] == 1.0
        assert result["positions"] == []

    def test_qty_scaling(self):
        positions = [{"qty": 100}]
        # Very low vol → scaling > 1 → qty increases
        returns = [0.001] * 20
        result = vol_target_positions(
            positions, returns, target_vol=0.10,
            min_scaling=0.25, max_scaling=2.0,
        )
        assert result["positions"][0]["qty"] >= 100

    def test_high_vol_reduces_qty(self):
        positions = [{"qty": 100}]
        # High vol → scaling < 1 → qty decreases
        returns = [0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05]
        result = vol_target_positions(
            positions, returns, target_vol=0.10,
            min_scaling=0.25, max_scaling=1.5,
        )
        assert result["positions"][0]["qty"] < 100

    def test_weight_scaling(self):
        positions = [{"weight": 0.30}]
        returns = [0.05, -0.05, 0.05, -0.05, 0.05, -0.05]
        result = vol_target_positions(
            positions, returns, target_vol=0.10,
            min_scaling=0.25, max_scaling=1.5,
        )
        assert result["positions"][0]["weight"] < 0.30  # High vol reduces

    def test_adjustment_pct(self):
        positions = [{"qty": 100}]
        returns = [0.05, -0.05, 0.05, -0.05, 0.05, -0.05]
        result = vol_target_positions(
            positions, returns, target_vol=0.10,
            min_scaling=0.25, max_scaling=1.5,
        )
        assert result["adjustment_pct"] != 0.0  # Some adjustment happened

    def test_lookback_window(self):
        positions = [{"qty": 100}]
        # Long return series, recent part is calm
        returns = [0.10] * 100 + [0.001] * 10
        result_short = vol_target_positions(positions, returns, lookback_days=10)
        result_long = vol_target_positions(positions, returns, lookback_days=60)
        # Short lookback sees calm period → higher scaling → more qty
        assert result_short["scaling"] >= result_long["scaling"]
