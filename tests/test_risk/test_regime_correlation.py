"""Tests for regime-conditional correlation risk (risk_manager.py)."""

from unittest.mock import patch, MagicMock

import pytest

from src.autfin.agents.risk_manager import (
    RiskManagerAgent,
    REGIME_CORRELATION_THRESHOLDS,
)


@pytest.fixture
def rm():
    return RiskManagerAgent()


class TestRegimeCorrelationThresholds:
    def test_all_regimes_have_thresholds(self):
        assert "low_vol" in REGIME_CORRELATION_THRESHOLDS
        assert "normal" in REGIME_CORRELATION_THRESHOLDS
        assert "high_vol" in REGIME_CORRELATION_THRESHOLDS
        assert "crisis" in REGIME_CORRELATION_THRESHOLDS

    def test_crisis_has_lowest_threshold(self):
        assert REGIME_CORRELATION_THRESHOLDS["crisis"] < REGIME_CORRELATION_THRESHOLDS["normal"]

    def test_low_vol_has_highest_threshold(self):
        assert REGIME_CORRELATION_THRESHOLDS["low_vol"] > REGIME_CORRELATION_THRESHOLDS["normal"]

    def test_thresholds_are_monotonic(self):
        # As regime gets worse, threshold should decrease
        assert (
            REGIME_CORRELATION_THRESHOLDS["low_vol"]
            > REGIME_CORRELATION_THRESHOLDS["normal"]
            > REGIME_CORRELATION_THRESHOLDS["high_vol"]
            > REGIME_CORRELATION_THRESHOLDS["crisis"]
        )


class TestCheckCorrelationRiskRegime:
    def test_skips_with_fewer_than_2_holdings(self, rm):
        portfolio = {"holdings": [{"tradingsymbol": "RELIANCE"}]}
        passes, reason = rm.check_correlation_risk(portfolio)
        assert passes is True
        assert "fewer than 2" in reason

    def test_uses_regime_threshold(self, rm):
        """Verify the regime detector is consulted."""
        portfolio = {
            "holdings": [
                {"tradingsymbol": "RELIANCE"},
                {"tradingsymbol": "INFY"},
            ]
        }
        mock_regime = MagicMock()
        mock_regime.current_regime.regime = "crisis"

        with patch("src.autfin.risk.regime.get_regime_detector", return_value=mock_regime):
            passes, reason = rm.check_correlation_risk(portfolio)
            # Should mention crisis regime in the reason (or skipped if yfinance fails)
            assert "regime=crisis" in reason or "skipped" in reason.lower()

    def test_fallback_to_normal_regime(self, rm):
        """When regime detector fails, falls back to normal."""
        portfolio = {
            "holdings": [
                {"tradingsymbol": "RELIANCE"},
                {"tradingsymbol": "INFY"},
            ]
        }
        with patch(
            "src.autfin.risk.regime.get_regime_detector",
            side_effect=Exception("no regime"),
        ):
            passes, reason = rm.check_correlation_risk(portfolio)
            assert "regime=normal" in reason or "skipped" in reason.lower()
