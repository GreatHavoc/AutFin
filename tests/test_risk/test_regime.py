"""Tests for regime detection."""

import pytest

from src.autfin.risk.regime import (
    RegimeDetector,
    RegimeState,
    get_regime_detector,
    reset_regime_detector,
    REGIME_KELLY_MULT,
    REGIME_CONV_THRESHOLD,
)


class TestRegimeDetector:
    def setup_method(self):
        self.detector = RegimeDetector()

    def test_default_regime(self):
        state = self.detector.current_regime
        assert state.regime == "normal"

    def test_low_vol_detected(self):
        state = self.detector.detect(vix=10.0)
        assert state.regime == "low_vol"
        assert state.kelly_mult == 1.0
        assert state.conviction_threshold == 0.50

    def test_normal_detected(self):
        state = self.detector.detect(vix=17.0)
        assert state.regime == "normal"
        assert state.kelly_mult == 0.75

    def test_high_vol_detected(self):
        state = self.detector.detect(vix=25.0)
        assert state.regime == "high_vol"
        assert state.kelly_mult == 0.50

    def test_crisis_detected(self):
        state = self.detector.detect(vix=35.0)
        assert state.regime == "crisis"
        assert state.kelly_mult == 0.25
        assert self.detector.should_de_gross()

    def test_crisis_by_drawdown(self):
        state = self.detector.detect(nifty_close_today=9500, nifty_close_5d_ago=10000)
        assert state.regime == "crisis"

    def test_fallback_to_realized_vol(self):
        import random
        random.seed(42)
        # Feed varied returns for realistic vol calculation
        for _ in range(25):
            self.detector.record_daily_return(random.gauss(0.02, 0.01))
        state = self.detector.detect()  # No VIX provided
        assert state.realized_vol > 0
        assert state.regime in ("low_vol", "normal", "high_vol", "crisis")

    def test_realized_vol_low(self):
        for _ in range(25):
            self.detector.record_daily_return(0.003)  # 0.3% daily = low vol
        state = self.detector.detect()
        assert state.regime == "low_vol"

    def test_regime_change_tracked(self):
        self.detector.detect(vix=10.0)
        self.detector.detect(vix=25.0)
        history = self.detector.get_history()
        assert len(history) == 1
        assert history[0]["from"] == "low_vol"
        assert history[0]["to"] == "high_vol"

    def test_to_dict(self):
        state = self.detector.detect(vix=15.0)
        d = state.to_dict()
        assert "regime" in d
        assert "kelly_mult" in d
        assert "conviction_threshold" in d

    def test_confidence_limited_data(self):
        state = self.detector.detect(vix=15.0)
        # With no return history, confidence should be reduced
        assert state.confidence <= 0.7


class TestRegimeModule:
    def test_singleton(self):
        reset_regime_detector()
        d1 = get_regime_detector()
        d2 = get_regime_detector()
        assert d1 is d2

    def test_reset(self):
        reset_regime_detector()
        d1 = get_regime_detector()
        reset_regime_detector()
        d2 = get_regime_detector()
        assert d1 is not d2

    def test_constants(self):
        assert "low_vol" in REGIME_KELLY_MULT
        assert "crisis" in REGIME_KELLY_MULT
        assert "low_vol" in REGIME_CONV_THRESHOLD
        assert "crisis" in REGIME_CONV_THRESHOLD
        assert REGIME_KELLY_MULT["low_vol"] > REGIME_KELLY_MULT["crisis"]
