"""Tests for regression detection."""

import pytest
from src.autfin.learning.regression import RegressionDetector, RegressionAlert


class TestRegressionDetectorInit:
    def test_default_thresholds(self):
        d = RegressionDetector()
        assert d.min_trades == 20
        assert d.win_rate_warning == 0.40
        assert d.win_rate_critical == 0.30

    def test_custom_thresholds(self):
        d = RegressionDetector(
            min_trades_for_alert=10,
            win_rate_warning=0.45,
            win_rate_critical=0.35,
        )
        assert d.min_trades == 10
        assert d.win_rate_warning == 0.45
        assert d.win_rate_critical == 0.35


class TestRegressionAlert:
    def test_alert_fields(self):
        alert = RegressionAlert(
            metric_name="win_rate",
            current_value=0.25,
            baseline_value=0.50,
            degradation_pct=50.0,
            severity="critical",
            recommendation="Pause trading",
        )
        assert alert.metric_name == "win_rate"
        assert alert.severity == "critical"
        assert alert.current_value == 0.25


class TestCheckWinRate:
    @pytest.mark.asyncio
    async def test_insufficient_trades_returns_no_alerts(self):
        d = RegressionDetector(min_trades_for_alert=100)
        alerts = await d._check_win_rate()
        assert alerts == []

    @pytest.mark.asyncio
    async def test_handles_db_error_gracefully(self):
        d = RegressionDetector()
        # Should not raise even if DB is unavailable
        alerts = await d._check_win_rate()
        assert isinstance(alerts, list)


class TestCheckLossStreak:
    @pytest.mark.asyncio
    async def test_handles_db_error_gracefully(self):
        d = RegressionDetector()
        alerts = await d._check_loss_streak()
        assert isinstance(alerts, list)


class TestCheckCalibration:
    @pytest.mark.asyncio
    async def test_handles_db_error_gracefully(self):
        d = RegressionDetector()
        alerts = await d._check_calibration()
        assert isinstance(alerts, list)


class TestShouldAutoPause:
    @pytest.mark.asyncio
    async def test_returns_tuple(self):
        d = RegressionDetector()
        should_pause, reason = await d.should_auto_pause()
        assert isinstance(should_pause, bool)
        assert isinstance(reason, str)


class TestCheckPerformanceRegression:
    @pytest.mark.asyncio
    async def test_scheduler_helper(self):
        """Test the helper function the scheduler calls."""
        from src.autfin.learning.regression import check_performance_regression
        # Just ensure it's callable
        assert callable(check_performance_regression)
