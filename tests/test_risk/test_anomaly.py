"""Tests for anomaly detection."""

import pytest

from src.autfin.risk.anomaly import RollingStats, AnomalyDetector


class TestRollingStats:
    def test_empty(self):
        rs = RollingStats(window=5)
        assert rs.mean == 0.0
        assert rs.std == 0.0
        assert rs.count == 0

    def test_single_value(self):
        rs = RollingStats(window=5)
        rs.add(10.0)
        assert rs.mean == 10.0
        assert rs.count == 1

    def test_mean(self):
        rs = RollingStats(window=5)
        for v in [10, 20, 30]:
            rs.add(v)
        assert rs.mean == 20.0

    def test_z_score(self):
        rs = RollingStats(window=10)
        for v in [10, 10, 10, 10, 10, 10, 10, 10, 10, 20]:
            rs.add(v)
        z = rs.z_score(20.0)
        assert z > 0  # 20 is above mean

    def test_window_limit(self):
        rs = RollingStats(window=3)
        for v in [10, 20, 30, 40, 50]:
            rs.add(v)
        assert rs.count == 3
        assert rs.mean == 40.0  # Only last 3 values


class TestAnomalyDetector:
    def test_no_anomalies_initially(self):
        ad = AnomalyDetector(window=5)
        anomalies = ad.check_anomalies()
        assert len(anomalies) == 0

    def test_low_win_rate_detected(self):
        ad = AnomalyDetector(window=20)
        for _ in range(25):
            ad.record_trade_result(won=False, pnl_pct=-2.0)
        anomalies = ad.check_anomalies()
        win_rate_anomalies = [a for a in anomalies if a["metric"] == "win_rate"]
        assert len(win_rate_anomalies) == 1
        assert win_rate_anomalies[0]["severity"] == "high"

    def test_high_win_rate_no_anomaly(self):
        ad = AnomalyDetector(window=5)
        for _ in range(15):
            ad.record_trade_result(won=True, pnl_pct=3.0)
        anomalies = ad.check_anomalies()
        win_rate_anomalies = [a for a in anomalies if a["metric"] == "win_rate"]
        assert len(win_rate_anomalies) == 0

    def test_negative_pnl_anomaly(self):
        ad = AnomalyDetector(window=20)
        for _ in range(15):
            ad.record_trade_result(won=False, pnl_pct=-5.0)
        anomalies = ad.check_anomalies()
        pnl_anomalies = [a for a in anomalies if a["metric"] == "daily_pnl"]
        assert len(pnl_anomalies) == 1

    def test_summary(self):
        ad = AnomalyDetector(window=5)
        ad.record_trade_result(won=True, pnl_pct=2.0)
        summary = ad.get_summary()
        assert "win_rate" in summary
        assert "calibration_error" in summary
