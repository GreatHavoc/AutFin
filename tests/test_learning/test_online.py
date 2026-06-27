"""Tests for online learning."""

import pytest

from src.autfin.learning.online import (
    OnlineLearner,
    BucketStats,
    get_online_learner,
    reset_online_learner,
    _bucket_for_conviction,
)


class TestBucketStats:
    def test_empty(self):
        b = BucketStats()
        assert b.mean_conviction == 0.0
        assert b.mean_outcome == 0.0
        assert b.calibration_error == 0.0

    def test_single_update(self):
        b = BucketStats()
        b.n = 1
        b.sum_conviction = 0.8
        b.sum_outcome = 1.0
        b.sum_conv_sq = 0.64
        b.sum_conv_outcome = 0.8
        assert b.mean_conviction == pytest.approx(0.8)
        assert b.mean_outcome == 1.0
        assert b.calibration_error == pytest.approx(0.2)

    def test_to_dict(self):
        b = BucketStats()
        d = b.to_dict()
        assert "n" in d
        assert "calibration_error" in d


class TestBucketMapping:
    def test_very_high(self):
        assert _bucket_for_conviction(0.9) == "very_high"
        assert _bucket_for_conviction(0.8) == "very_high"

    def test_high(self):
        assert _bucket_for_conviction(0.7) == "high"
        assert _bucket_for_conviction(0.6) == "high"

    def test_medium(self):
        assert _bucket_for_conviction(0.5) == "medium"
        assert _bucket_for_conviction(0.4) == "medium"

    def test_low(self):
        assert _bucket_for_conviction(0.3) == "low"
        assert _bucket_for_conviction(0.2) == "low"

    def test_very_low(self):
        assert _bucket_for_conviction(0.1) == "very_low"
        assert _bucket_for_conviction(0.0) == "very_low"


class TestOnlineLearner:
    def setup_method(self):
        self.learner = OnlineLearner()

    def test_update(self):
        self.learner.update(conviction=0.8, outcome=1.0)
        bucket = self.learner.get_bucket(0.8)
        assert bucket.n == 1
        assert bucket.mean_outcome == 1.0

    def test_multiple_updates(self):
        for _ in range(10):
            self.learner.update(conviction=0.7, outcome=1.0)
        for _ in range(5):
            self.learner.update(conviction=0.7, outcome=0.0)
        bucket = self.learner.get_bucket(0.7)
        assert bucket.n == 15
        assert bucket.mean_outcome == pytest.approx(10 / 15)

    def test_sector_tracking(self):
        self.learner.update(conviction=0.7, outcome=1.0, sector="IT")
        self.learner.update(conviction=0.7, outcome=0.0, sector="Banking")
        it_bucket = self.learner.get_sector_bucket("IT", 0.7)
        bank_bucket = self.learner.get_sector_bucket("Banking", 0.7)
        assert it_bucket.n == 1
        assert bank_bucket.n == 1

    def test_get_adjustment_insufficient_data(self):
        # With < 5 samples, returns raw conviction
        adj = self.learner.get_adjustment(0.7)
        assert adj == 0.7

    def test_get_adjustment_pulls_toward_actual(self):
        # If actual win rate for 0.7 conviction is 100%
        for _ in range(20):
            self.learner.update(conviction=0.7, outcome=1.0)
        adj = self.learner.get_adjustment(0.7)
        # Should be pulled upward toward 1.0
        assert adj > 0.7

    def test_get_adjustment_clamped(self):
        # Very high conviction, all losses
        for _ in range(20):
            self.learner.update(conviction=0.9, outcome=0.0)
        adj = self.learner.get_adjustment(0.9)
        assert 0.05 <= adj <= 0.95

    def test_drift_detection(self):
        # Stable: no drift
        for _ in range(30):
            self.learner.update(conviction=0.6, outcome=1.0)
        drift = self.learner.get_drift()
        assert not drift["drifting"]

    def test_get_report(self):
        self.learner.update(conviction=0.7, outcome=1.0, pnl_pct=3.0)
        report = self.learner.get_report()
        assert "total_trades" in report
        assert "avg_calibration_error" in report
        assert "buckets" in report
        assert "drift" in report
        assert report["total_trades"] == 1

    def test_all_buckets(self):
        self.learner.update(conviction=0.3, outcome=0.0)
        self.learner.update(conviction=0.7, outcome=1.0)
        self.learner.update(conviction=0.9, outcome=1.0)
        buckets = self.learner.get_all_buckets()
        assert "low" in buckets
        assert "high" in buckets
        assert "very_high" in buckets

    def test_get_sector_report(self):
        self.learner.update(conviction=0.7, outcome=1.0, sector="IT")
        report = self.learner.get_sector_report("IT")
        assert "high" in report

    def test_total_pnl_tracked(self):
        self.learner.update(conviction=0.7, outcome=1.0, pnl_pct=3.0)
        self.learner.update(conviction=0.5, outcome=0.0, pnl_pct=-2.0)
        assert self.learner._total_pnl == pytest.approx(1.0)


class TestModuleSingleton:
    def test_singleton(self):
        reset_online_learner()
        l1 = get_online_learner()
        l2 = get_online_learner()
        assert l1 is l2

    def test_reset(self):
        reset_online_learner()
        l1 = get_online_learner()
        reset_online_learner()
        l2 = get_online_learner()
        assert l1 is not l2
