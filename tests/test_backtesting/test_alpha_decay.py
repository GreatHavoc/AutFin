"""Tests for alpha decay tracker."""

import pytest

from src.autfin.backtesting.alpha_decay import AlphaDecayTracker, DecayReport, FoldSnapshot


class TestAlphaDecayTracker:
    def test_empty_report(self):
        tracker = AlphaDecayTracker()
        report = tracker.report()
        assert report.n_folds == 0
        assert report.alpha_decayed is False
        assert "No folds" in report.decay_reason

    def test_single_fold(self):
        tracker = AlphaDecayTracker(min_folds=2)
        tracker.add_fold({"fold": 0, "test_sharpe": 1.0, "test_return_pct": 10.0})
        report = tracker.report()
        assert report.n_folds == 1
        assert report.latest_sharpe == 1.0
        assert report.earliest_sharpe == 1.0
        # Not enough folds to trigger decay
        assert report.alpha_decayed is False

    def test_stable_sharpe_no_decay(self):
        """Stable Sharpe across folds should not trigger decay."""
        tracker = AlphaDecayTracker(min_folds=3, sharpe_decay_threshold=0.5, min_sharpe=0.3)
        for i in range(5):
            tracker.add_fold({"fold": i, "test_sharpe": 1.0, "test_return_pct": 8.0})
        report = tracker.report()
        assert report.n_folds == 5
        assert report.sharpe_drop_pct == pytest.approx(0.0, abs=0.1)
        assert report.alpha_decayed is False

    def test_sharpe_decay_triggers(self):
        """Sharpe dropping >50% should trigger decay."""
        tracker = AlphaDecayTracker(min_folds=3, sharpe_decay_threshold=0.5, min_sharpe=0.3)
        tracker.add_fold({"fold": 0, "test_sharpe": 1.0, "test_return_pct": 10.0})
        tracker.add_fold({"fold": 1, "test_sharpe": 0.7, "test_return_pct": 7.0})
        tracker.add_fold({"fold": 2, "test_sharpe": 0.4, "test_return_pct": 4.0})
        report = tracker.report()
        assert report.alpha_decayed is True
        assert "Sharpe dropped" in report.decay_reason
        assert report.sharpe_drop_pct > 50

    def test_min_sharpe_floor_triggers(self):
        """Sharpe below floor should trigger decay."""
        tracker = AlphaDecayTracker(min_folds=3, sharpe_decay_threshold=0.5, min_sharpe=0.5)
        tracker.add_fold({"fold": 0, "test_sharpe": 1.0, "test_return_pct": 10.0})
        tracker.add_fold({"fold": 1, "test_sharpe": 0.6, "test_return_pct": 6.0})
        tracker.add_fold({"fold": 2, "test_sharpe": 0.2, "test_return_pct": 2.0})
        report = tracker.report()
        assert report.alpha_decayed is True
        assert "Min Sharpe" in report.decay_reason

    def test_trend_computation(self):
        """Sharpe trend should be negative for decaying strategy."""
        tracker = AlphaDecayTracker()
        for i in range(5):
            tracker.add_fold({"fold": i, "test_sharpe": 1.0 - i * 0.2, "test_return_pct": 10.0 - i * 2})
        report = tracker.report()
        assert report.sharpe_trend < 0
        assert report.return_trend < 0

    def test_positive_trend(self):
        """Improving Sharpe should have positive trend."""
        tracker = AlphaDecayTracker()
        for i in range(5):
            tracker.add_fold({"fold": i, "test_sharpe": 0.5 + i * 0.1, "test_return_pct": 5.0 + i * 1.0})
        report = tracker.report()
        assert report.sharpe_trend > 0
        assert report.alpha_decayed is False

    def test_to_dict(self):
        """to_dict returns serializable dict with all fields."""
        tracker = AlphaDecayTracker(min_folds=2)
        tracker.add_fold({"fold": 0, "test_sharpe": 1.0, "test_return_pct": 8.0})
        tracker.add_fold({"fold": 1, "test_sharpe": 0.8, "test_return_pct": 6.0})
        d = tracker.to_dict()
        assert isinstance(d, dict)
        assert "n_folds" in d
        assert "alpha_decayed" in d
        assert "sharpe_trend" in d
        assert "recommendation" in d

    def test_fold_snapshot_fields(self):
        """FoldSnapshot stores all provided fields."""
        fs = FoldSnapshot(
            fold=0, test_sharpe=1.2, test_return_pct=8.5,
            test_max_drawdown=0.05, test_win_rate=0.6,
            best_params={"fast": 20}, n_bars=200,
        )
        assert fs.fold == 0
        assert fs.test_sharpe == 1.2
        assert fs.best_params == {"fast": 20}
        assert fs.n_bars == 200


class TestDecayReport:
    def test_report_fields(self):
        tracker = AlphaDecayTracker(min_folds=2)
        tracker.add_fold({"fold": 0, "test_sharpe": 1.0, "test_return_pct": 8.0})
        tracker.add_fold({"fold": 1, "test_sharpe": 0.8, "test_return_pct": 6.0})
        report = tracker.report()
        assert isinstance(report, DecayReport)
        assert report.n_folds == 2
        assert isinstance(report.recommendation, str)
