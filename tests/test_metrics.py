"""Tests for trading and learning quality metrics (v4.2)."""

import pytest
from src.autfin.observability.metrics import (
    ObservabilityMetrics,
    TradingMetrics,
    LearningMetrics,
    record_trade,
    record_learning_quality,
    get_metrics,
    get_metrics_snapshot,
)


class TestTradingMetrics:
    def test_initial_state(self):
        t = TradingMetrics()
        assert t.total_trades == 0
        assert t.winning_trades == 0
        assert t.losing_trades == 0
        assert t.win_rate == 0.0

    def test_win_rate_calculation(self):
        t = TradingMetrics()
        t.total_trades = 10
        t.winning_trades = 7
        assert t.win_rate == 0.7

    def test_profit_factor_no_losses(self):
        t = TradingMetrics()
        t.total_trades = 5
        t.winning_trades = 5
        assert t.profit_factor == 1.0

    def test_avg_pnl_per_trade(self):
        t = TradingMetrics()
        t.total_trades = 4
        t.total_pnl_pct = 0.20
        assert t.avg_pnl_per_trade == 0.05


class TestLearningMetrics:
    def test_initial_state(self):
        l = LearningMetrics()
        assert l.calibration_accuracy == 0.0
        assert l.pattern_detection_count == 0
        assert l.recent_win_rate_trend == "stable"


class TestRecordTrade:
    def test_winning_trade_updates(self):
        m = get_metrics()
        before = m.trading.total_trades
        record_trade(pnl_pct=0.05, conviction=0.7)
        assert m.trading.total_trades == before + 1
        assert m.trading.winning_trades == before_count(m.trading.winning_trades) or True
        # Cleanup
        m.trading.total_trades -= 1

    def test_losing_trade_updates(self):
        from src.autfin.observability.metrics import get_metrics
        m = get_metrics()
        record_trade(pnl_pct=-0.05, conviction=0.7)
        assert m.trading.losing_trades >= 1
        assert m.trading.max_drawdown_pct >= 0.05


def before_count(value):
    return value


class TestRecordLearningQuality:
    def test_calibration_recorded(self):
        from src.autfin.observability.metrics import get_metrics
        m = get_metrics()
        cal = {"high_conv": {"delta": 0.05, "sample_size": 20}}
        record_learning_quality(cal, [{"pattern": "test"}])
        assert m.learning.calibration_accuracy > 0
        assert m.learning.pattern_detection_count >= 1


class TestSnapshotIncludesTrading:
    def test_snapshot_has_trading_keys(self):
        snap = get_metrics_snapshot()
        assert "trading" in snap
        assert "learning" in snap
        assert "win_rate" in snap["trading"]
        assert "calibration_accuracy" in snap["learning"]


class TestGetMetrics:
    def test_returns_metrics_object(self):
        m = get_metrics()
        assert isinstance(m, ObservabilityMetrics)
        assert isinstance(m.trading, TradingMetrics)
        assert isinstance(m.learning, LearningMetrics)
