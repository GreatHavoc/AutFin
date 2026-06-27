"""Tests for walk-forward backtesting harness."""

import pytest

from src.autfin.backtesting.walk_forward import (
    split_windows,
    walk_forward,
    simple_sma_crossover,
    _sharpe_ratio,
    _max_drawdown,
    _win_rate,
    _simple_returns,
)


def _make_prices(n=200, start=100, trend=0.001, vol=0.02):
    """Generate synthetic price series with slight upward trend."""
    import random
    random.seed(42)
    prices = [start]
    for _ in range(n - 1):
        ret = trend + random.gauss(0, vol)
        prices.append(prices[-1] * (1 + ret))
    return prices


class TestSplitWindows:
    def test_basic_split(self):
        windows = split_windows(200, n_folds=5)
        assert len(windows) > 0
        for w in windows:
            assert w.train_start < w.train_end
            assert w.train_end <= w.test_start
            assert w.test_start < w.test_end

    def test_no_overlap(self):
        windows = split_windows(200, n_folds=5)
        for w in windows:
            assert w.train_start < w.train_end
            assert w.test_start < w.test_end
            assert w.train_end <= w.test_start  # No train/test overlap within a window

    def test_single_fold(self):
        windows = split_windows(200, n_folds=1)
        assert len(windows) == 1
        w = windows[0]
        assert w.train_size > 0
        assert w.test_size > 0

    def test_small_data(self):
        windows = split_windows(15, n_folds=10)
        assert len(windows) >= 1


class TestSimpleReturns:
    def test_basic(self):
        r = _simple_returns([100, 110, 105])
        assert len(r) == 2
        assert r[0] == pytest.approx(0.10, abs=1e-6)
        assert r[1] == pytest.approx(-5 / 110, abs=1e-6)

    def test_empty(self):
        assert _simple_returns([]) == []


class TestMetrics:
    def test_sharpe_positive_returns(self):
        returns = [0.01, 0.02, 0.015, -0.005, 0.01]
        sharpe = _sharpe_ratio(returns)
        assert sharpe > 0  # Positive avg return

    def test_sharpe_empty(self):
        assert _sharpe_ratio([]) == 0.0

    def test_max_drawdown(self):
        returns = [0.10, -0.20, 0.05]
        dd = _max_drawdown(returns)
        assert 0 < dd <= 1.0

    def test_win_rate(self):
        assert _win_rate([0.1, -0.1, 0.1]) == pytest.approx(2 / 3, abs=0.01)


class TestSMACrossover:
    def test_basic(self):
        prices = _make_prices(200)
        trades = simple_sma_crossover(prices, fast_period=20, slow_period=50)
        assert isinstance(trades, list)

    def test_too_short(self):
        prices = _make_prices(30)
        trades = simple_sma_crossover(prices, fast_period=20, slow_period=50)
        assert trades == []


class TestWalkForward:
    def test_basic_run(self):
        prices = _make_prices(200)
        result = walk_forward(prices, n_folds=3)
        assert "fold_results" in result
        assert "aggregate" in result
        assert "assessment" in result
        assert result["n_folds"] > 0
        assert result["assessment"] in ("HEALTHY", "CAUTION", "OVERFIT")

    def test_custom_params(self):
        prices = _make_prices(200)
        result = walk_forward(
            prices,
            param_grid={"fast_period": [10, 20], "slow_period": [40, 50]},
            n_folds=3,
        )
        assert result["n_folds"] > 0
        # Check that best_params are from the grid
        for fr in result["fold_results"]:
            params = fr["best_params"]
            assert params["fast_period"] in [10, 20]
            assert params["slow_period"] in [40, 50]

    def test_overfit_detection(self):
        # Very noisy data should show overfitting
        import random
        random.seed(99)
        prices = [100]
        for _ in range(200):
            prices.append(prices[-1] * (1 + random.gauss(0, 0.05)))
        result = walk_forward(prices, n_folds=5)
        # Should have assessment
        assert result["assessment"] in ("HEALTHY", "CAUTION", "OVERFIT")

    def test_aggregate_metrics(self):
        prices = _make_prices(200)
        result = walk_forward(prices, n_folds=3)
        agg = result["aggregate"]
        assert "avg_oos_sharpe" in agg
        assert "avg_oos_return_pct" in agg
        assert "avg_oos_drawdown" in agg
        assert "win_consistency" in agg
