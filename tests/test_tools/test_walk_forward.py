"""Tests for walk-forward backtesting + alpha decay (autfin.backtesting)."""
from __future__ import annotations

import asyncio
from datetime import datetime

from autfin.backtesting import (
    AlphaDecayTracker,
    fetch_price_data,
    simple_sma_crossover,
    split_windows,
    walk_forward,
)


def test_windows_tile_the_whole_series():
    windows = split_windows(n_bars=500, train_pct=0.7, n_folds=5)
    assert len(windows) == 5
    # The bug this guards: folds used to start at 0,30,60,90,120 and stop at
    # bar 220, silently ignoring 56% of the price history.
    assert windows[0].train_start == 0
    assert windows[-1].test_end == 500


def test_windows_are_contiguous_and_causal():
    windows = split_windows(n_bars=500, train_pct=0.7, n_folds=5)
    for prev, cur in zip(windows, windows[1:]):
        assert cur.train_start == prev.test_end  # tiled, no gap, no overlap
    for w in windows:
        assert w.train_end == w.test_start  # no lookahead: test starts after train
        assert w.train_size > 0 and w.test_size >= 3


def test_windows_handle_short_series():
    windows = split_windows(n_bars=20, n_folds=5)
    assert len(windows) == 1
    assert windows[0].test_end == 20


def test_sma_crossover_closes_the_open_position():
    # Monotonic uptrend: fast MA stays above slow MA, so the only trade is the
    # entry — it must be marked to market instead of returning no trades.
    closes = [100.0 + i for i in range(200)]
    trades = simple_sma_crossover(closes, fast_period=5, slow_period=20)
    assert len(trades) == 1
    assert trades[0] > 0


def test_walk_forward_flags_insufficient_data_instead_of_overfit():
    # 400 bars over 5 folds leaves 80-bar windows -> 24-bar test windows, too
    # short for the default slow_period grid (40-60). That is missing evidence,
    # not overfitting, and used to be reported as OVERFIT with 999 ratio.
    closes = [100 + (i % 11) * 0.9 + i * 0.15 for i in range(400)]
    result = walk_forward(closes, n_folds=5)
    assert result["n_usable_folds"] == 0
    assert result["assessment"] == "INSUFFICIENT_DATA"
    assert result["is_overfit"] is False


def test_walk_forward_reports_required_keys():
    closes = [100 + (i % 11) * 0.9 + i * 0.15 for i in range(400)]
    result = walk_forward(
        closes, n_folds=4, param_grid={"fast_period": [5], "slow_period": [20]}
    )
    assert set(result) >= {"n_folds", "fold_results", "aggregate", "assessment", "recommendation"}
    assert result["n_folds"] == 4
    assert set(result["aggregate"]) == {
        "avg_oos_sharpe",
        "avg_oos_return_pct",
        "avg_oos_drawdown",
        "avg_overfit_ratio",
        "win_consistency",
    }
    assert result["assessment"] in {"HEALTHY", "CAUTION", "OVERFIT", "INSUFFICIENT_DATA"}


def test_alpha_decay_flags_a_decaying_strategy():
    tracker = AlphaDecayTracker()
    for fold in range(4):
        tracker.add_fold({"fold": fold, "test_sharpe": 2.0 - 0.5 * fold, "test_return_pct": 10.0 - 3 * fold})
    report = tracker.report()
    assert report.n_folds == 4
    assert report.alpha_decayed is True
    assert report.sharpe_drop_pct > 0
    assert tracker.to_dict()["alpha_decayed"] is True


def test_alpha_decay_ignores_single_fold():
    tracker = AlphaDecayTracker(min_folds=3)
    tracker.add_fold({"test_sharpe": 0.1, "test_return_pct": -5.0})
    assert tracker.report().alpha_decayed is False


def test_fetch_price_data_pins_as_of_as_the_end_date(monkeypatch):
    """as_of must become yfinance's `end`, not a post-fetch index filter.

    Anchoring a relative period at `end` is what makes period+as_of return a
    window that ends at the cutoff instead of 0 bars.
    """
    import pandas as pd

    captured: dict = {}

    class FakeTicker:
        def __init__(self, symbol):
            captured["symbol"] = symbol

        def history(self, **kwargs):
            captured.update(kwargs)
            return pd.DataFrame({"Close": [1.0, 2.0]})

    monkeypatch.setattr("yfinance.Ticker", FakeTicker)

    closes = asyncio.run(fetch_price_data("RELIANCE", period="1y", as_of="2025-06-01"))

    assert closes == [1.0, 2.0]
    assert captured["symbol"] == "RELIANCE.NS"
    assert captured["period"] == "1y"
    assert captured["end"] == datetime(2025, 6, 1)


def test_fetch_price_data_keeps_earlier_explicit_end(monkeypatch):
    captured: dict = {}

    class FakeTicker:
        def __init__(self, symbol):
            pass

        def history(self, **kwargs):
            captured.update(kwargs)
            return None

    monkeypatch.setattr("yfinance.Ticker", FakeTicker)

    assert asyncio.run(
        fetch_price_data("X", start="2024-01-01", end="2024-06-01", as_of="2025-01-01")
    ) == []
    assert captured["end"] == datetime(2024, 6, 1)
