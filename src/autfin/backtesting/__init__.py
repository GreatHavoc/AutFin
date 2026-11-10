"""Walk-forward backtesting + alpha decay tracking.

Rolling train/test windows, out-of-sample evaluation, overfit detection.
Imported as ``autfin.backtesting``; the vectorbt suite lives in ``autfin.backtest``.
"""
from __future__ import annotations

from autfin.backtesting.alpha_decay import (
    AlphaDecayTracker,
    DecayReport,
    FoldSnapshot,
)
from autfin.backtesting.walk_forward import (
    BacktestMetrics,
    FoldResult,
    WalkForwardConfig,
    Window,
    fetch_price_data,
    simple_sma_crossover,
    split_windows,
    walk_forward,
    walk_forward_with_symbol,
)

__all__ = [
    "AlphaDecayTracker",
    "BacktestMetrics",
    "DecayReport",
    "FoldResult",
    "FoldSnapshot",
    "WalkForwardConfig",
    "Window",
    "fetch_price_data",
    "simple_sma_crossover",
    "split_windows",
    "walk_forward",
    "walk_forward_with_symbol",
]
