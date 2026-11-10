"""AutFin Backtest Suite — validate strategies before trading."""
from autfin.backtest.engine import (
    STRATEGIES,
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    strategy_bollinger_reversal,
    strategy_macd,
    strategy_multi_factor,
    strategy_rsi,
    strategy_sma_crossover,
)
from autfin.backtest.quick import print_stats, quick_backtest
from autfin.backtest.sweep import parameter_sweep, print_sweep_result
from autfin.backtesting import (
    AlphaDecayTracker,
    WalkForwardConfig,
    split_windows,
    walk_forward,
    walk_forward_with_symbol,
)

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "STRATEGIES",
    "quick_backtest",
    "print_stats",
    "parameter_sweep",
    "print_sweep_result",
    "strategy_sma_crossover",
    "strategy_rsi",
    "strategy_macd",
    "strategy_bollinger_reversal",
    "strategy_multi_factor",
    "AlphaDecayTracker",
    "WalkForwardConfig",
    "split_windows",
    "walk_forward",
    "walk_forward_with_symbol",
]
