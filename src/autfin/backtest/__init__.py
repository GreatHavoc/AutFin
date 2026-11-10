"""AutFin Backtest Suite — validate strategies before trading."""
from autfin.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    STRATEGIES,
    strategy_sma_crossover,
    strategy_rsi,
    strategy_macd,
    strategy_bollinger_reversal,
    strategy_multi_factor,
)
from autfin.backtest.quick import quick_backtest, print_stats
from autfin.backtest.sweep import parameter_sweep, print_sweep_result

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
]
