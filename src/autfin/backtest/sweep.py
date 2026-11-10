"""Parameter sweep backtest using VectorBT.

Tests strategy parameter combinations at speed (vectorized, NumPy/Numba).
Good for finding optimal SMA periods, RSI thresholds, etc.

Usage:
    from autfin.backtest.sweep import parameter_sweep
    result = parameter_sweep("RELIANCE", strategy="sma_crossover")
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from autfin.agents.yfinance_utils import normalize_symbol

logger = logging.getLogger(__name__)


def _fetch_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV from yfinance."""
    ticker = yf.Ticker(normalize_symbol(symbol.upper()))
    df = ticker.history(start=start, end=end, interval="1d")
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    return df


def sweep_sma_crossover(
    symbol: str,
    start: str = "2022-01-01",
    end: str = "2024-12-31",
    fast_range: tuple[int, int, int] = (5, 50, 5),
    slow_range: tuple[int, int, int] = (20, 200, 10),
    capital: float = 100_000,
    commission: float = 0.001,
) -> dict[str, Any]:
    """Sweep SMA crossover parameters.

    Args:
        fast_range: (start, end, step) for fast SMA period.
        slow_range: (start, end, step) for slow SMA period.
    """
    import vectorbt as vbt

    df = _fetch_data(symbol, start, end)
    close = df["Close"]

    fast_periods = np.arange(fast_range[0], fast_range[1], fast_range[2])
    slow_periods = np.arange(slow_range[0], slow_range[1], slow_range[2])

    # Generate SMA signals for all parameter combos
    sma_fast = vbt.MA.run(close, fast_periods)
    sma_slow = vbt.MA.run(close, slow_periods)

    # Entry: fast crosses above slow; Exit: fast crosses below
    entries = sma_fast.ma_crossed_above(sma_slow)
    exits = sma_fast.ma_crossed_below(sma_slow)

    # Run vectorized backtest
    pf = vbt.Portfolio.from_signals(
        close, entries, exits,
        init_cash=capital,
        fees=commission,
        freq="1D",
    )

    # Collect results
    total_returns = pf.total_return()
    sharpes = pf.sharpe_ratio()
    max_dds = pf.max_drawdown()

    # Find best
    if isinstance(total_returns, pd.Series):
        best_idx = total_returns.idxmax()
        best_return = float(total_returns.loc[best_idx])
        best_sharpe = float(sharpes.loc[best_idx]) if best_idx in sharpes.index else 0
        best_dd = float(max_dds.loc[best_idx]) if best_idx in max_dds.index else 0
    else:
        best_idx = "single"
        best_return = float(total_returns)
        best_sharpe = float(sharpes)
        best_dd = float(max_dds)

    return {
        "symbol": symbol.upper(),
        "strategy": "sma_crossover",
        "param_grid": {
            "fast": fast_periods.tolist(),
            "slow": slow_periods.tolist(),
        },
        "best": {
            "params": str(best_idx),
            "return": best_return,
            "sharpe": best_sharpe,
            "max_drawdown": best_dd,
        },
        "total_return": total_returns,
        "sharpe_ratio": sharpes,
        "max_drawdown": max_dds,
        "portfolio": pf,
    }


def sweep_rsi(
    symbol: str,
    start: str = "2022-01-01",
    end: str = "2024-12-31",
    period_range: tuple[int, int, int] = (7, 28, 7),
    oversold_range: tuple[int, int, int] = (20, 40, 5),
    overbought_range: tuple[int, int, int] = (60, 80, 5),
    capital: float = 100_000,
    commission: float = 0.001,
) -> dict[str, Any]:
    """Sweep RSI parameters."""
    import vectorbt as vbt

    df = _fetch_data(symbol, start, end)
    close = df["Close"]

    periods = np.arange(period_range[0], period_range[1], period_range[2])
    rsi = vbt.RSI.run(close, window=periods)

    # Test different threshold combos
    results = {}
    for os_lvl in np.arange(oversold_range[0], oversold_range[1], oversold_range[2]):
        for ob_lvl in np.arange(overbought_range[0], overbought_range[1], overbought_range[2]):
            entries = rsi.rsi_crossed_below(os_lvl)
            exits = rsi.rsi_crossed_above(ob_lvl)
            pf = vbt.Portfolio.from_signals(close, entries, exits, init_cash=capital, fees=commission, freq="1D")
            key = f"rsi_{periods[0] if len(periods) == 1 else 'sweep'}_os{os_lvl}_ob{ob_lvl}"
            tr = pf.total_return()
            results[key] = {
                "return": float(tr.iloc[0]) if hasattr(tr, "iloc") else float(tr),
                "sharpe": float(pf.sharpe_ratio().iloc[0]) if hasattr(pf.sharpe_ratio(), "iloc") else float(pf.sharpe_ratio()),
                "max_drawdown": float(pf.max_drawdown().iloc[0]) if hasattr(pf.max_drawdown(), "iloc") else float(pf.max_drawdown()),
            }

    # Find best
    best_key = max(results, key=lambda k: results[k]["return"])
    best = results[best_key]

    return {
        "symbol": symbol.upper(),
        "strategy": "rsi",
        "param_grid": {
            "period": periods.tolist(),
            "oversold": list(range(oversold_range[0], oversold_range[1], oversold_range[2])),
            "overbought": list(range(overbought_range[0], overbought_range[1], overbought_range[2])),
        },
        "best": {"params": best_key, **best},
        "results": results,
    }


def sweep_macd(
    symbol: str,
    start: str = "2022-01-01",
    end: str = "2024-12-31",
    fast_range: tuple[int, int, int] = (8, 16, 2),
    slow_range: tuple[int, int, int] = (20, 32, 2),
    signal_range: tuple[int, int, int] = (7, 12, 1),
    capital: float = 100_000,
    commission: float = 0.001,
) -> dict[str, Any]:
    """Sweep MACD parameters."""
    import vectorbt as vbt

    df = _fetch_data(symbol, start, end)
    close = df["Close"]

    fasts = np.arange(fast_range[0], fast_range[1], fast_range[2])
    slows = np.arange(slow_range[0], slow_range[1], slow_range[2])
    signals = np.arange(signal_range[0], signal_range[1], signal_range[2])

    # MACD indicator — broadcast across param combos
    macd = vbt.MACD.run(close, fast=fasts, slow=slows, signal=signals)

    entries = macd.macd_crossed_above(macd.signal)
    exits = macd.macd_crossed_below(macd.signal)

    pf = vbt.Portfolio.from_signals(close, entries, exits, init_cash=capital, fees=commission, freq="1D")

    total_returns = pf.total_return()
    best_idx = total_returns.idxmax() if isinstance(total_returns, pd.Series) else "single"

    return {
        "symbol": symbol.upper(),
        "strategy": "macd",
        "param_grid": {
            "fast": fasts.tolist(),
            "slow": slows.tolist(),
            "signal": signals.tolist(),
        },
        "best": {
            "params": str(best_idx),
            "return": float(total_returns.loc[best_idx]) if isinstance(total_returns, pd.Series) else float(total_returns),
        },
        "total_return": total_returns,
        "portfolio": pf,
    }


SWEEP_MAP = {
    "sma_crossover": sweep_sma_crossover,
    "rsi": sweep_rsi,
    "macd": sweep_macd,
}


def parameter_sweep(
    symbol: str,
    strategy: str = "sma_crossover",
    start: str = "2022-01-01",
    end: str = "2024-12-31",
    **kwargs,
) -> dict[str, Any]:
    """Run parameter sweep for a given strategy.

    Args:
        strategy: One of sma_crossover, rsi, macd.
        **kwargs: Strategy-specific parameter ranges.
    """
    sweep_fn = SWEEP_MAP.get(strategy)
    if not sweep_fn:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(SWEEP_MAP)}")
    return sweep_fn(symbol, start=start, end=end, **kwargs)


def print_sweep_result(result: dict[str, Any]) -> None:
    """Pretty-print sweep results."""
    print(f"\n{'='*50}")
    print(f"  PARAMETER SWEEP: {result['strategy']} on {result['symbol']}")
    print(f"{'='*50}")
    best = result.get("best", {})
    print(f"  Best Params:   {best.get('params', 'N/A')}")
    print(f"  Best Return:   {best.get('return', 0):>+8.2%}")
    print(f"  Best Sharpe:   {best.get('sharpe', 0):>8.2f}")
    print(f"  Max Drawdown:  {best.get('max_drawdown', 0):>8.2%}")
    print(f"{'='*50}")
