"""Quick single-asset backtest using backtesting.py.

Wraps the backtesting.py library to test AutFin strategies on individual stocks.
Generates HTML reports with equity curves, drawdowns, and trade stats.

Usage:
    from autfin.backtest.quick import quick_backtest
    result = quick_backtest("RELIANCE", start="2022-01-01", end="2024-12-31")
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy

from autfin.agents.yfinance_utils import normalize_symbol
from autfin.tools.technical_tools import (
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
)

logger = logging.getLogger(__name__)


# ─── Strategy classes (backtesting.py format) ───


class SMACrossover(Strategy):
    """SMA crossover strategy."""

    fast = 20
    slow = 50

    def init(self):
        close = pd.Series(self.data.Close)
        sma_f = calculate_sma(close.tolist(), self.fast)
        sma_s = calculate_sma(close.tolist(), self.slow)
        self.sma_fast = self.I(lambda: sma_f, name="SMA_fast")
        self.sma_slow = self.I(lambda: sma_s, name="SMA_slow")

    def next(self):
        if not self.position:
            if self.sma_fast[-1] > self.sma_slow[-1] and self.sma_fast[-2] <= self.sma_slow[-2]:
                self.buy(size=10)
        else:
            if self.sma_fast[-1] < self.sma_slow[-1] and self.sma_fast[-2] >= self.sma_slow[-2]:
                self.sell(size=10)


class RSIStrategy(Strategy):
    """RSI oversold/overbought strategy."""

    period = 14
    oversold = 30
    overbought = 70

    def init(self):
        close = pd.Series(self.data.Close)
        rsi = calculate_rsi(close.tolist(), self.period)
        self.rsi = self.I(lambda: rsi, name="RSI")

    def next(self):
        if not self.position:
            if self.rsi[-1] < self.oversold:
                self.buy(size=10)
        else:
            if self.rsi[-1] > self.overbought:
                self.sell(size=10)


class MACDStrategy(Strategy):
    """MACD crossover strategy."""

    fast = 12
    slow = 26
    signal = 9

    def init(self):
        close = pd.Series(self.data.Close)
        macd_data = calculate_macd(close.tolist(), self.fast, self.slow, self.signal)
        self.macd_line = self.I(lambda: macd_data["macd"], name="MACD")
        self.signal_line = self.I(lambda: macd_data["signal"], name="Signal")

    def next(self):
        if not self.position:
            if self.macd_line[-1] > self.signal_line[-1] and self.macd_line[-2] <= self.signal_line[-2]:
                self.buy(size=10)
        else:
            if self.macd_line[-1] < self.signal_line[-1] and self.macd_line[-2] >= self.signal_line[-2]:
                self.sell(size=10)


class BollingerStrategy(Strategy):
    """Bollinger band reversal strategy."""

    period = 20
    stddev = 2.0

    def init(self):
        close = pd.Series(self.data.Close)
        bb = calculate_bollinger_bands(close.tolist(), self.period, self.stddev)
        self.bb_upper = self.I(lambda: bb["upper"], name="BB_upper")
        self.bb_lower = self.I(lambda: bb["lower"], name="BB_lower")

    def next(self):
        if not self.position:
            if self.data.Close[-1] <= self.bb_lower[-1]:
                self.buy(size=10)
        else:
            if self.data.Close[-1] >= self.bb_upper[-1]:
                self.sell(size=10)


STRATEGY_MAP: dict[str, type[Strategy]] = {
    "sma_crossover": SMACrossover,
    "rsi": RSIStrategy,
    "macd": MACDStrategy,
    "bollinger": BollingerStrategy,
}


# ─── Quick backtest ───


def quick_backtest(
    symbol: str,
    strategy: str = "sma_crossover",
    start: str = "2022-01-01",
    end: str = "2024-12-31",
    capital: float = 100_000,
    commission: float = 0.001,
    strategy_params: Optional[dict] = None,
    optimize: bool = False,
    max_tries: int = 200,
) -> dict[str, Any]:
    """Run a quick backtest on a single stock.

    Args:
        symbol: Stock ticker (e.g., "RELIANCE").
        strategy: One of sma_crossover, rsi, macd, bollinger.
        start/end: Date range (YYYY-MM-DD).
        capital: Starting capital in INR.
        commission: Commission per trade (0.001 = 0.1%).
        strategy_params: Override strategy parameters.
        optimize: Run backtesting.py optimizer.
        max_tries: Max optimization iterations.

    Returns:
        dict with stats, backtest object, and optional optimization result.
    """
    ticker = yf.Ticker(normalize_symbol(symbol.upper()))
    df = ticker.history(start=start, end=end, interval="1d")
    if df.empty:
        raise ValueError(f"No data for {symbol}")

    # backtesting.py needs specific column names
    df = df.rename(columns={
        "Open": "Open", "High": "High", "Low": "Low",
        "Close": "Close", "Volume": "Volume",
    })
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    strat_class = STRATEGY_MAP.get(strategy)
    if not strat_class:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(STRATEGY_MAP)}")

    bt = Backtest(df, strat_class, cash=capital, commission=commission)

    # Apply params
    if strategy_params:
        stats = bt.run(**strategy_params)
    else:
        stats = bt.run()

    result: dict[str, Any] = {
        "symbol": symbol.upper(),
        "strategy": strategy,
        "stats": stats,
        "backtest": bt,
    }

    if optimize:
        opt_result = bt.optimize(
            **{k: v for k, v in (strategy_params or {}).items() if isinstance(v, (int, float))},
            max_tries=max_tries,
            random_state=42,
        )
        result["optimized"] = opt_result

    return result


def print_stats(result: dict[str, Any]) -> None:
    """Pretty-print backtest stats."""
    stats = result["stats"]
    print(f"\n{'='*50}")
    print(f"  QUICK BACKTEST: {result['strategy']} on {result['symbol']}")
    print(f"{'='*50}")
    print(f"  Start:           {stats.get('Start', 'N/A')}")
    print(f"  End:             {stats.get('End', 'N/A')}")
    print(f"  Duration:        {stats.get('Duration', 'N/A')}")
    print(f"  Return:          {stats.get('Return [%]', 0):>8.2f}%")
    print(f"  Buy & Hold:      {stats.get('Buy & Hold Return [%]', 0):>8.2f}%")
    print(f"  Sharpe Ratio:    {stats.get('Sharpe Ratio', 0):>8.2f}")
    print(f"  Max Drawdown:    {stats.get('Max. Drawdown [%]', 0):>8.2f}%")
    print(f"  Win Rate:        {stats.get('Win Rate [%]', 0):>8.1f}%")
    print(f"  # Trades:        {stats.get('# Trades', 0):>8d}")
    print(f"  Profit Factor:   {stats.get('Profit Factor', 0):>8.2f}")
    print(f"  Avg Trade:       {stats.get('Avg. Trade [%]', 0):>8.2f}%")
    print(f"{'='*50}")
