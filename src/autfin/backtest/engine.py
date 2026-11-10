"""AutFin Backtest Engine — multi-factor strategy validation.

Backtests AutFin's combined analysis pipeline (technical + fundamental + sentiment)
against historical data. Zero external backtesting dependencies — uses existing
yfinance + pandas + technical_tools.py.

Usage:
    from autfin.backtest.engine import BacktestEngine

    engine = BacktestEngine("RELIANCE", start="2022-01-01", end="2024-12-31")
    result = engine.run()
    print(result.summary())
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from autfin.tools.technical_tools import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_sma,
)

logger = logging.getLogger(__name__)


# ─── Configuration ───


@dataclass
class BacktestConfig:
    """Backtest parameters."""

    initial_capital: float = 100_000.0
    position_size: float = 0.95  # fraction of capital per trade
    commission: float = 0.001  # 0.1% per trade (Zerodha-like)
    slippage: float = 0.0005  # 0.05%
    max_positions: int = 1  # max concurrent positions
    stop_loss_atr_mult: float = 2.0  # stop loss = entry - 2x ATR
    take_profit_atr_mult: float = 3.0  # take profit = entry + 3x ATR


# ─── Strategies ───


def strategy_sma_crossover(
    df: pd.DataFrame, fast: int = 20, slow: int = 50
) -> pd.Series:
    """SMA crossover: buy when fast > slow, sell when fast < slow."""
    sma_fast = calculate_sma(df["Close"].tolist(), fast)
    sma_slow = calculate_sma(df["Close"].tolist(), slow)
    sig = pd.Series(0, index=df.index, dtype=int)
    for i in range(len(df)):
        if sma_fast[i] is not None and sma_slow[i] is not None:
            if sma_fast[i] > sma_slow[i]:
                sig.iloc[i] = 1  # long
            elif sma_fast[i] < sma_slow[i]:
                sig.iloc[i] = -1  # flat
    return sig


def strategy_rsi(
    df: pd.DataFrame, period: int = 14, oversold: int = 30, overbought: int = 70
) -> pd.Series:
    """RSI: buy when oversold, sell when overbought."""
    rsi = calculate_rsi(df["Close"].tolist(), period)
    sig = pd.Series(0, index=df.index, dtype=int)
    for i in range(len(df)):
        if rsi[i] is not None:
            if rsi[i] < oversold:
                sig.iloc[i] = 1
            elif rsi[i] > overbought:
                sig.iloc[i] = -1
    return sig


def strategy_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.Series:
    """MACD: buy on bullish crossover, sell on bearish."""
    macd_data = calculate_macd(df["Close"].tolist(), fast, slow, signal)
    hist = macd_data["histogram"]
    sig = pd.Series(0, index=df.index, dtype=int)
    for i in range(1, len(df)):
        if hist[i] is not None and hist[i - 1] is not None:
            if hist[i] > 0 and hist[i - 1] <= 0:
                sig.iloc[i] = 1
            elif hist[i] < 0 and hist[i - 1] >= 0:
                sig.iloc[i] = -1
    return sig


def strategy_bollinger_reversal(
    df: pd.DataFrame, period: int = 20, stddev: float = 2.0
) -> pd.Series:
    """Bollinger: buy at lower band, sell at upper band."""
    bb = calculate_bollinger_bands(df["Close"].tolist(), period, stddev)
    sig = pd.Series(0, index=df.index, dtype=int)
    for i in range(len(df)):
        if bb["lower"][i] is not None and bb["upper"][i] is not None:
            if df["Close"].iloc[i] <= bb["lower"][i]:
                sig.iloc[i] = 1
            elif df["Close"].iloc[i] >= bb["upper"][i]:
                sig.iloc[i] = -1
    return sig


def strategy_multi_factor(
    df: pd.DataFrame,
    sma_fast: int = 20,
    sma_slow: int = 50,
    rsi_period: int = 14,
    rsi_oversold: int = 40,
    rsi_overbought: int = 60,
) -> pd.Series:
    """Combined: SMA trend + RSI filter. Buy when SMA bullish AND RSI not overbought."""
    sma_f = calculate_sma(df["Close"].tolist(), sma_fast)
    sma_s = calculate_sma(df["Close"].tolist(), sma_slow)
    rsi = calculate_rsi(df["Close"].tolist(), rsi_period)
    sig = pd.Series(0, index=df.index, dtype=int)
    for i in range(len(df)):
        if sma_f[i] is not None and sma_s[i] is not None and rsi[i] is not None:
            trend_up = sma_f[i] > sma_s[i]
            rsi_ok = rsi[i] < rsi_overbought
            rsi_oversold_hit = rsi[i] < rsi_oversold
            if trend_up and rsi_ok:
                sig.iloc[i] = 1
            elif not trend_up or rsi_oversold_hit:
                sig.iloc[i] = -1
    return sig


STRATEGIES: dict[str, Callable] = {
    "sma_crossover": strategy_sma_crossover,
    "rsi": strategy_rsi,
    "macd": strategy_macd,
    "bollinger": strategy_bollinger_reversal,
    "multi_factor": strategy_multi_factor,
}


# ─── Trade tracking ───


@dataclass
class Trade:
    entry_date: str
    entry_price: float
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    shares: int = 0
    pnl: float = 0.0
    return_pct: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestResult:
    strategy: str
    symbol: str
    start: str
    end: str
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    benchmark_curve: list[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"{'='*50}",
            f"  BACKTEST: {self.strategy} on {self.symbol}",
            f"  {self.start} → {self.end}",
            f"{'='*50}",
            f"  Total Return:    {self.total_return:>+8.2%}",
            f"  Annual Return:   {self.annual_return:>+8.2%}",
            f"  Sharpe Ratio:    {self.sharpe_ratio:>8.2f}",
            f"  Max Drawdown:    {self.max_drawdown:>8.2%}",
            f"  Win Rate:        {self.win_rate:>8.1%}",
            f"  Profit Factor:   {self.profit_factor:>8.2f}",
            f"  Total Trades:    {self.total_trades:>8d}",
            f"  Avg Trade:       {self.avg_trade_return:>+8.2%}",
            f"  Best Trade:      {self.best_trade:>+8.2%}",
            f"  Worst Trade:     {self.worst_trade:>+8.2%}",
            f"{'='*50}",
        ]
        return "\n".join(lines)


# ─── Engine ───


class BacktestEngine:
    """Runs a strategy against historical OHLCV data."""

    def __init__(
        self,
        symbol: str,
        start: str = "2020-01-01",
        end: str = "2024-12-31",
        config: Optional[BacktestConfig] = None,
    ):
        self.symbol = symbol.upper()
        self.start = start
        self.end = end
        self.config = config or BacktestConfig()
        self._df: Optional[pd.DataFrame] = None

    def fetch_data(self) -> pd.DataFrame:
        """Download OHLCV from yfinance."""
        if self._df is not None:
            return self._df
        from autfin.agents.yfinance_utils import normalize_symbol
        ticker = yf.Ticker(normalize_symbol(self.symbol))
        df = ticker.history(start=self.start, end=self.end, interval="1d")
        if df.empty:
            raise ValueError(f"No data for {self.symbol} between {self.start} and {self.end}")
        self._df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        return self._df

    def run(
        self,
        strategy: str | Callable = "sma_crossover",
        strategy_kwargs: Optional[dict] = None,
    ) -> BacktestResult:
        """Run backtest.

        Args:
            strategy: Name from STRATEGIES dict or a callable(df) -> pd.Series of signals.
            strategy_kwargs: Extra kwargs to pass to the strategy function.
        """
        df = self.fetch_data()
        kwargs = strategy_kwargs or {}

        # Get signal series
        if isinstance(strategy, str):
            if strategy not in STRATEGIES:
                raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(STRATEGIES)}")
            signals = STRATEGIES[strategy](df, **kwargs)
            strategy_name = strategy
        else:
            signals = strategy(df, **kwargs)
            strategy_name = strategy.__name__

        # Simulate trades
        cfg = self.config
        capital = cfg.initial_capital
        position = 0  # shares held
        entry_price = 0.0
        entry_date = ""
        trades: list[Trade] = []
        equity = []

        for i in range(len(df)):
            price = df["Close"].iloc[i]
            date_str = str(df.index[i].date())
            sig = signals.iloc[i]

            # Stop loss / take profit check (if in position)
            if position > 0:
                pnl_pct = (price - entry_price) / entry_price
                atr_val = None
                atr_list = calculate_atr(
                    df["High"].iloc[: i + 1].tolist(),
                    df["Low"].iloc[: i + 1].tolist(),
                    df["Close"].iloc[: i + 1].tolist(),
                    14,
                )
                if atr_list:
                    atr_val = atr_list[-1]

                # Stop loss
                if atr_val and price <= entry_price - cfg.stop_loss_atr_mult * atr_val:
                    exit_price = price * (1 - cfg.slippage)
                    cost = position * exit_price * cfg.commission
                    pnl = (exit_price - entry_price) * position - cost
                    capital += position * exit_price - cost
                    trades.append(
                        Trade(
                            entry_date=entry_date,
                            entry_price=entry_price,
                            exit_date=date_str,
                            exit_price=exit_price,
                            shares=position,
                            pnl=pnl,
                            return_pct=(exit_price - entry_price) / entry_price,
                            exit_reason="stop_loss",
                        )
                    )
                    position = 0

                # Take profit
                elif atr_val and price >= entry_price + cfg.take_profit_atr_mult * atr_val:
                    exit_price = price * (1 - cfg.slippage)
                    cost = position * exit_price * cfg.commission
                    pnl = (exit_price - entry_price) * position - cost
                    capital += position * exit_price - cost
                    trades.append(
                        Trade(
                            entry_date=entry_date,
                            entry_price=entry_price,
                            exit_date=date_str,
                            exit_price=exit_price,
                            shares=position,
                            pnl=pnl,
                            return_pct=(exit_price - entry_price) / entry_price,
                            exit_reason="take_profit",
                        )
                    )
                    position = 0

            # Signal-based entries/exits
            if sig == 1 and position == 0:
                # Buy
                entry_price = price * (1 + cfg.slippage)
                investable = capital * cfg.position_size
                shares = int(investable / entry_price)
                if shares > 0:
                    cost = shares * entry_price * cfg.commission
                    capital -= shares * entry_price + cost
                    position = shares
                    entry_date = date_str

            elif sig == -1 and position > 0:
                # Sell
                exit_price = price * (1 - cfg.slippage)
                cost = position * exit_price * cfg.commission
                pnl = (exit_price - entry_price) * position - cost
                capital += position * exit_price - cost
                trades.append(
                    Trade(
                        entry_date=entry_date,
                        entry_price=entry_price,
                        exit_date=date_str,
                        exit_price=exit_price,
                        shares=position,
                        pnl=pnl,
                        return_pct=(exit_price - entry_price) / entry_price,
                        exit_reason="signal",
                    )
                )
                position = 0

            # Track equity
            equity_val = capital + position * price
            equity.append(equity_val)

        # Close open position at end
        if position > 0:
            exit_price = df["Close"].iloc[-1] * (1 - cfg.slippage)
            cost = position * exit_price * cfg.commission
            pnl = (exit_price - entry_price) * position - cost
            capital += position * exit_price - cost
            trades.append(
                Trade(
                    entry_date=entry_date,
                    entry_price=entry_price,
                    exit_date=str(df.index[-1].date()),
                    exit_price=exit_price,
                    shares=position,
                    pnl=pnl,
                    return_pct=(exit_price - entry_price) / entry_price,
                    exit_reason="end_of_data",
                )
            )
            position = 0

        # Calculate metrics
        result = BacktestResult(
            strategy=strategy_name,
            symbol=self.symbol,
            start=self.start,
            end=self.end,
            trades=trades,
            equity_curve=equity,
        )

        if equity:
            result.total_return = (equity[-1] / cfg.initial_capital) - 1
            years = max((df.index[-1] - df.index[0]).days / 365.25, 0.01)
            result.annual_return = (1 + result.total_return) ** (1 / years) - 1

        if len(equity) > 1:
            returns = pd.Series(equity).pct_change().dropna()
            if returns.std() > 0:
                result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

            # Max drawdown
            peak = pd.Series(equity).cummax()
            dd = (pd.Series(equity) - peak) / peak
            result.max_drawdown = dd.min()

        # Trade stats
        result.total_trades = len(trades)
        if trades:
            wins = [t for t in trades if t.pnl > 0]
            losses = [t for t in trades if t.pnl <= 0]
            result.win_rate = len(wins) / len(trades) if trades else 0
            result.avg_trade_return = np.mean([t.return_pct for t in trades])
            result.best_trade = max(t.return_pct for t in trades)
            result.worst_trade = min(t.return_pct for t in trades)
            total_profit = sum(t.pnl for t in wins)
            total_loss = abs(sum(t.pnl for t in losses))
            result.profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        # Benchmark (buy & hold)
        start_price = df["Close"].iloc[0]
        end_price = df["Close"].iloc[-1]
        result.benchmark_curve = (
            (df["Close"] / start_price * cfg.initial_capital).tolist()
        )

        return result
