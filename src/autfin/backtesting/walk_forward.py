"""Walk-forward backtesting harness for AutFin.

Prevents overfitting by splitting historical data into rolling
train/test windows. Strategy parameters are optimized on the training
window, then evaluated on the out-of-sample test window. The walk
advances forward, and the process repeats.

Walk-forward analysis (WFA) is the gold standard for evaluating
trading strategies because it simulates real-world conditions where
you optimize on past data and trade on unseen future data.

Architecture:
  1. split_windows() — divide price data into (train, test) pairs
  2. optimize_on_train() — find best parameters on training data
  3. evaluate_on_test() — run strategy with optimized params on test
  4. walk_forward() — orchestrate the full loop
  5. summarize_results() — aggregate OOS metrics, detect overfitting

All arithmetic is deterministic — no LLM involved.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Window:
    """A single train/test window pair."""
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    fold: int = 0

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    train_return_pct: float = 0.0
    test_return_pct: float = 0.0
    train_max_drawdown: float = 0.0
    test_max_drawdown: float = 0.0
    train_win_rate: float = 0.0
    test_win_rate: float = 0.0
    train_trades: int = 0
    test_trades: int = 0
    best_params: Dict[str, Any] = field(default_factory=dict)
    overfit_ratio: float = 0.0  # train_sharpe / test_sharpe (>1.5 = overfit)


def split_windows(
    n_bars: int,
    train_pct: float = 0.7,
    n_folds: int = 5,
    step_pct: float = 0.0,
) -> List[Window]:
    """Split n_bars into rolling train/test windows.

    Args:
        n_bars: Total number of data points
        train_pct: Fraction of each window used for training
        n_folds: Number of walk-forward folds
        step_pct: Step size as fraction of window (0 = auto)

    Returns:
        List of Window objects
    """
    if n_folds < 1:
        n_folds = 1

    window_size = n_bars // n_folds
    if window_size < 10:
        # Too few bars, single fold
        train_end = int(n_bars * train_pct)
        return [Window(0, train_end, train_end, n_bars, fold=0)]

    step = max(1, int(window_size * (1 - train_pct))) if step_pct <= 0 else max(1, int(n_bars * step_pct))

    windows = []
    for i in range(n_folds):
        start = i * step
        if start + window_size > n_bars:
            break
        train_end = start + int(window_size * train_pct)
        test_end = min(start + window_size, n_bars)
        if train_end >= test_end or test_end - train_end < 3:
            continue
        windows.append(Window(
            train_start=start,
            train_end=train_end,
            test_start=train_end,
            test_end=test_end,
            fold=i,
        ))

    if not windows:
        train_end = int(n_bars * train_pct)
        windows = [Window(0, train_end, train_end, n_bars, fold=0)]

    return windows


def _simple_returns(closes: List[float]) -> List[float]:
    """Compute simple returns from close prices."""
    if len(closes) < 2:
        return []
    return [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]


def _sharpe_ratio(returns: List[float], risk_free: float = 0.0) -> float:
    """Annualized Sharpe ratio (assume 252 trading days)."""
    if not returns:
        return 0.0
    excess = [r - risk_free for r in returns]
    mean_r = sum(excess) / len(excess)
    if len(excess) < 2:
        return 0.0
    var = sum((r - mean_r) ** 2 for r in excess) / (len(excess) - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    if std == 0:
        return 0.0
    return (mean_r / std) * math.sqrt(252)


def _max_drawdown(returns: List[float]) -> float:
    """Maximum drawdown from return series."""
    if not returns:
        return 0.0
    cumulative = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        cumulative *= (1 + r)
        peak = max(peak, cumulative)
        dd = (peak - cumulative) / peak
        max_dd = max(max_dd, dd)
    return max_dd


def _win_rate(returns: List[float]) -> float:
    """Fraction of positive-return periods."""
    if not returns:
        return 0.0
    wins = sum(1 for r in returns if r > 0)
    return wins / len(returns)


def simple_sma_crossover(
    closes: List[float],
    fast_period: int = 20,
    slow_period: int = 50,
) -> List[float]:
    """Simple SMA crossover strategy. Returns list of trade returns."""
    if len(closes) < slow_period + 1:
        return []

    trades = []
    position = None  # None = flat, >0 = long

    for i in range(slow_period, len(closes)):
        fast_ma = sum(closes[i - fast_period:i]) / fast_period
        slow_ma = sum(closes[i - slow_period:i]) / slow_period

        if fast_ma > slow_ma and position is None:
            position = closes[i]
        elif fast_ma < slow_ma and position is not None:
            ret = (closes[i] - position) / position
            trades.append(ret)
            position = None

    return trades


def _default_optimize(
    closes: List[float],
    param_grid: Dict[str, List[Any]],
    strategy_fn: Callable,
) -> Dict[str, Any]:
    """Grid search for best params on training data. Maximizes Sharpe."""
    best_sharpe = -999
    best_params: Dict[str, Any] = {}

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    def _grid(idx: int, current: Dict[str, Any]):
        nonlocal best_sharpe, best_params
        if idx == len(keys):
            returns = strategy_fn(closes, **current)
            sharpe = _sharpe_ratio(returns)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = dict(current)
            return
        for v in values[idx]:
            current[keys[idx]] = v
            _grid(idx + 1, current)

    _grid(0, {})
    return best_params


@dataclass
class BacktestMetrics:
    """Metrics from a single backtest run."""
    returns: List[float] = field(default_factory=list)
    sharpe: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    n_trades: int = 0


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis with symbol + date range."""
    symbol: str
    start_date: str | None = None  # ISO date string
    end_date: str | None = None    # ISO date string
    as_of: str | None = None       # Point-in-time cutoff for backtests
    strategy_fn: Callable = simple_sma_crossover
    param_grid: Dict[str, List[Any]] | None = None
    n_folds: int = 5
    train_pct: float = 0.7
    period: str = "2y"  # Fallback period if no start/end dates
    interval: str = "1d"


def _run_backtest(
    closes: List[float],
    strategy_fn: Callable,
    params: Dict[str, Any],
) -> BacktestMetrics:
    """Run strategy on price data and compute metrics."""
    returns = strategy_fn(closes, **params)
    if not returns:
        return BacktestMetrics()

    total_return = 1.0
    for r in returns:
        total_return *= (1 + r)

    return BacktestMetrics(
        returns=returns,
        sharpe=_sharpe_ratio(returns),
        total_return_pct=round((total_return - 1) * 100, 2),
        max_drawdown=round(_max_drawdown(returns), 4),
        win_rate=round(_win_rate(returns), 4),
        n_trades=len(returns),
    )


def walk_forward(
    closes: List[float],
    strategy_fn: Callable = simple_sma_crossover,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    n_folds: int = 5,
    train_pct: float = 0.7,
) -> Dict[str, Any]:
    """Run walk-forward analysis.

    Args:
        closes: Full series of close prices
        strategy_fn: Function(closes, **params) -> List[float] of trade returns
        param_grid: Dict of param_name -> list of values to try
        n_folds: Number of walk-forward folds
        train_pct: Fraction of each window for training

    Returns:
        Dict with fold results, aggregate OOS metrics, and overfitting assessment
    """
    if param_grid is None:
        param_grid = {"fast_period": [10, 20, 30], "slow_period": [40, 50, 60]}

    windows = split_windows(len(closes), train_pct=train_pct, n_folds=n_folds)
    fold_results: List[FoldResult] = []

    for w in windows:
        train_data = closes[w.train_start:w.train_end]
        test_data = closes[w.test_start:w.test_end]

        # Optimize on train
        best_params = _default_optimize(train_data, param_grid, strategy_fn)

        # Evaluate on train (in-sample)
        train_bt = _run_backtest(train_data, strategy_fn, best_params)

        # Evaluate on test (out-of-sample)
        test_bt = _run_backtest(test_data, strategy_fn, best_params)

        # Overfit ratio: IS Sharpe / OOS Sharpe
        overfit_ratio = (
            train_bt.sharpe / test_bt.sharpe
            if test_bt.sharpe > 0
            else 999.0
        )

        fr = FoldResult(
            fold=w.fold,
            train_start=w.train_start,
            train_end=w.train_end,
            test_start=w.test_start,
            test_end=w.test_end,
            train_sharpe=round(train_bt.sharpe, 3),
            test_sharpe=round(test_bt.sharpe, 3),
            train_return_pct=train_bt.total_return_pct,
            test_return_pct=test_bt.total_return_pct,
            train_max_drawdown=train_bt.max_drawdown,
            test_max_drawdown=test_bt.max_drawdown,
            train_win_rate=train_bt.win_rate,
            test_win_rate=test_bt.win_rate,
            train_trades=train_bt.n_trades,
            test_trades=test_bt.n_trades,
            best_params=best_params,
            overfit_ratio=round(overfit_ratio, 2),
        )
        fold_results.append(fr)

    # Aggregate OOS metrics
    oos_returns: List[float] = []
    oos_sharpes: List[float] = []
    oos_returns_pct: List[float] = []
    oos_drawdowns: List[float] = []

    for fr in fold_results:
        oos_sharpes.append(fr.test_sharpe)
        oos_returns_pct.append(fr.test_return_pct)
        oos_drawdowns.append(fr.test_max_drawdown)

    avg_oos_sharpe = sum(oos_sharpes) / max(len(oos_sharpes), 1)
    avg_oos_return = sum(oos_returns_pct) / max(len(oos_returns_pct), 1)
    avg_oos_drawdown = sum(oos_drawdowns) / max(len(oos_drawdowns), 1)
    avg_overfit = sum(fr.overfit_ratio for fr in fold_results) / max(len(fold_results), 1)

    # Overfitting assessment
    is_overfit = avg_overfit > 1.5 or avg_oos_sharpe < 0.5
    assessment = (
        "OVERFIT" if is_overfit
        else "CAUTION" if avg_overfit > 1.2
        else "HEALTHY"
    )

    # Win consistency: how many folds had positive OOS return
    positive_folds = sum(1 for r in oos_returns_pct if r > 0)
    win_consistency = positive_folds / max(len(fold_results), 1)

    return {
        "n_folds": len(fold_results),
        "fold_results": [
            {
                "fold": fr.fold,
                "train_sharpe": fr.train_sharpe,
                "test_sharpe": fr.test_sharpe,
                "train_return_pct": fr.train_return_pct,
                "test_return_pct": fr.test_return_pct,
                "train_max_drawdown": fr.train_max_drawdown,
                "test_max_drawdown": fr.test_max_drawdown,
                "best_params": fr.best_params,
                "overfit_ratio": fr.overfit_ratio,
            }
            for fr in fold_results
        ],
        "aggregate": {
            "avg_oos_sharpe": round(avg_oos_sharpe, 3),
            "avg_oos_return_pct": round(avg_oos_return, 2),
            "avg_oos_drawdown": round(avg_oos_drawdown, 4),
            "avg_overfit_ratio": round(avg_overfit, 2),
            "win_consistency": round(win_consistency, 3),
        },
        "assessment": assessment,
        "is_overfit": is_overfit,
        "recommendation": (
            "Strategy shows overfitting — reduce complexity or add regularization"
            if is_overfit
            else "Strategy passes WFA — acceptable for paper trading"
        ),
    }


async def fetch_price_data(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    as_of: str | None = None,
    period: str = "2y",
    interval: str = "1d",
) -> List[float]:
    """Fetch close prices for walk-forward backtesting.

    Uses yfinance_utils.fetch_history for point-in-time safety.

    Args:
        symbol: Stock symbol (e.g. "RELIANCE")
        start: ISO date string for data start (e.g. "2023-01-01")
        end: ISO date string for data end (e.g. "2024-06-01")
        as_of: Point-in-time cutoff — data after this date is excluded
        period: yfinance period if no start/end (default "2y")
        interval: yfinance interval (default "1d")

    Returns:
        List of close prices. Empty list if no data.
    """
    import yfinance as yf
    from src.autfin.agents.yfinance_utils import normalize_symbol

    ticker = yf.Ticker(normalize_symbol(symbol))

    kwargs: dict = {"interval": interval}
    if start and end:
        kwargs["start"] = start
        kwargs["end"] = end
    else:
        kwargs["period"] = period

    df = await asyncio.to_thread(ticker.history, **kwargs)
    if df is None or df.empty:
        return []

    # Point-in-time filtering
    if as_of is not None:
        if isinstance(as_of, str):
            as_of_dt = datetime.fromisoformat(as_of)
        else:
            as_of_dt = as_of
        df = df[df.index <= pd.Timestamp(as_of_dt)]

    return df["Close"].tolist()


async def walk_forward_with_symbol(
    config: WalkForwardConfig,
) -> Dict[str, Any]:
    """Run walk-forward analysis on a symbol with point-in-time safety.

    Fetches historical data using yfinance_utils.fetch_history (with as_of
    for backtest safety), then runs the standard walk_forward analysis.

    Args:
        config: WalkForwardConfig with symbol, dates, strategy, etc.

    Returns:
        Same dict as walk_forward(), augmented with symbol + date metadata.
    """
    closes = await fetch_price_data(
        symbol=config.symbol,
        start=config.start_date,
        end=config.end_date,
        as_of=config.as_of,
        period=config.period,
        interval=config.interval,
    )

    if not closes:
        return {
            "symbol": config.symbol,
            "error": "No price data available",
            "n_folds": 0,
            "fold_results": [],
            "aggregate": {},
            "assessment": "NO_DATA",
            "is_overfit": False,
            "recommendation": "Cannot evaluate — no historical data for this symbol/date range",
        }

    result = walk_forward(
        closes=closes,
        strategy_fn=config.strategy_fn,
        param_grid=config.param_grid,
        n_folds=config.n_folds,
        train_pct=config.train_pct,
    )

    result["symbol"] = config.symbol
    result["start_date"] = config.start_date
    result["end_date"] = config.end_date
    result["as_of"] = config.as_of
    result["n_bars"] = len(closes)

    return result
