"""Performance metrics for backtesting."""

from dataclasses import dataclass, field
from typing import Any
import math


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics."""

    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    annualized_volatility: float = 0.0
    avg_holding_period_days: float = 0.0
    expectancy: float = 0.0
    payoff_ratio: float = 0.0
    max_consecutive_losses: int = 0
    recovery_factor: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_return_pct": round(self.total_return * 100, 2),
            "annualized_return_pct": round(self.annualized_return * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "calmar_ratio": round(self.calmar_ratio, 3),
            "win_rate_pct": round(self.win_rate * 100, 2),
            "profit_factor": round(self.profit_factor, 3),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "avg_win": round(self.avg_win, 2),
            "avg_loss": round(self.avg_loss, 2),
            "annualized_volatility_pct": round(self.annualized_volatility * 100, 2),
            "avg_holding_period_days": round(self.avg_holding_period_days, 1),
            "expectancy_pct": round(self.expectancy * 100, 2),
            "payoff_ratio": round(self.payoff_ratio, 3),
            "max_consecutive_losses": self.max_consecutive_losses,
            "recovery_factor": round(self.recovery_factor, 3),
        }


def calculate_metrics(
    equity_curve: list[float],
    trades: list[dict[str, Any]],
    trading_days_per_year: int = 252,
    risk_free_rate: float = 0.06,
) -> BacktestMetrics:
    """Calculate comprehensive backtest metrics.

    Args:
        equity_curve: Portfolio value at each time step (starting with initial capital).
        trades: List of completed trades, each with 'pnl' (float) and 'holding_days' (int).
        trading_days_per_year: Trading days in a year (252 for equities).
        risk_free_rate: Annual risk-free rate (default 6% for India).

    Returns:
        BacktestMetrics with all computed metrics.
    """
    metrics = BacktestMetrics()

    if len(equity_curve) < 2:
        return metrics

    initial_capital = equity_curve[0]
    final_capital = equity_curve[-1]

    # --- Return metrics ---
    metrics.total_return = (final_capital - initial_capital) / initial_capital if initial_capital else 0.0

    n_periods = len(equity_curve) - 1
    years = n_periods / trading_days_per_year
    if years > 0 and initial_capital > 0:
        metrics.annualized_return = (final_capital / initial_capital) ** (1 / years) - 1

    # --- Daily returns ---
    daily_returns = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i - 1]
        if prev > 0:
            daily_returns.append((equity_curve[i] - prev) / prev)
        else:
            daily_returns.append(0.0)

    # --- Volatility ---
    if len(daily_returns) > 1:
        mean_ret = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_ret) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        daily_vol = math.sqrt(variance)
        metrics.annualized_volatility = daily_vol * math.sqrt(trading_days_per_year)

    # --- Sharpe Ratio ---
    if metrics.annualized_volatility > 0:
        metrics.sharpe_ratio = (metrics.annualized_return - risk_free_rate) / metrics.annualized_volatility

    # --- Sortino Ratio ---
    downside_returns = [r for r in daily_returns if r < 0]
    if downside_returns:
        downside_variance = sum(r ** 2 for r in downside_returns) / len(downside_returns)
        downside_vol = math.sqrt(downside_variance) * math.sqrt(trading_days_per_year)
        if downside_vol > 0:
            metrics.sortino_ratio = (metrics.annualized_return - risk_free_rate) / downside_vol

    # --- Max Drawdown ---
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        if peak > 0:
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
    metrics.max_drawdown = max_dd

    # --- Calmar Ratio ---
    if metrics.max_drawdown > 0:
        metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown

    # --- Recovery Factor ---
    if metrics.max_drawdown > 0:
        metrics.recovery_factor = metrics.total_return / metrics.max_drawdown

    # --- Trade metrics ---
    if trades:
        metrics.total_trades = len(trades)
        pnls = [t.get("pnl", 0.0) for t in trades]
        holding_days = [t.get("holding_days", 0) for t in trades]

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        metrics.winning_trades = len(wins)
        metrics.losing_trades = len(losses)
        metrics.win_rate = len(wins) / len(pnls) if pnls else 0.0

        metrics.avg_win = sum(wins) / len(wins) if wins else 0.0
        metrics.avg_loss = sum(losses) / len(losses) if losses else 0.0

        # Profit factor: gross profits / gross losses
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

        # Payoff ratio
        metrics.payoff_ratio = abs(metrics.avg_win / metrics.avg_loss) if metrics.avg_loss != 0 else float("inf")

        # Expectancy
        metrics.expectancy = (
            metrics.win_rate * metrics.avg_win + (1 - metrics.win_rate) * metrics.avg_loss
        )

        # Average holding period
        metrics.avg_holding_period_days = sum(holding_days) / len(holding_days) if holding_days else 0.0

        # Max consecutive losses
        max_losses = 0
        current_losses = 0
        for p in pnls:
            if p <= 0:
                current_losses += 1
                max_losses = max(max_losses, current_losses)
            else:
                current_losses = 0
        metrics.max_consecutive_losses = max_losses

    return metrics
