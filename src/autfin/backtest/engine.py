"""Core backtesting engine for AutFin."""

from dataclasses import dataclass, field
from typing import Any
import math

from .metrics import BacktestMetrics, calculate_metrics
from .strategies import BaseStrategy, StrategySignal, get_strategy


@dataclass
class TradeRecord:
    """Record of a completed trade."""

    symbol: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    quantity: int
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_days: int = 0
    entry_cost: float = 0.0
    exit_cost: float = 0.0
    total_cost: float = 0.0


@dataclass
class BacktestConfig:
    """Configuration for backtest run."""

    initial_capital: float = 100000.0  # ₹1 lakh
    position_size_pct: float = 10.0  # % of capital per trade
    max_positions: int = 5
    brokerage_pct: float = 0.03  # Zerodha: 0.03% or ₹20 whichever is lower
    stt_pct: float = 0.1  # STT on sell side
    gst_pct: float = 18.0  # GST on brokerage
    sebi_charges_pct: float = 0.0001  # SEBI turnover fee
    stamp_duty_pct: float = 0.015  # Stamp duty on buy side
    slippage_pct: float = 0.05  # Estimated slippage per trade
    risk_free_rate: float = 0.06  # 6% for India


@dataclass
class BacktestResult:
    """Complete backtest result."""

    symbol: str
    strategy_name: str
    config: BacktestConfig
    metrics: BacktestMetrics
    trades: list[TradeRecord]
    equity_curve: list[float]
    daily_returns: list[float]
    signals: list[StrategySignal]


class BacktestEngine:
    """Core backtesting engine.

    Simulates trading strategy on historical data with realistic
    transaction costs, slippage, and position management.
    """

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        data: list[dict[str, Any]],
        strategy: str | BaseStrategy = "momentum",
        symbol: str = "UNKNOWN",
    ) -> BacktestResult:
        """Run backtest on historical data.

        Args:
            data: List of OHLCV dicts with keys: date, open, high, low, close, volume.
                  Sorted chronologically (oldest first).
            strategy: Strategy name string or BaseStrategy instance.
            symbol: Stock symbol for trade records.

        Returns:
            BacktestResult with metrics, trades, and equity curve.
        """
        if isinstance(strategy, str):
            strat = get_strategy(strategy)
        else:
            strat = strategy

        # Generate signals
        signals = strat.generate_signals(data)

        # Simulate trading
        trades, equity_curve = self._simulate(data, signals, symbol)

        # Calculate metrics
        metrics = calculate_metrics(
            equity_curve=equity_curve,
            trades=[{"pnl": t.pnl, "holding_days": t.holding_days} for t in trades],
            risk_free_rate=self.config.risk_free_rate,
        )

        # Daily returns
        daily_returns = []
        for i in range(1, len(equity_curve)):
            prev = equity_curve[i - 1]
            daily_returns.append((equity_curve[i] - prev) / prev if prev > 0 else 0.0)

        return BacktestResult(
            symbol=symbol,
            strategy_name=strat.name,
            config=self.config,
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            signals=signals,
        )

    def _simulate(
        self,
        data: list[dict[str, Any]],
        signals: list[StrategySignal],
        symbol: str,
    ) -> tuple[list[TradeRecord], list[float]]:
        """Simulate trading based on signals.

        Args:
            data: Historical OHLCV data.
            signals: Strategy signals.
            symbol: Stock symbol.

        Returns:
            Tuple of (list of trades, equity curve).
        """
        capital = self.config.initial_capital
        equity_curve = [capital]
        trades: list[TradeRecord] = []
        positions: list[dict[str, Any]] = []

        for i, (bar, signal) in enumerate(zip(data, signals)):
            # Process exits first
            new_positions = []
            for pos in positions:
                current_price = bar["close"]
                # Check stop loss or sell signal
                stop_loss = pos.get("stop_loss")
                if signal.action == "sell" or (stop_loss and stop_loss > 0 and current_price <= stop_loss):
                    # Calculate exit cost
                    exit_value = pos["quantity"] * current_price
                    exit_cost = self._calculate_sell_cost(exit_value)

                    # Calculate P&L
                    entry_value = pos["quantity"] * pos["entry_price"]
                    entry_cost = pos["entry_cost"]
                    pnl = exit_value - entry_value - entry_cost - exit_cost

                    # Record trade
                    trade = TradeRecord(
                        symbol=symbol,
                        entry_date=pos["entry_date"],
                        entry_price=pos["entry_price"],
                        exit_date=bar.get("date", f"day_{i}"),
                        exit_price=current_price,
                        quantity=pos["quantity"],
                        pnl=pnl,
                        pnl_pct=pnl / entry_value if entry_value > 0 else 0,
                        holding_days=i - pos["entry_index"],
                        entry_cost=entry_cost,
                        exit_cost=exit_cost,
                        total_cost=entry_cost + exit_cost,
                    )
                    trades.append(trade)
                    capital += exit_value - exit_cost
                else:
                    new_positions.append(pos)

            positions = new_positions

            # Process entries
            if signal.action == "buy" and len(positions) < self.config.max_positions:
                # Calculate position size
                available = capital * (self.config.position_size_pct / 100)
                entry_price = bar["close"]

                if available > entry_price:
                    quantity = int(available / entry_price)
                    if quantity > 0:
                        entry_value = quantity * entry_price
                        entry_cost = self._calculate_buy_cost(entry_value)

                        if capital >= entry_value + entry_cost:
                            capital -= entry_value + entry_cost
                            positions.append({
                                "entry_price": entry_price,
                                "quantity": quantity,
                                "entry_date": bar.get("date", f"day_{i}"),
                                "entry_index": i,
                                "entry_cost": entry_cost,
                                "stop_loss": entry_price * 0.95,  # 5% stop loss
                            })

            # Update equity curve
            portfolio_value = capital + sum(
                pos["quantity"] * bar["close"] for pos in positions
            )
            equity_curve.append(portfolio_value)

        # Close any remaining positions at last bar
        if positions and data:
            last_bar = data[-1]
            for pos in positions:
                exit_value = pos["quantity"] * last_bar["close"]
                exit_cost = self._calculate_sell_cost(exit_value)
                entry_value = pos["quantity"] * pos["entry_price"]
                pnl = exit_value - entry_value - pos["entry_cost"] - exit_cost

                trade = TradeRecord(
                    symbol=symbol,
                    entry_date=pos["entry_date"],
                    entry_price=pos["entry_price"],
                    exit_date=last_bar.get("date", "end"),
                    exit_price=last_bar["close"],
                    quantity=pos["quantity"],
                    pnl=pnl,
                    pnl_pct=pnl / entry_value if entry_value > 0 else 0,
                    holding_days=len(data) - 1 - pos["entry_index"],
                    entry_cost=pos["entry_cost"],
                    exit_cost=exit_cost,
                    total_cost=pos["entry_cost"] + exit_cost,
                )
                trades.append(trade)

        return trades, equity_curve

    def _calculate_buy_cost(self, value: float) -> float:
        """Calculate buy-side transaction costs (Indian markets)."""
        brokerage = min(value * self.config.brokerage_pct / 100, 20.0)
        gst = brokerage * self.config.gst_pct / 100
        sebi = value * self.config.sebi_charges_pct / 100
        stamp = value * self.config.stamp_duty_pct / 100
        slippage = value * self.config.slippage_pct / 100
        return brokerage + gst + sebi + stamp + slippage

    def _calculate_sell_cost(self, value: float) -> float:
        """Calculate sell-side transaction costs (Indian markets)."""
        brokerage = min(value * self.config.brokerage_pct / 100, 20.0)
        gst = brokerage * self.config.gst_pct / 100
        sebi = value * self.config.sebi_charges_pct / 100
        stt = value * self.config.stt_pct / 100
        stamp = value * self.config.stamp_duty_pct / 100
        slippage = value * self.config.slippage_pct / 100
        return brokerage + gst + sebi + stt + stamp + slippage


def fetch_historical_data(
    symbol: str,
    period: str = "2y",
) -> list[dict[str, Any]]:
    """Fetch historical OHLCV data using yfinance.

    Args:
        symbol: Stock symbol (e.g., "RELIANCE.NS" for NSE).
        period: Data period (1y, 2y, 5y, max).

    Returns:
        List of OHLCV dicts sorted chronologically.

    Raises:
        ImportError: If yfinance is not installed.
        ValueError: If no data returned.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for fetching historical data. "
            "Install with: pip install yfinance"
        )

    # Ensure .NS suffix for NSE stocks
    if not symbol.endswith((".NS", ".BO")):
        symbol = f"{symbol.removesuffix('.NS')}.NS"

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)

    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    data = []
    for idx, row in df.iterrows():
        data.append({
            "date": idx.strftime("%Y-%m-%d"),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"]),
        })

    return data
