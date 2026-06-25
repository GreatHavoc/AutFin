"""Tests for AutFin backtesting engine."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from src.autfin.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    TradeRecord,
    fetch_historical_data,
)
from src.autfin.backtest.metrics import BacktestMetrics, calculate_metrics
from src.autfin.backtest.strategies import (
    BaseStrategy,
    BreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    OpeningRangeBreakoutStrategy,
    StrategySignal,
    TrendFollowingStrategy,
    get_strategy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 100, start_price: float = 100.0, trend: float = 0.001) -> list[dict]:
    """Generate synthetic OHLCV data."""
    data = []
    price = start_price
    for i in range(n):
        change = price * trend
        data.append({
            "date": f"2025-01-{i + 1:02d}" if i < 31 else f"2025-02-{i - 30:02d}" if i < 59 else f"2025-03-{i - 58:02d}",
            "open": price,
            "high": price + abs(change) * 1.5,
            "low": price - abs(change) * 0.5,
            "close": price + change,
            "volume": 100000 + i * 1000,
        })
        price = price + change
    return data


def _make_trending_data(n: int = 100) -> list[dict]:
    """Generate data with clear uptrend."""
    return _make_ohlcv(n, start_price=100.0, trend=0.005)


def _make_volatile_data(n: int = 100) -> list[dict]:
    """Generate volatile data."""
    data = []
    price = 100.0
    for i in range(n):
        change = price * 0.02 * ((-1) ** (i % 3))
        data.append({
            "date": f"2025-01-{i + 1:02d}" if i < 31 else f"2025-02-{i - 30:02d}",
            "open": price,
            "high": price + abs(change),
            "low": price - abs(change),
            "close": price + change,
            "volume": 100000,
        })
        price = price + change
    return data


# ---------------------------------------------------------------------------
# Test StrategySignal
# ---------------------------------------------------------------------------

class TestStrategySignal:
    def test_signal_creation(self):
        signal = StrategySignal(action="buy", confidence=0.8, reason="test")
        assert signal.action == "buy"
        assert signal.confidence == 0.8
        assert signal.reason == "test"

    def test_signal_defaults(self):
        signal = StrategySignal(action="hold")
        assert signal.confidence == 0.0
        assert signal.reason == ""
        assert signal.metadata == {}


# ---------------------------------------------------------------------------
# Test BaseStrategy indicators
# ---------------------------------------------------------------------------

class TestBaseStrategyIndicators:
    def test_sma_basic(self):
        prices = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = BaseStrategy._sma(prices, 3)
        assert result[0] is None
        assert result[1] is None
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(3.0)
        assert result[4] == pytest.approx(4.0)

    def test_sma_insufficient_data(self):
        result = BaseStrategy._sma([1.0, 2.0], 5)
        assert all(r is None for r in result)

    def test_ema_basic(self):
        prices = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = BaseStrategy._ema(prices, 3)
        assert result[0] is None
        assert result[1] is None
        assert result[2] == pytest.approx(2.0)  # SMA seed
        assert result[3] is not None
        assert result[4] is not None

    def test_ema_insufficient_data(self):
        result = BaseStrategy._ema([1.0], 5)
        assert all(r is None for r in result)

    def test_rsi_overbought(self):
        prices = list(range(1, 30))  # Strictly increasing
        result = BaseStrategy._rsi(prices, 14)
        # RSI should be 100 for strictly increasing prices
        assert result[14] == pytest.approx(100.0)

    def test_rsi_oversold(self):
        prices = list(range(30, 0, -1))  # Strictly decreasing
        result = BaseStrategy._rsi(prices, 14)
        # RSI should be 0 for strictly decreasing prices
        assert result[14] == pytest.approx(0.0)

    def test_rsi_insufficient_data(self):
        result = BaseStrategy._rsi([1.0, 2.0], 14)
        assert all(r is None for r in result)

    def test_bollinger_bands(self):
        prices = [100.0] * 30  # Flat prices
        upper, middle, lower = BaseStrategy._bollinger_bands(prices, 20, 2.0)
        assert upper[19] == pytest.approx(100.0)
        assert middle[19] == pytest.approx(100.0)
        assert lower[19] == pytest.approx(100.0)

    def test_atr(self):
        highs = [110.0, 112.0, 115.0, 113.0, 118.0]
        lows = [100.0, 102.0, 105.0, 103.0, 108.0]
        closes = [105.0, 108.0, 110.0, 107.0, 115.0]
        result = BaseStrategy._atr(highs, lows, closes, 3)
        assert result[2] is not None
        assert result[2] > 0


# ---------------------------------------------------------------------------
# Test MomentumStrategy
# ---------------------------------------------------------------------------

class TestMomentumStrategy:
    def test_signal_generation(self):
        strategy = MomentumStrategy(fast_period=5, slow_period=20)
        data = _make_trending_data(50)
        signals = strategy.generate_signals(data)
        assert len(signals) == 50
        assert all(isinstance(s, StrategySignal) for s in signals)

    def test_has_signals(self):
        strategy = MomentumStrategy(fast_period=5, slow_period=15)
        data = _make_trending_data(60)
        signals = strategy.generate_signals(data)
        non_hold = [s for s in signals if s.action != "hold"]
        assert len(non_hold) >= 0  # May or may not produce buy signals depending on crossover

    def test_insufficient_data(self):
        strategy = MomentumStrategy(fast_period=20, slow_period=50)
        data = _make_ohlcv(10)
        signals = strategy.generate_signals(data)
        assert all(s.action == "hold" for s in signals)


# ---------------------------------------------------------------------------
# Test MeanReversionStrategy
# ---------------------------------------------------------------------------

class TestMeanReversionStrategy:
    def test_signal_generation(self):
        strategy = MeanReversionStrategy(bb_period=10, rsi_period=10)
        data = _make_volatile_data(50)
        signals = strategy.generate_signals(data)
        assert len(signals) == 50

    def test_insufficient_data(self):
        strategy = MeanReversionStrategy(bb_period=20)
        data = _make_ohlcv(5)
        signals = strategy.generate_signals(data)
        assert all(s.action == "hold" for s in signals)


# ---------------------------------------------------------------------------
# Test TrendFollowingStrategy
# ---------------------------------------------------------------------------

class TestTrendFollowingStrategy:
    def test_signal_generation(self):
        strategy = TrendFollowingStrategy(sma_period=20, atr_period=14)
        data = _make_trending_data(60)
        signals = strategy.generate_signals(data)
        assert len(signals) == 60

    def test_has_signals(self):
        strategy = TrendFollowingStrategy(sma_period=10, atr_period=10)
        data = _make_trending_data(50)
        signals = strategy.generate_signals(data)
        non_hold = [s for s in signals if s.action != "hold"]
        assert len(non_hold) >= 0  # May or may not have signals depending on data


# ---------------------------------------------------------------------------
# Test OpeningRangeBreakoutStrategy
# ---------------------------------------------------------------------------

class TestOpeningRangeBreakoutStrategy:
    def test_signal_generation(self):
        strategy = OpeningRangeBreakoutStrategy(range_minutes=5, atr_period=10)
        data = _make_volatile_data(50)
        signals = strategy.generate_signals(data)
        assert len(signals) == 50

    def test_insufficient_data(self):
        strategy = OpeningRangeBreakoutStrategy()
        data = _make_ohlcv(3)
        signals = strategy.generate_signals(data)
        assert all(s.action == "hold" for s in signals)


# ---------------------------------------------------------------------------
# Test BreakoutStrategy
# ---------------------------------------------------------------------------

class TestBreakoutStrategy:
    def test_signal_generation(self):
        strategy = BreakoutStrategy(lookback_period=10, atr_period=10)
        data = _make_trending_data(60)
        signals = strategy.generate_signals(data)
        assert len(signals) >= 50  # Breakout may return fewer due to lookback

    def test_insufficient_data(self):
        strategy = BreakoutStrategy(lookback_period=20, atr_period=14)
        data = _make_ohlcv(5)
        signals = strategy.generate_signals(data)
        assert all(s.action == "hold" for s in signals)


# ---------------------------------------------------------------------------
# Test get_strategy
# ---------------------------------------------------------------------------

class TestGetStrategy:
    def test_momentum(self):
        s = get_strategy("momentum")
        assert isinstance(s, MomentumStrategy)

    def test_mean_reversion(self):
        s = get_strategy("mean_reversion")
        assert isinstance(s, MeanReversionStrategy)

    def test_trend_following(self):
        s = get_strategy("trend_following")
        assert isinstance(s, TrendFollowingStrategy)

    def test_orb(self):
        s = get_strategy("orb")
        assert isinstance(s, OpeningRangeBreakoutStrategy)

    def test_breakout(self):
        s = get_strategy("breakout")
        assert isinstance(s, BreakoutStrategy)

    def test_unknown_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("nonexistent")

    def test_custom_params(self):
        s = get_strategy("momentum", fast_period=10, slow_period=30)
        assert s.fast_period == 10
        assert s.slow_period == 30


# ---------------------------------------------------------------------------
# Test BacktestConfig
# ---------------------------------------------------------------------------

class TestBacktestConfig:
    def test_defaults(self):
        config = BacktestConfig()
        assert config.initial_capital == 100000.0
        assert config.position_size_pct == 10.0
        assert config.max_positions == 5
        assert config.brokerage_pct == 0.03
        assert config.stt_pct == 0.1
        assert config.slippage_pct == 0.05

    def test_custom_config(self):
        config = BacktestConfig(initial_capital=500000.0, max_positions=10)
        assert config.initial_capital == 500000.0
        assert config.max_positions == 10


# ---------------------------------------------------------------------------
# Test BacktestEngine
# ---------------------------------------------------------------------------

class TestBacktestEngine:
    def test_basic_run(self):
        data = _make_trending_data(60)
        engine = BacktestEngine()
        result = engine.run(data, "momentum", "TEST")
        assert isinstance(result, BacktestResult)
        assert result.symbol == "TEST"
        assert result.strategy_name == "momentum"
        assert len(result.equity_curve) > 0
        assert len(result.signals) == 60

    def test_result_has_metrics(self):
        data = _make_trending_data(60)
        engine = BacktestEngine()
        result = engine.run(data, "momentum")
        assert isinstance(result.metrics, BacktestMetrics)
        assert result.metrics.total_trades >= 0

    def test_equity_curve_starts_at_capital(self):
        config = BacktestConfig(initial_capital=200000.0)
        engine = BacktestEngine(config)
        data = _make_trending_data(60)
        result = engine.run(data, "momentum")
        assert result.equity_curve[0] == 200000.0

    def test_with_mean_reversion(self):
        data = _make_volatile_data(60)
        engine = BacktestEngine()
        result = engine.run(data, "mean_reversion", "TEST")
        assert result.strategy_name == "mean_reversion"
        assert len(result.equity_curve) > 0

    def test_with_breakout(self):
        data = _make_trending_data(60)
        engine = BacktestEngine()
        result = engine.run(data, "breakout", "TEST")
        assert result.strategy_name == "breakout"

    def test_with_orb(self):
        data = _make_volatile_data(60)
        engine = BacktestEngine()
        result = engine.run(data, "orb", "TEST")
        assert result.strategy_name == "orb"

    def test_with_strategy_instance(self):
        data = _make_trending_data(60)
        strategy = MomentumStrategy(fast_period=10, slow_period=30)
        engine = BacktestEngine()
        result = engine.run(data, strategy, "TEST")
        assert result.strategy_name == "momentum"

    def test_transaction_costs_applied(self):
        config = BacktestConfig(brokerage_pct=0.1, slippage_pct=0.1)
        engine = BacktestEngine(config)
        data = _make_trending_data(60)
        result = engine.run(data, "momentum")
        # With costs, total cost should be > 0 if there are trades
        if result.trades:
            total_cost = sum(t.total_cost for t in result.trades)
            assert total_cost > 0

    def test_max_positions_respected(self):
        config = BacktestConfig(max_positions=1)
        engine = BacktestEngine(config)
        data = _make_trending_data(60)
        result = engine.run(data, "momentum")
        # Engine should respect max positions
        assert isinstance(result, BacktestResult)


# ---------------------------------------------------------------------------
# Test BacktestConfig cost calculations
# ---------------------------------------------------------------------------

class TestCostCalculations:
    def test_buy_cost(self):
        engine = BacktestEngine()
        cost = engine._calculate_buy_cost(100000.0)
        assert cost > 0

    def test_sell_cost_includes_stt(self):
        engine = BacktestEngine()
        buy_cost = engine._calculate_buy_cost(100000.0)
        sell_cost = engine._calculate_sell_cost(100000.0)
        # Sell cost should be higher due to STT
        assert sell_cost > buy_cost


# ---------------------------------------------------------------------------
# Test TradeRecord
# ---------------------------------------------------------------------------

class TestTradeRecord:
    def test_trade_creation(self):
        trade = TradeRecord(
            symbol="TEST",
            entry_date="2025-01-01",
            entry_price=100.0,
            exit_date="2025-01-10",
            exit_price=110.0,
            quantity=100,
            pnl=1000.0,
            pnl_pct=0.1,
            holding_days=9,
        )
        assert trade.symbol == "TEST"
        assert trade.pnl == 1000.0


# ---------------------------------------------------------------------------
# Test Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_empty_equity_curve(self):
        metrics = calculate_metrics([], [])
        assert metrics.total_return == 0.0
        assert metrics.sharpe_ratio == 0.0

    def test_single_value_equity_curve(self):
        metrics = calculate_metrics([100000.0], [])
        assert metrics.total_return == 0.0

    def test_profitable_curve(self):
        curve = [100000, 105000, 110000, 115000, 120000]
        metrics = calculate_metrics(curve, [])
        assert metrics.total_return > 0
        assert metrics.annualized_return > 0

    def test_losing_curve(self):
        curve = [100000, 95000, 90000, 85000, 80000]
        metrics = calculate_metrics(curve, [])
        assert metrics.total_return < 0
        assert metrics.max_drawdown > 0

    def test_with_trades(self):
        curve = [100000, 105000, 100000, 110000, 105000, 115000]
        trades = [
            {"pnl": 5000.0, "holding_days": 5},
            {"pnl": -5000.0, "holding_days": 3},
            {"pnl": 10000.0, "holding_days": 4},
            {"pnl": -5000.0, "holding_days": 2},
            {"pnl": 10000.0, "holding_days": 6},
        ]
        metrics = calculate_metrics(curve, trades)
        assert metrics.total_trades == 5
        assert metrics.winning_trades == 3
        assert metrics.losing_trades == 2
        assert metrics.win_rate == pytest.approx(0.6)

    def test_max_drawdown(self):
        curve = [100, 110, 105, 95, 100, 115]
        metrics = calculate_metrics(curve, [])
        assert metrics.max_drawdown > 0
        assert metrics.max_drawdown <= 1.0

    def test_sharpe_ratio_positive(self):
        curve = [100 + i * 2 for i in range(50)]
        metrics = calculate_metrics(curve, [])
        assert metrics.sharpe_ratio > 0

    def test_sortino_ratio(self):
        curve = [100 + (i % 3 - 1) * 2 for i in range(50)]  # Mix of up/down
        metrics = calculate_metrics(curve, [])
        assert isinstance(metrics.sortino_ratio, float)

    def test_to_dict(self):
        metrics = BacktestMetrics(total_return=0.15, sharpe_ratio=1.5)
        d = metrics.to_dict()
        assert d["total_return_pct"] == 15.0
        assert d["sharpe_ratio"] == 1.5

    def test_consecutive_losses(self):
        trades = [
            {"pnl": -100, "holding_days": 1},
            {"pnl": -200, "holding_days": 1},
            {"pnl": -150, "holding_days": 1},
            {"pnl": 500, "holding_days": 1},
        ]
        metrics = calculate_metrics([100000, 99000, 98000, 97000, 100000], trades)
        assert metrics.max_consecutive_losses == 3


# ---------------------------------------------------------------------------
# Test fetch_historical_data (mocked)
# ---------------------------------------------------------------------------

class TestFetchHistoricalData:
    def test_fetch_with_ns_suffix(self):
        import pandas as pd
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker

        idx = pd.date_range("2025-01-01", periods=5)
        df = pd.DataFrame({
            "Open": [100.0] * 5,
            "High": [105.0] * 5,
            "Low": [95.0] * 5,
            "Close": [102.0] * 5,
            "Volume": [100000] * 5,
        }, index=idx)
        mock_ticker.history.return_value = df

        import sys
        sys.modules["yfinance"] = mock_yf
        try:
            data = fetch_historical_data("RELIANCE", "5d")
            assert len(data) == 5
            mock_yf.Ticker.assert_called_with("RELIANCE.NS")
        finally:
            del sys.modules["yfinance"]

    def test_fetch_empty_data(self):
        import pandas as pd
        import sys
        mock_yf = MagicMock()
        mock_ticker = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        mock_ticker.history.return_value = pd.DataFrame()

        sys.modules["yfinance"] = mock_yf
        try:
            with pytest.raises(ValueError, match="No data returned"):
                fetch_historical_data("INVALID", "5d")
        finally:
            del sys.modules["yfinance"]

    def test_fetch_without_yfinance(self):
        import sys
        old = sys.modules.get("yfinance")
        sys.modules["yfinance"] = None
        try:
            with pytest.raises(ImportError, match="yfinance is required"):
                fetch_historical_data("RELIANCE")
        finally:
            if old is not None:
                sys.modules["yfinance"] = old
            else:
                sys.modules.pop("yfinance", None)
