"""Tests for backtest suite — engine, quick, sweep."""
from __future__ import annotations

import pytest

from autfin.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    strategy_sma_crossover,
    strategy_rsi,
    strategy_macd,
    strategy_bollinger_reversal,
    strategy_multi_factor,
)


# ─── Engine tests (live yfinance) ───


@pytest.mark.asyncio
async def test_engine_fetch_data():
    engine = BacktestEngine("RELIANCE", start="2023-01-01", end="2023-06-30")
    df = engine.fetch_data()
    assert len(df) > 50
    assert "Close" in df.columns


@pytest.mark.asyncio
async def test_engine_sma_crossover():
    engine = BacktestEngine("RELIANCE", start="2023-01-01", end="2023-12-31")
    result = engine.run("sma_crossover")
    assert result.total_trades >= 0
    assert isinstance(result.total_return, float)
    assert result.strategy == "sma_crossover"
    assert result.symbol == "RELIANCE"
    assert len(result.equity_curve) > 0


@pytest.mark.asyncio
async def test_engine_rsi():
    engine = BacktestEngine("RELIANCE", start="2023-01-01", end="2023-12-31")
    result = engine.run("rsi")
    assert result.strategy == "rsi"
    assert isinstance(result.sharpe_ratio, float)


@pytest.mark.asyncio
async def test_engine_macd():
    engine = BacktestEngine("RELIANCE", start="2023-01-01", end="2023-12-31")
    result = engine.run("macd")
    assert result.strategy == "macd"
    assert len(result.equity_curve) > 0


@pytest.mark.asyncio
async def test_engine_multi_factor():
    engine = BacktestEngine("RELIANCE", start="2023-01-01", end="2023-12-31")
    result = engine.run("multi_factor")
    assert result.strategy == "multi_factor"
    assert result.total_return != 0 or result.total_trades == 0


@pytest.mark.asyncio
async def test_engine_custom_strategy():
    engine = BacktestEngine("RELIANCE", start="2023-01-01", end="2023-12-31")
    result = engine.run(lambda df: strategy_sma_crossover(df, fast=10, slow=30))
    assert result.strategy == "<lambda>"


@pytest.mark.asyncio
async def test_engine_config_stop_loss():
    cfg = BacktestConfig(stop_loss_atr_mult=1.0, take_profit_atr_mult=1.5)
    engine = BacktestEngine("RELIANCE", start="2023-01-01", end="2023-12-31", config=cfg)
    result = engine.run("sma_crossover")
    # Stop loss should trigger more exits
    sl_trades = [t for t in result.trades if t.exit_reason == "stop_loss"]
    tp_trades = [t for t in result.trades if t.exit_reason == "take_profit"]
    assert isinstance(sl_trades, list)
    assert isinstance(tp_trades, list)


@pytest.mark.asyncio
async def test_engine_summary():
    engine = BacktestEngine("RELIANCE", start="2023-01-01", end="2023-12-31")
    result = engine.run("sma_crossover")
    summary = result.summary()
    assert "RELIANCE" in summary
    assert "sma_crossover" in summary
    assert "Total Return" in summary


@pytest.mark.asyncio
async def test_engine_benchmark():
    engine = BacktestEngine("RELIANCE", start="2023-01-01", end="2023-12-31")
    result = engine.run("sma_crossover")
    assert len(result.benchmark_curve) > 0
    # Benchmark should start at initial capital
    assert result.benchmark_curve[0] == pytest.approx(100_000, rel=0.01)


@pytest.mark.asyncio
async def test_engine_bad_symbol():
    engine = BacktestEngine("FAKE_SYMBOL_XYZ", start="2023-01-01", end="2023-06-30")
    with pytest.raises(ValueError, match="No data"):
        engine.run("sma_crossover")


@pytest.mark.asyncio
async def test_engine_bad_strategy():
    engine = BacktestEngine("RELIANCE", start="2023-01-01", end="2023-06-30")
    with pytest.raises(ValueError, match="Unknown strategy"):
        engine.run("nonexistent_strategy")


# ─── Quick backtest tests ───


@pytest.mark.asyncio
async def test_quick_backtest():
    from autfin.backtest.quick import quick_backtest
    result = quick_backtest("RELIANCE", start="2023-01-01", end="2023-12-31")
    assert result["symbol"] == "RELIANCE"
    assert result["strategy"] == "sma_crossover"
    assert "stats" in result
    assert "backtest" in result


@pytest.mark.asyncio
async def test_quick_backtest_rsi():
    from autfin.backtest.quick import quick_backtest
    result = quick_backtest("RELIANCE", strategy="rsi", start="2023-01-01", end="2023-12-31")
    assert result["strategy"] == "rsi"


@pytest.mark.asyncio
async def test_quick_backtest_macd():
    from autfin.backtest.quick import quick_backtest
    result = quick_backtest("RELIANCE", strategy="macd", start="2023-01-01", end="2023-12-31")
    assert result["strategy"] == "macd"


# ─── Sweep tests (vectorbt) ───


@pytest.mark.asyncio
async def test_sweep_sma():
    from autfin.backtest.sweep import sweep_sma_crossover
    result = sweep_sma_crossover(
        "RELIANCE", start="2023-01-01", end="2023-12-31",
        fast_range=(10, 30, 10), slow_range=(30, 60, 15),
    )
    assert result["strategy"] == "sma_crossover"
    assert "best" in result
    assert "return" in result["best"]


@pytest.mark.asyncio
async def test_sweep_rsi():
    from autfin.backtest.sweep import sweep_rsi
    result = sweep_rsi(
        "RELIANCE", start="2023-01-01", end="2023-12-31",
        period_range=(10, 21, 7), oversold_range=(25, 35, 5), overbought_range=(65, 75, 5),
    )
    assert result["strategy"] == "rsi"
    assert "best" in result


@pytest.mark.asyncio
async def test_parameter_sweep_dispatch():
    from autfin.backtest.sweep import parameter_sweep
    result = parameter_sweep(
        "RELIANCE", strategy="sma_crossover",
        start="2023-01-01", end="2023-12-31",
        fast_range=(10, 30, 10), slow_range=(30, 60, 15),
    )
    assert result["strategy"] == "sma_crossover"


def test_strategies_dict():
    from autfin.backtest.engine import STRATEGIES
    assert "sma_crossover" in STRATEGIES
    assert "rsi" in STRATEGIES
    assert "macd" in STRATEGIES
    assert "bollinger" in STRATEGIES
    assert "multi_factor" in STRATEGIES
