"""Shared test fixtures for AutFin tool tests."""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


@pytest.fixture
def mock_deps():
    """Minimal AgentDeps for tool calls."""
    deps = MagicMock()
    deps.capital = 1_000_000.0
    deps.max_drawdown_pct = 0.08
    deps.kite_client = None
    return deps


@pytest.fixture
def mock_run_context(mock_deps):
    """RunContext-like object for tool functions."""
    ctx = MagicMock()
    ctx.deps = mock_deps
    return ctx


@pytest.fixture
def fake_yf_info():
    """Fake yfinance info dict for RELIANCE."""
    return {
        "symbol": "RELIANCE.NS",
        "shortName": "Reliance Industries Ltd",
        "sector": "Energy",
        "marketCap": 17_000_000_000_000,
        "trailingPE": 25.0,
        "priceToBook": 2.5,
        "returnOnEquity": 0.095,
        "returnOnAssets": 0.05,
        "debtToEquity": 35.0,
        "currentRatio": 1.2,
        "freeCashflow": 500_000_000_000,
        "earningsYield": 0.04,
        "dividendYield": 0.003,
        "revenueGrowth": 0.12,
        "earningsGrowth": 0.08,
        "operatingMargins": 0.15,
        "profitMargins": 0.10,
        "regularMarketPrice": 2500.0,
        "previousClose": 2480.0,
        "fiftyDayAverage": 2450.0,
        "twoHundredDayAverage": 2350.0,
        "beta": 1.1,
    }


@pytest.fixture
def fake_yf_history():
    """Fake yfinance OHLCV DataFrame-like data."""
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    n = 250
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="B")
    base = 2500
    returns = np.random.normal(0.0005, 0.015, n)
    close = base * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    opn = close * (1 + np.random.normal(0, 0.003, n))
    volume = np.random.randint(1_000_000, 10_000_000, n)

    df = pd.DataFrame({
        "Open": opn,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)
    return df
