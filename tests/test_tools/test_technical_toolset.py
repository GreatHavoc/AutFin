"""Tests for technical_toolset.py — live yfinance calls, no mocks."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta_classic as pta

import pytest


@pytest.mark.asyncio
async def test_sma_reliance():
    import yfinance as yf
    from autfin.agents.yfinance_utils import normalize_symbol

    ticker = yf.Ticker(normalize_symbol("RELIANCE"))
    df = ticker.history(period="6mo")
    assert not df.empty

    result = pta.sma(df["Close"], length=20)
    assert result is not None
    assert len(result) == len(df)
    assert result.iloc[-1] > 0


@pytest.mark.asyncio
async def test_rsi_reliance():
    import yfinance as yf
    from autfin.agents.yfinance_utils import normalize_symbol

    ticker = yf.Ticker(normalize_symbol("RELIANCE"))
    df = ticker.history(period="6mo")

    result = pta.rsi(df["Close"], length=14)
    valid = result.dropna()
    assert all(0 <= v <= 100 for v in valid)


@pytest.mark.asyncio
async def test_macd_reliance():
    import yfinance as yf
    from autfin.agents.yfinance_utils import normalize_symbol

    ticker = yf.Ticker(normalize_symbol("RELIANCE"))
    df = ticker.history(period="1y")

    result = pta.macd(df["Close"], fast=12, slow=26, signal=9)
    assert result is not None
    assert not result.empty


@pytest.mark.asyncio
async def test_bollinger_reliance():
    import yfinance as yf
    from autfin.agents.yfinance_utils import normalize_symbol

    ticker = yf.Ticker(normalize_symbol("RELIANCE"))
    df = ticker.history(period="6mo")

    result = pta.bbands(df["Close"], length=20, std=2)
    assert result is not None
    upper = [c for c in result.columns if c.startswith("BBU_")][0]
    lower = [c for c in result.columns if c.startswith("BBL_")][0]
    valid = result.dropna()
    assert (valid[upper] >= valid[lower]).all()


@pytest.mark.asyncio
async def test_atr_reliance():
    import yfinance as yf
    from autfin.agents.yfinance_utils import normalize_symbol

    ticker = yf.Ticker(normalize_symbol("RELIANCE"))
    df = ticker.history(period="6mo")

    result = pta.atr(df["High"], df["Low"], df["Close"], length=14)
    valid = result.dropna()
    assert (valid > 0).all()


def test_toolset_creation():
    from autfin.tools.technical_toolset import create_technical_toolset
    assert create_technical_toolset() is not None
