"""
Centralized yfinance helpers — point-in-time data fetching.

Phase 3.2: All historical data requests accept an optional `as_of` datetime
to prevent look-ahead bias during backtests. The `normalize_symbol` helper
eliminates the .NS suffix duplication across 16+ call sites.

Usage:
    from autfin.agents.yfinance_utils import fetch_history, normalize_symbol

    df = await fetch_history("RELIANCE", as_of="2024-06-01", period="1y")
    sym = normalize_symbol("RELIANCE")  # -> "RELIANCE.NS"
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def normalize_symbol(symbol: str) -> str:
    """Normalize a stock symbol to NSE format (.NS suffix).

    Examples:
        normalize_symbol("RELIANCE")     -> "RELIANCE.NS"
        normalize_symbol("RELIANCE.NS")  -> "RELIANCE.NS"
        normalize_symbol("RELIANCE.BO")  -> "RELIANCE.BO"  (BSE kept as-is)
    """
    if symbol.endswith((".NS", ".BO")):
        return symbol
    return f"{symbol.removesuffix('.NS')}.NS"


async def fetch_history(
    symbol: str,
    as_of: Optional[str | datetime] = None,
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch OHLCV data with point-in-time filtering.

    Args:
        symbol: Stock symbol (e.g. "RELIANCE" or "RELIANCE.NS")
        as_of: End date for data fetch. ISO string or datetime.
               None = live (up to now). Used for backtests.
        period: yfinance period string ("3mo", "6mo", "1y", "2y")
        interval: yfinance interval ("1d", "1h", etc.)

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume].
        Empty DataFrame if no data available.
    """
    import yfinance as yf

    ticker = yf.Ticker(normalize_symbol(symbol))

    kwargs: dict = {"period": period, "interval": interval}
    if as_of is not None:
        if isinstance(as_of, str):
            as_of = datetime.fromisoformat(as_of)
        kwargs["end"] = as_of

    df = await asyncio.to_thread(ticker.history, **kwargs)
    if df is None:
        return pd.DataFrame()
    return df


async def fetch_snapshot() -> dict:
    """Fetch live market snapshot for emergency trigger detection.

    Returns:
        {
            "nifty_change_pct": float,  # Nifty 50 intraday % change
            "india_vix": float,         # India VIX level
        }
    """
    import yfinance as yf

    result = {"nifty_change_pct": 0.0, "india_vix": 0.0}

    try:
        nifty = yf.Ticker("^NSEI")
        info = await asyncio.to_thread(lambda: nifty.info)
        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
        current = info.get("regularMarketPrice") or info.get("currentPrice")
        if prev_close and current and prev_close > 0:
            result["nifty_change_pct"] = round(((current - prev_close) / prev_close) * 100, 2)
    except Exception as exc:
        logger.warning("Failed to fetch Nifty data: %s", exc)

    try:
        vix = yf.Ticker("^INDIAVIX")
        vix_info = await asyncio.to_thread(lambda: vix.info)
        result["india_vix"] = vix_info.get("regularMarketPrice", 0.0) or 0.0
    except Exception as exc:
        logger.warning("Failed to fetch India VIX: %s", exc)

    return result
