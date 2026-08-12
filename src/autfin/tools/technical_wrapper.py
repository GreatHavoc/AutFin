"""AutFin v7.0 — Technical Tools Wrapper

Wraps existing technical indicators as async functions for Pydantic AI agents.
"""
from __future__ import annotations

import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta_classic as pta

from typing import Any

from autfin.agents.yfinance_utils import normalize_symbol


async def fetch_ohlcv(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV data from yfinance."""
    ticker = yf.Ticker(normalize_symbol(symbol.upper()))
    df = ticker.history(period=period, interval=interval)
    return df


async def get_technical_data(symbol: str, as_of: str | None = None) -> dict[str, Any]:
    """Get comprehensive technical analysis for a symbol."""
    try:
        df = await fetch_ohlcv(symbol)
        
        if df.empty:
            return {"symbol": symbol, "error": "No data available"}
        
        close = df["Close"].tolist()
        high = df["High"].tolist()
        low = df["Low"].tolist()
        volume = df["Volume"].tolist()
        
        # Calculate indicators
        sma_20 = _to_list(pta.sma(pd.Series(close), length=20))
        sma_50 = _to_list(pta.sma(pd.Series(close), length=50))
        sma_200 = _to_list(pta.sma(pd.Series(close), length=200))
        rsi = _to_list(pta.rsi(pd.Series(close), length=14))
        
        macd_df = pta.macd(pd.Series(close), fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            macd_col = [c for c in macd_df.columns if c.startswith("MACD_")][0]
            sig_col = [c for c in macd_df.columns if c.startswith("MACDs_")][0]
            hist_col = [c for c in macd_df.columns if c.startswith("MACDh_")][0]
            macd = macd_df[macd_col].tolist()
            macd_signal = macd_df[sig_col].tolist()
            macd_histogram = macd_df[hist_col].tolist()
        else:
            macd = macd_signal = macd_histogram = [float('nan')] * len(close)
        
        bb_df = pta.bbands(pd.Series(close), length=20, std=2)
        if bb_df is not None and not bb_df.empty:
            upper_col = [c for c in bb_df.columns if c.startswith("BBU_")][0]
            mid_col = [c for c in bb_df.columns if c.startswith("BBM_")][0]
            lower_col = [c for c in bb_df.columns if c.startswith("BBL_")][0]
            bb_upper = bb_df[upper_col].tolist()
            bb_middle = bb_df[mid_col].tolist()
            bb_lower = bb_df[lower_col].tolist()
        else:
            bb_upper = bb_middle = bb_lower = [float('nan')] * len(close)
        
        atr = _to_list(pta.atr(pd.Series(high), pd.Series(low), pd.Series(close), length=14))
        
        # Volume trend
        vol_sma = _to_list(pta.sma(pd.Series(volume), length=20))
        last_vol = volume[-1] if volume else 0
        last_vol_sma = vol_sma[-1] if vol_sma and not np.isnan(vol_sma[-1]) else 1
        if last_vol > last_vol_sma * 1.5:
            volume_trend = "increasing"
        elif last_vol < last_vol_sma * 0.5:
            volume_trend = "decreasing"
        else:
            volume_trend = "stable"
        
        # Current values
        current_close = close[-1] if close else 0
        current_sma_20 = sma_20[-1] if sma_20 and not np.isnan(sma_20[-1]) else 0
        current_sma_50 = sma_50[-1] if sma_50 and not np.isnan(sma_50[-1]) else 0
        current_sma_200 = sma_200[-1] if sma_200 and not np.isnan(sma_200[-1]) else 0
        current_rsi = rsi[-1] if rsi and not np.isnan(rsi[-1]) else 50
        current_macd = macd[-1] if macd and not np.isnan(macd[-1]) else 0
        current_macd_signal = macd_signal[-1] if macd_signal and not np.isnan(macd_signal[-1]) else 0
        current_atr = atr[-1] if atr and not np.isnan(atr[-1]) else 0
        
        # Generate signal
        signal = "hold"
        strength = 50
        conviction = 0.5
        
        if current_rsi < 30 and current_close > current_sma_20:
            signal = "buy"
            strength = 70
            conviction = 0.7
        elif current_rsi > 70 and current_close < current_sma_20:
            signal = "sell"
            strength = 70
            conviction = 0.7
        elif current_macd > current_macd_signal and current_close > current_sma_50:
            signal = "buy"
            strength = 65
            conviction = 0.65
        elif current_macd < current_macd_signal and current_close < current_sma_50:
            signal = "sell"
            strength = 65
            conviction = 0.65
        
        # Candlestick patterns (simplified)
        patterns = []
        if len(close) >= 3:
            # Hammer
            if (high[-2] - low[-2]) > 0:
                body = abs(close[-2] - df["Open"].iloc[-2])
                lower_shadow = min(close[-2], df["Open"].iloc[-2]) - low[-2]
                upper_shadow = high[-2] - max(close[-2], df["Open"].iloc[-2])
                if lower_shadow > 2 * body and upper_shadow < body:
                    patterns.append("hammer")
        
        return {
            "symbol": symbol,
            "sma_20": current_sma_20,
            "sma_50": current_sma_50,
            "sma_200": current_sma_200,
            "rsi": current_rsi,
            "macd": current_macd,
            "macd_signal": current_macd_signal,
            "macd_histogram": current_macd - current_macd_signal,
            "bollinger_upper": bb_upper[-1] if bb_upper and not np.isnan(bb_upper[-1]) else 0,
            "bollinger_middle": bb_middle[-1] if bb_middle and not np.isnan(bb_middle[-1]) else 0,
            "bollinger_lower": bb_lower[-1] if bb_lower and not np.isnan(bb_lower[-1]) else 0,
            "atr": current_atr,
            "volume_trend": volume_trend,
            "candlestick_patterns": patterns,
            "signal": signal,
            "strength": strength,
            "conviction": conviction,
            "summary": f"RSI: {current_rsi:.1f}, MACD: {current_macd:.2f}, Trend: {'Bullish' if current_close > current_sma_200 else 'Bearish'}",
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}


async def get_atr(symbol: str, period: int = 14) -> float:
    """Get ATR for a symbol."""
    try:
        df = await fetch_ohlcv(symbol)
        high = df["High"].tolist()
        low = df["Low"].tolist()
        close = df["Close"].tolist()
        atr = _to_list(pta.atr(pd.Series(high), pd.Series(low), pd.Series(close), length=period))
        return atr[-1] if atr and not np.isnan(atr[-1]) else 0
    except Exception:
        return 0


def _to_list(series: pd.Series | None) -> list[float]:
    """Convert pandas Series to list, handling NaN."""
    if series is None:
        return []
    return series.fillna(float('nan')).tolist()