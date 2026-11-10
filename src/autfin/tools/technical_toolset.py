"""AutFin v7.0 — Technical Toolset

Wraps technical indicators as a reusable FunctionToolset for Pydantic AI agents.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pandas_ta_classic as pta
import yfinance as yf
from pydantic_ai import FunctionToolset, RunContext

from autfin.types import AgentDeps


def create_technical_toolset() -> FunctionToolset[AgentDeps]:
    """Create a FunctionToolset with all technical analysis tools."""
    toolset = FunctionToolset[AgentDeps]()
    
    async def _fetch_ohlcv(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Fetch OHLCV data from yfinance."""
        from autfin.agents.yfinance_utils import normalize_symbol
        ticker = yf.Ticker(normalize_symbol(symbol.upper()))
        df = ticker.history(period=period, interval=interval)
        return df
    
    def _to_list(series: pd.Series | None) -> list[float]:
        """Convert pandas Series to list, handling NaN."""
        if series is None:
            return []
        return series.fillna(float('nan')).tolist()
    
    def _latest(arr: list[float]) -> float | None:
        """Return the last non-NaN value from a list."""
        for v in reversed(arr):
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                return float(v)
        return None
    
    @toolset.tool
    async def get_technical_data(ctx: RunContext[AgentDeps], symbol: str) -> dict[str, Any]:
        """Get comprehensive technical analysis for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
        """
        try:
            df = await _fetch_ohlcv(symbol)
            
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
            current_sma_20 = _latest(sma_20)
            current_sma_50 = _latest(sma_50)
            current_sma_200 = _latest(sma_200)
            current_rsi = _latest(rsi)
            current_macd = _latest(macd)
            current_macd_signal = _latest(macd_signal)
            current_atr = _latest(atr)
            
            # Generate signal
            signal = "hold"
            strength = 50
            conviction = 0.5
            
            if current_rsi and current_rsi < 30 and current_sma_20 and current_close > current_sma_20:
                signal = "buy"
                strength = 70
                conviction = 0.7
            elif current_rsi and current_rsi > 70 and current_sma_20 and current_close < current_sma_20:
                signal = "sell"
                strength = 70
                conviction = 0.7
            elif current_macd and current_macd_signal and current_sma_50 and current_macd > current_macd_signal and current_close > current_sma_50:
                signal = "buy"
                strength = 65
                conviction = 0.65
            elif current_macd and current_macd_signal and current_sma_50 and current_macd < current_macd_signal and current_close < current_sma_50:
                signal = "sell"
                strength = 65
                conviction = 0.65
            
            # Candlestick patterns (simplified)
            patterns = []
            if len(close) >= 3:
                # Hammer
                open_ = df["Open"].iloc[-2]
                body = abs(close[-2] - open_)
                lower_shadow = min(close[-2], open_) - low[-2]
                upper_shadow = high[-2] - max(close[-2], open_)
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
                "macd_histogram": (current_macd - current_macd_signal) if current_macd and current_macd_signal else 0,
                "bollinger_upper": _latest(bb_upper),
                "bollinger_middle": _latest(bb_middle),
                "bollinger_lower": _latest(bb_lower),
                "atr": current_atr,
                "volume_trend": volume_trend,
                "candlestick_patterns": patterns,
                "signal": signal,
                "strength": strength,
                "conviction": conviction,
                "summary": f"RSI: {current_rsi:.1f}, MACD: {current_macd:.2f}, Trend: {'Bullish' if current_sma_200 and current_close > current_sma_200 else 'Bearish'}",
            }
        except Exception as e:
            return {"symbol": symbol, "error": str(e)}

    @toolset.tool
    async def get_atr(ctx: RunContext[AgentDeps], symbol: str, period: int = 14) -> float:
        """Get ATR for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            period: ATR period (default 14)
        """
        try:
            df = await _fetch_ohlcv(symbol)
            high = df["High"].tolist()
            low = df["Low"].tolist()
            close = df["Close"].tolist()
            atr = _to_list(pta.atr(pd.Series(high), pd.Series(low), pd.Series(close), length=period))
            return _latest(atr) or 0
        except Exception:
            return 0

    @toolset.tool
    async def calculate_sma(ctx: RunContext[AgentDeps], data: list[float], period: int) -> list[float]:
        """Calculate Simple Moving Average.
        
        Args:
            data: List of price data
            period: SMA period
        """
        return _to_list(pta.sma(pd.Series(data), length=period))

    @toolset.tool
    async def calculate_ema(ctx: RunContext[AgentDeps], data: list[float], period: int) -> list[float]:
        """Calculate Exponential Moving Average.
        
        Args:
            data: List of price data
            period: EMA period
        """
        return _to_list(pta.ema(pd.Series(data), length=period))

    @toolset.tool
    async def calculate_rsi(ctx: RunContext[AgentDeps], data: list[float], period: int = 14) -> list[float]:
        """Calculate Relative Strength Index.
        
        Args:
            data: List of price data
            period: RSI period (default 14)
        """
        return _to_list(pta.rsi(pd.Series(data), length=period))

    @toolset.tool
    async def calculate_macd(
        ctx: RunContext[AgentDeps],
        data: list[float],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> dict[str, list[float]]:
        """Calculate MACD indicator.
        
        Args:
            data: List of price data
            fast: Fast period (default 12)
            slow: Slow period (default 26)
            signal: Signal period (default 9)
        """
        s = pd.Series(data)
        df = pta.macd(s, fast=fast, slow=slow, signal=signal)
        nans = [float(np.nan)] * len(data)
        if df is None or df.empty:
            return {"macd": nans, "signal": nans, "histogram": nans}
        macd_col = [c for c in df.columns if c.startswith("MACD_")][0]
        sig_col = [c for c in df.columns if c.startswith("MACDs_")][0]
        hist_col = [c for c in df.columns if c.startswith("MACDh_")][0]
        return {
            "macd": df[macd_col].tolist(),
            "signal": df[sig_col].tolist(),
            "histogram": df[hist_col].tolist(),
        }

    @toolset.tool
    async def calculate_bollinger_bands(
        ctx: RunContext[AgentDeps],
        data: list[float],
        period: int = 20,
        stddev: float = 2.0,
    ) -> dict[str, list[float]]:
        """Calculate Bollinger Bands.
        
        Args:
            data: List of price data
            period: Period (default 20)
            stddev: Standard deviation multiplier (default 2.0)
        """
        nans = [float(np.nan)] * len(data)
        bb_df = pta.bbands(pd.Series(data), length=period, std=stddev)
        if bb_df is None or bb_df.empty:
            return {"upper": nans, "middle": nans, "lower": nans}
        upper_col = [c for c in bb_df.columns if c.startswith("BBU_")][0]
        mid_col = [c for c in bb_df.columns if c.startswith("BBM_")][0]
        lower_col = [c for c in bb_df.columns if c.startswith("BBL_")][0]
        return {
            "upper": bb_df[upper_col].tolist(),
            "middle": bb_df[mid_col].tolist(),
            "lower": bb_df[lower_col].tolist(),
        }

    @toolset.tool
    async def calculate_atr(
        ctx: RunContext[AgentDeps],
        high: list[float],
        low: list[float],
        close: list[float],
        period: int = 14,
    ) -> list[float]:
        """Calculate Average True Range.
        
        Args:
            high: List of high prices
            low: List of low prices
            close: List of close prices
            period: ATR period (default 14)
        """
        return _to_list(pta.atr(pd.Series(high), pd.Series(low), pd.Series(close), length=period))

    return toolset


# Create the default toolset instance
technical_toolset = create_technical_toolset()


def get_technical_toolset() -> FunctionToolset[AgentDeps]:
    """Get the shared technical toolset instance."""
    return technical_toolset