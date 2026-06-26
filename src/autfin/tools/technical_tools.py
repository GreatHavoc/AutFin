"""Technical indicator computation module for AutFin.

Uses pandas-ta-classic (pure Python) for all indicator calculations.
Provides standalone compute functions (list[float] in/out) plus async
wrappers that fetch OHLCV from yfinance and return JSON-serialized results.
"""

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta_classic as pta

logger = logging.getLogger(__name__)


# =============================================================================
# HELPER — list <-> pd.Series conversion
# =============================================================================

def _to_series(data: list[float]) -> pd.Series:
    return pd.Series(data, dtype=np.float64)


def _to_list(series: pd.Series) -> list[float]:
    if series is None:
        return []
    return series.tolist()


def _latest(arr: list[float]) -> float | None:
    """Return the last non-NaN value from a list."""
    for v in reversed(arr):
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return float(v)
    return None


# =============================================================================
# BASIC INDICATORS
# =============================================================================


def calculate_sma(data: list[float], period: int) -> list[float]:
    """Simple Moving Average."""
    return _to_list(pta.sma(_to_series(data), length=period))


def calculate_ema(data: list[float], period: int) -> list[float]:
    """Exponential Moving Average."""
    return _to_list(pta.ema(_to_series(data), length=period))


def calculate_rsi(data: list[float], period: int = 14) -> list[float]:
    """Relative Strength Index."""
    return _to_list(pta.rsi(_to_series(data), length=period))


def calculate_macd(
    data: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, list[float]]:
    """MACD indicator.

    Returns:
        Dict with ``macd``, ``signal``, ``histogram`` lists.
    """
    s = _to_series(data)
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


# =============================================================================
# BANDS & VOLATILITY
# =============================================================================


def calculate_bollinger_bands(
    data: list[float], period: int = 20, stddev: float = 2.0
) -> dict[str, list[float]]:
    """Bollinger Bands.

    Returns:
        Dict with ``upper``, ``middle``, ``lower`` lists.
    """
    nans = [float(np.nan)] * len(data)
    s = _to_series(data)
    df = pta.bbands(s, length=period, std=stddev)
    if df is None or df.empty:
        return {"upper": nans, "middle": nans, "lower": nans}
    upper_col = [c for c in df.columns if c.startswith("BBU_")][0]
    mid_col = [c for c in df.columns if c.startswith("BBM_")][0]
    lower_col = [c for c in df.columns if c.startswith("BBL_")][0]
    return {
        "upper": df[upper_col].tolist(),
        "middle": df[mid_col].tolist(),
        "lower": df[lower_col].tolist(),
    }


def calculate_atr(
    high: list[float],
    low: list[float],
    close: list[float],
    period: int = 14,
) -> list[float]:
    """Average True Range."""
    return _to_list(pta.atr(
        _to_series(high), _to_series(low), _to_series(close), length=period
    ))


# =============================================================================
# VOLUME INDICATORS
# =============================================================================


def calculate_obv(close: list[float], volume: list[float]) -> list[float]:
    """On Balance Volume."""
    return _to_list(pta.obv(_to_series(close), _to_series(volume)))


def calculate_vwap(
    high: list[float],
    low: list[float],
    close: list[float],
    volume: list[float],
) -> list[float]:
    """Volume Weighted Average Price (cumulative)."""
    h = _to_series(high)
    l = _to_series(low)
    c = _to_series(close)
    v = _to_series(volume)
    typical = (h + l + c) / 3.0
    cum_pv = np.cumsum(typical * v)
    cum_v = np.cumsum(v)
    vwap = np.divide(cum_pv, cum_v, out=np.full_like(cum_pv, np.nan), where=cum_v > 0)
    return vwap.tolist()


# =============================================================================
# PATTERN DETECTION
# =============================================================================


def detect_support_resistance(
    data: list[float], window: int = 5
) -> dict[str, list[float]]:
    """Detect local support (minima) and resistance (maxima) levels."""
    arr = np.array(data, dtype=np.float64)
    supports = np.full(len(arr), np.nan)
    resistances = np.full(len(arr), np.nan)

    if len(arr) < 2 * window + 1:
        return {"support": supports.tolist(), "resistance": resistances.tolist()}

    for i in range(window, len(arr) - window):
        left = arr[i - window : i]
        right = arr[i + 1 : i + 1 + window]
        if np.all(arr[i] <= left) and np.all(arr[i] <= right):
            supports[i] = arr[i]
        if np.all(arr[i] >= left) and np.all(arr[i] >= right):
            resistances[i] = arr[i]

    return {"support": supports.tolist(), "resistance": resistances.tolist()}


def detect_crossover(series1: list[float], series2: list[float]) -> list[str | None]:
    """Detect crossover points between two indicator lines."""
    s1 = np.array(series1, dtype=np.float64)
    s2 = np.array(series2, dtype=np.float64)
    result: list[str | None] = [None] * len(s1)

    for i in range(1, len(s1)):
        v1_prev, v2_prev = s1[i - 1], s2[i - 1]
        v1_curr, v2_curr = s1[i], s2[i]
        if np.isnan(v1_prev) or np.isnan(v2_prev) or np.isnan(v1_curr) or np.isnan(v2_curr):
            continue
        if v1_prev <= v2_prev and v1_curr > v2_curr:
            result[i] = "bullish"
        elif v1_prev >= v2_prev and v1_curr < v2_curr:
            result[i] = "bearish"

    return result


def calculate_stochastic(
    high: list[float],
    low: list[float],
    close: list[float],
    k_period: int = 14,
    d_period: int = 3,
) -> dict[str, list[float]]:
    """Stochastic Oscillator (%K and %D)."""
    df = pta.stoch(
        _to_series(high), _to_series(low), _to_series(close),
        k=k_period, d=d_period,
    )
    nans = [float(np.nan)] * len(close)
    if df is None or df.empty:
        return {"k": nans, "d": nans}
    k_col = [c for c in df.columns if c.startswith("STOCHk_")][0]
    d_col = [c for c in df.columns if c.startswith("STOCHd_")][0]
    return {"k": df[k_col].tolist(), "d": df[d_col].tolist()}


# =============================================================================
# ASYNC WRAPPERS — yfinance -> compute -> JSON
# =============================================================================


async def fetch_technical_indicators(symbol: str) -> str:
    """Fetch OHLCV from yfinance and compute indicators.

    Computes: SMA (20), EMA (20), RSI (14), MACD, Bollinger Bands, ATR,
    OBV, VWAP, and Stochastic. Returns the latest values as JSON.
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(f"{symbol.upper().removesuffix('.NS')}.NS")
        hist = ticker.history(period="6mo", interval="1d")
        if hist.empty:
            return json.dumps({"error": f"No price data for {symbol}", "symbol": symbol})

        close = hist["Close"].tolist()
        high = hist["High"].tolist()
        low = hist["Low"].tolist()
        volume = hist["Volume"].tolist()

        sma_20 = calculate_sma(close, 20)
        ema_20 = calculate_ema(close, 20)
        rsi_14 = calculate_rsi(close, 14)
        macd = calculate_macd(close)
        bb = calculate_bollinger_bands(close, 20, 2.0)
        atr = calculate_atr(high, low, close, 14)
        obv = calculate_obv(close, volume)
        vwap = calculate_vwap(high, low, close, volume)
        stoch = calculate_stochastic(high, low, close)

        result = {
            "symbol": symbol,
            "period": "6mo",
            "data_points": len(close),
            "latest": {
                "sma_20": _latest(sma_20),
                "ema_20": _latest(ema_20),
                "rsi_14": _latest(rsi_14),
                "macd": _latest(macd["macd"]),
                "macd_signal": _latest(macd["signal"]),
                "macd_histogram": _latest(macd["histogram"]),
                "bollinger_upper": _latest(bb["upper"]),
                "bollinger_middle": _latest(bb["middle"]),
                "bollinger_lower": _latest(bb["lower"]),
                "atr_14": _latest(atr),
                "obv": _latest(obv),
                "vwap": _latest(vwap),
                "stoch_k": _latest(stoch["k"]),
                "stoch_d": _latest(stoch["d"]),
            },
        }

        return json.dumps(result)
    except Exception as e:
        logger.exception("fetch_technical_indicators failed for %s", symbol)
        return json.dumps({"error": str(e), "symbol": symbol})


async def compute_custom_indicator(
    symbol: str, indicator: str, period: int = 14
) -> str:
    """Compute a specific technical indicator for a symbol.

    Supported: ``sma``, ``ema``, ``rsi``, ``macd``, ``bb``, ``atr``,
    ``obv``, ``vwap``, ``stochastic``, ``sr`` (support/resistance).
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(f"{symbol.upper().removesuffix('.NS')}.NS")
        hist = ticker.history(period="6mo", interval="1d")
        if hist.empty:
            return json.dumps({"error": f"No price data for {symbol}", "symbol": symbol})

        close = hist["Close"].tolist()
        high = hist["High"].tolist()
        low = hist["Low"].tolist()
        volume = hist["Volume"].tolist()

        indicator = indicator.lower().strip()

        if indicator == "sma":
            data = calculate_sma(close, period)
            return json.dumps({"symbol": symbol, "indicator": "sma", "period": period, "values": data})
        elif indicator == "ema":
            data = calculate_ema(close, period)
            return json.dumps({"symbol": symbol, "indicator": "ema", "period": period, "values": data})
        elif indicator == "rsi":
            data = calculate_rsi(close, period)
            return json.dumps({"symbol": symbol, "indicator": "rsi", "period": period, "values": data})
        elif indicator == "macd":
            data = calculate_macd(close)
            return json.dumps({"symbol": symbol, "indicator": "macd", "values": data})
        elif indicator == "bb":
            data = calculate_bollinger_bands(close, period)
            return json.dumps({"symbol": symbol, "indicator": "bollinger_bands", "period": period, "values": data})
        elif indicator == "atr":
            data = calculate_atr(high, low, close, period)
            return json.dumps({"symbol": symbol, "indicator": "atr", "period": period, "values": data})
        elif indicator == "obv":
            data = calculate_obv(close, volume)
            return json.dumps({"symbol": symbol, "indicator": "obv", "values": data})
        elif indicator == "vwap":
            data = calculate_vwap(high, low, close, volume)
            return json.dumps({"symbol": symbol, "indicator": "vwap", "values": data})
        elif indicator == "stochastic":
            data = calculate_stochastic(high, low, close)
            return json.dumps({"symbol": symbol, "indicator": "stochastic", "values": data})
        elif indicator == "sr":
            data = detect_support_resistance(close, window=period)
            return json.dumps({"symbol": symbol, "indicator": "support_resistance", "window": period, "values": data})
        else:
            valid = ["sma", "ema", "rsi", "macd", "bb", "atr", "obv", "vwap", "stochastic", "sr"]
            return json.dumps({
                "error": f"Unknown indicator '{indicator}'. Valid: {valid}",
                "symbol": symbol,
            })
    except Exception as e:
        logger.exception("compute_custom_indicator failed for %s (%s)", symbol, indicator)
        return json.dumps({"error": str(e), "symbol": symbol, "indicator": indicator})


async def detect_patterns(symbol: str) -> str:
    """Detect candlestick patterns and support/resistance for a symbol.

    Computes support/resistance, SMA crossovers, and RSI status.
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(f"{symbol.upper().removesuffix('.NS')}.NS")
        hist = ticker.history(period="6mo", interval="1d")
        if hist.empty:
            return json.dumps({"error": f"No price data for {symbol}", "symbol": symbol})

        close = hist["Close"].tolist()
        high = hist["High"].tolist()
        low = hist["Low"].tolist()
        volume = hist["Volume"].tolist()

        sr = detect_support_resistance(close, window=5)

        def _latest_non_nan(series: list[float]) -> float | None:
            for v in reversed(series):
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    return float(v)
            return None

        sma20 = calculate_sma(close, 20)
        sma50 = calculate_sma(close, 50)
        crossovers = detect_crossover(sma20, sma50)

        latest_cross: str | None = None
        for v in reversed(crossovers):
            if v is not None:
                latest_cross = v
                break

        rsi_vals = calculate_rsi(close, 14)
        latest_rsi = _latest_non_nan(rsi_vals)
        rsi_signal = "neutral"
        if latest_rsi is not None:
            if latest_rsi > 70:
                rsi_signal = "overbought"
            elif latest_rsi < 30:
                rsi_signal = "oversold"

        result = {
            "symbol": symbol,
            "data_points": len(close),
            "support_latest": _latest_non_nan(sr["support"]),
            "resistance_latest": _latest_non_nan(sr["resistance"]),
            "sma_20_latest": _latest_non_nan(sma20),
            "sma_50_latest": _latest_non_nan(sma50),
            "latest_crossover": latest_cross,
            "rsi_14_latest": latest_rsi,
            "rsi_signal": rsi_signal,
            "crossovers_total": sum(1 for v in crossovers if v is not None),
            "crossovers_bullish": sum(1 for v in crossovers if v == "bullish"),
            "crossovers_bearish": sum(1 for v in crossovers if v == "bearish"),
        }

        return json.dumps(result)
    except Exception as e:
        logger.exception("detect_patterns failed for %s", symbol)
        return json.dumps({"error": str(e), "symbol": symbol})
