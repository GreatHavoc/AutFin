"""Pure Python technical indicator computation module for AutFin.

No TA-Lib dependency — all indicators implemented from scratch with numpy.
Provides standalone compute functions (list[float] in/out) plus async wrappers
that fetch OHLCV from yfinance and return JSON-serialized results.
"""

import json
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# BASIC INDICATORS
# =============================================================================


def calculate_sma(data: list[float], period: int) -> list[float]:
    """Compute Simple Moving Average.

    Returns a list of the same length as input; the first ``period - 1``
    entries are NaN.

    Args:
        data: List of price values.
        period: Look-back window length.

    Returns:
        SMA values (NaN-padded at the start).
    """
    arr = np.array(data, dtype=np.float64)
    result = np.full(len(arr), np.nan)
    if len(arr) < period:
        return result.tolist()

    kernel = np.ones(period) / period
    result[period - 1 :] = np.convolve(arr, kernel, mode="valid")
    return result.tolist()


def calculate_ema(data: list[float], period: int) -> list[float]:
    """Compute Exponential Moving Average.

    The first ``period - 1`` entries are NaN.  The first valid value is the
    SMA of the first ``period`` elements; subsequent values use exponential
    smoothing.

    Args:
        data: List of price values.
        period: Look-back window length.

    Returns:
        EMA values (NaN-padded at the start).
    """
    arr = np.array(data, dtype=np.float64)
    result = np.full(len(arr), np.nan)
    if len(arr) < period:
        return result.tolist()

    multiplier = 2.0 / (period + 1)
    result[period - 1] = float(np.mean(arr[:period]))
    for i in range(period, len(arr)):
        result[i] = (arr[i] - result[i - 1]) * multiplier + result[i - 1]
    return result.tolist()


def calculate_rsi(data: list[float], period: int = 14) -> list[float]:
    """Compute Relative Strength Index (Wilder's smoothing).

    The first ``period`` entries are NaN.

    Args:
        data: List of close prices.
        period: RSI period (default 14).

    Returns:
        RSI values in [0, 100] (NaN-padded at the start).
    """
    arr = np.array(data, dtype=np.float64)
    result = np.full(len(arr), np.nan)
    if len(arr) <= period:
        return result.tolist()

    delta = np.diff(arr)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.full(len(arr), np.nan)
    avg_loss = np.full(len(arr), np.nan)

    idx = period
    avg_gain[idx] = float(np.mean(gains[:period]))
    avg_loss[idx] = float(np.mean(losses[:period]))

    for i in range(idx + 1, len(arr)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss > 0)
    result[period:] = 100.0 - (100.0 / (1.0 + rs[period:]))
    return result.tolist()


def calculate_macd(
    data: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, list[float]]:
    """Compute MACD (Moving Average Convergence Divergence).

    Args:
        data: List of close prices.
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).
        signal: Signal-line EMA period (default 9).

    Returns:
        Dict with keys ``macd``, ``signal``, ``histogram`` — each a list
        of the same length as input, NaN-padded where insufficient data.
    """
    ema_fast = np.array(calculate_ema(data, fast), dtype=np.float64)
    ema_slow = np.array(calculate_ema(data, slow), dtype=np.float64)
    macd_line = ema_fast - ema_slow

    nans = [float(np.nan)] * len(data)
    if len(data) < slow:
        return {"macd": nans, "signal": nans, "histogram": nans}

    first_valid = slow - 1
    macd_valid = macd_line[first_valid:]
    if len(macd_valid) < signal:
        return {"macd": macd_line.tolist(), "signal": nans, "histogram": nans}

    signal_valid = np.array(
        calculate_ema(macd_valid.tolist(), signal), dtype=np.float64
    )
    signal_arr = np.full(len(data), np.nan)
    signal_arr[first_valid:] = signal_valid

    histogram = macd_line - signal_arr

    return {
        "macd": macd_line.tolist(),
        "signal": signal_arr.tolist(),
        "histogram": histogram.tolist(),
    }


# =============================================================================
# BANDS & VOLATILITY
# =============================================================================


def calculate_bollinger_bands(
    data: list[float], period: int = 20, stddev: float = 2.0
) -> dict[str, list[float]]:
    """Compute Bollinger Bands.

    Args:
        data: List of close prices.
        period: SMA window length (default 20).
        stddev: Number of standard deviations for the bands (default 2.0).

    Returns:
        Dict with keys ``upper``, ``middle``, ``lower`` — each a list of
        the same length as input, NaN-padded at the start.
    """
    arr = np.array(data, dtype=np.float64)
    result_upper = np.full(len(arr), np.nan)
    result_mid = np.full(len(arr), np.nan)
    result_lower = np.full(len(arr), np.nan)

    if len(arr) < period:
        return {
            "upper": result_upper.tolist(),
            "middle": result_mid.tolist(),
            "lower": result_lower.tolist(),
        }

    for i in range(period - 1, len(arr)):
        window = arr[i - period + 1 : i + 1]
        mu = float(np.mean(window))
        sigma = float(np.std(window, ddof=1))
        result_mid[i] = mu
        result_upper[i] = mu + stddev * sigma
        result_lower[i] = mu - stddev * sigma

    return {
        "upper": result_upper.tolist(),
        "middle": result_mid.tolist(),
        "lower": result_lower.tolist(),
    }


def calculate_atr(
    high: list[float],
    low: list[float],
    close: list[float],
    period: int = 14,
) -> list[float]:
    """Compute Average True Range (Wilder's smoothing).

    Args:
        high: List of high prices.
        low: List of low prices.
        close: List of close prices.
        period: ATR period (default 14).

    Returns:
        ATR values, NaN-padded at the start.
    """
    if not high:
        return []

    h = np.array(high, dtype=np.float64)
    l = np.array(low, dtype=np.float64)
    c = np.array(close, dtype=np.float64)

    prev_close = np.roll(c, 1)
    prev_close[0] = c[0]

    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close)))

    result = np.full(len(tr), np.nan)
    if len(tr) < period:
        return result.tolist()

    result[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, len(tr)):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period

    return result.tolist()


# =============================================================================
# VOLUME INDICATORS
# =============================================================================


def calculate_obv(close: list[float], volume: list[float]) -> list[float]:
    """Compute On-Balance Volume.

    OBV starts at the first period's volume and adds / subtracts subsequent
    volumes depending on close-to-close direction.

    Args:
        close: List of close prices.
        volume: List of volumes.

    Returns:
        Cumulative OBV values.
    """
    if not close:
        return []

    c = np.array(close, dtype=np.float64)
    v = np.array(volume, dtype=np.float64)

    obv = np.zeros(len(c))
    obv[0] = v[0]

    for i in range(1, len(c)):
        if c[i] > c[i - 1]:
            obv[i] = obv[i - 1] + v[i]
        elif c[i] < c[i - 1]:
            obv[i] = obv[i - 1] - v[i]
        else:
            obv[i] = obv[i - 1]

    return obv.tolist()


def calculate_vwap(
    high: list[float],
    low: list[float],
    close: list[float],
    volume: list[float],
) -> list[float]:
    """Compute Volume Weighted Average Price (cumulative).

    Uses the typical price ``(high + low + close) / 3``.

    Args:
        high: List of high prices.
        low: List of low prices.
        close: List of close prices.
        volume: List of volumes.

    Returns:
        Cumulative VWAP values.
    """
    h = np.array(high, dtype=np.float64)
    l = np.array(low, dtype=np.float64)
    c = np.array(close, dtype=np.float64)
    v = np.array(volume, dtype=np.float64)

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
    """Detect local support (minima) and resistance (maxima) levels.

    A point is a support if it is the lowest value in a neighbourhood of
    size ``window`` on each side.  Resistance is the mirror condition.

    Args:
        data: List of price values.
        window: Number of neighbours to compare against on each side.

    Returns:
        Dict with keys ``support`` and ``resistance``, each a list of the
        same length as input with the detected level or NaN.
    """
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
    """Detect crossover points between two indicator lines.

    Args:
        series1: First indicator line (e.g. fast EMA).
        series2: Second indicator line (e.g. slow EMA).

    Returns:
        List of the same length — ``"bullish"`` when series1 crosses
        **above** series2, ``"bearish"`` when series1 crosses **below**
        series2, or ``None`` otherwise.
    """
    s1 = np.array(series1, dtype=np.float64)
    s2 = np.array(series2, dtype=np.float64)

    result: list[str | None] = [None] * len(s1)

    for i in range(1, len(s1)):
        v1_prev = s1[i - 1]
        v2_prev = s2[i - 1]
        v1_curr = s1[i]
        v2_curr = s2[i]

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
    """Compute the Stochastic Oscillator (%K and %D).

    %K = 100 * (Close - LowestLow) / (HighestHigh - LowestLow) over ``k_period``.
    %D = SMA of %K over ``d_period``.

    Args:
        high: List of high prices.
        low: List of low prices.
        close: List of close prices.
        k_period: %K look-back (default 14).
        d_period: %D smoothing period (default 3).

    Returns:
        Dict with keys ``k`` and ``d``, NaN-padded at the start.
    """
    h = np.array(high, dtype=np.float64)
    l = np.array(low, dtype=np.float64)
    c = np.array(close, dtype=np.float64)

    k_values = np.full(len(c), np.nan)

    for i in range(k_period - 1, len(c)):
        w_high = float(np.max(h[i - k_period + 1 : i + 1]))
        w_low = float(np.min(l[i - k_period + 1 : i + 1]))
        if w_high != w_low:
            k_values[i] = 100.0 * (c[i] - w_low) / (w_high - w_low)
        else:
            k_values[i] = 50.0

    d_values = np.array(calculate_sma(k_values.tolist(), d_period), dtype=np.float64)

    return {"k": k_values.tolist(), "d": d_values.tolist()}


# =============================================================================
# ASYNC WRAPPERS — yfinance → compute → JSON
# =============================================================================


async def fetch_technical_indicators(symbol: str) -> str:
    """Fetch OHLCV from yfinance and compute a comprehensive set of indicators.

    Computes: SMA (20), EMA (20), RSI (14), MACD, Bollinger Bands, ATR,
    OBV, VWAP, and Stochastic. Returns the latest values plus the full
    series templates as a JSON string.  The ``latest`` object contains the
    most recent non-NaN value for each indicator.

    Args:
        symbol: Trading symbol (e.g. ``"RELIANCE"``).

    Returns:
        JSON string with indicator results.
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

        def _latest(arr: list[float]) -> float | None:
            for v in reversed(arr):
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    return float(v)
            return None

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

    Fetches OHLCV data from yfinance and returns only the requested indicator.

    Supported indicators:
        ``sma``, ``ema``, ``rsi``, ``macd``, ``bb`` (Bollinger Bands),
        ``atr``, ``obv``, ``vwap``, ``stochastic``, ``sr`` (support/resistance).

    Args:
        symbol: Trading symbol (e.g. ``"TCS"``).
        indicator: Indicator slug (see above).
        period: Period for indicators that accept one (default 14).

    Returns:
        JSON string with the indicator result.
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

    Fetches OHLCV from yfinance and computes:
    - Support and resistance levels (latest non-NaN values).
    - Bullish and bearish crossovers between SMA 20 and SMA 50.
    - RSI overbought / oversold status.

    Args:
        symbol: Trading symbol (e.g. ``"INFY"``).

    Returns:
        JSON string with detected patterns.
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
        open_ = hist["Open"].tolist()
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
