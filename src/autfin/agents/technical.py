"""Technical analysis agent for AutFin — agentic with tool-calling loop.

Uses pandas-ta-classic (pure Python) for all indicator calculations.
If TA-Lib (C library) is installed, pandas-ta-classic can optionally use it
for speed, but it is NOT required.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import pandas_ta_classic as pta

from langgraph.config import RunnableConfig

from src.autfin.state.agent_state import AgentState
from src.autfin.llm.client import get_llm_client
from src.autfin.agents.tools import TECHNICAL_TOOLS, TECHNICAL_CALLABLES

logger = logging.getLogger(__name__)

# ================== TECHNICAL INDICATOR CONFIGURATION ==================

# Trend indicators (5)
TREND_INDICATORS = {
    "sma": "Simple Moving Average",
    "ema": "Exponential Moving Average",
    "wma": "Weighted Moving Average",
    "macd": "Moving Average Convergence Divergence",
    "adx": "Average Directional Index",
}

# Momentum indicators (5)
MOMENTUM_INDICATORS = {
    "rsi": "Relative Strength Index",
    "stoch": "Stochastic Oscillator",
    "cci": "Commodity Channel Index",
    "williams_r": "Williams %R",
    "ao": "Awesome Oscillator",
}

# Volatility indicators (3)
VOLATILITY_INDICATORS = {
    "bollinger": "Bollinger Bands",
    "atr": "Average True Range",
    "keltner": "Keltner Channel",
}

# Volume indicators (3)
VOLUME_INDICATORS = {
    "obv": "On Balance Volume",
    "mfi": "Money Flow Index",
    "vwma": "Volume Weighted Moving Average",
}

# Pattern recognition indicators (20+ candlestick patterns)
PATTERN_INDICATORS = {
    # 2-candle patterns (8)
    "engulfing_bullish": "Bullish Engulfing Pattern",
    "engulfing_bearish": "Bearish Engulfing Pattern",
    "harami_bullish": "Bullish Harami Pattern",
    "harami_bearish": "Bearish Harami Pattern",
    "piercing_bullish": "Piercing Pattern (Bullish)",
    "dark_cloud_cover": "Dark Cloud Cover (Bearish)",
    "morning_star": "Morning Star Pattern",
    "evening_star": "Evening Star Pattern",
    # 1-candle patterns (7)
    "hammer": "Hammer Pattern",
    "hanging_man": "Hanging Man Pattern",
    "shooting_star": "Shooting Star Pattern",
    "inverted_hammer": "Inverted Hammer Pattern",
    "doji": "Doji Pattern",
    "dragonfly_doji": "Dragonfly Doji",
    "gravestone_doji": "Gravestone Doji",
    # 3-candle patterns (6)
    "three_white_soldiers": "Three White Soldiers",
    "three_black_crows": "Three Black Crows",
    "three_inside_up": "Three Inside Up Pattern",
    "three_inside_down": "Three Inside Down Pattern",
    "three_line_strike": "Three Line Strike Pattern",
    "advanced_bloody_kick": "Advanced Bloody Kick",
}


# ================== INDICATOR CALCULATION (pandas-ta-classic) ==================

def calculate_sma(prices: pd.Series, period: int = 20) -> pd.Series:
    """Simple Moving Average."""
    return pta.sma(prices, length=period)


def calculate_multi_sma(prices: pd.Series, periods: list[int] | None = None) -> dict[int, pd.Series]:
    """SMA for multiple periods (20, 50, 200 by default)."""
    if periods is None:
        periods = [20, 50, 200]
    return {p: pta.sma(prices, length=p) for p in periods}


def calculate_ema(prices: pd.Series, period: int = 20) -> pd.Series:
    """Exponential Moving Average."""
    return pta.ema(prices, length=period)


def calculate_wma(prices: pd.Series, period: int = 20) -> pd.Series:
    """Weighted Moving Average."""
    return pta.wma(prices, length=period)


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """MACD indicator. Returns dict with 'macd', 'signal', 'histogram' pd.Series."""
    df = pta.macd(prices, fast=fast, slow=slow, signal=signal)
    if df is None or df.empty:
        n = len(prices)
        nan = pd.Series([float("nan")] * n, index=prices.index)
        return {"macd": nan, "signal": nan, "histogram": nan}
    macd_col = [c for c in df.columns if c.startswith("MACD_")][0]
    signal_col = [c for c in df.columns if c.startswith("MACDs_")][0]
    hist_col = [c for c in df.columns if c.startswith("MACDh_")][0]
    return {"macd": df[macd_col], "signal": df[signal_col], "histogram": df[hist_col]}


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> dict:
    """Average Directional Index. Returns dict with 'adx', 'plus_di', 'minus_di'."""
    df = pta.adx(high, low, close, length=period)
    if df is None or df.empty:
        n = len(close)
        nan = pd.Series([float("nan")] * n, index=close.index)
        return {"adx": nan, "plus_di": nan, "minus_di": nan}
    adx_col = [c for c in df.columns if c.startswith("ADX_")][0]
    dip_col = [c for c in df.columns if c.startswith("DMP_")][0]
    dim_col = [c for c in df.columns if c.startswith("DMN_")][0]
    return {"adx": df[adx_col], "plus_di": df[dip_col], "minus_di": df[dim_col]}


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    return pta.rsi(prices, length=period)


def calculate_stoch(high: pd.Series, low: pd.Series, close: pd.Series,
                    k_period: int = 14, d_period: int = 3) -> dict:
    """Stochastic Oscillator. Returns dict with 'slowk', 'slowd'."""
    df = pta.stoch(high, low, close, k=k_period, d=d_period)
    if df is None or df.empty:
        n = len(close)
        nan = pd.Series([float("nan")] * n, index=close.index)
        return {"slowk": nan, "slowd": nan}
    k_col = [c for c in df.columns if c.startswith("STOCHk_")][0]
    d_col = [c for c in df.columns if c.startswith("STOCHd_")][0]
    return {"slowk": df[k_col], "slowd": df[d_col]}


def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    return pta.cci(high, low, close, length=period)


def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Williams %R."""
    return pta.willr(high, low, close, length=period)


def calculate_awesome(high: pd.Series, low: pd.Series) -> pd.Series:
    """Awesome Oscillator."""
    return pta.ao(high, low)


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, nbdevup: float = 2.0,
                               nbdevdn: float = 2.0) -> dict:
    """Bollinger Bands. Returns dict with 'upper', 'middle', 'lower'."""
    df = pta.bbands(prices, length=period, std=nbdevup)
    if df is None or df.empty:
        n = len(prices)
        nan = pd.Series([float("nan")] * n, index=prices.index)
        return {"upper": nan, "middle": nan, "lower": nan}
    upper_col = [c for c in df.columns if c.startswith("BBU_")][0]
    mid_col = [c for c in df.columns if c.startswith("BBM_")][0]
    lower_col = [c for c in df.columns if c.startswith("BBL_")][0]
    return {"upper": df[upper_col], "middle": df[mid_col], "lower": df[lower_col]}


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    return pta.atr(high, low, close, length=period)


def calculate_keltner_channel(high: pd.Series, low: pd.Series, close: pd.Series,
                              period: int = 20, multiplier: float = 2.0) -> dict:
    """Keltner Channel. Returns dict with 'upper', 'middle', 'lower'."""
    df = pta.kc(high, low, close, length=period, scalar=multiplier)
    if df is None or df.empty:
        n = len(close)
        nan = pd.Series([float("nan")] * n, index=close.index)
        return {"upper": nan, "middle": nan, "lower": nan}
    upper_col = [c for c in df.columns if "U" in c.split("_") or c.endswith("U")][0]
    mid_col = [c for c in df.columns if "M" in c.split("_") or c.endswith("M")][0]
    lower_col = [c for c in df.columns if "L" in c.split("_") or c.endswith("L")][0]
    return {"upper": df[upper_col], "middle": df[mid_col], "lower": df[lower_col]}


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On Balance Volume."""
    return pta.obv(close, volume)


def calculate_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
                  period: int = 14) -> pd.Series:
    """Money Flow Index."""
    return pta.mfi(high, low, close, volume, length=period)


def calculate_vwma(close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume Weighted Moving Average."""
    return pta.vwma(close, volume, length=period)


# ================== CANDLESTICK PATTERN DETECTION ==================

def _detect_patterns(open_: pd.Series, high: pd.Series, low: pd.Series,
                     close: pd.Series) -> dict[str, pd.Series]:
    """Detect all candlestick patterns using pandas-ta-classic.

    pandas-ta-classic exposes CDL patterns via ``pta.cdl_pattern()`` which
    returns a DataFrame with one column per pattern (value: +100 bullish,
    -100 bearish, 0 neutral).
    """
    try:
        # cdl_pattern returns a DataFrame; each col is a pattern name
        df = pta.cdl_pattern(open_, high, low, close, name="all")
    except Exception:
        df = None

    n = len(close)
    nan = pd.Series([False] * n, index=close.index)

    # Map pandas-ta pattern names to our internal names
    _PATTERN_MAP = {
        "CDL_ENGULFING": ("engulfing_bullish", "engulfing_bearish"),
        "CDL_HARAMI": ("harami_bullish", "harami_bearish"),
        "CDL_PIERCING": ("piercing_bullish", None),
        "CDL_DARKCLOUDCOVER": (None, "dark_cloud_cover"),
        "CDL_MORNINGSTAR": ("morning_star", None),
        "CDL_EVENINGSTAR": (None, "evening_star"),
        "CDL_HAMMER": ("hammer", None),
        "CDL_HANGINGMAN": ("hanging_man", None),
        "CDL_SHOOTINGSTAR": ("shooting_star", None),
        "CDL_INVERTEDHAMMER": ("inverted_hammer", None),
        "CDL_DOJI": ("doji", None),
        "CDL_DRAGONFLYDOJI": ("dragonfly_doji", None),
        "CDL_GRAVESTONEDOJI": ("gravestone_doji", None),
        "CDL_THREEWHITESOLDIERS": ("three_white_soldiers", None),
        "CDL_THREEBLACKCROWS": ("three_black_crows", None),
        "CDL_THREEINSIDEUP": ("three_inside_up", None),
        "CDL_THREEINSIDEDOWN": ("three_inside_down", None),
        "CDL_THREELINESTRIKE": ("three_line_strike", None),
        "CDL_ADVANCEDBLOCK": ("advanced_bloody_kick", None),
    }

    result: dict[str, pd.Series] = {}

    if df is not None and not df.empty:
        for cdl_name, (bull_key, bear_key) in _PATTERN_MAP.items():
            matching = [c for c in df.columns if cdl_name in c.upper()]
            if not matching:
                continue
            col = df[matching[0]]
            if bull_key:
                result[bull_key] = col > 0
            if bear_key:
                result[bear_key] = col < 0
    else:
        # Fallback: return all-False series if pattern detection unavailable
        for bull_key, bear_key in _PATTERN_MAP.values():
            if bull_key:
                result[bull_key] = nan.copy()
            if bear_key:
                result[bear_key] = nan.copy()

    return result


# ================== MAIN INDICATOR CALCULATION ==================

def calculate_indicators(data: pd.DataFrame) -> dict:
    """Calculate all technical indicators using pandas-ta-classic.

    Args:
        data: DataFrame with columns: 'open', 'high', 'low', 'close', 'volume'

    Returns:
        Dictionary containing all calculated indicators.
    """
    if data.empty:
        raise ValueError("Input data is empty")

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    o = data["open"]
    h = data["high"]
    l = data["low"]
    c = data["close"]
    v = data["volume"]

    indicators: dict[str, Any] = {}

    # Trend (6)
    indicators["sma"] = calculate_sma(c)
    indicators["multi_sma"] = calculate_multi_sma(c)
    indicators["ema"] = calculate_ema(c)
    indicators["wma"] = calculate_wma(c)
    indicators["macd"] = calculate_macd(c)
    indicators["adx"] = calculate_adx(h, l, c)

    # Momentum (5)
    indicators["rsi"] = calculate_rsi(c)
    indicators["stoch"] = calculate_stoch(h, l, c)
    indicators["cci"] = calculate_cci(h, l, c)
    indicators["williams_r"] = calculate_williams_r(h, l, c)
    indicators["ao"] = calculate_awesome(h, l)

    # Volatility (3)
    indicators["bollinger_bands"] = calculate_bollinger_bands(c)
    indicators["atr"] = calculate_atr(h, l, c)
    indicators["keltner_channel"] = calculate_keltner_channel(h, l, c)

    # Volume (3)
    indicators["obv"] = calculate_obv(c, v)
    indicators["mfi"] = calculate_mfi(h, l, c, v)
    indicators["vwma"] = calculate_vwma(c, v)

    # Candlestick patterns (20+)
    indicators["patterns"] = _detect_patterns(o, h, l, c)

    logger.info("Calculated %d indicator groups", len(indicators))
    return indicators


# ================== SIGNAL GENERATION ==================

def generate_signal(indicators: dict) -> dict:
    """Generate buy/sell/hold signals based on indicator combinations.

    Args:
        indicators: Dictionary of calculated indicators

    Returns:
        Dictionary containing signal, confidence, and reasoning
    """
    signal_data: dict[str, Any] = {
        "signal": "HOLD",
        "confidence": 0.0,
        "reasoning": [],
        "indicator_scores": {},
        "pattern_signals": [],
    }

    trend_score = 0
    trend_max = 0
    momentum_score = 0
    momentum_max = 0
    volatility_score = 0
    volatility_max = 0

    # --- Trend indicators ---

    if "sma" in indicators and "ema" in indicators:
        sma = indicators["sma"]
        ema = indicators["ema"]
        sma_val = sma.iloc[-1] if len(sma) > 0 and not pd.isna(sma.iloc[-1]) else None
        ema_val = ema.iloc[-1] if len(ema) > 0 and not pd.isna(ema.iloc[-1]) else None
        close_series = indicators.get("sma", pd.Series()).dropna()
        price = close_series.iloc[-1] if not close_series.empty else None

        if sma_val and ema_val and price is not None:
            trend_max += 2
            if price > max(sma_val, ema_val):
                trend_score += 1
                signal_data["reasoning"].append("Price above SMA and EMA (Bullish)")
            elif price < min(sma_val, ema_val):
                trend_score -= 1
                signal_data["reasoning"].append("Price below SMA and EMA (Bearish)")

    if "multi_sma" in indicators:
        multi_sma = indicators["multi_sma"]
        sma20 = multi_sma.get(20)
        sma50 = multi_sma.get(50)
        sma200 = multi_sma.get(200)

        close_prices = indicators.get("sma", pd.Series()).dropna()
        price = close_prices.iloc[-1] if not close_prices.empty else None

        if price is not None:
            if sma50 is not None and sma200 is not None:
                s50_c = sma50.iloc[-1] if len(sma50) > 0 and not pd.isna(sma50.iloc[-1]) else None
                s50_p = sma50.iloc[-2] if len(sma50) > 1 and not pd.isna(sma50.iloc[-2]) else None
                s200_c = sma200.iloc[-1] if len(sma200) > 0 and not pd.isna(sma200.iloc[-1]) else None
                s200_p = sma200.iloc[-2] if len(sma200) > 1 and not pd.isna(sma200.iloc[-2]) else None

                if all(v is not None for v in [s50_c, s50_p, s200_c, s200_p]):
                    trend_max += 2
                    if s50_p <= s200_p and s50_c > s200_c:
                        trend_score += 2
                        signal_data["reasoning"].append("Golden Cross (SMA50 > SMA200) — Strong bullish")
                    elif s50_p >= s200_p and s50_c < s200_c:
                        trend_score -= 2
                        signal_data["reasoning"].append("Death Cross (SMA50 < SMA200) — Strong bearish")
                    elif s50_c > s200_c:
                        trend_score += 1
                        signal_data["reasoning"].append("SMA50 above SMA200 (Bullish trend)")
                    else:
                        trend_score -= 1
                        signal_data["reasoning"].append("SMA50 below SMA200 (Bearish trend)")

            if sma20 is not None:
                s20_val = sma20.iloc[-1] if len(sma20) > 0 and not pd.isna(sma20.iloc[-1]) else None
                if s20_val is not None:
                    trend_max += 1
                    if price > s20_val:
                        trend_score += 1
                        signal_data["reasoning"].append("Price above SMA20 (Short-term bullish)")
                    else:
                        trend_score -= 1
                        signal_data["reasoning"].append("Price below SMA20 (Short-term bearish)")

    # MACD
    if "macd" in indicators and indicators["macd"]:
        m = indicators["macd"]
        macd_val = m["macd"].iloc[-1] if len(m["macd"]) > 0 and not pd.isna(m["macd"].iloc[-1]) else None
        sig_val = m["signal"].iloc[-1] if len(m["signal"]) > 0 and not pd.isna(m["signal"].iloc[-1]) else None
        if macd_val is not None and sig_val is not None:
            trend_max += 2
            if macd_val > sig_val:
                trend_score += 1
                signal_data["reasoning"].append("MACD above signal line (Bullish)")
            elif macd_val < sig_val:
                trend_score -= 1
                signal_data["reasoning"].append("MACD below signal line (Bearish)")

    # ADX
    if "adx" in indicators and indicators["adx"]:
        adx_val = indicators["adx"]["adx"].iloc[-1] if len(indicators["adx"]["adx"]) > 0 and not pd.isna(indicators["adx"]["adx"].iloc[-1]) else None
        if adx_val is not None:
            trend_max += 1
            if adx_val > 25:
                signal_data["reasoning"].append(f"Strong trend (ADX: {adx_val:.2f})")
            else:
                signal_data["reasoning"].append(f"Weak trend (ADX: {adx_val:.2f})")

    # --- Momentum indicators ---

    if "rsi" in indicators:
        rsi_val = indicators["rsi"].iloc[-1] if len(indicators["rsi"]) > 0 and not pd.isna(indicators["rsi"].iloc[-1]) else None
        if rsi_val is not None:
            momentum_max += 2
            if rsi_val > 70:
                momentum_score -= 1
                signal_data["reasoning"].append(f"RSI overbought ({rsi_val:.2f}) - Potential sell signal")
            elif rsi_val < 30:
                momentum_score += 1
                signal_data["reasoning"].append(f"RSI oversold ({rsi_val:.2f}) - Potential buy signal")
            else:
                signal_data["reasoning"].append(f"RSI neutral ({rsi_val:.2f})")

    if "stoch" in indicators and indicators["stoch"]:
        st = indicators["stoch"]
        k_val = st["slowk"].iloc[-1] if len(st["slowk"]) > 0 and not pd.isna(st["slowk"].iloc[-1]) else None
        d_val = st["slowd"].iloc[-1] if len(st["slowd"]) > 0 and not pd.isna(st["slowd"].iloc[-1]) else None
        if k_val is not None and d_val is not None:
            momentum_max += 2
            if k_val > d_val and k_val < 20:
                momentum_score += 1
                signal_data["reasoning"].append("Stochastic crossover in oversold zone (Bullish)")
            elif k_val < d_val and k_val > 80:
                momentum_score -= 1
                signal_data["reasoning"].append("Stochastic crossover in overbought zone (Bearish)")

    # --- Volatility indicators ---

    if "bollinger_bands" in indicators:
        bb = indicators["bollinger_bands"]
        up = bb["upper"].iloc[-1] if len(bb["upper"]) > 0 and not pd.isna(bb["upper"].iloc[-1]) else None
        lo = bb["lower"].iloc[-1] if len(bb["lower"]) > 0 and not pd.isna(bb["lower"].iloc[-1]) else None
        mid = bb["middle"].iloc[-1] if len(bb["middle"]) > 0 and not pd.isna(bb["middle"].iloc[-1]) else None
        bb_close = bb["upper"].dropna()
        cur = bb_close.iloc[-1] if not bb_close.empty else None

        if up and lo and mid and cur:
            volatility_max += 2
            if cur > up:
                volatility_score -= 1
                signal_data["reasoning"].append("Price above upper Bollinger Band (Overbought)")
            elif cur < lo:
                volatility_score += 1
                signal_data["reasoning"].append("Price below lower Bollinger Band (Oversold)")
            elif lo < cur < up:
                signal_data["reasoning"].append("Price within Bollinger Bands")

    if "atr" in indicators:
        atr_val = indicators["atr"].iloc[-1] if len(indicators["atr"]) > 0 and not pd.isna(indicators["atr"].iloc[-1]) else None
        if atr_val is not None:
            volatility_max += 1
            signal_data["reasoning"].append(f"ATR volatility: {atr_val:.2f}")

    # --- Pattern signals ---

    pattern_score = 0
    pattern_max = 0
    if "patterns" in indicators and indicators["patterns"]:
        patterns = indicators["patterns"]

        bullish = [
            "engulfing_bullish", "harami_bullish", "piercing_bullish",
            "morning_star", "hammer", "inverted_hammer", "dragonfly_doji",
            "three_white_soldiers", "three_inside_up", "three_line_strike",
        ]
        bearish = [
            "engulfing_bearish", "harami_bearish", "dark_cloud_cover",
            "evening_star", "hanging_man", "shooting_star", "gravestone_doji",
            "three_black_crows", "three_inside_down", "advanced_bloody_kick",
        ]

        for name in bullish:
            if name in patterns and not patterns[name].empty and patterns[name].iloc[-1]:
                pattern_score += 1
                pattern_max += 1
                signal_data["pattern_signals"].append({
                    "type": "bullish",
                    "pattern": PATTERN_INDICATORS.get(name, name),
                })

        for name in bearish:
            if name in patterns and not patterns[name].empty and patterns[name].iloc[-1]:
                pattern_score -= 1
                pattern_max += 1
                signal_data["pattern_signals"].append({
                    "type": "bearish",
                    "pattern": PATTERN_INDICATORS.get(name, name),
                })

    # --- Aggregate ---

    total_score = trend_score + momentum_score + volatility_score + pattern_score
    max_score = trend_max + momentum_max + volatility_max + pattern_max

    if max_score > 0:
        confidence = abs(total_score) / max_score
        if total_score > 0.5:
            signal_data["signal"] = "BUY"
        elif total_score < -0.5:
            signal_data["signal"] = "SELL"
        else:
            signal_data["signal"] = "HOLD"
        signal_data["confidence"] = round(confidence, 2)

    signal_data["indicator_scores"] = {
        "trend": trend_score,
        "momentum": momentum_score,
        "volatility": volatility_score,
        "patterns": len(signal_data["pattern_signals"]),
    }

    if not signal_data["reasoning"]:
        signal_data["reasoning"].append("Insufficient data for analysis")

    return signal_data


# ================== TECHNICAL AGENT ==================

class TechnicalAgent:
    """Performs technical analysis using pandas-ta-classic indicators."""

    def __init__(self):
        """Initialize the Technical Agent."""
        self.indicator_config = {
            **TREND_INDICATORS,
            **MOMENTUM_INDICATORS,
            **VOLATILITY_INDICATORS,
            **VOLUME_INDICATORS,
            **PATTERN_INDICATORS,
        }

    async def analyze(self, symbol: str, data: Optional[pd.DataFrame] = None) -> dict:
        """Perform comprehensive technical analysis for a symbol.

        Args:
            symbol: Trading symbol to analyze
            data: DataFrame with OHLCV columns

        Returns:
            Dictionary containing all technical indicators and signals
        """
        try:
            if data is None:
                raise ValueError("Price data must be provided for analysis")

            indicators = calculate_indicators(data)
            signal = generate_signal(indicators)

            report = {
                "symbol": symbol,
                "timestamp": pd.Timestamp.now().isoformat(),
                "indicators": indicators,
                "signal": signal,
                "indicators_count": len(self.indicator_config),
            }
            return report

        except Exception as e:
            logger.error("Technical analysis failed for %s: %s", symbol, e)
            return {
                "symbol": symbol,
                "error": str(e),
                "timestamp": pd.Timestamp.now().isoformat(),
            }

    async def generate_signals(self, symbol: str, data: Optional[pd.DataFrame] = None) -> dict:
        """Generate trading signals based on technical analysis."""
        analysis = await self.analyze(symbol, data)
        if "error" in analysis:
            return {
                "symbol": symbol,
                "signal": "HOLD",
                "confidence": 0.0,
                "error": analysis["error"],
            }
        return analysis["signal"]

    def get_indicator_names(self) -> list:
        """Get list of all indicator names."""
        return list(self.indicator_config.keys())

    async def _agentic_analyze(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Tool-calling agentic analysis: LLM decides which indicators to compute."""
        llm = get_llm_client()
        if not llm.is_available:
            return None

        system_prompt = (
            f"You are an expert technical analyst. Analyze {symbol} on the NSE.\n"
            "Use your tools to fetch price data, compute indicators, and detect patterns.\n"
            "Produce a COMPLETE analysis as a single JSON object with these fields:\n"
            "sma_20, sma_50, sma_200, rsi, macd, macd_signal, macd_histogram,\n"
            "bollinger_upper, bollinger_middle, bollinger_lower, atr, volume_trend,\n"
            "candlestick_patterns (list), signal (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL),\n"
            "strength (-5 to +5), conviction (0.0-1.0), summary (2-3 sentences).\n"
            "Output ONLY the JSON object — no markdown, no explanation."
        )

        raw = await llm.generate_with_tools(
            system_prompt=system_prompt,
            tools=TECHNICAL_TOOLS,
            tool_callables=TECHNICAL_CALLABLES,
            model=llm.deep_model,
            max_tokens=4096,
            max_iterations=20,
        )
        if not raw:
            return None

        from src.autfin.utils import extract_json
        result = extract_json(raw)
        if result:
            return result
        logger.warning("Technical agentic output not JSON for %s", symbol)
        return None


# ================== TECHNICAL AGENT ASYNC WORKER ==================

async def technical_agent(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Orchestrate technical analysis workflow.

    Args:
        state: Current agent state containing market data

    Returns:
        Updated state with technical analysis results
    """
    errors = state.get("errors", [])
    technical_reports = state.get("technical_reports", {})

    try:
        market_snapshot = state.get("market_snapshot", {})
        quotes = market_snapshot.get("quotes", {})

        if not quotes:
            errors.append("No market data available for technical analysis")
            return {"technical_reports": technical_reports, "errors": errors}

        symbol = next(iter(quotes), None)
        if symbol is None:
            errors.append("No symbols available for technical analysis")
            return {"technical_reports": technical_reports, "errors": errors}

        quote = quotes[symbol]

        # Fetch real historical OHLCV data via yfinance
        price_data = None
        try:
            import yfinance as yf
            ticker = yf.Ticker(f"{symbol.removesuffix('.NS')}.NS")
            hist = ticker.history(period="6mo", interval="1d")
            if hist is not None and not hist.empty:
                price_data = hist.rename(columns={
                    "Open": "open", "High": "high", "Low": "low",
                    "Close": "close", "Volume": "volume",
                })[["open", "high", "low", "close", "volume"]]
        except Exception as yf_exc:
            logger.warning("yfinance failed for %s: %s", symbol, yf_exc)

        if price_data is None or price_data.empty:
            technical_reports[symbol] = {
                "error": "yfinance_ohlcv_unavailable",
                "message": f"Cannot analyze {symbol} — OHLCV data unavailable from yfinance",
                "symbol": symbol,
                "status": "error",
            }
            return {"technical_reports": technical_reports, "errors": errors}

        agent = TechnicalAgent()
        analysis = await agent._agentic_analyze(symbol)
        if analysis is None:
            analysis = await agent.analyze(symbol, price_data)

        if "error" not in analysis:
            technical_reports[symbol] = analysis
        else:
            errors.append(analysis.get("error", "Unknown error"))

        return {
            "technical_reports": technical_reports,
            "next_agent": None,
            "errors": errors,
        }

    except Exception as e:
        errors.append(f"Technical agent failed: {str(e)}")
        return {
            "technical_reports": technical_reports,
            "next_agent": None,
            "errors": errors,
        }


# ================== UTILITY FUNCTIONS ==================

def get_supported_indicators() -> dict:
    """Get dictionary of all supported indicators with descriptions."""
    return {
        "trend": TREND_INDICATORS,
        "momentum": MOMENTUM_INDICATORS,
        "volatility": VOLATILITY_INDICATORS,
        "volume": VOLUME_INDICATORS,
        "patterns": PATTERN_INDICATORS,
    }


def validate_data(data: pd.DataFrame) -> bool:
    """Validate that DataFrame has required columns for technical analysis."""
    required = ["open", "high", "low", "close", "volume"]
    return all(col in data.columns for col in required)


def get_indicator_count() -> int:
    """Get total count of supported indicators."""
    return (len(TREND_INDICATORS) + len(MOMENTUM_INDICATORS) +
            len(VOLATILITY_INDICATORS) + len(VOLUME_INDICATORS) +
            len(PATTERN_INDICATORS))
