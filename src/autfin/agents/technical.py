"""Technical analysis agent for AutFin — agentic with tool-calling loop."""

import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import talib

from langgraph.config import RunnableConfig

from src.autfin.state.agent_state import AgentState
from src.autfin.llm.client import get_llm_client
from src.autfin.agents.tools import TECHNICAL_TOOLS, TECHNICAL_CALLABLES

# Configure logging
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


# ================== INDICATOR CALCULATION FUNCTIONS ==================

def calculate_sma(prices: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Simple Moving Average."""
    return talib.SMA(prices, timeperiod=period)


def calculate_multi_sma(prices: pd.Series, periods: list[int] | None = None) -> dict[int, pd.Series]:
    """Calculate SMA for multiple periods (20, 50, 200 by default)."""
    if periods is None:
        periods = [20, 50, 200]
    return {p: talib.SMA(prices, timeperiod=p) for p in periods}


def calculate_ema(prices: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return talib.EMA(prices, timeperiod=period)


def calculate_wma(prices: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Weighted Moving Average."""
    return talib.WMA(prices, timeperiod=period)


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """Calculate MACD indicator."""
    macd, macd_signal, macd_hist = talib.MACD(
        prices, fastperiod=fast, slowperiod=slow, signalperiod=signal
    )
    return {
        "macd": macd,
        "signal": macd_signal,
        "histogram": macd_hist,
    }


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> dict:
    """Calculate ADX indicator."""
    adx = talib.ADX(high, low, close, timeperiod=period)
    plus_di = talib.PLUS_DI(high, low, close, timeperiod=period)
    minus_di = talib.MINUS_DI(high, low, close, timeperiod=period)
    return {
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
    }


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    return talib.RSI(prices, timeperiod=period)


def calculate_stoch(high: pd.Series, low: pd.Series, close: pd.Series,
                    k_period: int = 14, d_period: int = 3) -> dict:
    """Calculate Stochastic Oscillator."""
    slowk, slowd = talib.STOCH(
        high, low, close,
        fastk_period=k_period,
        slowk_period=d_period,
        slowd_period=d_period
    )
    return {
        "slowk": slowk,
        "slowd": slowd,
    }


def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Commodity Channel Index."""
    return talib.CCI(high, low, close, timeperiod=period)


def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Williams %R."""
    return talib.WILLR(high, low, close, timeperiod=period)


def calculate_awesome(high: pd.Series, low: pd.Series) -> pd.Series:
    """Calculate Awesome Oscillator."""
    # AO = (High + Low) / 2 - SMA((High + Low) / 2, 34)
    mid_prices = (high + low) / 2
    sma_34 = talib.SMA(mid_prices, timeperiod=34)
    return mid_prices - sma_34


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, nbdevup: float = 2.0,
                               nbdevdn: float = 2.0) -> dict:
    """Calculate Bollinger Bands."""
    upper, middle, lower = talib.BBANDS(
        prices, timeperiod=period, nbdevup=nbdevup, nbdevdn=nbdevdn
    )
    return {
        "upper": upper,
        "middle": middle,
        "lower": lower,
    }


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    return talib.ATR(high, low, close, timeperiod=period)


def calculate_keltner_channel(high: pd.Series, low: pd.Series, close: pd.Series,
                              period: int = 20, multiplier: float = 2.0) -> dict:
    """Calculate Keltner Channel using ATR."""
    # Keltner Channel = EMA(base) ± multiplier * ATR
    base = talib.EMA(close, timeperiod=period)
    atr = talib.ATR(high, low, close, timeperiod=period)
    return {
        "upper": base + (multiplier * atr),
        "middle": base,
        "lower": base - (multiplier * atr),
    }


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On Balance Volume."""
    return talib.OBV(close, volume)


def calculate_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
                  period: int = 14) -> pd.Series:
    """Calculate Money Flow Index."""
    return talib.MFI(high, low, close, volume, timeperiod=period)


def calculate_vwma(close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Volume Weighted Moving Average."""
    # VWMA = SUM(Close * Volume) / SUM(Volume)
    cumulative_pv = talib.SMA(close * volume, timeperiod=period) * period
    cumulative_v = talib.SMA(volume, timeperiod=period) * period
    return cumulative_pv / cumulative_v


# ================== CANDLESTICK PATTERN DETECTION ==================

def detect_engulfing_pattern(open: pd.Series, close: pd.Series) -> dict:
    """Detect Bullish and Bearish Engulfing patterns."""
    bullish = talib.CDL_ENGULFING(open, close) > 0
    bearish = talib.CDL_ENGULFING(open, close) < 0
    return {
        "engulfing_bullish": bullish,
        "engulfing_bearish": bearish,
    }


def detect_harami_pattern(open: pd.Series, close: pd.Series) -> dict:
    """Detect Bullish and Bearish Harami patterns."""
    bullish = talib.CDL_HARAMI(open, close) > 0
    bearish = talib.CDL_HARAMI(open, close) < 0
    return {
        "harami_bullish": bullish,
        "harami_bearish": bearish,
    }


def detect_piercing_pattern(open: pd.Series, close: pd.Series) -> dict:
    """Detect Piercing pattern (Bullish) and Dark Cloud Cover (Bearish)."""
    bullish = talib.CDL_PIERCING(open, close) > 0
    bearish = talib.CDL_DARKCLOUDCOVER(open, close) > 0
    return {
        "piercing_bullish": bullish,
        "dark_cloud_cover": bearish,
    }


def detect_morning_star(open: pd.Series, close: pd.Series) -> dict:
    """Detect Morning Star and Evening Star patterns."""
    bullish = talib.CDL_MORNINGSTAR(open, close) > 0
    bearish = talib.CDL_EVENINGSTAR(open, close) > 0
    return {
        "morning_star": bullish,
        "evening_star": bearish,
    }


def detect_hammer_patterns(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> dict:
    """Detect Hammer, Hanging Man, Shooting Star, Inverted Hammer patterns."""
    hammer = talib.CDL_HAMMER(open, high, low, close) > 0
    hanging_man = talib.CDL_HANGINGMAN(open, high, low, close) > 0
    shooting_star = talib.CDL_SHOOTINGSTAR(open, high, low, close) > 0
    inverted_hammer = talib.CDL_INVERTEDHAMMER(open, high, low, close) > 0
    return {
        "hammer": hammer,
        "hanging_man": hanging_man,
        "shooting_star": shooting_star,
        "inverted_hammer": inverted_hammer,
    }


def detect_doji_patterns(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> dict:
    """Detect Doji patterns."""
    doji = talib.CDL_DOJI(open, high, low, close) > 0
    dragonfly_doji = talib.CDL_DRAGONFLYDOJI(open, high, low, close) > 0
    gravestone_doji = talib.CDL_GRAVESTONEDOJI(open, high, low, close) > 0
    return {
        "doji": doji,
        "dragonfly_doji": dragonfly_doji,
        "gravestone_doji": gravestone_doji,
    }


def detect_three_candle_patterns(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> dict:
    """Detect three-candle patterns."""
    white_soldiers = talib.CDL_THREEWHITESOLDIERS(open, high, low, close) > 0
    black_crows = talib.CDL_THREEBLACKCROWS(open, high, low, close) > 0
    inside_up = talib.CDL_THREEINSIDEUP(open, high, low, close) > 0
    inside_down = talib.CDL_THREEINSIDEDOWN(open, high, low, close) > 0
    line_strike = talib.CDL_THREELINESTRIKE(open, high, low, close) > 0
    advanced_bloody_kick = talib.CDL_ADVANCEDBLOCK(open, high, low, close) > 0
    return {
        "three_white_soldiers": white_soldiers,
        "three_black_crows": black_crows,
        "three_inside_up": inside_up,
        "three_inside_down": inside_down,
        "three_line_strike": line_strike,
        "advanced_bloody_kick": advanced_bloody_kick,
    }


# ================== MAIN INDICATOR CALCULATION ==================

def calculate_indicators(data: pd.DataFrame) -> dict:
    """
    Calculate all 20+ technical indicators using TA-Lib.
    
    Args:
        data: DataFrame with columns: 'open', 'high', 'low', 'close', 'volume'
    
    Returns:
        Dictionary containing all calculated indicators
    """
    if data.empty:
        raise ValueError("Input data is empty")
    
    # Validate required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    indicators = {}
    
    open_prices = data['open']
    high_prices = data['high']
    low_prices = data['low']
    close_prices = data['close']
    volumes = data['volume']
    
    try:
        # Trend Indicators (6)
        indicators['sma'] = calculate_sma(close_prices)
        indicators['multi_sma'] = calculate_multi_sma(close_prices)
        indicators['ema'] = calculate_ema(close_prices)
        indicators['wma'] = calculate_wma(close_prices)
        indicators['macd'] = calculate_macd(close_prices)
        indicators['adx'] = calculate_adx(high_prices, low_prices, close_prices)
        
        # Momentum Indicators (5)
        indicators['rsi'] = calculate_rsi(close_prices)
        indicators['stoch'] = calculate_stoch(high_prices, low_prices, close_prices)
        indicators['cci'] = calculate_cci(high_prices, low_prices, close_prices)
        indicators['williams_r'] = calculate_williams_r(high_prices, low_prices, close_prices)
        indicators['ao'] = calculate_awesome(high_prices, low_prices)
        
        # Volatility Indicators (3)
        indicators['bollinger_bands'] = calculate_bollinger_bands(close_prices)
        indicators['atr'] = calculate_atr(high_prices, low_prices, close_prices)
        indicators['keltner_channel'] = calculate_keltner_channel(high_prices, low_prices, close_prices)
        
        # Volume Indicators (3)
        indicators['obv'] = calculate_obv(close_prices, volumes)
        indicators['mfi'] = calculate_mfi(high_prices, low_prices, close_prices, volumes)
        indicators['vwma'] = calculate_vwma(close_prices, volumes)
        
        # Pattern Recognition (20+ patterns)
        indicators['patterns'] = {
            # 2-candle patterns
            **detect_engulfing_pattern(open_prices, close_prices),
            **detect_harami_pattern(open_prices, close_prices),
            **detect_piercing_pattern(open_prices, close_prices),
            **detect_morning_star(open_prices, close_prices),
            # 1-candle patterns
            **detect_hammer_patterns(open_prices, high_prices, low_prices, close_prices),
            **detect_doji_patterns(open_prices, high_prices, low_prices, close_prices),
            # 3-candle patterns
            **detect_three_candle_patterns(open_prices, high_prices, low_prices, close_prices),
        }
        
        logger.info(f"Calculated {len(indicators)} indicator groups")
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        raise
    
    return indicators


# ================== SIGNAL GENERATION ==================

def generate_signal(indicators: dict) -> dict:
    """
    Generate buy/sell/hold signals based on indicator combinations.
    
    Args:
        indicators: Dictionary of calculated indicators
    
    Returns:
        Dictionary containing signal, confidence, and reasoning
    """
    signal_data = {
        "signal": "HOLD",
        "confidence": 0.0,
        "reasoning": [],
        "indicator_scores": {},
        "pattern_signals": [],
    }
    
    total_score = 0
    max_score = 0
    
    # Category-specific scores
    trend_score = 0
    trend_max = 0
    momentum_score = 0
    momentum_max = 0
    volatility_score = 0
    volatility_max = 0
    
    # Trend indicators scoring
    if 'sma' in indicators and 'ema' in indicators:
        sma = indicators['sma'].iloc[-1] if not pd.isna(indicators['sma'].iloc[-1]) else None
        ema = indicators['ema'].iloc[-1] if not pd.isna(indicators['ema'].iloc[-1]) else None
        # Use close_prices if provided in indicators, else fall back to SMA (legacy)
        close_series = indicators.get('close_prices', indicators.get('sma', pd.Series())).dropna()
        price = close_series.iloc[-1] if not close_series.empty else None

        if sma and ema and price is not None:
            trend_max += 2
            if price > max(sma, ema):
                trend_score += 1
                signal_data["reasoning"].append("Price above SMA and EMA (Bullish)")
            elif price < min(sma, ema):
                trend_score -= 1
                signal_data["reasoning"].append("Price below SMA and EMA (Bearish)")

    # Multi-period SMA analysis (20/50/200)
    if 'multi_sma' in indicators:
        multi_sma = indicators['multi_sma']
        sma20 = multi_sma.get(20)
        sma50 = multi_sma.get(50)
        sma200 = multi_sma.get(200)
        
        # Get current price from close prices (prefer actual close_prices over SMA)
        close_prices = indicators.get('close_prices', indicators.get('sma', pd.Series())).dropna()
        price = close_prices.iloc[-1] if not close_prices.empty else None
        
        if price is not None:
            # Check for golden cross (SMA50 crosses above SMA200)
            if sma50 is not None and sma200 is not None:
                sma50_curr = sma50.iloc[-1] if not pd.isna(sma50.iloc[-1]) else None
                sma50_prev = sma50.iloc[-2] if len(sma50) > 1 and not pd.isna(sma50.iloc[-2]) else None
                sma200_curr = sma200.iloc[-1] if not pd.isna(sma200.iloc[-1]) else None
                sma200_prev = sma200.iloc[-2] if len(sma200) > 1 and not pd.isna(sma200.iloc[-2]) else None
                
                if all(v is not None for v in [sma50_curr, sma50_prev, sma200_curr, sma200_prev]):
                    trend_max += 2
                    if sma50_prev <= sma200_prev and sma50_curr > sma200_curr:
                        trend_score += 2
                        signal_data["reasoning"].append("Golden Cross (SMA50 > SMA200) — Strong bullish")
                    elif sma50_prev >= sma200_prev and sma50_curr < sma200_curr:
                        trend_score -= 2
                        signal_data["reasoning"].append("Death Cross (SMA50 < SMA200) — Strong bearish")
                    elif sma50_curr > sma200_curr:
                        trend_score += 1
                        signal_data["reasoning"].append("SMA50 above SMA200 (Bullish trend)")
                    else:
                        trend_score -= 1
                        signal_data["reasoning"].append("SMA50 below SMA200 (Bearish trend)")
            
            # Price position relative to SMAs
            if sma20 is not None:
                sma20_val = sma20.iloc[-1] if not pd.isna(sma20.iloc[-1]) else None
                if sma20_val is not None:
                    trend_max += 1
                    if price > sma20_val:
                        trend_score += 1
                        signal_data["reasoning"].append("Price above SMA20 (Short-term bullish)")
                    else:
                        trend_score -= 1
                        signal_data["reasoning"].append("Price below SMA20 (Short-term bearish)")
    
    # MACD
    if 'macd' in indicators and indicators['macd']:
        macd = indicators['macd']['macd'].iloc[-1] if not pd.isna(indicators['macd']['macd'].iloc[-1]) else None
        signal_line = indicators['macd']['signal'].iloc[-1] if not pd.isna(indicators['macd']['signal'].iloc[-1]) else None
        
        if macd is not None and signal_line is not None:
            trend_max += 2
            if macd > signal_line:
                trend_score += 1
                signal_data["reasoning"].append("MACD above signal line (Bullish)")
            elif macd < signal_line:
                trend_score -= 1
                signal_data["reasoning"].append("MACD below signal line (Bearish)")
    
    # ADX
    if 'adx' in indicators and indicators['adx']:
        adx = indicators['adx']['adx'].iloc[-1] if not pd.isna(indicators['adx']['adx'].iloc[-1]) else None
        
        if adx is not None:
            trend_max += 1
            if adx > 25:
                signal_data["reasoning"].append(f"Strong trend (ADX: {adx:.2f})")
            else:
                signal_data["reasoning"].append(f"Weak trend (ADX: {adx:.2f})")
    
    # RSI
    if 'rsi' in indicators:
        rsi = indicators['rsi'].iloc[-1] if not pd.isna(indicators['rsi'].iloc[-1]) else None
        
        if rsi is not None:
            momentum_max += 2
            if rsi > 70:
                momentum_score -= 1
                signal_data["reasoning"].append(f"RSI overbought ({rsi:.2f}) - Potential sell signal")
            elif rsi < 30:
                momentum_score += 1
                signal_data["reasoning"].append(f"RSI oversold ({rsi:.2f}) - Potential buy signal")
            else:
                signal_data["reasoning"].append(f"RSI neutral ({rsi:.2f})")
    
    # Stochastic
    if 'stoch' in indicators and indicators['stoch']:
        stoch_k = indicators['stoch']['slowk'].iloc[-1] if not pd.isna(indicators['stoch']['slowk'].iloc[-1]) else None
        stoch_d = indicators['stoch']['slowd'].iloc[-1] if not pd.isna(indicators['stoch']['slowd'].iloc[-1]) else None
        
        if stoch_k is not None and stoch_d is not None:
            momentum_max += 2
            if stoch_k > stoch_d and stoch_k < 20:
                momentum_score += 1
                signal_data["reasoning"].append("Stochastic crossover in oversold zone (Bullish)")
            elif stoch_k < stoch_d and stoch_k > 80:
                momentum_score -= 1
                signal_data["reasoning"].append("Stochastic crossover in overbought zone (Bearish)")
    
    # Bollinger Bands
    if 'bollinger_bands' in indicators:
        upper = indicators['bollinger_bands']['upper'].iloc[-1] if not pd.isna(indicators['bollinger_bands']['upper'].iloc[-1]) else None
        lower = indicators['bollinger_bands']['lower'].iloc[-1] if not pd.isna(indicators['bollinger_bands']['lower'].iloc[-1]) else None
        middle = indicators['bollinger_bands']['middle'].iloc[-1] if not pd.isna(indicators['bollinger_bands']['middle'].iloc[-1]) else None
        # Use close price from the data instead of DataFrame index
        bb_close = indicators['bollinger_bands']['upper'].dropna()
        current_price = bb_close.iloc[-1] if not bb_close.empty else None
        
        if upper and lower and middle and current_price:
            volatility_max += 2
            if current_price > upper:
                volatility_score -= 1
                signal_data["reasoning"].append("Price above upper Bollinger Band (Overbought)")
            elif current_price < lower:
                volatility_score += 1
                signal_data["reasoning"].append("Price below lower Bollinger Band (Oversold)")
            elif lower < current_price < upper:
                signal_data["reasoning"].append("Price within Bollinger Bands")
    
    # ATR - Volatility
    if 'atr' in indicators:
        atr = indicators['atr'].iloc[-1] if not pd.isna(indicators['atr'].iloc[-1]) else None
        
        if atr is not None:
            volatility_max += 1
            signal_data["reasoning"].append(f"ATR volatility: {atr:.2f}")
    
    # Pattern signals
    pattern_score = 0
    pattern_max = 0
    if 'patterns' in indicators and indicators['patterns']:
        patterns = indicators['patterns']
        
        bullish_patterns = [
            'engulfing_bullish', 'harami_bullish', 'piercing_bullish',
            'morning_star', 'hammer', 'inverted_hammer', 'dragonfly_doji',
            'three_white_soldiers', 'three_inside_up', 'three_line_strike'
        ]
        
        bearish_patterns = [
            'engulfing_bearish', 'harami_bearish', 'dark_cloud_cover',
            'evening_star', 'hanging_man', 'shooting_star', 'gravestone_doji',
            'three_black_crows', 'three_inside_down', 'advanced_bloody_kick'
        ]
        
        for pattern_name in bullish_patterns:
            if pattern_name in patterns and patterns[pattern_name].iloc[-1]:
                pattern_score += 1
                pattern_max += 1
                signal_data["pattern_signals"].append({
                    "type": "bullish",
                    "pattern": PATTERN_INDICATORS.get(pattern_name, pattern_name),
                })
        
        for pattern_name in bearish_patterns:
            if pattern_name in patterns and patterns[pattern_name].iloc[-1]:
                pattern_score -= 1
                pattern_max += 1
                signal_data["pattern_signals"].append({
                    "type": "bearish",
                    "pattern": PATTERN_INDICATORS.get(pattern_name, pattern_name),
                })
    
    # Aggregate category scores into total
    total_score = trend_score + momentum_score + volatility_score + pattern_score
    max_score = trend_max + momentum_max + volatility_max + pattern_max
    
    # Calculate final signal
    if max_score > 0:
        confidence = abs(total_score) / max_score
        
        if total_score > 0.5:
            signal_data["signal"] = "BUY"
        elif total_score < -0.5:
            signal_data["signal"] = "SELL"
        else:
            signal_data["signal"] = "HOLD"
        
        signal_data["confidence"] = round(confidence, 2)
    
    # Add indicator scores (category-specific)
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
    """Performs technical analysis using 20+ TA-Lib indicators."""
    
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
        """
        Perform comprehensive technical analysis for a symbol.
        
        Args:
            symbol: Trading symbol to analyze
            data: Optional price data DataFrame. If not provided, uses market data.
        
        Returns:
            Dictionary containing all technical indicators and signals
        """
        try:
            if data is None:
                raise ValueError("Price data must be provided for analysis")
            
            # Calculate all indicators
            indicators = calculate_indicators(data)
            
            # Generate signals
            signal = generate_signal(indicators)
            
            # Build technical report
            report = {
                "symbol": symbol,
                "timestamp": pd.Timestamp.now().isoformat(),
                "indicators": indicators,
                "signal": signal,
                "indicators_count": len(self.indicator_config),
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Technical analysis failed for {symbol}: {str(e)}")
            return {
                "symbol": symbol,
                "error": str(e),
                "timestamp": pd.Timestamp.now().isoformat(),
            }
    
    async def generate_signals(self, symbol: str, data: Optional[pd.DataFrame] = None) -> dict:
        """
        Generate trading signals based on technical analysis.
        
        Args:
            symbol: Trading symbol
            data: Optional price data DataFrame
        
        Returns:
            Dictionary containing signal, confidence, and reasoning
        """
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
            max_tokens=2048,
        )
        if not raw:
            return None

        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"Technical agentic output not JSON for {symbol}")
            return None


# ================== TECHNICAL AGENT ASYNC WORKER ==================

async def technical_agent(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """
    Orchestrate technical analysis workflow.
    
    Args:
        state: Current agent state containing market data
    
    Returns:
        Updated state with technical analysis results
    """
    errors = state.get("errors", [])
    technical_reports = state.get("technical_reports", {})
    
    try:
        # Get market snapshot from state
        market_snapshot = state.get("market_snapshot", {})
        quotes = market_snapshot.get("quotes", {})
        
        if not quotes:
            errors.append("No market data available for technical analysis")
            return {
                **state,
                "technical_reports": technical_reports,
                "errors": errors,
            }
        
        # Get symbol to analyze (first available)
        symbol = next(iter(quotes), None)
        
        if symbol is None:
            errors.append("No symbols available for technical analysis")
            return {
                **state,
                "technical_reports": technical_reports,
                "errors": errors,
            }
        
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
            logger.warning("yfinance failed for %s: %s — falling back to quote data", symbol, yf_exc)

        # yfinance unavailable — return error, do not fabricate fake data
        if price_data is None or price_data.empty:
            technical_reports[symbol] = {
                "error": "yfinance_ohlcv_unavailable",
                "message": f"Cannot analyze {symbol} — OHLCV data unavailable from yfinance",
                "symbol": symbol,
                "status": "error",
            }
            return {**state, "technical_reports": technical_reports, "errors": errors}

        # Perform technical analysis
        agent = TechnicalAgent()
        analysis = await agent._agentic_analyze(symbol)
        if analysis is None:
            analysis = await agent.analyze(symbol, price_data)
        
        if "error" not in analysis:
            technical_reports[symbol] = analysis
        else:
            errors.append(analysis.get("error", "Unknown error"))
        
        return {
            **state,
            "technical_reports": technical_reports,
            "next_agent": None,  # Will be set by supervisor
            "errors": errors,
        }
        
    except Exception as e:
        errors.append(f"Technical agent failed: {str(e)}")
        return {
            **state,
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
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    return all(col in data.columns for col in required_columns)


def get_indicator_count() -> int:
    """Get total count of supported indicators."""
    return (len(TREND_INDICATORS) + len(MOMENTUM_INDICATORS) +
            len(VOLATILITY_INDICATORS) + len(VOLUME_INDICATORS) +
            len(PATTERN_INDICATORS))
