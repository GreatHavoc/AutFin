"""
Price-Distortion Detection — deterministic analyzers for market anomalies.

Detects fake breakouts, pump patterns, distribution signals, and other
anomalies that pure technical analysis misses.

All functions are deterministic — no LLM calls.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------


def detect_fake_breakout(
    price_data: pd.DataFrame,
    lookback: int = 20,
    volume_threshold: float = 1.2,
) -> Dict[str, Any]:
    """Detect price breakouts without volume confirmation.

    A breakout above the N-day high with volume < threshold × average
    is suspicious — likely low-conviction or engineered.

    Args:
        price_data: DataFrame with 'close', 'high', 'volume' columns
        lookback: Number of days for the high/avg-volume lookback
        volume_threshold: Minimum volume ratio for a "real" breakout

    Returns:
        {flag: bool, severity, reason, details}
    """
    try:
        if len(price_data) < lookback + 1:
            return {"flag": False, "severity": "none", "reason": "Insufficient data"}

        recent = price_data.tail(lookback)
        prior = price_data.iloc[-(lookback + 1):-1]

        current_close = float(price_data["close"].iloc[-1])
        prior_high = float(prior["high"].max())
        current_volume = float(price_data["volume"].iloc[-1])
        avg_volume = float(recent["volume"].mean())

        broke_above = current_close > prior_high
        vol_ratio = current_volume / max(avg_volume, 1)

        if broke_above and vol_ratio < volume_threshold:
            severity = "high" if vol_ratio < 0.8 else "medium"
            return {
                "flag": True,
                "severity": severity,
                "reason": f"Breakout above {lookback}D high (₹{prior_high:,.1f}) with weak volume ({vol_ratio:.1f}x avg)",
                "details": {
                    "current_close": current_close,
                    "prior_high": prior_high,
                    "volume_ratio": vol_ratio,
                    "avg_volume": avg_volume,
                },
            }
        return {"flag": False, "severity": "none", "reason": "No fake breakout detected"}
    except Exception as e:
        logger.warning("detect_fake_breakout error: %s", e)
        return {"flag": False, "severity": "none", "reason": f"Error: {e}"}


def detect_distribution(
    price_data: pd.DataFrame,
    lookback_52w: int = 252,
    obv_lookback: int = 20,
) -> Dict[str, Any]:
    """Detect institutional distribution (selling into strength).

    Signals: near 52-week high + OBV falling + price making higher highs
    while OBV makes lower highs = classic distribution pattern.

    Args:
        price_data: DataFrame with 'close', 'high', 'low', 'volume' columns
        lookback_52w: Trading days for 52-week high calculation
        obv_lookback: Days for OBV trend comparison

    Returns:
        {flag: bool, severity, reason, details}
    """
    try:
        if len(price_data) < max(lookback_52w, obv_lookback) + 1:
            return {"flag": False, "severity": "none", "reason": "Insufficient data"}

        close = price_data["close"]
        volume = price_data["volume"]

        # Calculate OBV (On-Balance Volume)
        obv = (volume * close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))).cumsum()

        # Check if near 52-week high
        high_52w = float(close.tail(min(lookback_52w, len(close))).max())
        current = float(close.iloc[-1])
        pct_from_high = (current - high_52w) / high_52w * 100 if high_52w > 0 else 0
        near_high = pct_from_high >= -5  # within 5% of 52w high

        # Check OBV divergence: price trending up, OBV trending down
        price_trend = float(close.tail(obv_lookback).iloc[-1]) - float(close.tail(obv_lookback).iloc[0])
        obv_trend = float(obv.tail(obv_lookback).iloc[-1]) - float(obv.tail(obv_lookback).iloc[0])

        price_up = price_trend > 0
        obv_down = obv_trend < 0
        divergence = price_up and obv_down

        if near_high and divergence:
            return {
                "flag": True,
                "severity": "high",
                "reason": f"Distribution signal: near 52w high ({pct_from_high:+.1f}%) with OBV declining — institutional selling into strength",
                "details": {
                    "pct_from_52w_high": pct_from_high,
                    "price_trend": price_trend,
                    "obv_trend": obv_trend,
                    "high_52w": high_52w,
                },
            }
        return {"flag": False, "severity": "none", "reason": "No distribution detected"}
    except Exception as e:
        logger.warning("detect_distribution error: %s", e)
        return {"flag": False, "severity": "none", "reason": f"Error: {e}"}


def detect_pump_pattern(
    price_data: pd.DataFrame,
    volume_spike_threshold: float = 5.0,
    price_move_threshold: float = 5.0,
) -> Dict[str, Any]:
    """Detect pump-and-dump patterns.

    Signals: volume spike >5x average + large price move + low prior volume
    = classic pump setup.

    Args:
        price_data: DataFrame with 'close', 'volume' columns
        volume_spike_threshold: Volume ratio to flag as spike
        price_move_threshold: % move to flag as suspicious

    Returns:
        {flag: bool, severity, reason, details}
    """
    try:
        if len(price_data) < 21:
            return {"flag": False, "severity": "none", "reason": "Insufficient data"}

        close = price_data["close"]
        volume = price_data["volume"]

        avg_vol_20d = float(volume.iloc[-21:-1].mean())
        current_vol = float(volume.iloc[-1])
        vol_ratio = current_vol / max(avg_vol_20d, 1)

        daily_return = (float(close.iloc[-1]) / float(close.iloc[-2]) - 1) * 100 if float(close.iloc[-2]) > 0 else 0
        abs_return = abs(daily_return)

        # Pump: volume spike + large move
        if vol_ratio >= volume_spike_threshold and abs_return >= price_move_threshold:
            return {
                "flag": True,
                "severity": "high",
                "reason": f"Pump pattern: volume {vol_ratio:.1f}x avg + {daily_return:+.1f}% move in one session",
                "details": {
                    "volume_ratio": vol_ratio,
                    "daily_return_pct": daily_return,
                    "avg_volume_20d": avg_vol_20d,
                    "current_volume": current_vol,
                },
            }
        return {"flag": False, "severity": "none", "reason": "No pump pattern detected"}
    except Exception as e:
        logger.warning("detect_pump_pattern error: %s", e)
        return {"flag": False, "severity": "none", "reason": f"Error: {e}"}


def detect_circuit_suspicion(
    price_data: pd.DataFrame,
    circuit_pct: float = 20.0,
    min_hits: int = 2,
) -> Dict[str, Any]:
    """Detect repeated circuit hits without clear catalyst.

    If a stock hits upper circuit multiple sessions in a row without
    a clear news catalyst, it may be thin-float momentum trading.

    Args:
        price_data: DataFrame with 'close', 'open' columns
        circuit_pct: Circuit limit percentage (default 20% for NSE)
        min_hits: Minimum circuit hits to flag

    Returns:
        {flag: bool, severity, reason, details}
    """
    try:
        if len(price_data) < 5:
            return {"flag": False, "severity": "none", "reason": "Insufficient data"}

        close = price_data["close"]
        circuit_hits = 0
        for i in range(-1, -min(len(close), 6), -1):
            prev_close = float(close.iloc[i - 1]) if abs(i) < len(close) else float(close.iloc[0])
            curr_close = float(close.iloc[i])
            if prev_close > 0:
                pct_change = (curr_close / prev_close - 1) * 100
                if pct_change >= circuit_pct * 0.95:  # within 95% of circuit
                    circuit_hits += 1
                else:
                    break
            else:
                break

        if circuit_hits >= min_hits:
            return {
                "flag": True,
                "severity": "high",
                "reason": f"Hit upper circuit {circuit_hits} sessions in a row — thin-float momentum risk",
                "details": {"circuit_hits": circuit_hits},
            }
        return {"flag": False, "severity": "none", "reason": "No circuit suspicion"}
    except Exception as e:
        logger.warning("detect_circuit_suspicion error: %s", e)
        return {"flag": False, "severity": "none", "reason": f"Error: {e}"}


def detect_sector_relative_weakness(
    stock_return_20d: float,
    sector_return_20d: float,
    threshold: float = -3.0,
) -> Dict[str, Any]:
    """Detect if stock is underperforming its sector despite a rally.

    A stock up 3% when its sector is up 7% is a laggard — not worth
    entering even if it looks "cheap" on fundamentals.

    Args:
        stock_return_20d: Stock's 20-day return %
        sector_return_20d: Sector's 20-day return %
        threshold: Max underperformance before flagging (negative number)

    Returns:
        {flag: bool, severity, reason, details}
    """
    try:
        underperformance = stock_return_20d - sector_return_20d

        if underperformance <= threshold and sector_return_20d > 0:
            return {
                "flag": True,
                "severity": "medium",
                "reason": f"Laggard: stock {stock_return_20d:+.1f}% vs sector {sector_return_20d:+.1f}% ({underperformance:+.1f}% underperformance)",
                "details": {
                    "stock_return_20d": stock_return_20d,
                    "sector_return_20d": sector_return_20d,
                    "underperformance": underperformance,
                },
            }
        return {"flag": False, "severity": "none", "reason": "No relative weakness"}
    except Exception as e:
        logger.warning("detect_sector_relative_weakness error: %s", e)
        return {"flag": False, "severity": "none", "reason": f"Error: {e}"}


def check_promoter_pledge(
    current_pledge_pct: float,
    prior_pledge_pct: float,
    threshold: float = 5.0,
) -> Dict[str, Any]:
    """Check if promoter pledge % increased significantly.

    Rising promoter pledge is a distribution/stress signal — the promoter
    is borrowing against their shares, which can lead to forced selling.

    Args:
        current_pledge_pct: Current promoter pledge %
        prior_pledge_pct: Prior period promoter pledge %
        threshold: Minimum increase to flag

    Returns:
        {flag: bool, severity, reason, details}
    """
    try:
        increase = current_pledge_pct - prior_pledge_pct

        if increase >= threshold:
            severity = "high" if increase >= 10 else "medium"
            return {
                "flag": True,
                "severity": severity,
                "reason": f"Promoter pledge increased by {increase:.1f}pp ({prior_pledge_pct:.1f}% → {current_pledge_pct:.1f}%) — stress signal",
                "details": {
                    "current_pledge_pct": current_pledge_pct,
                    "prior_pledge_pct": prior_pledge_pct,
                    "increase": increase,
                },
            }
        return {"flag": False, "severity": "none", "reason": "No pledge increase"}
    except Exception as e:
        logger.warning("check_promoter_pledge error: %s", e)
        return {"flag": False, "severity": "none", "reason": f"Error: {e}"}


# ---------------------------------------------------------------------------
# Composite distortion audit
# ---------------------------------------------------------------------------


async def distortion_audit(
    symbol: str,
    depth: str = "standard",
) -> Dict[str, Any]:
    """Run all distortion detectors on a stock.

    Fetches OHLCV data via yfinance, then runs all detectors.
    Returns composite result with conviction adjustment.

    Args:
        symbol: Stock symbol (e.g. 'RELIANCE')
        depth: quick (fewer checks), standard (all), deep (all + extended)

    Returns:
        {
            status, symbol, flags: [...], conviction_adjustment: float,
            summary: str
        }
    """
    try:
        import yfinance as yf

        period_map = {"quick": "3mo", "standard": "6mo", "deep": "1y"}
        period = period_map.get(depth, "6mo")

        ticker = yf.Ticker(f"{symbol.removesuffix('.NS')}.NS")
        hist = ticker.history(period=period, interval="1d")

        if hist is None or hist.empty:
            return {
                "status": "error",
                "symbol": symbol,
                "message": f"OHLCV data unavailable for {symbol}",
            }

        price_data = hist.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })[["open", "high", "low", "close", "volume"]]

        flags = []

        # 1. Fake breakout detection
        fb = detect_fake_breakout(price_data)
        if fb["flag"]:
            flags.append({"detector": "fake_breakout", **fb})

        # 2. Distribution detection
        dist = detect_distribution(price_data)
        if dist["flag"]:
            flags.append({"detector": "distribution", **dist})

        # 3. Pump pattern
        pump = detect_pump_pattern(price_data)
        if pump["flag"]:
            flags.append({"detector": "pump_pattern", **pump})

        # 4. Circuit suspicion
        circuit = detect_circuit_suspicion(price_data)
        if circuit["flag"]:
            flags.append({"detector": "circuit_suspicion", **circuit})

        # 5. Sector relative weakness (needs sector data)
        try:
            sector_map = {
                "IT": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
                "Banks": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK"],
                "Auto": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO"],
                "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "AUROPHARMA", "DIVISLAB"],
            }
            stock_sector = None
            stock_sym = symbol.upper().removesuffix(".NS")
            for sector, syms in sector_map.items():
                if stock_sym in syms:
                    stock_sector = sector
                    break

            if stock_sector:
                sector_syms = sector_map[stock_sector]
                stock_ret = (float(price_data["close"].iloc[-1]) / float(price_data["close"].iloc[-21]) - 1) * 100
                # Use a representative sector ETF or average
                sector_ticker = yf.Ticker(f"^{stock_sector.upper()}500.NS" if stock_sector != "IT" else "^CNXIT")
                try:
                    sector_hist = sector_ticker.history(period="1mo", interval="1d")
                    if sector_hist is not None and len(sector_hist) > 1:
                        sector_ret = (float(sector_hist["Close"].iloc[-1]) / float(sector_hist["Close"].iloc[0]) - 1) * 100
                    else:
                        sector_ret = 0.0
                except Exception:
                    sector_ret = 0.0

                srw = detect_sector_relative_weakness(stock_ret, sector_ret)
                if srw["flag"]:
                    flags.append({"detector": "sector_weakness", **srw})
        except Exception:
            pass  # Sector comparison is best-effort

        # Conviction adjustment: -0.1 per flag
        conviction_adj = -0.1 * len(flags)

        high_flags = [f for f in flags if f.get("severity") == "high"]
        medium_flags = [f for f in flags if f.get("severity") == "medium"]

        if high_flags:
            summary = f"WARNING: {len(high_flags)} high-severity distortion(s) detected for {symbol}"
        elif medium_flags:
            summary = f"CAUTION: {len(medium_flags)} medium-severity distortion(s) for {symbol}"
        else:
            summary = f"No distortions detected for {symbol}"

        return {
            "status": "success",
            "symbol": symbol,
            "flags": flags,
            "flag_count": len(flags),
            "high_count": len(high_flags),
            "medium_count": len(medium_flags),
            "conviction_adjustment": conviction_adj,
            "summary": summary,
            "depth": depth,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.warning("distortion_audit failed for %s: %s", symbol, e)
        return {"status": "error", "symbol": symbol, "message": str(e)}
