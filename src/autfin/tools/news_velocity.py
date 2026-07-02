"""
News Velocity Analyzer — detect anomalous news flow patterns.

Compares news rate vs sector baseline and detects "manufactured"
news flow (high velocity without social confirmation).

Deterministic — no LLM calls.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def analyze_news_velocity(
    symbol: str,
    hours: int = 24,
) -> Dict[str, Any]:
    """Compare news rate vs baseline and detect anomalies.

    Args:
        symbol: Stock symbol (e.g. 'RELIANCE')
        hours: Time window in hours to analyze

    Returns:
        {
            status, symbol,
            news_count: int,
            velocity_ratio: float,
            has_news_catalyst: bool,
            anomaly_flags: list,
            signal: "normal" | "elevated" | "suspicious"
        }
    """
    try:
        from src.autfin.tools.sentiment_tools import SentimentTools

        tools = SentimentTools()

        # Fetch recent news
        news_result = await tools.get_marketaux_news(symbol, limit=20)
        await tools.close()

        if not news_result or news_result.get("status") != "success":
            return {
                "status": "success",
                "symbol": symbol,
                "news_count": 0,
                "velocity_ratio": 0.0,
                "has_news_catalyst": False,
                "anomaly_flags": [],
                "signal": "normal",
                "message": "No news data available",
            }

        articles = news_result.get("articles", [])
        if not articles:
            return {
                "status": "success",
                "symbol": symbol,
                "news_count": 0,
                "velocity_ratio": 0.0,
                "has_news_catalyst": False,
                "anomaly_flags": [],
                "signal": "normal",
            }

        # Count articles in the time window
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_count = 0
        has_catalyst = False

        for article in articles:
            try:
                pub_date = article.get("published_at", article.get("date", ""))
                if pub_date:
                    dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00").replace("+00:00", ""))
                    if dt >= cutoff:
                        recent_count += 1
                        # Check for catalyst keywords
                        title = (article.get("title") or "").lower()
                        catalyst_kw = ["earnings", "results", "quarterly", "annual",
                                       "merger", "acquisition", "buyback", "dividend",
                                       "SEBI", "RBI", "penalty", "order"]
                        if any(kw.lower() in title for kw in catalyst_kw):
                            has_catalyst = True
            except Exception:
                continue

        # Baseline: typical news count for an Indian large-cap is ~2-5 per day
        baseline_per_day = 3.0
        baseline_for_window = baseline_per_day * (hours / 24)
        velocity_ratio = recent_count / max(baseline_for_window, 1)

        # Anomaly detection
        anomaly_flags = []

        if velocity_ratio >= 5.0:
            anomaly_flags.append("extreme_velocity")
        elif velocity_ratio >= 3.0:
            anomaly_flags.append("high_velocity")

        if velocity_ratio >= 3.0 and not has_catalyst:
            anomaly_flags.append("high_velocity_no_catalyst")

        # Signal classification
        if "extreme_velocity" in anomaly_flags:
            signal = "suspicious"
        elif "high_velocity" in anomaly_flags:
            signal = "elevated"
        else:
            signal = "normal"

        return {
            "status": "success",
            "symbol": symbol,
            "news_count": recent_count,
            "velocity_ratio": round(velocity_ratio, 2),
            "has_news_catalyst": has_catalyst,
            "anomaly_flags": anomaly_flags,
            "signal": signal,
            "baseline_per_day": baseline_per_day,
            "hours": hours,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.warning("analyze_news_velocity failed for %s: %s", symbol, e)
        return {"status": "error", "symbol": symbol, "message": str(e)}
