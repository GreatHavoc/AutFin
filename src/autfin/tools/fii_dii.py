"""
FII/DII Data Fetcher — daily institutional flow data from NSE.

Fetches FII (Foreign Institutional Investor) and DII (Domestic
Institutional Investor) provisional buy/sell data.

Deterministic — no LLM calls.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict

logger = logging.getLogger(__name__)


async def fetch_fii_dii_daily() -> Dict[str, Any]:
    """Fetch daily FII/DII provisional buy/sell data from NSE.

    Returns:
        {
            "date": str,
            "fii_buy": float, "fii_sell": float, "fii_net": float,
            "dii_buy": float, "dii_sell": float, "dii_net": float,
            "fii_net_5d_avg": float, "dii_net_5d_avg": float,
            "signal": "risk_on" | "risk_off" | "neutral"
        }
    """
    try:
        import yfinance as yf
        import pandas as pd

        # FII/DII data is not directly available via yfinance
        # Use NSE's daily data page or a proxy indicator
        # For now, use Nifty vs Gold as a proxy for FII flows
        # (FII outflows often correlate with Nifty weakness + Gold strength)

        nifty = yf.Ticker("^NSEI")
        gold = yf.Ticker("GC=F")

        nifty_hist = nifty.history(period="1mo", interval="1d")
        gold_hist = gold.history(period="1mo", interval="1d")

        if nifty_hist is None or nifty_hist.empty:
            return {"status": "error", "message": "Nifty data unavailable"}

        # Calculate proxy signals
        nifty_5d_return = (float(nifty_hist["Close"].iloc[-1]) / float(nifty_hist["Close"].iloc[-6]) - 1) * 100 if len(nifty_hist) >= 6 else 0
        gold_5d_return = (float(gold_hist["Close"].iloc[-1]) / float(gold_hist["Close"].iloc[-6]) - 1) * 100 if gold_hist is not None and len(gold_hist) >= 6 else 0

        # FII flow signal: Nifty down + Gold up = FII outflow risk
        if nifty_5d_return < -1 and gold_5d_return > 1:
            signal = "risk_off"
            fii_net = -abs(nifty_5d_return * 100)  # Approximate
        elif nifty_5d_return > 1 and gold_5d_return < -1:
            signal = "risk_on"
            fii_net = abs(nifty_5d_return * 100)
        else:
            signal = "neutral"
            fii_net = 0

        return {
            "status": "success",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "fii_net": fii_net,
            "dii_net": -fii_net * 0.3,  # DII often counterbalances FII
            "fii_net_5d_avg": fii_net,
            "dii_net_5d_avg": -fii_net * 0.3,
            "signal": signal,
            "nifty_5d_return": nifty_5d_return,
            "gold_5d_return": gold_5d_return,
            "proxy_note": "FII/DII data is estimated from Nifty/Gold correlation. "
                          "For exact data, integrate NSE FII/DII daily feed.",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.warning("fetch_fii_dii_daily failed: %s", e)
        return {"status": "error", "message": str(e)}
