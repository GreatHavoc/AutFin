"""
Watchlist Curator — deterministic watchlist builder.

Runs BEFORE the supervisor to build today's watchlist from multiple sources:
1. Holdings (from Kite SDK)
2. Manual watchlist (from user input)
3. Pre-earnings (within 14 days, NSE calendar)
4. Large moves >3% (LTP scan)
5. Top movers by volume (NSE data)
6. Macro shock → broaden to ALL holdings

Deterministic — no LLM calls.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WatchlistCurator:
    """Deterministic watchlist builder. Runs BEFORE supervisor."""

    async def curate(self, state: Dict[str, Any]) -> List[str]:
        """Build today's watchlist from multiple sources.

        Args:
            state: Current agent state (holdings, positions, etc.)

        Returns:
            List of stock symbols to analyze today
        """
        watchlist = set()

        # 1. Holdings — always include what we own
        holdings = state.get("holdings", []) or []
        for h in holdings:
            sym = h.get("tradingsymbol") or h.get("symbol")
            if sym:
                watchlist.add(sym.upper().replace(".NS", ""))

        # 2. Manual watchlist (from state)
        manual = state.get("candidates", []) or []
        for sym in manual:
            if sym:
                watchlist.add(sym.upper().replace(".NS", ""))

        # 3. Alert symbol (price alert context)
        alert_sym = state.get("alert_symbol")
        if alert_sym:
            watchlist.add(alert_sym.upper().replace(".NS", ""))

        # 4. Pre-earnings — stocks with earnings in next 14 days
        try:
            pre_earnings = await self._find_pre_earnings(list(watchlist))
            watchlist.update(pre_earnings)
        except Exception as e:
            logger.debug("Pre-earnings scan skipped: %s", e)

        # 5. Top movers — large moves >3%
        try:
            movers = await self._find_top_movers()
            watchlist.update(movers)
        except Exception as e:
            logger.debug("Top movers scan skipped: %s", e)

        # 6. Macro shock — if macro is bearish, include all holdings
        macro = state.get("macro_signals") or {}
        if macro.get("impact_score", "").startswith("high_negative"):
            # Already included all holdings in step 1
            logger.info("Macro shock detected — all holdings already in watchlist")

        # Remove empty strings and sort
        result = sorted([s for s in watchlist if s and len(s) > 0])

        logger.info("Watchlist curated: %d symbols", len(result))
        return result

    async def _find_pre_earnings(self, current_symbols: List[str]) -> List[str]:
        """Find stocks with earnings in the next 14 days.

        Uses a static list of known quarterly result dates.
        In production, this would query NSE/BSE corporate calendar.
        """
        # This is a placeholder — in production, integrate with
        # NSE corporate calendar or MoneyControl earnings calendar
        return []

    async def _find_top_movers(self, threshold_pct: float = 3.0) -> List[str]:
        """Find stocks with large moves (>threshold%) today.

        Uses Nifty 500 universe and checks for significant movers.
        """
        try:
            from src.autfin.tools.universe import NIFTY_500
            import yfinance as yf

            # Check a subset of the universe (top 50 by market cap)
            # to avoid rate limits
            sample = NIFTY_500[:50]

            movers = []
            for sym in sample:
                try:
                    ticker = yf.Ticker(f"{sym}.NS")
                    hist = ticker.history(period="2d", interval="1d")
                    if hist is not None and len(hist) >= 2:
                        prev_close = float(hist["Close"].iloc[-2])
                        curr_close = float(hist["Close"].iloc[-1])
                        if prev_close > 0:
                            pct_change = abs(curr_close / prev_close - 1) * 100
                            if pct_change >= threshold_pct:
                                movers.append(sym)
                except Exception:
                    continue

            return movers[:10]  # Cap at 10
        except ImportError:
            return []
        except Exception as e:
            logger.debug("Top movers scan failed: %s", e)
            return []


async def curate_watchlist(state: Dict[str, Any]) -> List[str]:
    """Convenience function to run the watchlist curator."""
    curator = WatchlistCurator()
    return await curator.curate(state)
