"""
Near-real-time market data feed for AutFin v3.0.

Free approach:
- Kite SDK holdings poll (every 2s) — includes LTP for owned stocks
- yfinance fast_info poll (every 30s) — for watchlist and indices

Google Finance: CONFIRMED BLOCKED (302 redirect for all NSE symbols).

The feed is an in-memory singleton that agents read from.
"""
from __future__ import annotations
import asyncio, logging, time
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# In-memory price store
_price_cache: dict[str, dict] = {}
_watchlist: list[str] = []
_last_poll: float = 0
_running: bool = False


def get_price(symbol: str) -> Optional[dict[str, Any]]:
    """Get the latest cached price for a symbol."""
    entry = _price_cache.get(symbol.upper())
    if entry:
        age = time.monotonic() - entry.get("polled_at", 0)
        if age < 120:  # Cache valid for 2 minutes
            return entry
    return None


def get_watchlist_prices() -> dict[str, dict]:
    """Get all cached watchlist prices."""
    now = time.monotonic()
    return {
        sym: data for sym, data in _price_cache.items()
        if now - data.get("polled_at", 0) < 120
    }


def add_to_watchlist(symbols: list[str]):
    """Add symbols to the polling watchlist."""
    global _watchlist
    for s in symbols:
        if s.upper() not in _watchlist:
            _watchlist.append(s.upper())


def remove_from_watchlist(symbol: str):
    global _watchlist
    _watchlist = [s for s in _watchlist if s != symbol.upper()]


async def _poll_kite():
    """Poll Kite holdings for LTP every 2 seconds."""
    try:
        from src.autfin.tools.kite_tools import create_kite_tools
        kite = create_kite_tools()
        holdings = await kite.get_holdings()
        if holdings.get("status") == "success":
            now = time.monotonic()
            for h in holdings.get("data", []):
                sym = h.get("tradingsymbol", "")
                if sym:
                    _price_cache[sym] = {
                        "ltp": h.get("last_price", 0),
                        "change_pct": h.get("pnl", 0),
                        "volume": 0,
                        "source": "kite",
                        "polled_at": now,
                    }
    except Exception as e:
        logger.debug("Kite poll failed: %s", e)


async def _poll_yfinance():
    """Poll yfinance for watchlist prices every 30 seconds."""
    if not _watchlist:
        return
    try:
        import yfinance as yf
        now = time.monotonic()
        for sym in _watchlist[:20]:  # Cap at 20 per poll
            try:
                t = yf.Ticker(f"{sym.removesuffix('.NS')}.NS")
                price = t.fast_info.get("lastPrice", 0)
                if price and price > 0:
                    _price_cache[sym] = {
                        "ltp": price,
                        "change_pct": 0,
                        "volume": 0,
                        "source": "yfinance",
                        "polled_at": now,
                    }
            except Exception:
                pass  # Skip failing symbols
    except Exception as e:
        logger.debug("yfinance poll failed: %s", e)


async def _feed_loop():
    """Main feed loop — alternates between Kite and yfinance polls."""
    global _running, _last_poll
    kite_interval = 2.0
    yf_interval = 30.0
    last_yf = 0.0
    
    while _running:
        try:
            await _poll_kite()
            await asyncio.sleep(kite_interval)
            
            now = time.monotonic()
            if now - last_yf >= yf_interval:
                await _poll_yfinance()
                last_yf = now
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning("Feed loop error: %s", e)
            await asyncio.sleep(5)


async def start_feed(watchlist: list[str] = None) -> bool:
    """Start the market data feed in the background."""
    global _running, _watchlist
    if _running:
        return True
    
    if watchlist:
        add_to_watchlist(watchlist)
    
    _running = True
    asyncio.ensure_future(_feed_loop())
    logger.info("Market feed started (Kite 2s + yfinance 30s)")
    return True


def stop_feed():
    """Stop the market data feed."""
    global _running
    _running = False
    logger.info("Market feed stopped")


def get_feed_status() -> dict:
    """Get feed status for dashboard."""
    return {
        "running": _running,
        "cached_symbols": len(_price_cache),
        "watchlist": _watchlist[:10],
        "last_poll": _last_poll,
        "sample": {
            sym: {"ltp": data["ltp"], "source": data["source"]}
            for sym, data in list(_price_cache.items())[:5]
        },
    }
