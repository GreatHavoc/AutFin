"""AutFin v7.0 — Kite Tools Wrapper

Wraps existing KiteTools class as async functions for Pydantic AI agents.
"""
from __future__ import annotations

import os

from autfin.tools.kite_tools import KiteTools

# Global singleton
_kite_instance: KiteTools | None = None


def get_kite_instance() -> KiteTools:
    """Get or create KiteTools singleton."""
    global _kite_instance
    if _kite_instance is None:
        _kite_instance = KiteTools(
            api_key=os.getenv("KITE_API_KEY"),
            api_secret=os.getenv("KITE_API_SECRET"),
        )
    return _kite_instance


async def get_portfolio() -> dict:
    """Fetch portfolio holdings from Kite."""
    kite = get_kite_instance()
    result = await kite.get_holdings()
    if result.get("status") == "success":
        return {"holdings": result.get("data", []), "error": None}
    return {"holdings": [], "error": result.get("message", "Unknown error")}


async def get_quotes(symbols: str) -> dict:
    """Fetch real-time quotes for comma-separated symbols."""
    kite = get_kite_instance()
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    
    # Format symbols for Kite (NSE:SYMBOL)
    instruments = [f"NSE:{sym}" for sym in symbol_list]
    
    result = await kite.get_quotes(instruments)
    if result.get("status") == "success":
        return {"quotes": result.get("data", {}), "error": None}
    return {"quotes": {}, "error": result.get("message", "Unknown error")}


async def get_market_status() -> dict:
    """Get current market status (from NSE API)."""
    import httpx
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("https://www.nseindia.com/api/marketStatus")
            if resp.status_code == 200:
                data = resp.json()
                market_state = data.get("marketState", [])
                is_open = any(m.get("status") == "Open" for m in market_state)
                
                phase = "closed"
                if is_open:
                    phase = "open"
                elif any(m.get("status") == "Pre-Open" for m in market_state):
                    phase = "pre_market"
                elif any(m.get("status") == "Post-Close" for m in market_state):
                    phase = "post_market"
                
                return {
                    "is_open": is_open,
                    "phase": phase,
                    "market_state": market_state,
                }
    except Exception:
        pass
    
    # Fallback based on time
    from datetime import datetime, time
    now = datetime.now().time()
    if time(9, 15) <= now <= time(15, 30):
        phase = "open"
        is_open = True
    elif time(9, 0) <= now < time(9, 15):
        phase = "pre_market"
        is_open = False
    else:
        phase = "closed"
        is_open = False
    
    return {
        "is_open": is_open,
        "phase": phase,
        "market_state": [],
    }


async def get_margins() -> dict:
    """Get available margins and balances."""
    kite = get_kite_instance()
    result = await kite.get_margins()
    if result.get("status") == "success":
        return {"margins": result.get("data", {}), "error": None}
    return {"margins": {}, "error": result.get("message", "Unknown error")}


async def get_positions() -> dict:
    """Get current positions."""
    kite = get_kite_instance()
    result = await kite.get_positions()
    if result.get("status") == "success":
        return {"positions": result.get("data", {}), "error": None}
    return {"positions": {}, "error": result.get("message", "Unknown error")}


async def place_order(
    tradingsymbol: str,
    transaction_type: str,
    quantity: int,
    order_type: str = "MARKET",
    price: float | None = None,
    trigger_price: float | None = None,
    product: str = "CNC",
    variety: str = "regular",
    tag: str = "autfin",
) -> dict:
    """Place an order via Kite."""
    kite = get_kite_instance()
    result = await kite.place_order(
        tradingsymbol=tradingsymbol,
        transaction_type=transaction_type.upper(),
        quantity=quantity,
        order_type=order_type.upper(),
        price=price,
        trigger_price=trigger_price,
        product=product.upper(),
        variety=variety,
        tag=tag,
    )
    return result


async def get_instruments(exchange: str = "NSE") -> list:
    """Get list of instruments for an exchange."""
    kite = get_kite_instance()
    result = await kite.get_instruments(exchange)
    if result.get("status") == "success":
        return result.get("data", [])
    return []