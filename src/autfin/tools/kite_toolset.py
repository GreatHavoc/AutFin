"""AutFin v7.0 — Kite Toolset

Wraps existing KiteTools as a reusable FunctionToolset for Pydantic AI agents.
"""
from __future__ import annotations

import os
from typing import Any

from pydantic_ai import FunctionToolset, RunContext
from pydantic_ai.toolsets import CombinedToolset

from autfin.tools.kite_tools import KiteTools
from autfin.types import AgentDeps


def create_kite_toolset() -> FunctionToolset[AgentDeps]:
    """Create a FunctionToolset with all Kite tools."""
    toolset = FunctionToolset[AgentDeps]()
    
    # Get KiteTools instance from deps or singleton
    def _get_kite(ctx: RunContext[AgentDeps]) -> KiteTools:
        kite = ctx.deps.kite_client
        if not kite:
            from autfin.tools.kite_wrapper import get_kite_instance
            kite = get_kite_instance()
        return kite
    
    @toolset.tool
    async def get_portfolio(ctx: RunContext[AgentDeps]) -> dict:
        """Fetch portfolio holdings from Kite."""
        kite = _get_kite(ctx)
        result = await kite.get_holdings()
        if result.get("status") == "success":
            return {"holdings": result.get("data", []), "error": None}
        return {"holdings": [], "error": result.get("message", "Unknown error")}

    @toolset.tool
    async def get_quotes(ctx: RunContext[AgentDeps], symbols: str) -> dict:
        """Fetch real-time quotes for comma-separated symbols.
        
        Args:
            symbols: Comma-separated list of symbols (e.g., "RELIANCE,TCS,HDFCBANK")
        """
        kite = _get_kite(ctx)
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        instruments = [f"NSE:{sym}" for sym in symbol_list]
        
        result = await kite.get_quotes(instruments)
        if result.get("status") == "success":
            return {"quotes": result.get("data", {}), "error": None}
        return {"quotes": {}, "error": result.get("message", "Unknown error")}

    @toolset.tool
    async def get_margins(ctx: RunContext[AgentDeps]) -> dict:
        """Get available margins and balances."""
        kite = _get_kite(ctx)
        result = await kite.get_margins()
        if result.get("status") == "success":
            return {"margins": result.get("data", {}), "error": None}
        return {"margins": {}, "error": result.get("message", "Unknown error")}

    @toolset.tool
    async def get_positions(ctx: RunContext[AgentDeps]) -> dict:
        """Get current positions (day and net)."""
        kite = _get_kite(ctx)
        result = await kite.get_positions()
        if result.get("status") == "success":
            return {"positions": result.get("data", {}), "error": None}
        return {"positions": {}, "error": result.get("message", "Unknown error")}

    @toolset.tool
    async def place_order(
        ctx: RunContext[AgentDeps],
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
        """Place an order via Kite.
        
        Args:
            tradingsymbol: Trading symbol (e.g., "RELIANCE")
            transaction_type: "BUY" or "SELL"
            quantity: Order quantity
            order_type: "MARKET", "LIMIT", "SL", "SL-M"
            price: Limit price (required for LIMIT, SL, SL-M)
            trigger_price: Trigger price for stop loss orders
            product: "MIS", "NRML", "CNC", "BO", "CO"
            variety: "regular", "bo", "co", "amo"
            tag: Custom tag for tracking orders
        """
        kite = _get_kite(ctx)
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

    @toolset.tool
    async def get_instruments(ctx: RunContext[AgentDeps], exchange: str = "NSE") -> list:
        """Get list of instruments for an exchange."""
        kite = _get_kite(ctx)
        result = await kite.get_instruments(exchange)
        if result.get("status") == "success":
            return result.get("data", [])
        return []

    @toolset.tool
    async def get_market_status(ctx: RunContext[AgentDeps]) -> dict:
        """Get current market status from NSE API."""
        import httpx
        from datetime import datetime, time
        
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

    return toolset


# Create the default toolset instance
kite_toolset = create_kite_toolset()


# Also provide individual tool functions for direct agent registration
def get_kite_toolset() -> FunctionToolset[AgentDeps]:
    """Get the shared Kite toolset instance."""
    return kite_toolset