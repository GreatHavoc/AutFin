"""
Market Data Agent - Fetches balances, quotes, margins from Kite SDK
"""
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from langgraph.config import RunnableConfig

from src.autfin.state.agent_state import AgentState
from src.autfin.tools.kite_tools import KiteTools, create_kite_tools


# ================== CACHING ==================

class MarketDataCache:
    """Caches market data to avoid redundant API calls."""
    
    def __init__(self, ttl_seconds: int = 60):
        """
        Initialize cache with TTL.
        
        Args:
            ttl_seconds: Time-to-live for cache entries (default 60 seconds)
        """
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, datetime] = {}
        self._ttl = timedelta(seconds=ttl_seconds)
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self._timestamps:
            return True
        return datetime.now() - self._timestamps[key] > self._ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if valid."""
        if key not in self._cache:
            return None
        if self._is_expired(key):
            del self._cache[key]
            del self._timestamps[key]
            return None
        return self._cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache with timestamp."""
        self._cache[key] = value
        self._timestamps[key] = datetime.now()
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._timestamps.clear()


# Global cache instance
_cache = MarketDataCache(ttl_seconds=60)


# ================== BALANCE FETCHER ==================

async def balance_fetcher(state: AgentState, config: RunnableConfig | None = None) -> AgentState:
    """
    Fetch portfolio balance and margin information from Kite.
    
    Retrieves:
    - Available cash
    - Utilized margin
    - Holdings
    - Positions
    
    Updates state with balance data and caches results for efficiency.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with balance data
    """
    errors = state.get("errors", [])
    
    try:
        # Initialize KiteTools
        kite = create_kite_tools()
        
        # Check cache first
        cache_key = "balance_data"
        cached_data = _cache.get(cache_key)
        
        if cached_data is not None:
            # Use cached data
            return {
                "available_cash": cached_data.get("available_cash", 0),
                "utilized_margin": cached_data.get("utilized_margin", 0),
                "holdings": cached_data.get("holdings", []),
                "positions": cached_data.get("positions", []),
                "errors": errors,
            }
        
        # Fetch data from Kite
        margins, holdings, positions = await asyncio.gather(
            kite.get_margins(),
            kite.get_holdings(),
            kite.get_positions()
        )
        
        # Process margins
        if margins.get("status") == "success":
            data = margins.get("data", {})
            equity = data.get("equity", {})
            available_cash = equity.get("available", {}).get("cash", 0)
            utilized_margin = equity.get("utilised", {}).get("debits", 0)
        else:
            errors.append(f"Failed to fetch margins: {margins.get('message', 'Unknown error')}")
            available_cash = 0
            utilized_margin = 0
        
        # Process holdings
        if holdings.get("status") == "success":
            holdings_data = holdings.get("data", [])
        else:
            errors.append(f"Failed to fetch holdings: {holdings.get('message', 'Unknown error')}")
            holdings_data = []
        
        # Process positions
        if positions.get("status") == "success":
            positions_data = positions.get("data", {}).get("day", [])
        else:
            errors.append(f"Failed to fetch positions: {positions.get('message', 'Unknown error')}")
            positions_data = []
        
        # Cache results
        cache_data = {
            "available_cash": available_cash,
            "utilized_margin": utilized_margin,
            "holdings": holdings_data,
            "positions": positions_data,
        }
        _cache.set(cache_key, cache_data)
        
        return {
            "available_cash": available_cash,
            "utilized_margin": utilized_margin,
            "holdings": holdings_data,
            "positions": positions_data,
            "errors": errors,
        }
        
    except Exception as e:
        errors.append(f"Balance fetcher failed: {str(e)}")
        return {
            "available_cash": 0,
            "utilized_margin": 0,
            "holdings": [],
            "positions": [],
            "errors": errors,
        }


# ================== PRICE FETCHER ==================

async def _fetch_prices_yfinance(
    holdings: list, positions: list
) -> Dict[str, Any]:
    """
    Fallback: fetch LTP from yfinance when Kite quotes fail.

    Args:
        holdings: List of holding dicts from Kite
        positions: List of position dicts from Kite

    Returns:
        Dict mapping symbol → price quote dict
    """
    try:
        import yfinance as yf
    except ImportError:
        return {}

    # Collect unique symbols
    symbols = []
    for h in holdings:
        sym = h.get("tradingsymbol", "")
        if sym and sym not in symbols:
            symbols.append(sym)
    for p in positions:
        sym = p.get("tradingsymbol", "")
        if sym and sym not in symbols:
            symbols.append(sym)

    if not symbols:
        return {}

    quotes = {}
    for sym in symbols:
        try:
            ticker = yf.Ticker(f"{sym.removesuffix('.NS')}.NS")
            info = ticker.info
            ltp = getattr(info, "last_price", None) or 0
            prev = getattr(info, "previous_close", None) or ltp
            change = ltp - prev if prev else 0
            change_pct = (change / prev * 100) if prev else 0
            quotes[sym] = {
                "symbol": sym,
                "ltp": round(float(ltp), 2),
                "change": round(float(change), 2),
                "change_pct": round(float(change_pct), 2),
                "volume": 0,
                "avg_volume": 0,
                "high_52w": 0,
                "low_52w": 0,
                "source": "yfinance_fallback",
            }
        except Exception:
            continue

    return quotes


async def price_fetcher(state: AgentState, config: RunnableConfig | None = None) -> AgentState:
    """
    Fetch real-time price quotes for portfolio holdings.
    
    Retrieves live prices (LTP), change percentages, volume, and 52-week highs/lows.
    Updates state with price data and caches results.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with price data
    """
    errors = state.get("errors", [])
    
    try:
        # Initialize KiteTools
        kite = create_kite_tools()
        
        # Check cache first
        cache_key = "price_data"
        cached_data = _cache.get(cache_key)
        
        if cached_data is not None:
            # Use cached data
            return {
                "market_snapshot": cached_data.get("market_snapshot", {}),
                "errors": errors,
            }
        
        # Get holdings to know which symbols to fetch prices for
        holdings = state.get("holdings", [])
        positions = state.get("positions", [])
        
        # Extract instrument tokens and trading symbols
        instruments = []
        
        # From holdings
        for holding in holdings:
            if "instrument_token" in holding:
                instruments.append(holding["instrument_token"])
            elif "tradingsymbol" in holding:
                exchange = holding.get("exchange", "NSE")
                instruments.append(f"{exchange}:{holding['tradingsymbol']}")
        
        # From positions (intraday)
        for position in positions:
            if "instrument_token" in position:
                instruments.append(position["instrument_token"])
            elif "tradingsymbol" in position:
                exchange = position.get("exchange", "NSE")
                instruments.append(f"{exchange}:{position['tradingsymbol']}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_instruments = []
        for inst in instruments:
            if inst not in seen:
                seen.add(inst)
                unique_instruments.append(inst)
        instruments = unique_instruments
        
        # If no instruments, return empty snapshot
        if not instruments:
            _cache.set(cache_key, {"market_snapshot": {}})
            return {
                "market_snapshot": {},
                "errors": errors,
            }
        
        # Fetch quotes from Kite (skip if blocked on Personal tier)
        from src.autfin.config.capabilities import can_use
        if can_use("kite", "quotes"):
            quotes_result = await kite.get_quotes(instruments)
        else:
            quotes_result = {"status": "error", "message": "Kite quotes blocked on Personal tier — using yfinance fallback"}
        
        # Build market snapshot (initialized before branching so else path has it)
        market_snapshot = {
            "timestamp": datetime.now().isoformat(),
            "indices": {},
            "quotes": {},
            "advance_decline": {},
        }

        if quotes_result.get("status") == "success":
            quotes_data = quotes_result.get("data", {})
            
            # Process quotes
            for instrument_token, quote in quotes_data.items():
                symbol = quote.get("tradingsymbol", str(instrument_token))
                
                # Build price quote
                price_quote = {
                    "symbol": symbol,
                    "ltp": quote.get("last_price", 0),
                    "change": quote.get("change", 0),
                    "change_pct": quote.get("percent_change", 0),
                    "volume": quote.get("volume", 0),
                    "avg_volume": 0,  # Would need historical data to calculate
                    "high_52w": quote.get("day_high_52_week", 0),
                    "low_52w": quote.get("day_low_52_week", 0),
                }
                
                market_snapshot["quotes"][symbol] = price_quote
            
            # Cache results
            _cache.set(cache_key, {"market_snapshot": market_snapshot})
            
            return {
                "market_snapshot": market_snapshot,
                "errors": errors,
            }
        else:
            errors.append(f"Failed to fetch prices: {quotes_result.get('message', 'Unknown error')}")
            
            market_snapshot = {
                "timestamp": datetime.now().isoformat(),
                "indices": {},
                "quotes": {},
                "advance_decline": {},
            }
            
            # Fallback: try yfinance for each symbol
            yf_quotes = await _fetch_prices_yfinance(holdings, positions)
            if yf_quotes:
                market_snapshot["quotes"] = yf_quotes
                _cache.set(cache_key, {"market_snapshot": market_snapshot})
                return {
                    "market_snapshot": market_snapshot,
                    "errors": errors,
                }
            
            # Cache empty result to avoid repeated failures
            _cache.set(cache_key, {"market_snapshot": {}})
            
            return {
                "market_snapshot": {},
                "errors": errors,
            }
        
    except Exception as e:
        errors.append(f"Price fetcher failed: {str(e)}")
        
        # Cache empty result to avoid repeated failures
        _cache.set("price_data", {"market_snapshot": {}})
        
        return {
            "market_snapshot": {},
            "errors": errors,
        }


# ================== MAIN AGENT ==================

async def market_data_agent(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """
    Orchestrate market data fetching workflow.
    
    Runs balance_fetcher and price_fetcher in sequence,
    aggregating results into a complete market data snapshot.
    
    Args:
        state: Current agent state
        
    Returns:
        Complete market data state
    """
    # Step 1: Fetch balance data
    balance_state =     await balance_fetcher(state, config)
    
    # Step 2: Fetch price data (uses holdings from balance fetch)
    full_state = {**state, **balance_state}
    price_state = await price_fetcher(full_state)
    
    # Step 3: Merge results
    return {
        **balance_state,
        **price_state,
        "agent": "market_data",
        "next": None,  # Will be set by supervisor
    }


# ================== UTILITY FUNCTIONS ==================

def clear_cache() -> None:
    """Clear the market data cache."""
    _cache.clear()


def get_cache_status() -> Dict[str, Any]:
    """Get cache status for debugging."""
    return {
        "cache_size": len(_cache._cache),
        "cache_keys": list(_cache._cache.keys()),
    }