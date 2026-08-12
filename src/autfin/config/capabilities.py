"""
Data source capabilities registry for AutFin v3.0.

Agents check this before calling external APIs to avoid errors.
Updated automatically when a source fails — agents degrade gracefully.
"""
from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Known limitations discovered through testing
# Each source maps to {capability: available(bool), detail: str}

CAPABILITIES: dict[str, dict[str, Any]] = {
    # ── Kite SDK (Personal tier ≠ Connect tier) ──
    "kite": {
        "auto_login": True,
        "holdings": True,           # Works on Personal
        "positions": True,          # Works on Personal
        "margins": True,            # Works on Personal
        "orders": True,             # Works on Personal
        "order_place": True,        # Works on Personal
        "order_modify": True,       # Works on Personal
        "order_cancel": True,       # Works on Personal
        "trades": True,             # Works on Personal
        "profile": True,            # Works on Personal
        "ltp": False,               # BLOCKED on Personal — needs Connect (₹500/mo)
        "quotes": False,            # BLOCKED on Personal — needs Connect (₹500/mo)
        "ohlc": False,              # BLOCKED on Personal
        "historical_data": False,   # BLOCKED on Personal
        "gtt": True,                # Works on Personal
        "detail": "Personal tier: ltp/quote/ohlc blocked. Use holdings LTP or yfinance fallback.",
    },
    # ── yfinance ──
    "yfinance": {
        "ohlcv": True,
        "ratios": True,             # PE, PB, ROE, D/E, market cap (reliable)
        "financials": "partial",    # Balance sheet/income/cashflow: structure exists,
                                    # values can be 0 for some Indian stocks
        "global_indices": True,
        "forex": True,
        "commodities": True,
        "detail": "No API key needed. 15-min delay for NSE. Financial statements partially populated.",
    },
    # ── LLM Providers ──
    "openrouter": {
        "available": True,
        "rate_limited": "often",    # 50 free/day shared across all users
        "best_model": "meta/muse-glimmer-30b",
        "detail": "Rate-limited often. Use NVIDIA NIM as primary instead.",
    },
    "nvidia_nim": {
        "available": True,
        "rate_limited": False,      # Separately quota per model
        "best_models": [
            "thinkingmachines/inkling",          # reasoning
            "meta/muse-glimmer-30b",             # general
        ],
        "detail": "NVIDIA-hosted, free tier. Better reliability than OpenRouter.",
    },
    "cerebras": {
        "available": False,         # No API key set
        "detail": "Needs CEREBRAS_API_KEY in .env",
    },
    "ollama": {
        "available": False,         # Not running locally
        "detail": "Run 'ollama serve' and 'ollama pull llama3.2' to enable",
    },
    # ── News / Macro / Scraping ──
    "marketaux": {
        "available": True,
        "rate_limit": "100 articles/day",
        "detail": "Working. 5000+ news sources.",
    },
    "fred": {
        "available": True,
        "rate_limit": "120 requests/min",
        "detail": "Working. US macro indicators only.",
    },
    "firecrawl": {
        "available": True,
        "rate_limit": "500 credits/month",
        "detail": "Working. Used for Screener.in, NSE, Moneycontrol scraping.",
    },
    # ── NSE Data (unofficial) ──
    "nsepython": {
        "symbols": True,            # nse_eq_symbols() works (2372 symbols)
        "quotes": False,            # NSE blocks unofficial quote access
        "ohlcv": False,             # NSE blocks unofficial history access
        "detail": "Only symbol list works. Quotes/OHLCV blocked by NSE Cloudflare.",
    },
    # ── Observability ──
    "langsmith": {
        "available": True,
        "configured": True,         # LangSmith API key present
        "detail": "LangSmith cloud tracing with auto-tracing for Pydantic AI.",
    },
}


def can_use(source: str, capability: str) -> bool:
    """Check if a capability is available. Returns True/False."""
    cap = CAPABILITIES.get(source, {}).get(capability)
    if cap is True:
        return True
    if cap == "partial":
        return True  # Partial still works, caller should handle empty results
    return False


def get_fallback(source: str, capability: str) -> str | None:
    """Return the recommended fallback for a blocked capability."""
    if source == "kite" and capability in ("ltp", "quotes", "ohlc"):
        return "Use holdings.last_price for owned stocks, yfinance for others"
    if source == "kite" and capability == "historical_data":
        return "Use yfinance.history() for OHLCV"
    if source == "nsepython" and capability in ("quotes", "ohlcv"):
        return "Use yfinance (15-min delayed) or Kite holdings LTP"
    if source == "openrouter" and capability == "available":
        return "Use nvidia_nim instead"
    return None


def mark_failed(source: str, capability: str) -> None:
    """Mark a capability as failed so agents skip it next time."""
    CAPABILITIES.setdefault(source, {})[capability] = False
    logger.warning("Capability %s.%s marked as failed", source, capability)


def mark_available(source: str, capability: str) -> None:
    """Mark a capability as available after successful use."""
    CAPABILITIES.setdefault(source, {})[capability] = True


def get_status() -> dict:
    """Return a summary of all capabilities for dashboard/system page."""
    return {
        source: {
            cap: val for cap, val in caps.items() if cap != "detail"
        }
        for source, caps in CAPABILITIES.items()
    }


def print_capabilities():
    """Print human-readable capability summary."""
    print("\nData Source Capabilities:")
    for source, caps in sorted(CAPABILITIES.items()):
        detail = caps.pop("detail", "")
        failed = [c for c, v in caps.items() if v is False]
        partial = [c for c, v in caps.items() if v == "partial"]
        ok = [c for c, v in caps.items() if v is True]
        
        status = "OK" if not failed else f"BLOCKED: {', '.join(failed)}"
        print(f"  {source:15s} [{status}]")
        if partial:
            print(f"    partial: {', '.join(partial)}")
        if detail:
            print(f"    {detail}")
    print()
