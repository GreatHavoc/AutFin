"""Peer Comparison Agent — 8-metric sector ranking vs competitors."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Peer symbol lookup maps
# ---------------------------------------------------------------------------

PEER_MAP: dict[str, list[str]] = {
    "RELIANCE": ["ONGC", "BPCL", "HINDPETRO", "IOC"],
    "TCS": ["INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM"],
    "INFY": ["TCS", "HCLTECH", "WIPRO", "TECHM"],
    "HCLTECH": ["TCS", "INFY", "WIPRO", "TECHM"],
    "WIPRO": ["TCS", "INFY", "HCLTECH", "TECHM"],
    "HDFCBANK": ["ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN", "INDUSINDBK"],
    "ICICIBANK": ["HDFCBANK", "KOTAKBANK", "AXISBANK", "SBIN"],
    "KOTAKBANK": ["HDFCBANK", "ICICIBANK", "AXISBANK", "SBIN"],
    "SBIN": ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "BANKBARODA", "PNB"],
    "HINDUNILVR": ["NESTLEIND", "BRITANNIA", "DABUR", "GODREJCP", "MARICO"],
    "ITC": ["HINDUNILVR", "NESTLEIND", "BRITANNIA", "VBL"],
    "SUNPHARMA": ["DRREDDY", "CIPLA", "DIVISLAB", "BIOCON"],
    "DRREDDY": ["SUNPHARMA", "CIPLA", "DIVISLAB", "LUPIN"],
    "TATASTEEL": ["JSWSTEEL", "HINDZINC", "VEDL", "SAIL", "JINDALSTEL"],
    "JSWSTEEL": ["TATASTEEL", "HINDZINC", "VEDL", "SAIL"],
    "TATAMOTORS": ["MARUTI", "ASHOKLEY", "EICHERMOT", "BAJAJ-AUTO"],
    "MARUTI": ["TATAMOTORS", "ASHOKLEY", "EICHERMOT", "BAJAJ-AUTO", "M&M"],
    "BAJFINANCE": ["CHOLAFIN", "SRTRANSFIN", "MUTHOOTFIN", "BAJAJFINSV"],
    "NTPC": ["POWERGRID", "TATAPOWER", "ADANIPOWER", "NHPC"],
    "ASIANPAINT": ["BERGEPAINT", "INDIGOPNTS", "SHALPAINTS"],
    "TITAN": ["KALYANKJIL", "RAJESHEXPO", "PCJEWELLER"],
    "BHARTIARTL": ["RELIANCE", "IDEA"],
    "LT": ["SIEMENS", "ABB", "BHEL", "THERMAX"],
    "DIVISLAB": ["SUNPHARMA", "DRREDDY", "CIPLA", "LUPIN"],
}


async def peer_compare_agent(state: dict, symbol: str | None = None) -> dict[str, Any]:
    """
    Compare stock against sector peers on 8 metrics.

    Uses yfinance to fetch peer data, then ranks.

    Returns:
        {peer_comparisons: {symbol: {rankings, percentile, peers, comparison_summary}}}
    """
    errors: list[str] = list(state.get("errors", []) or [])

    if symbol is None:
        holders = state.get("holdings", []) or []
        candidates = state.get("candidates", []) or []
        symbol = (holders[0].get("tradingsymbol") if holders else None) or (candidates[0] if candidates else None) or "NIFTY"

    try:
        import yfinance as yf

        from src.autfin.agents.yfinance_utils import normalize_symbol

        ticker = yf.Ticker(normalize_symbol(symbol))
        info = ticker.info or {}

        peer_symbols = _get_peer_symbols(symbol)

        peers: dict[str, dict[str, Any]] = {symbol: _extract_metrics(info)}
        for peer_sym in peer_symbols:
            try:
                pt = yf.Ticker(normalize_symbol(peer_sym))
                pi = pt.info or {}
                peers[peer_sym] = _extract_metrics(pi)
            except Exception:
                logger.debug("Could not fetch peer data for %s", peer_sym)

        if len(peers) < 2:
            return {
                "peer_comparisons": {symbol: {"error": "Insufficient peer data", "peers_count": len(peers)}},
                "errors": errors,
            }

        rankings = _rank_metrics(peers, symbol)

        return {
            "peer_comparisons": {symbol: rankings},
            "errors": errors,
        }
    except Exception as e:
        errors.append(f"Peer comparison failed for {symbol}: {str(e)}")
        return {
            "peer_comparisons": {symbol: {"error": str(e)}},
            "errors": errors,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_metrics(info: dict) -> dict[str, Any]:
    """Extract 8 key comparison metrics from yfinance info."""
    return {
        "pe": info.get("trailingPE") or info.get("forwardPE"),
        "pb": info.get("priceToBook"),
        "roe": info.get("returnOnEquity"),
        "fcf_yield": (info.get("freeCashflow", 0) or 0) / max(info.get("marketCap", 1) or 1, 1),
        "revenue_growth": info.get("revenueGrowth"),
        "net_margin": (info.get("profitMargins", 0) or 0) * 100,
        "debt_equity": -(info.get("debtToEquity") or 0),
        "earnings_yield": (1 / info["trailingPE"]) if info.get("trailingPE") and info["trailingPE"] > 0 else 0,
    }


def _rank_metrics(peers: dict[str, dict[str, Any]], target_symbol: str) -> dict[str, Any]:
    """Rank target against peers on all 8 metrics."""
    metrics_list = [
        "pe", "pb", "roe", "fcf_yield",
        "revenue_growth", "net_margin", "debt_equity", "earnings_yield",
    ]
    metric_names = [
        "P/E", "P/B", "ROE", "FCF Yield",
        "Rev Growth", "Net Margin", "D/E (inverse)", "Earnings Yield",
    ]

    ascend: set[str] = {"pe", "pb"}

    target_metrics = peers.get(target_symbol, {})
    if not target_metrics:
        return {"error": "Target not found in peer group"}

    results: dict[str, dict[str, Any]] = {}
    for i, metric in enumerate(metrics_list):
        values = [(sym, m.get(metric)) for sym, m in peers.items() if m.get(metric) is not None]
        if not values:
            continue

        if metric in ascend:
            values.sort(key=lambda x: x[1])
        else:
            values.sort(key=lambda x: x[1], reverse=True)

        rank = next((j + 1 for j, (sym, _) in enumerate(values) if sym == target_symbol), len(values))
        percentile = (len(values) - rank) / max(len(values) - 1, 1) * 100

        results[metric_names[i]] = {
            "value": target_metrics.get(metric),
            "rank": f"{rank}/{len(values)}",
            "percentile": round(percentile, 0),
            "best_peer": values[0][0],
            "best_value": values[0][1],
        }

    percentiles = [r["percentile"] for r in results.values()]
    avg_percentile = sum(percentiles) / max(len(percentiles), 1)

    if avg_percentile >= 70:
        position = "LEADER"
    elif avg_percentile >= 40:
        position = "MID-PACK"
    else:
        position = "LAGGARD"

    return {
        "rankings": results,
        "overall_percentile": round(avg_percentile, 0),
        "position": position,
        "peers_count": len(peers),
        "peer_symbols": list(peers.keys()),
        "sector": peers.get(target_symbol, {}).get("sector", "Unknown"),
    }


def _get_peer_symbols(symbol: str) -> list[str]:
    """Return 4-6 peer symbols for a given stock based on sector maps."""
    return PEER_MAP.get(symbol.upper(), [])
