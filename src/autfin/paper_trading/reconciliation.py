"""Paper-live reconciliation for AutFin.

Compares paper trading positions against live Kite holdings to detect drift.
Runs nightly to ensure paper trading accuracy.

Returns discrepancies sorted by severity.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def reconcile(
    paper_positions: List[Dict[str, Any]],
    live_holdings: List[Dict[str, Any]],
    tolerance_pct: float = 0.01,
) -> Dict[str, Any]:
    """
    Compare paper vs live positions and report discrepancies.

    Args:
        paper_positions: Positions from paper trading engine
        live_holdings: Holdings from Kite API
        tolerance_pct: Max acceptable difference (1% = 0.01)

    Returns:
        Dict with matches, discrepancies, and summary
    """
    # Build lookup by symbol
    paper_map: Dict[str, Dict[str, Any]] = {}
    for p in paper_positions:
        sym = p.get("tradingsymbol") or p.get("symbol") or ""
        if sym:
            paper_map[sym.upper()] = {
                "qty": int(p.get("qty", p.get("quantity", 0)) or 0),
                "avg_price": float(p.get("avg_price", 0) or 0),
                "ltp": float(p.get("last_price", p.get("ltp", 0)) or 0),
            }

    live_map: Dict[str, Dict[str, Any]] = {}
    for h in live_holdings:
        sym = h.get("tradingsymbol") or h.get("symbol") or ""
        if sym:
            live_map[sym.upper()] = {
                "qty": int(h.get("qty", h.get("quantity", 0)) or 0),
                "avg_price": float(h.get("avg_price", 0) or 0),
                "ltp": float(h.get("last_price", h.get("ltp", 0)) or 0),
            }

    all_symbols = set(paper_map.keys()) | set(live_map.keys())
    matches = []
    discrepancies = []

    for sym in all_symbols:
        paper = paper_map.get(sym)
        live = live_map.get(sym)

        if paper and live:
            qty_match = paper["qty"] == live["qty"]
            price_diff = abs(paper["avg_price"] - live["avg_price"]) / max(live["avg_price"], 1)
            price_match = price_diff <= tolerance_pct

            if qty_match and price_match:
                matches.append({"symbol": sym, "qty": paper["qty"], "avg_price": paper["avg_price"]})
            else:
                severity = "high" if not qty_match else "low"
                discrepancies.append({
                    "symbol": sym,
                    "type": "quantity_mismatch" if not qty_match else "price_drift",
                    "paper": paper,
                    "live": live,
                    "qty_diff": paper["qty"] - live["qty"],
                    "price_diff_pct": round(price_diff * 100, 2),
                    "severity": severity,
                })
        elif paper and not live:
            discrepancies.append({
                "symbol": sym,
                "type": "paper_only",
                "paper": paper,
                "live": None,
                "severity": "medium",
                "message": f"Position {sym} exists in paper but not in live holdings",
            })
        elif live and not paper:
            discrepancies.append({
                "symbol": sym,
                "type": "live_only",
                "paper": None,
                "live": live,
                "severity": "medium",
                "message": f"Holding {sym} exists in live but not in paper trading",
            })

    discrepancies.sort(key=lambda d: {"high": 0, "medium": 1, "low": 2}.get(d.get("severity", "low"), 3))

    return {
        "status": "success",
        "matches": len(matches),
        "discrepancies": len(discrepancies),
        "discrepancy_details": discrepancies,
        "match_rate": round(len(matches) / max(len(all_symbols), 1) * 100, 1),
        "needs_attention": any(d.get("severity") == "high" for d in discrepancies),
    }
