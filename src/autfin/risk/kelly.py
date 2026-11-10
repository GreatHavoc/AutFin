"""Kelly criterion position sizing with regime gates for AutFin.

Implements fractional Kelly with adjustments for:
- Drawdown-based de-grossing
- Regime-conditional sizing (risk-on vs risk-off)
- Conviction-weighted allocation
- Per-position and per-portfolio caps

All arithmetic is deterministic — no LLM involved.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Kelly parameters
MAX_KELLY_FRACTION = 0.25  # Cap single position at 25% of Kelly
MAX_POSITION_PCT = 0.15    # Max 15% of portfolio per position
MIN_KELLY_PCT = 0.01       # Don't bother with < 1% allocation
DRAWDOWN_DEGROSS_THRESHOLD = 0.05  # De-gross when drawdown > 5%
REGIME_ADJUSTMENTS = {
    "risk_on": 1.0,     # Full Kelly
    "neutral": 0.75,    # 75% Kelly
    "risk_off": 0.50,   # 50% Kelly
}


def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    regime: str = "neutral",
    drawdown_pct: float = 0.0,
    conviction: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute Kelly-optimal position size.

    Kelly formula: f* = (p * b - q) / b
    where p = win probability, q = 1-p, b = avg_win/avg_loss

    Args:
        win_rate: Historical win rate (0.0-1.0)
        avg_win: Average winning trade return (e.g. 0.05 = 5%)
        avg_loss: Average losing trade return (e.g. 0.03 = 3%)
        regime: "risk_on", "neutral", or "risk_off"
        drawdown_pct: Current portfolio drawdown (0.0 = at peak)
        conviction: Signal conviction (0.0-1.0)

    Returns:
        Dict with kelly_raw, kelly_adjusted, position_pct, rationale
    """
    if win_rate <= 0 or win_rate >= 1 or avg_win <= 0 or avg_loss <= 0:
        return {
            "kelly_raw": 0.0,
            "kelly_adjusted": 0.0,
            "position_pct": 0.0,
            "rationale": "Insufficient data for Kelly calculation",
        }

    # Raw Kelly
    b = avg_win / avg_loss  # Odds ratio
    p = win_rate
    q = 1 - p
    kelly_raw = (p * b - q) / b

    if kelly_raw <= 0:
        return {
            "kelly_raw": round(kelly_raw, 4),
            "kelly_adjusted": 0.0,
            "position_pct": 0.0,
            "rationale": f"Negative Kelly ({kelly_raw:.2%}) — edge is negative, do not trade",
        }

    # Apply fractional Kelly (half-Kelly is standard)
    kelly_half = kelly_raw * 0.5

    # Regime adjustment
    regime_mult = REGIME_ADJUSTMENTS.get(regime, 0.75)
    kelly_regime = kelly_half * regime_mult

    # Drawdown de-grossing
    dd_mult = max(0.0, 1.0 - (drawdown_pct / DRAWDOWN_DEGROSS_THRESHOLD)) if drawdown_pct > 0 else 1.0
    kelly_dd = kelly_regime * dd_mult

    # Conviction scaling
    conv_mult = min(1.0, conviction / 0.5)  # 0.5 conviction = 1x, higher = up to 2x
    kelly_final = kelly_dd * conv_mult

    # Apply caps
    position_pct = min(kelly_final, MAX_KELLY_FRACTION, MAX_POSITION_PCT)
    position_pct = max(position_pct, 0.0)

    if position_pct < MIN_KELLY_PCT:
        position_pct = 0.0

    rationale_parts = [
        f"Raw Kelly: {kelly_raw:.2%}",
        f"Half-Kelly: {kelly_half:.2%}",
        f"Regime ({regime}): {regime_mult:.0%} -> {kelly_regime:.2%}",
    ]
    if drawdown_pct > 0:
        rationale_parts.append(f"Drawdown ({drawdown_pct:.1%}): {dd_mult:.0%} -> {kelly_dd:.2%}")
    rationale_parts.append(f"Conviction ({conviction:.2f}): {conv_mult:.2f}x -> {kelly_final:.2%}")
    rationale_parts.append(f"Final: {position_pct:.2%}")

    return {
        "kelly_raw": round(kelly_raw, 4),
        "kelly_adjusted": round(kelly_final, 4),
        "position_pct": round(position_pct, 4),
        "rationale": " | ".join(rationale_parts),
    }


def kelly_portfolio(
    opportunities: list[Dict[str, Any]],
    total_capital: float,
    current_drawdown: float = 0.0,
    regime: str = "neutral",
) -> Dict[str, Any]:
    """
    Allocate across multiple opportunities using Kelly.

    Args:
        opportunities: List of {symbol, win_rate, avg_win, avg_loss, conviction}
        total_capital: Total portfolio capital in INR
        current_drawdown: Current drawdown (0.0 = at peak)
        regime: Market regime

    Returns:
        Dict with allocations per symbol and total allocated
    """
    allocations = []
    total_allocated = 0.0

    for opp in sorted(opportunities, key=lambda x: x.get("conviction", 0), reverse=True):
        kelly = kelly_fraction(
            win_rate=opp.get("win_rate", 0.5),
            avg_win=opp.get("avg_win", 0.05),
            avg_loss=opp.get("avg_loss", 0.03),
            regime=regime,
            drawdown_pct=current_drawdown,
            conviction=opp.get("conviction", 0.5),
        )

        alloc_amount = total_capital * kelly["position_pct"]
        remaining = total_capital - total_allocated

        if alloc_amount > remaining:
            alloc_amount = remaining

        if alloc_amount > 0:
            allocations.append({
                "symbol": opp.get("symbol", ""),
                "position_pct": kelly["position_pct"],
                "alloc_amount": round(alloc_amount, 2),
                "kelly_raw": kelly["kelly_raw"],
                "rationale": kelly["rationale"],
            })
            total_allocated += alloc_amount

    return {
        "allocations": allocations,
        "total_allocated": round(total_allocated, 2),
        "total_pct": round(total_allocated / max(total_capital, 1) * 100, 2),
        "regime": regime,
        "drawdown": current_drawdown,
    }
