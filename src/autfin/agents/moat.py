"""Moat Assessment Agent — Morningstar 5-source competitive advantage framework."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def moat_agent(state: dict, symbol: str | None = None) -> dict[str, Any]:
    """
    Assess economic moat using Morningstar's 5-source framework.

    Scores 0-100 across 5 dimensions. Returns rating (WIDE/NARROW/NONE).

    Args:
        state: AgentState dict
        symbol: Stock symbol to analyze (defaults to first holding or first candidate)

    Returns:
        {moat_assessments: {symbol: {score: 0-100, rating: "WIDE"|"NARROW"|"NONE", sources: {dimension: {score, evidence}}}}}
    """
    errors: list[str] = list(state.get("errors", []) or [])

    if symbol is None:
        holders = state.get("holdings", []) or []
        candidates = state.get("candidates", []) or []
        symbol = (holders[0].get("tradingsymbol") if holders else None) or (candidates[0] if candidates else None) or "NIFTY"

    try:
        fundamental_reports = state.get("fundamental_reports", {}) or {}
        fund_data = fundamental_reports.get(symbol, {})
        info = fund_data.get("ratios", {})

        score = 0
        sources: dict[str, dict[str, Any]] = {}

        # 1. Intangible Assets (0-20): brands, patents, regulatory licenses
        brand_score = _assess_intangible_assets(fund_data, info)
        score += brand_score
        sources["intangible_assets"] = {"score": brand_score, "evidence": _intangible_evidence(fund_data, info)}

        # 2. Switching Costs (0-20): expense/inconvenience to switch
        switch_score = _assess_switching_costs(fund_data, info)
        score += switch_score
        sources["switching_costs"] = {"score": switch_score, "evidence": _switching_evidence(fund_data, info)}

        # 3. Network Effect (0-20): more users = more value
        network_score = _assess_network_effect(fund_data, info)
        score += network_score
        sources["network_effect"] = {"score": network_score, "evidence": _network_evidence(fund_data, info)}

        # 4. Cost Advantage (0-20): structurally lower costs
        cost_score = _assess_cost_advantage(fund_data, info)
        score += cost_score
        sources["cost_advantage"] = {"score": cost_score, "evidence": _cost_evidence(fund_data, info)}

        # 5. Efficient Scale (0-20): niche market, no incentive for new entrants
        scale_score = _assess_efficient_scale(fund_data, info)
        score += scale_score
        sources["efficient_scale"] = {"score": scale_score, "evidence": _scale_evidence(fund_data, info)}

        # Rating
        if score >= 75:
            rating = "WIDE"
        elif score >= 50:
            rating = "NARROW"
        else:
            rating = "NONE"

        return {
            "moat_assessments": {symbol: {"score": score, "rating": rating, "sources": sources}},
            "errors": errors,
        }
    except Exception as e:
        errors.append(f"Moat assessment failed for {symbol}: {str(e)}")
        return {
            "moat_assessments": {symbol: {"score": 0, "rating": "ERROR", "error": str(e)}},
            "errors": errors,
        }


# ---------------------------------------------------------------------------
# Scoring helpers — each returns 0-20
# ---------------------------------------------------------------------------

def _assess_intangible_assets(data: dict, info: dict) -> int:
    parts = 0
    roe = info.get("roe") or 0
    gm = info.get("debt_equity") or 99
    if roe > 0.25:
        parts += 8
    elif roe > 0.15:
        parts += 5
    else:
        parts += 2
    pb = info.get("pb") or 1
    if pb > 5:
        parts += 8
    elif pb > 2:
        parts += 5
    else:
        parts += 2
    if gm < 0.3:
        parts += 4
    return min(parts, 20)


def _assess_switching_costs(data: dict, info: dict) -> int:
    gm = info.get("debt_equity") or 99
    roe = info.get("roe") or 0
    if roe > 0.20 and gm < 0.5:
        return 18
    if roe > 0.15:
        return 12
    if roe > 0.10:
        return 8
    return 4


def _assess_network_effect(data: dict, info: dict) -> int:
    rev_growth = info.get("revenue_growth") or 0
    if not isinstance(rev_growth, (int, float)):
        rev_growth = 0
    if rev_growth > 0.25:
        return 15
    if rev_growth > 0.15:
        return 10
    if rev_growth > 0.05:
        return 6
    return 3


def _assess_cost_advantage(data: dict, info: dict) -> int:
    gm = info.get("fcf_yield") or 0
    if not isinstance(gm, (int, float)):
        gm = 0
    if gm > 0.08:
        return 16
    if gm > 0.04:
        return 12
    if gm > 0:
        return 8
    return 4


def _assess_efficient_scale(data: dict, info: dict) -> int:
    mcap = data.get("market_cap", 0) or 0
    if mcap > 5_00_000_00_00_000:
        return 12
    if mcap > 1_00_000_00_00_000:
        return 8
    return 5


# ---------------------------------------------------------------------------
# Evidence text helpers
# ---------------------------------------------------------------------------

def _intangible_evidence(data: dict, info: dict) -> str:
    return f"ROE: {info.get('roe', 'N/A')}, P/B: {info.get('pb', 'N/A')}"


def _switching_evidence(data: dict, info: dict) -> str:
    return f"ROE: {info.get('roe', 'N/A')}, D/E: {info.get('debt_equity', 'N/A')}"


def _network_evidence(data: dict, info: dict) -> str:
    return f"Revenue growth: {info.get('revenue_growth', 'N/A')}"


def _cost_evidence(data: dict, info: dict) -> str:
    return f"FCF yield: {info.get('fcf_yield', 'N/A')}"


def _scale_evidence(data: dict, info: dict) -> str:
    return f"Market cap: {data.get('market_cap', 'N/A')}"
