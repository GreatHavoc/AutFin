"""Moat Assessment — Morningstar's 5-source competitive advantage framework.

The dimension scores are a judgement ("is this a wide moat?"), so the model
makes them; the ratio extraction stays deterministic. This replaces five
hand-written `if roe > 0.25: parts += 8` scorers whose thresholds were invented
per dimension and could disagree about the same company (two of the five read
`debt_equity` where the variable and threshold meant gross margin).

Output shape is unchanged: {moat_assessments: {symbol: {score, rating, sources}},
errors: [...]}.
"""
from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from autfin.agents.base import create_agent

logger = logging.getLogger(__name__)

DIMENSIONS = (
    "intangible_assets",
    "switching_costs",
    "network_effect",
    "cost_advantage",
    "efficient_scale",
)


class DimensionScore(BaseModel):
    score: int = Field(ge=0, le=20, description="0-20 for this dimension")
    evidence: str = Field(description="The specific ratio or fact behind the score")


class MoatReport(BaseModel):
    """One symbol's moat assessment."""

    score: int = Field(ge=0, le=100)
    rating: str = Field(pattern="^(WIDE|NARROW|NONE)$")
    sources: dict[str, DimensionScore]


def _resolve_symbol(state: dict, symbol: str | None) -> str:
    holders = state.get("holdings", []) or []
    candidates = state.get("candidates", []) or []
    return (
        symbol
        or (holders[0].get("tradingsymbol") if holders else None)
        or (candidates[0] if candidates else None)
        or "NIFTY"
    )


def _extract_evidence(state: dict, symbol: str) -> dict[str, Any]:
    """Deterministic: pull the ratios the model needs to judge the five sources."""
    fundamental_reports = state.get("fundamental_reports", {}) or {}
    fund_data = fundamental_reports.get(symbol, {}) or {}
    info = fund_data.get("ratios", {}) or {}
    return {
        "symbol": symbol,
        "market_cap": fund_data.get("market_cap", info.get("market_cap")),
        "roe": info.get("roe"),
        "gross_margin": info.get("gross_margin"),
        "operating_margin": info.get("operating_margin"),
        "debt_equity": info.get("debt_equity"),
        "pb": info.get("pb"),
        "revenue_growth": info.get("revenue_growth"),
        "fcf_yield": info.get("fcf_yield"),
        "sector": info.get("sector"),
    }


_moat_agent = create_agent(
    name="moat",
    output_type=MoatReport,
    model_type="deep",
    instructions=(
        "You are a Moat analyst using Morningstar's five sources of competitive "
        "advantage. Score each dimension 0-20 from the evidence you are given:\n"
        "- intangible_assets: brand, patents, licences\n"
        "- switching_costs: expense or inconvenience of moving away\n"
        "- network_effect: value that grows with users\n"
        "- cost_advantage: structurally lower cost than rivals\n"
        "- efficient_scale: niche where a new entrant earns nothing\n\n"
        "Total is the sum (0-100). Rating: WIDE at 75+, NARROW at 50+, else NONE.\n"
        "Every score needs a one-line evidence string quoting the actual number.\n"
        "If the evidence is missing or the ratios are extreme, score low and say so — "
        "a low score from thin data is the correct answer, not a zero."
    ),
)


async def moat_agent(state: dict, symbol: str | None = None) -> dict[str, Any]:
    """Assess economic moat for `symbol` (defaults to the first holding/candidate)."""
    errors: list[str] = list(state.get("errors", []) or [])
    symbol = _resolve_symbol(state, symbol)
    evidence = _extract_evidence(state, symbol)

    try:
        result = await _moat_agent.run(
            f"Assess the economic moat of {symbol} from this evidence: {evidence}"
        )
        report = result.output
        return {
            "moat_assessments": {
                symbol: {
                    "score": report.score,
                    "rating": report.rating,
                    "sources": {
                        dim: {"score": d.score, "evidence": d.evidence}
                        for dim, d in report.sources.items()
                    },
                }
            },
            "errors": errors,
        }
    except Exception as e:  # the framework raises UserError on invalid output
        errors.append(f"Moat assessment failed for {symbol}: {e}")
        return {
            "moat_assessments": {
                symbol: {"score": 0, "rating": "ERROR", "error": str(e)}
            },
            "errors": errors,
        }
