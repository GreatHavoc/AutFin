"""
Response schemas for agent tools (v5.3 Subagents pattern).

Defines the expected JSON structure each agent tool returns.
Used for:
- Prompt injection (tells the LLM what JSON to output)
- Validation (post-LLM sanity check before returning to supervisor)
- Documentation (single source of truth for inter-agent contracts)
"""
from __future__ import annotations

from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Fundamentals analysis output
# ---------------------------------------------------------------------------

FUNDAMENTALS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "pe_ratio": {"type": "number"},
        "pb_ratio": {"type": "number"},
        "ev_ebitda": {"type": "number"},
        "roe": {"type": "number"},
        "roa": {"type": "number"},
        "net_margin": {"type": "number"},
        "operating_margin": {"type": "number"},
        "debt_equity": {"type": "number"},
        "revenue_growth": {"type": "number"},
        "eps_growth": {"type": "number"},
        "sector": {"type": "string"},
        "market_cap": {"type": "number"},
        "health_score": {"type": "integer"},
        "moat_level": {"type": "string", "enum": ["wide", "narrow", "none"]},
        "recommendation": {"type": "string", "enum": ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]},
        "conviction": {"type": "number"},
        "strength": {"type": "integer"},
        "risks": {"type": "array", "items": {"type": "string"}},
        "catalysts": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["recommendation", "conviction", "summary"],
}

# ---------------------------------------------------------------------------
# Technical analysis output
# ---------------------------------------------------------------------------

TECHNICAL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "sma_20": {"type": "number"},
        "sma_50": {"type": "number"},
        "sma_200": {"type": "number"},
        "rsi": {"type": "number"},
        "macd": {"type": "number"},
        "macd_signal": {"type": "number"},
        "macd_histogram": {"type": "number"},
        "bollinger_upper": {"type": "number"},
        "bollinger_middle": {"type": "number"},
        "bollinger_lower": {"type": "number"},
        "atr": {"type": "number"},
        "volume_trend": {"type": "string", "enum": ["increasing", "decreasing", "stable"]},
        "candlestick_patterns": {"type": "array", "items": {"type": "string"}},
        "signal": {"type": "string", "enum": ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]},
        "strength": {"type": "integer"},
        "conviction": {"type": "number"},
        "summary": {"type": "string"},
    },
    "required": ["signal", "conviction", "summary"],
}

# ---------------------------------------------------------------------------
# Sentiment analysis output
# ---------------------------------------------------------------------------

SENTIMENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "sentiment_score": {"type": "number"},
        "sentiment_label": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
        "news_sentiment": {"type": "number"},
        "announcement_sentiment": {"type": "number"},
        "catalysts": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["sentiment_score", "sentiment_label", "summary"],
}

# ---------------------------------------------------------------------------
# Global macro output
# ---------------------------------------------------------------------------

MACRO_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "impact_signal": {"type": "string", "enum": ["risk_on", "risk_off", "neutral"]},
        "vix_level": {"type": "number"},
        "oil_price": {"type": "number"},
        "dxy_trend": {"type": "string"},
        "inr_usd": {"type": "number"},
        "fed_stance": {"type": "string"},
        "sector_impacts": {"type": "object"},
        "india_specific": {"type": "object"},
        "narrative": {"type": "string"},
        "conviction": {"type": "number"},
    },
    "required": ["impact_signal", "narrative", "conviction"],
}

# ---------------------------------------------------------------------------
# Research synthesis output
# ---------------------------------------------------------------------------

RESEARCH_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "stocks": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "bull_case": {"type": "string"},
                    "bear_case": {"type": "string"},
                    "rating": {"type": "string"},
                    "conviction": {"type": "number"},
                    "target_price": {"type": "number"},
                    "summary": {"type": "string"},
                },
            },
        },
        "portfolio_impact": {"type": "string"},
        "key_risks": {"type": "array", "items": {"type": "string"}},
        "top_picks": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["stocks", "portfolio_impact"],
}

# ---------------------------------------------------------------------------
# Risk assessment output
# ---------------------------------------------------------------------------

RISK_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["approved", "rejected", "conditional"]},
        "checks": {"type": "array"},
        "approved_size": {"type": "integer"},
        "risk_reward": {"type": "number"},
        "stop_loss_pct": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["verdict", "rationale"],
}

# ---------------------------------------------------------------------------
# Portfolio output
# ---------------------------------------------------------------------------

PORTFOLIO_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "total_value": {"type": "number"},
        "total_pnl": {"type": "number"},
        "cash_available": {"type": "number"},
        "sector_allocation": {"type": "object"},
        "holdings": {"type": "array"},
        "rebalancing_suggestions": {"type": "array"},
        "summary": {"type": "string"},
    },
    "required": ["total_value", "cash_available"],
}

# ---------------------------------------------------------------------------
# Schema registry — maps agent tool name → expected output schema
# ---------------------------------------------------------------------------

AGENT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "fundamentals_analysis": FUNDAMENTALS_SCHEMA,
    "technical_analysis": TECHNICAL_SCHEMA,
    "sentiment_analysis": SENTIMENT_SCHEMA,
    "macro_analysis": MACRO_SCHEMA,
    "research_synthesis": RESEARCH_SCHEMA,
    "risk_assessment": RISK_SCHEMA,
    "portfolio_allocation": PORTFOLIO_SCHEMA,
}


def validate_agent_response(agent_name: str, data: Dict[str, Any]) -> List[str]:
    """Validate an agent's response against its schema. Returns list of issues."""
    schema = AGENT_SCHEMAS.get(agent_name)
    if not schema:
        return [f"Unknown agent: {agent_name}"]

    issues = []
    required = schema.get("required", [])
    for field in required:
        if field not in data:
            issues.append(f"Missing required field: {field}")

    return issues
