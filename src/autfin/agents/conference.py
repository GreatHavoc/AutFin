"""AutFin v7.0 — Conference Agent

Multi-agent coordination conferences for capital allocation.
"""
from __future__ import annotations

# ponytail: removed Decimal — NIM xgrammar chokes on its regex pattern
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import Thinking

from autfin.agents.base import create_agent
from autfin.types import (
    AgentDeps,
    AnalysisReport,
    ConferenceResult,
)

conference_agent = create_agent(
    name="conference",
    output_type=ConferenceResult,
    model_type="deep",
    instructions="You are the ConferenceAgent for AutFin v7.0 — multi-agent coordination conferences.",
    capabilities=[
        Thinking(effort="high"),
    ],
)


@conference_agent.tool
async def run_morning_conference(
    ctx: RunContext[AgentDeps],
    market_snapshot: Any,
    candidates: list[str],
    analyses: dict[str, AnalysisReport],
    portfolio: list[Any],
) -> ConferenceResult:
    """Run morning capital allocation conference."""
    from autfin.agents.debate import debate_agent
    from autfin.agents.risk import risk_agent

    deps = ctx.deps
    capital = deps.capital

    decisions = []
    capital_allocation = {}
    remaining_capital = capital * 0.9  # Keep 10% cash

    # Process each candidate through debate + risk
    for symbol in candidates[:10]:  # Max 10 candidates
        analysis = analyses.get(symbol)
        if not analysis:
            continue

        # Get current price from fundamental data
        current_price = 0.0
        if analysis.fundamental and analysis.fundamental.market_cap:
            # Estimate price from market cap and PE
            pe = analysis.fundamental.pe_ratio or 20
            eps = (analysis.fundamental.market_cap / pe) / 1e7 if pe > 0 else 0
            current_price = eps * pe if eps > 0 else 0
        if current_price <= 0:
            continue

        # Run debate via agent
        debate_prompt = (
            f"Analyze {symbol} for a BUY decision. "
            f"Current price: ₹{current_price:.2f}. "
            f"Fundamental score: {analysis.fundamental.health_score if analysis.fundamental else 'N/A'}. "
            f"Technical signal: {analysis.technical.signal if analysis.technical else 'N/A'}. "
            f"Sentiment: {analysis.sentiment.sentiment_label if analysis.sentiment else 'N/A'}."
        )
        debate_result = await debate_agent.run(debate_prompt, deps=deps)

        if debate_result.output.decision != "buy" or debate_result.output.confidence < 0.5:
            decisions.append({
                "symbol": symbol,
                "action": "skip",
                "reason": f"Debate: {debate_result.output.decision}, confidence {debate_result.output.confidence:.2f}",
            })
            continue

        # Run risk check via agent
        risk_prompt = (
            f"Assess trade for {symbol}: BUY at ₹{current_price:.2f}. "
            f"Conviction: {debate_result.output.confidence:.2f}. "
            f"Portfolio capital: ₹{capital:,.0f}. "
            f"Existing positions: {len(portfolio)}."
        )
        risk_result = await risk_agent.run(risk_prompt, deps=deps)

        if risk_result.output.verdict == "reject":
            decisions.append({
                "symbol": symbol,
                "action": "reject",
                "reason": risk_result.output.rationale,
            })
            continue

        # Approved - allocate capital
        position_size = risk_result.output.position_size
        position_value = position_size * current_price

        if position_value > remaining_capital:
            # Scale down
            position_size = int(remaining_capital / current_price)
            position_value = position_size * current_price

        if position_size <= 0:
            continue

        decisions.append({
            "symbol": symbol,
            "action": "buy",
            "quantity": position_size,
            "entry_price": current_price,
            "stop_loss": float(risk_result.output.stop_loss),
            "target": float(risk_result.output.target),
            "risk_reward": risk_result.output.risk_reward,
            "conviction": debate_result.output.confidence,
        })

        capital_allocation[symbol] = position_value
        remaining_capital -= position_value

    # Determine regime
    regime = determine_regime(market_snapshot)

    return ConferenceResult(
        conference_type="morning",
        regime=regime,
        decisions=decisions,
        capital_allocation=capital_allocation,
        risk_alerts=market_snapshot.get("risk_alerts", []) if isinstance(market_snapshot, dict) else [],
        next_meeting=None,
    )


@conference_agent.tool
async def run_emergency_conference(
    ctx: RunContext[AgentDeps],
    symbol: str,
    price_move_pct: float,
    current_positions: list[Any],
) -> ConferenceResult:
    """Run emergency conference for large intraday moves."""
    decisions = []

    # Quick assessment: trim, hedge, or hold
    if price_move_pct > 5:
        action = "trim"
        reason = f"Extreme move {price_move_pct:.1f}% - reduce risk"
    elif price_move_pct > 3:
        action = "review"
        reason = f"Significant move {price_move_pct:.1f}% - assess"
    else:
        action = "monitor"
        reason = f"Move {price_move_pct:.1f}% within normal range"

    decisions.append({
        "symbol": symbol,
        "action": action,
        "reason": reason,
    })

    return ConferenceResult(
        conference_type="emergency",
        regime="high_volatility",
        decisions=decisions,
        capital_allocation={},
        risk_alerts=[f"{symbol} moved {price_move_pct:.1f}%"],
        next_meeting=None,
    )


@conference_agent.tool
async def run_trade_review_conference(
    ctx: RunContext[AgentDeps],
    execution_reports: list[Any],
    updated_portfolio: list[Any],
) -> ConferenceResult:
    """Run post-trade review conference."""
    decisions = []

    for report in execution_reports:
        if report.status == "complete":
            decisions.append({
                "symbol": report.order.symbol,
                "action": "confirm",
                "filled_qty": report.filled_qty,
                "avg_price": float(report.avg_price) if report.avg_price else 0,
            })
        elif report.status == "rejected":
            decisions.append({
                "symbol": report.order.symbol,
                "action": "retry_or_cancel",
                "reason": report.message,
            })

    return ConferenceResult(
        conference_type="trade_review",
        regime="normal",
        decisions=decisions,
        capital_allocation={},
        risk_alerts=[],
        next_meeting=None,
    )


@conference_agent.tool
async def run_experience_conference(
    ctx: RunContext[AgentDeps],
    recent_trades: list[dict],
    market_memory: list[dict],
) -> ConferenceResult:
    """Run EOD experience conference for learning."""
    # This would integrate with MemoryAgent
    decisions = [{
        "action": "update_calibration",
        "trades_reviewed": len(recent_trades),
        "patterns_found": len(market_memory),
    }]

    return ConferenceResult(
        conference_type="experience",
        regime="normal",
        decisions=decisions,
        capital_allocation={},
        risk_alerts=[],
        next_meeting=None,
    )


def determine_regime(market_snapshot: Any) -> str:
    """Determine market regime from snapshot."""
    if isinstance(market_snapshot, dict):
        alerts = market_snapshot.get("risk_alerts", [])
        quotes = market_snapshot.get("quotes", {})

        # Check for high volatility
        large_moves = sum(1 for q in quotes.values() if abs(q.get("change_pct", 0)) > 2)
        if large_moves > len(quotes) * 0.3:
            return "high_volatility"

        # Check for crash/rally
        avg_change = sum(q.get("change_pct", 0) for q in quotes.values()) / max(len(quotes), 1)
        if avg_change < -3:
            return "crash"
        elif avg_change > 3:
            return "rally"

    return "normal"