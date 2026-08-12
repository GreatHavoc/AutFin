"""AutFin v7.0 — Conference Agent

Multi-agent coordination conferences for capital allocation.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Thinking

from autfin.agents.base import create_agent, get_model
from autfin.agents.managed_prompts import conference_instructions
from autfin.agents.observability import create_debug_instrumentation
from autfin.types import (
    AgentDeps,
    ConferenceResult,
    AnalysisReport,
    RiskAssessment,
    DebateResult,
)


conference_agent = create_agent(
    name="conference",
    output_type=ConferenceResult,
    model_type="deep",
    instructions="You are the ConferenceAgent for AutFin v7.0 — multi-agent coordination conferences.",
    capabilities=[
        Thinking(effort="high"),
        create_debug_instrumentation(),
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
    import asyncio

    deps = ctx.deps
    capital = deps.capital

    decisions = []
    capital_allocation = {}
    remaining_capital = capital * Decimal("0.1")  # Keep 10% cash
    risk_budget = capital * Decimal(str(deps.max_drawdown_pct))

    # Process each candidate through debate + risk
    for symbol in candidates[:10]:  # Max 10 candidates
        analysis = analyses.get(symbol)
        if not analysis:
            continue

        # Run debate
        from autfin.agents.debate import run_debate
        debate_result = await run_debate(ctx, symbol, analysis, "buy")

        if debate_result.decision != "buy" or debate_result.confidence < 0.5:
            decisions.append({
                "symbol": symbol,
                "action": "skip",
                "reason": f"Debate: {debate_result.decision}, confidence {debate_result.confidence:.2f}",
            })
            continue

        # Run risk check
        from autfin.agents.risk import assess_trade
        risk_result = await assess_trade(
            ctx,
            symbol=symbol,
            action="buy",
            entry_price=float(analysis.fundamental.market_cap / 1e7) if analysis.fundamental else 0,
            conviction=debate_result.confidence,
            analysis_data=analysis.model_dump() if hasattr(analysis, 'model_dump') else analysis,
        )

        if risk_result.verdict == "reject":
            decisions.append({
                "symbol": symbol,
                "action": "reject",
                "reason": risk_result.rationale,
            })
            continue

        # Approved - allocate capital
        position_size = risk_result.position_size
        position_value = position_size * Decimal(str(risk_result.stop_loss))  # Approximate

        if position_value > remaining_capital:
            # Scale down
            position_size = int(remaining_capital / risk_result.stop_loss)
            position_value = position_size * risk_result.stop_loss

        decisions.append({
            "symbol": symbol,
            "action": "buy",
            "quantity": position_size,
            "entry_price": float(risk_result.target),
            "stop_loss": float(risk_result.stop_loss),
            "target": float(risk_result.target),
            "risk_reward": risk_result.risk_reward,
            "conviction": debate_result.confidence,
        })

        capital_allocation[symbol] = position_value
        remaining_capital -= position_value
        risk_budget -= position_value * Decimal("0.08")  # 8% risk per position

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