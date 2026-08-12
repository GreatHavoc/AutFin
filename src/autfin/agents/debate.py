"""AutFin v7.0 — Debate Agent

Bull/Bear adversarial debate for trade decisions.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Thinking

from autfin.agents.base import create_agent, get_model
from autfin.agents.managed_prompts import debate_instructions
from autfin.agents.observability import create_debug_instrumentation
from autfin.types import (
    AgentDeps,
    DebateResult,
    DebateArgument,
    AnalysisReport,
)


debate_agent = create_agent(
    name="debate",
    output_type=DebateResult,
    model_type="deep",
    instructions=debate_instructions.default,
    capabilities=[
        Thinking(effort="high"),
        create_debug_instrumentation(),
    ],
)


@debate_agent.tool
async def run_debate(
    ctx: RunContext[AgentDeps],
    symbol: str,
    analysis_data: AnalysisReport | dict,
    trade_direction: Literal["buy", "sell"] = "buy",
) -> DebateResult:
    """Run full bull/bear debate on a symbol."""
    import asyncio

    # Generate bull and bear arguments in parallel
    bull_task = generate_bull_case(ctx, symbol, analysis_data, trade_direction)
    bear_task = generate_bear_case(ctx, symbol, analysis_data, trade_direction)

    bull_args, bear_args = await asyncio.gather(bull_task, bear_task)

    # Synthesize
    synthesis = synthesize_debate(bull_args, bear_args)
    decision, confidence = make_decision(bull_args, bear_args, trade_direction)

    return DebateResult(
        symbol=symbol,
        bull_thesis=bull_args[0].argument if bull_args else "",
        bear_thesis=bear_args[0].argument if bear_args else "",
        bull_arguments=bull_args,
        bear_arguments=bear_args,
        rebuttal="",
        synthesis=synthesis,
        decision=decision,
        confidence=confidence,
    )


async def generate_bull_case(
    ctx: RunContext[AgentDeps],
    symbol: str,
    analysis_data: AnalysisReport | dict,
    trade_direction: str,
) -> list[DebateArgument]:
    """Generate bull arguments."""
    args = []

    if isinstance(analysis_data, AnalysisReport):
        data = analysis_data
    else:
        data = analysis_data

    # Fundamental bull case
    if data.get("fundamental"):
        f = data["fundamental"]
        if f.health_score > 60:
            args.append(DebateArgument(
                symbol=symbol,
                side="bull",
                argument=f"Strong fundamentals: health score {f.health_score}/100, ROE {f.roe}%",
                evidence=[f"ROE: {f.roe}%", f"Revenue growth: {f.revenue_growth}%"],
                confidence=0.7,
            ))

    # Technical bull case
    if data.get("technical"):
        t = data["technical"]
        if t.signal in ("buy", "strong_buy"):
            args.append(DebateArgument(
                symbol=symbol,
                side="bull",
                argument=f"Technical signal: {t.signal}, RSI {t.rsi}, trend positive",
                evidence=[f"Signal: {t.signal}", f"RSI: {t.rsi}"],
                confidence=0.65,
            ))

    # Valuation bull case
    if data.get("valuation") and data["valuation"].upside_pct:
        v = data["valuation"]
        if v.upside_pct > 15:
            args.append(DebateArgument(
                symbol=symbol,
                side="bull",
                argument=f"Undervalued: {v.upside_pct:.1f}% upside to fair value",
                evidence=[f"Fair value range: {v.fair_value_range}"],
                confidence=0.7,
            ))

    # Moat
    if data.get("moat") and data["moat"].rating == "wide":
        m = data["moat"]
        args.append(DebateArgument(
            symbol=symbol,
            side="bull",
            argument=f"Wide moat: {m.score}/100 - sustainable competitive advantage",
            evidence=m.sources,
            confidence=0.75,
        ))

    # Sentiment
    if data.get("sentiment") and data["sentiment"].sentiment_label == "bullish":
        s = data["sentiment"]
        args.append(DebateArgument(
            symbol=symbol,
            side="bull",
            argument=f"Positive sentiment: {s.sentiment_score:.2f}, catalysts: {s.catalysts}",
            evidence=s.catalysts,
            confidence=0.6,
        ))

    return args


async def generate_bear_case(
    ctx: RunContext[AgentDeps],
    symbol: str,
    analysis_data: AnalysisReport | dict,
    trade_direction: str,
) -> list[DebateArgument]:
    """Generate bear arguments."""
    args = []

    if isinstance(analysis_data, AnalysisReport):
        data = analysis_data
    else:
        data = analysis_data

    # Fundamental bear case
    if data.get("fundamental"):
        f = data["fundamental"]
        if f.debt_equity and f.debt_equity > 1.5:
            args.append(DebateArgument(
                symbol=symbol,
                side="bear",
                argument=f"High leverage: D/E {f.debt_equity:.1f}, financial risk elevated",
                evidence=[f"D/E: {f.debt_equity:.1f}", f"Interest coverage: {f.get('interest_coverage', 'N/A')}"],
                confidence=0.7,
            ))
        if f.revenue_growth and f.revenue_growth < 5:
            args.append(DebateArgument(
                symbol=symbol,
                side="bear",
                argument=f"Slowing growth: Revenue growth only {f.revenue_growth}%",
                evidence=[f"Revenue growth: {f.revenue_growth}%"],
                confidence=0.6,
            ))

    # Technical bear case
    if data.get("technical"):
        t = data["technical"]
        if t.signal in ("sell", "strong_sell"):
            args.append(DebateArgument(
                symbol=symbol,
                side="bear",
                argument=f"Technical breakdown: {t.signal}, below key MAs",
                evidence=[f"Signal: {t.signal}", f"Below SMA200: {t.sma_200}"],
                confidence=0.65,
            ))
        if t.rsi and t.rsi > 75:
            args.append(DebateArgument(
                symbol=symbol,
                side="bear",
                argument=f"Overbought: RSI {t.rsi:.1f}, pullback risk high",
                evidence=[f"RSI: {t.rsi:.1f}"],
                confidence=0.6,
            ))

    # Valuation bear case
    if data.get("valuation") and data["valuation"].upside_pct:
        v = data["valuation"]
        if v.upside_pct < -10:
            args.append(DebateArgument(
                symbol=symbol,
                side="bear",
                argument=f"Overvalued: {abs(v.upside_pct):.1f}% above fair value",
                evidence=[f"Fair value: {v.fair_value_range}"],
                confidence=0.7,
            ))

    # Risks from all dimensions
    all_risks = []
    for key in ["fundamental", "technical", "sentiment"]:
        if data.get(key) and data[key].risks:
            all_risks.extend(data[key].risks)

    if all_risks:
        args.append(DebateArgument(
            symbol=symbol,
            side="bear",
            argument=f"Key risks: {'; '.join(all_risks[:3])}",
            evidence=all_risks[:5],
            confidence=0.65,
        ))

    return args


def synthesize_debate(bull_args: list[DebateArgument], bear_args: list[DebateArgument]) -> str:
    """Synthesize bull and bear cases into balanced view."""
    bull_points = [a.argument for a in bull_args]
    bear_points = [a.argument for a in bear_args]

    return f"""
BULL CASE: {'; '.join(bull_points[:3])}
BEAR CASE: {'; '.join(bear_points[:3])}

SYNTHESIS: The bull case rests on {len(bull_args)} arguments (avg confidence: {sum(a.confidence for a in bull_args)/len(bull_args):.2f if bull_args else 0:.2f}).
The bear case rests on {len(bear_args)} arguments (avg confidence: {sum(a.confidence for a in bear_args)/len(bear_args):.2f if bear_args else 0:.2f}).
""".strip()


def make_decision(
    bull_args: list[DebateArgument],
    bear_args: list[DebateArgument],
    trade_direction: str,
) -> tuple[str, float]:
    """Make final decision from debate."""
    bull_strength = sum(a.confidence for a in bull_args)
    bear_strength = sum(a.confidence for a in bear_args)

    if bull_strength > bear_strength * 1.5:
        return ("buy" if trade_direction == "buy" else "sell", min(bull_strength / (bull_strength + bear_strength), 0.9))
    elif bear_strength > bull_strength * 1.5:
        return ("sell" if trade_direction == "buy" else "buy", min(bear_strength / (bull_strength + bear_strength), 0.9))
    else:
        return ("hold", 0.5)