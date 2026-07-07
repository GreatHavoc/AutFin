"""
Debate System — Adversarial Bull/Bear Research.

Two researcher agents explicitly argue opposing positions before every
trade decision. Based on TradingAgents (Columbia/NYU) — ablation studies
show systems without debate underperform by 20-40%.

Architecture:
  1. Bull Researcher argues FOR the trade
  2. Bear Researcher argues AGAINST the trade
  3. Synthesizer reads both arguments, produces final verdict

The debate forces counter-arguments and stress-tests every assumption.
No trade proceeds without explicit bear case being addressed.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from src.autfin.llm.client import get_client, primary_model

logger = logging.getLogger(__name__)


@dataclass
class DebateArgument:
    """One side's argument in the debate."""
    side: str  # "bull" or "bear"
    thesis: str
    key_points: list[str]
    risk_factors: list[str]
    conviction: float  # 0.0 to 1.0
    evidence: list[dict]  # supporting data points


@dataclass
class DebateResult:
    """Final output of a bull/bear debate."""
    symbol: str
    bull_argument: DebateArgument
    bear_argument: DebateArgument
    verdict: str  # "buy" | "sell" | "hold" | "insufficient_evidence"
    final_conviction: float  # adjusted after debate
    key_insights: list[str]  # most important points from both sides
    unresolved_questions: list[str]  # what couldn't be resolved
    confidence_in_verdict: float  # how confident we are in the verdict itself


BULL_PROMPT = """You are the BULL RESEARCHER for {symbol}. Your job is to argue FOR buying/holding this stock.

You have access to analysis data. Build the strongest possible case for {symbol}.

Your argument must include:
1. THESIS: One-sentence investment thesis (why this stock will outperform)
2. KEY POINTS: 3-5 strongest arguments with supporting data
3. RISK FACTORS: Acknowledge the top 1-2 risks (shows intellectual honesty)
4. CONVICTION: Your conviction score (0.0-1.0) based on the evidence
5. EVIDENCE: Specific data points that support your case

Be rigorous. Don't exaggerate. A strong bull case acknowledges weaknesses.

ANALYSIS DATA:
{analysis_data}

{additional_context}

Respond in JSON:
{{
    "thesis": "...",
    "key_points": ["...", "..."],
    "risk_factors": ["..."],
    "conviction": 0.75,
    "evidence": [{{"type": "...", "detail": "...", "strength": "strong|moderate|weak"}}]
}}"""


BEAR_PROMPT = """You are the BEAR RESEARCHER for {symbol}. Your job is to argue AGAINST buying/holding this stock.

You have access to analysis data. Build the strongest possible case against {symbol}.

Your argument must include:
1. THESIS: One-sentence thesis for why this stock will underperform or is overvalued
2. KEY POINTS: 3-5 strongest arguments against with supporting data
3. STRENGTHS: Acknowledge 1-2 genuine strengths (shows you're not biased)
4. CONVICTION: Your conviction score for the bear case (0.0-1.0)
5. EVIDENCE: Specific data points that support your bear case

Be rigorous. Don't cherry-pick. A strong bear case acknowledges genuine strengths.

ANALYSIS DATA:
{analysis_data}

{additional_context}

Respond in JSON:
{{
    "thesis": "...",
    "key_points": ["...", "..."],
    "strengths_acknowledged": ["..."],
    "conviction": 0.65,
    "evidence": [{{"type": "...", "detail": "...", "strength": "strong|moderate|weak"}}]
}}"""


SYNTHESIS_PROMPT = """You are the DEBATE SYNTHESIZER for {symbol}. You have read arguments from both the Bull and Bear researchers.

Your job is to produce a final, balanced verdict. Consider:
- Which arguments are strongest (backed by concrete evidence)?
- Where did the debate produce new insights?
- What questions remain unresolved?
- What is the risk/reward profile?

BULL ARGUMENT:
{bull_argument}

BEAR ARGUMENT:
{bear_argument}

ADDITIONAL CONTEXT:
{additional_context}

Respond in JSON:
{{
    "verdict": "buy" | "sell" | "hold" | "insufficient_evidence",
    "final_conviction": 0.65,
    "reasoning": "Brief explanation of why this verdict",
    "key_insights": ["Most important insight from the debate", "..."],
    "unresolved_questions": ["What we couldn't determine", "..."],
    "confidence_in_verdict": 0.8,
    "bull_score": 0.7,
    "bear_score": 0.6,
    "risk_reward_ratio": "2.5:1"
}}"""


async def run_bull_argument(
    symbol: str,
    analysis_data: str,
    additional_context: str = "",
) -> DebateArgument:
    """Run the Bull Researcher agent."""
    client = get_client()
    prompt = BULL_PROMPT.format(
        symbol=symbol,
        analysis_data=analysis_data,
        additional_context=additional_context or "No additional context.",
    )

    try:
        response = await client.chat.completions.create(
            model=primary_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000,
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)

        return DebateArgument(
            side="bull",
            thesis=data.get("thesis", ""),
            key_points=data.get("key_points", []),
            risk_factors=data.get("risk_factors", []),
            conviction=float(data.get("conviction", 0.5)),
            evidence=data.get("evidence", []),
        )
    except Exception as e:
        logger.error("Bull researcher failed for %s: %s", symbol, e)
        return DebateArgument(
            side="bull",
            thesis=f"Error: {e}",
            key_points=[],
            risk_factors=[],
            conviction=0.0,
            evidence=[],
        )


async def run_bear_argument(
    symbol: str,
    analysis_data: str,
    additional_context: str = "",
) -> DebateArgument:
    """Run the Bear Researcher agent."""
    client = get_client()
    prompt = BEAR_PROMPT.format(
        symbol=symbol,
        analysis_data=analysis_data,
        additional_context=additional_context or "No additional context.",
    )

    try:
        response = await client.chat.completions.create(
            model=primary_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000,
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)

        return DebateArgument(
            side="bear",
            thesis=data.get("thesis", ""),
            key_points=data.get("key_points", []),
            risk_factors=data.get("strengths_acknowledged", []),
            conviction=float(data.get("conviction", 0.5)),
            evidence=data.get("evidence", []),
        )
    except Exception as e:
        logger.error("Bear researcher failed for %s: %s", symbol, e)
        return DebateArgument(
            side="bear",
            thesis=f"Error: {e}",
            key_points=[],
            risk_factors=[],
            conviction=0.0,
            evidence=[],
        )


async def run_debate(
    symbol: str,
    analysis_data: str,
    additional_context: str = "",
) -> DebateResult:
    """
    Run a full bull/bear debate on a stock.

    1. Bull argues FOR
    2. Bear argues AGAINST (sees bull's argument for counter-points)
    3. Synthesizer reads both, produces verdict

    Returns DebateResult with both arguments and final verdict.
    """
    import asyncio

    # Step 1: Run bull and bear in parallel
    bull_task = run_bull_argument(symbol, analysis_data, additional_context)
    bear_task = run_bear_argument(symbol, analysis_data, additional_context)

    bull_arg, bear_arg = await asyncio.gather(bull_task, bear_task)

    # Step 2: Synthesize — bear sees bull's argument for counter-points
    bull_text = json.dumps({
        "thesis": bull_arg.thesis,
        "key_points": bull_arg.key_points,
        "risk_factors": bull_arg.risk_factors,
        "conviction": bull_arg.conviction,
    }, indent=2)

    bear_text = json.dumps({
        "thesis": bear_arg.thesis,
        "key_points": bear_arg.key_points,
        "strengths_acknowledged": bear_arg.risk_factors,
        "conviction": bear_arg.conviction,
    }, indent=2)

    client = get_client()
    synth_prompt = SYNTHESIS_PROMPT.format(
        symbol=symbol,
        bull_argument=bull_text,
        bear_argument=bear_text,
        additional_context=additional_context or "No additional context.",
    )

    try:
        response = await client.chat.completions.create(
            model=primary_model(),
            messages=[{"role": "user", "content": synth_prompt}],
            temperature=0.2,
            max_tokens=2000,
        )
        content = response.choices[0].message.content or "{}"
        synth = json.loads(content)
    except Exception as e:
        logger.error("Debate synthesis failed for %s: %s", symbol, e)
        synth = {
            "verdict": "insufficient_evidence",
            "final_conviction": 0.0,
            "reasoning": f"Synthesis failed: {e}",
            "key_insights": [],
            "unresolved_questions": ["Synthesis failed"],
            "confidence_in_verdict": 0.0,
        }

    return DebateResult(
        symbol=symbol,
        bull_argument=bull_arg,
        bear_argument=bear_arg,
        verdict=synth.get("verdict", "insufficient_evidence"),
        final_conviction=float(synth.get("final_conviction", 0.5)),
        key_insights=synth.get("key_insights", []),
        unresolved_questions=synth.get("unresolved_questions", []),
        confidence_in_verdict=float(synth.get("confidence_in_verdict", 0.5)),
    )


def format_debate_result(result: DebateResult) -> str:
    """Format a DebateResult for display or storage."""
    lines = [
        f"=== DEBATE: {result.symbol} ===",
        "",
        f"VERDICT: {result.verdict.upper()} (conviction: {result.final_conviction:.0%})",
        f"Confidence in verdict: {result.confidence_in_verdict:.0%}",
        "",
        "--- BULL CASE ---",
        f"Thesis: {result.bull_argument.thesis}",
        "Key points:",
    ]
    for pt in result.bull_argument.key_points:
        lines.append(f"  + {pt}")
    lines.append(f"Conviction: {result.bull_argument.conviction:.0%}")

    lines.extend([
        "",
        "--- BEAR CASE ---",
        f"Thesis: {result.bear_argument.thesis}",
        "Key points:",
    ])
    for pt in result.bear_argument.key_points:
        lines.append(f"  - {pt}")
    lines.append(f"Conviction: {result.bear_argument.conviction:.0%}")

    if result.key_insights:
        lines.extend(["", "--- KEY INSIGHTS ---"])
        for insight in result.key_insights:
            lines.append(f"  * {insight}")

    if result.unresolved_questions:
        lines.extend(["", "--- UNRESOLVED ---"])
        for q in result.unresolved_questions:
            lines.append(f"  ? {q}")

    return "\n".join(lines)
