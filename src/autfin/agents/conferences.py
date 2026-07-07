"""
Conference System — Multi-Agent Coordination Mechanisms.

Inspired by HedgeAgents (2025) — their ablation study shows removing
coordination mechanisms drops performance by 17-38%. Coordination is
not optional overhead; it's the source of edge.

Four conference types:
  1. Morning Conference — daily: all analysts report, allocate capital
  2. Trade Review Conference — on-demand: bull+bear debate, risk gates
  3. Emergency Conference — triggered: when market moves > 3% intraday
  4. Experience Conference — weekly: agents share lessons, update memory
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from src.autfin.llm.client import get_client, primary_model

logger = logging.getLogger(__name__)


class ConferenceType(str, Enum):
    MORNING = "morning"
    TRADE_REVIEW = "trade_review"
    EMERGENCY = "emergency"
    EXPERIENCE = "experience"


class MarketRegime(str, Enum):
    NORMAL = "normal"
    HIGH_VOLATILITY = "high_volatility"
    CRASH = "crash"
    RALLY = "rally"
    LOW_LIQUIDITY = "low_liquidity"


@dataclass
class AnalystReport:
    """Report from one analyst agent."""
    agent: str  # "global", "india", "sentiment", "technical"
    timestamp: datetime
    findings: list[dict]
    confidence: float
    regime: MarketRegime
    alerts: list[str] = field(default_factory=list)


@dataclass
class ConferenceResult:
    """Output of a conference."""
    conference_type: ConferenceType
    timestamp: datetime
    regime: MarketRegime
    reports: list[AnalystReport]
    decisions: list[dict]  # specific action items
    capital_allocation: dict  # {symbol: amount}
    risk_alerts: list[str]
    next_meeting: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Morning Conference — daily capital allocation
# ---------------------------------------------------------------------------

MORNING_CONFERENCE_PROMPT = """You are running a MORNING CONFERENCE for a trading desk.

All 4 analysts have reported their findings. Your job:
1. Synthesize all analyst reports into a unified market view
2. Identify the MARKET REGIME (normal, high_volatility, crash, rally, low_liquidity)
3. Allocate capital across sectors/symbols based on conviction
4. Flag any risk alerts that need immediate attention
5. Set the meeting schedule for the day

ANALYST REPORTS:
{analyst_reports}

CURRENT PORTFOLIO:
{portfolio}

AVAILABLE CASH: ₹{cash:,.2f}

RULES:
- Max 30% in any single sector
- Keep 10% cash reserve minimum
- Only allocate to symbols with conviction ≥ 0.5
- Flag any position exceeding 20% of portfolio
- Emergency meeting trigger: if any analyst flags > 3 alerts

Respond in JSON:
{{
    "regime": "normal" | "high_volatility" | "crash" | "rally" | "low_liquidity",
    "regime_reasoning": "...",
    "capital_allocation": {{"RELIANCE": 50000, "TCS": 30000}},
    "sector_allocation": {{"IT": 0.3, "Banking": 0.25}},
    "risk_alerts": ["...", "..."],
    "decisions": [{{"action": "...", "symbol": "...", "reasoning": "..."}}],
    "next_meeting_hours": 4
}}"""


async def run_morning_conference(
    analyst_reports: list[AnalystReport],
    portfolio: list[dict],
    available_cash: float,
) -> ConferenceResult:
    """
    Run the Morning Conference.

    All 4 analysts report findings. The conference:
    1. Synthesizes into a unified market view
    2. Identifies market regime
    3. Allocates capital across sectors/symbols
    4. Flags risk alerts
    5. Sets next meeting time
    """
    # Format reports for the LLM
    reports_text = "\n\n".join([
        f"--- {r.agent.upper()} ANALYST ---\n"
        f"Regime: {r.regime.value}\n"
        f"Confidence: {r.confidence:.0%}\n"
        f"Findings:\n" + "\n".join(f"  - {f}" for f in r.findings) +
        (f"\nAlerts: {r.alerts}" if r.alerts else "")
        for r in analyst_reports
    ])

    portfolio_text = json.dumps(portfolio[:10], indent=2, default=str)

    client = get_client()
    prompt = MORNING_CONFERENCE_PROMPT.format(
        analyst_reports=reports_text,
        portfolio=portfolio_text,
        cash=available_cash,
    )

    try:
        response = await client.chat.completions.create(
            model=primary_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=3000,
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
    except Exception as e:
        logger.error("Morning conference failed: %s", e)
        data = {
            "regime": "normal",
            "regime_reasoning": f"Conference failed: {e}",
            "capital_allocation": {},
            "sector_allocation": {},
            "risk_alerts": [f"Conference error: {e}"],
            "decisions": [],
            "next_meeting_hours": 2,
        }

    regime_str = data.get("regime", "normal")
    try:
        regime = MarketRegime(regime_str)
    except ValueError:
        regime = MarketRegime.NORMAL

    next_hours = data.get("next_meeting_hours", 4)
    next_meeting = datetime.now().replace(
        hour=datetime.now().hour + next_hours
    )

    return ConferenceResult(
        conference_type=ConferenceType.MORNING,
        timestamp=datetime.now(),
        regime=regime,
        reports=analyst_reports,
        decisions=data.get("decisions", []),
        capital_allocation=data.get("capital_allocation", {}),
        risk_alerts=data.get("risk_alerts", []),
        next_meeting=next_meeting,
    )


# ---------------------------------------------------------------------------
# Trade Review Conference — on-demand bull+bear debate
# ---------------------------------------------------------------------------

TRADE_REVIEW_PROMPT = """You are running a TRADE REVIEW CONFERENCE for {symbol}.

The Bull and Bear researchers have debated. The Risk Manager has run checks.
Your job: make the final go/no-go decision.

BULL ARGUMENT:
{bull_argument}

BEAR ARGUMENT:
{bear_argument}

RISK ASSESSMENT:
{risk_assessment}

PORTFOLIO CONTEXT:
- Current position: {current_position}
- Available cash: ₹{cash:,.2f}
- Sector exposure: {sector_exposure}

RULES:
- No trade if risk/reward < 2:1
- No trade if conviction < 0.5
- No trade if sector concentration would exceed 30%
- If bull conviction > bear conviction + 0.2: lean toward BUY
- If bear conviction > bull conviction + 0.2: lean toward SELL
- Otherwise: HOLD and wait for clearer signal

Respond in JSON:
{{
    "decision": "buy" | "sell" | "hold" | "reject",
    "quantity": 0,
    "order_type": "MARKET" | "LIMIT",
    "limit_price": null,
    "reasoning": "...",
    "conditions": ["If X happens, reconsider"],
    "next_review_date": "YYYY-MM-DD"
}}"""


async def run_trade_review(
    symbol: str,
    bull_argument: str,
    bear_argument: str,
    risk_assessment: str,
    current_position: str,
    available_cash: float,
    sector_exposure: str,
) -> dict:
    """
    Run a Trade Review Conference for a specific stock.

    Combines bull/bear debate with risk assessment to make a final decision.
    """
    client = get_client()
    prompt = TRADE_REVIEW_PROMPT.format(
        symbol=symbol,
        bull_argument=bull_argument,
        bear_argument=bear_argument,
        risk_assessment=risk_assessment,
        current_position=current_position,
        cash=available_cash,
        sector_exposure=sector_exposure,
    )

    try:
        response = await client.chat.completions.create(
            model=primary_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)
    except Exception as e:
        logger.error("Trade review conference failed for %s: %s", symbol, e)
        return {
            "decision": "reject",
            "reasoning": f"Conference failed: {e}",
            "conditions": ["Retry after error resolved"],
        }


# ---------------------------------------------------------------------------
# Emergency Conference — triggered by extreme market moves
# ---------------------------------------------------------------------------

EMERGENCY_PROMPT = """EMERGENCY CONFERENCE TRIGGERED — Market move detected.

TRIGGER: {trigger_description}
REGIME: {regime}
TIME: {timestamp}

Current portfolio:
{portfolio}

Available cash: ₹{cash:,.2f}

IMMEDIATE ACTIONS NEEDED:
1. Assess damage to existing positions
2. Identify hedging opportunities
3. Determine if any positions need emergency exits
4. Check if any buy opportunities exist (panic = opportunity)

RULES:
- Capital preservation is priority #1
- Reduce position sizes by 50% in high-volatility regime
- Exit any position down > 10% unless thesis intact
- Maximum 3 emergency trades per conference
- Human approval required for any trade > ₹100,000

Respond in JSON:
{{
    "urgency": "critical" | "high" | "medium",
    "portfolio_impact": "severe" | "moderate" | "mild",
    "immediate_actions": [
        {{"action": "exit" | "reduce" | "hold" | "hedge", "symbol": "...", "reasoning": "..."}}
    ],
    "hedging_opportunities": [
        {{"symbol": "...", "instrument": "...", "reasoning": "..."}}
    ],
    "human_approval_needed": true | false,
    "human_approval_reason": "..."
}}"""


async def run_emergency_conference(
    trigger_description: str,
    regime: MarketRegime,
    portfolio: list[dict],
    available_cash: float,
) -> ConferenceResult:
    """
    Run an Emergency Conference triggered by extreme market moves.

    Priority: capital preservation over profit.
    """
    client = get_client()
    prompt = EMERGENCY_PROMPT.format(
        trigger_description=trigger_description,
        regime=regime.value,
        timestamp=datetime.now().isoformat(),
        portfolio=json.dumps(portfolio[:10], indent=2, default=str),
        cash=available_cash,
    )

    try:
        response = await client.chat.completions.create(
            model=primary_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
    except Exception as e:
        logger.error("Emergency conference failed: %s", e)
        data = {
            "urgency": "critical",
            "portfolio_impact": "unknown",
            "immediate_actions": [],
            "hedging_opportunities": [],
            "human_approval_needed": True,
            "human_approval_reason": f"Emergency conference failed: {e}",
        }

    return ConferenceResult(
        conference_type=ConferenceType.EMERGENCY,
        timestamp=datetime.now(),
        regime=regime,
        reports=[],
        decisions=data.get("immediate_actions", []),
        capital_allocation={},
        risk_alerts=[data.get("human_approval_reason", "")],
    )


# ---------------------------------------------------------------------------
# Experience Conference — weekly knowledge sharing
# ---------------------------------------------------------------------------

EXPERIENCE_PROMPT = """EXPERIENCE CONFERENCE — Weekly knowledge sharing session.

All agents share what they learned this week. Goals:
1. Identify patterns that worked/failed
2. Update conviction calibration
3. Share cross-domain insights
4. Update memory with new learnings

AGENT EXPERIENCES:
{agent_experiences}

RECENT TRADE OUTCOMES:
{trade_outcomes}

RULES:
- Be specific: "RELIANCE broke support at 2400 because of X" not "RELIANCE went down"
- Extract generalizable patterns, not just one-off events
- Update calibration if conviction scores were consistently wrong
- Flag any systematic biases detected

Respond in JSON:
{{
    "patterns_identified": [
        {{"pattern": "...", "evidence": "...", "confidence": "high|medium|low", "actionable": true}}
    ],
    "calibration_updates": [
        {{"tier": "...", "old_rate": 0.6, "new_rate": 0.65, "reasoning": "..."}}
    ],
    "cross_domain_insights": ["..."],
    "memory_updates": [
        {{"key": "...", "value": "...", "type": "market|reflection|experience"}}
    ]
}}"""


async def run_experience_conference(
    agent_experiences: list[dict],
    trade_outcomes: list[dict],
) -> dict:
    """
    Run the Experience Conference — weekly knowledge sharing.

    Agents share lessons, update calibration, extract patterns.
    """
    exp_text = "\n".join([
        f"--- {e.get('agent', 'unknown')} ---\n{e.get('experience', '')}"
        for e in agent_experiences
    ])

    outcomes_text = json.dumps(trade_outcomes[:20], indent=2, default=str)

    client = get_client()
    prompt = EXPERIENCE_PROMPT.format(
        agent_experiences=exp_text,
        trade_outcomes=outcomes_text,
    )

    try:
        response = await client.chat.completions.create(
            model=primary_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=3000,
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)
    except Exception as e:
        logger.error("Experience conference failed: %s", e)
        return {
            "patterns_identified": [],
            "calibration_updates": [],
            "cross_domain_insights": [],
            "memory_updates": [],
        }


# ---------------------------------------------------------------------------
# Emergency trigger detection
# ---------------------------------------------------------------------------

def should_trigger_emergency(
    market_data: dict,
    current_regime: MarketRegime = MarketRegime.NORMAL,
) -> tuple[bool, str]:
    """
    Check if market conditions warrant an Emergency Conference.

    Returns (should_trigger, reason).
    """
    # Check for crash conditions
    nifty_change = abs(market_data.get("nifty_change_pct", 0))
    if nifty_change >= 3.0:
        direction = "crashed" if market_data.get("nifty_change_pct", 0) < 0 else "surged"
        return True, f"Nifty {direction} {nifty_change:.1f}% intraday"

    # Check for extreme volatility
    india_vix = market_data.get("india_vix", 0)
    if india_vix >= 25:
        return True, f"India VIX at {india_vix:.1f} (extreme fear)"

    # Check for regime change
    if current_regime == MarketRegime.NORMAL and india_vix >= 20:
        return True, f"Regime shift to high volatility (VIX {india_vix:.1f})"

    return False, ""
