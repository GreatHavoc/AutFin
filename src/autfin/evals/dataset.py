"""AutFin v7.0 — Evaluation Framework (Enhanced)

Pydantic Evals configuration with LLM judges, span-based evaluation,
financial domain evaluators, and Logfire integration.
"""
from __future__ import annotations

import asyncio

# ponytail: removed Decimal — NIM xgrammar chokes on its regex pattern
from typing import Any

import logfire
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    Evaluator,
    EvaluatorContext,
    HasMatchingSpan,
    IsInstance,
    LLMJudge,
)

from autfin.types import AgentDeps

# ─── Logfire Configuration for Evals ───

def configure_evals_logfire(project_name: str = "autfin-evals"):
    """Configure Logfire for evals with span capture."""
    logfire.configure(
        project=project_name,
        send_to_logfire=False,  # Set True in CI/CD
        scrubbing_options=logfire.ScrubbingOptions(
            redact_credit_cards=True,
            redact_email_addresses=True,
            redact_phone_numbers=True,
            redact_ip_addresses=True,
            redact_us_social_security_numbers=True,
        ),
    )
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx(capture_all=True)


# ─── Test Case Factories ───

def create_morning_routine_cases() -> list[Case]:
    """Create morning routine test cases."""
    return [
        Case(
            name="morning_routine_basic",
            inputs={
                "context": "morning_routine",
                "holdings": [
                    {"tradingsymbol": "RELIANCE", "qty": 100, "avg_price": 2500, "ltp": 2600, "pnl": 10000, "sector": "ENERGY"},
                    {"tradingsymbol": "TCS", "qty": 50, "avg_price": 3500, "ltp": 3600, "pnl": 5000, "sector": "IT"},
                ],
                "cash": 100000,
            },
            expected_output={"action": "continue", "contains": ["RELIANCE", "TCS", "portfolio"]},
            metadata={"category": "morning_routine", "priority": "high"},
        ),
        Case(
            name="morning_routine_empty_portfolio",
            inputs={
                "context": "morning_routine",
                "holdings": [],
                "cash": 1000000,
            },
            expected_output={"action": "complete", "contains": ["portfolio", "cash"]},
            metadata={"category": "morning_routine", "priority": "medium"},
        ),
        Case(
            name="morning_routine_full_scan",
            inputs={
                "context": "morning_routine",
                "holdings": [
                    {"tradingsymbol": "RELIANCE", "qty": 100, "avg_price": 2500, "ltp": 2600, "pnl": 10000, "sector": "ENERGY"},
                    {"tradingsymbol": "TCS", "qty": 50, "avg_price": 3500, "ltp": 3600, "pnl": 5000, "sector": "IT"},
                    {"tradingsymbol": "HDFCBANK", "qty": 75, "avg_price": 1600, "ltp": 1650, "pnl": 3750, "sector": "FINANCIAL"},
                ],
                "cash": 500000,
            },
            expected_output={"action": "continue", "contains": ["RELIANCE", "TCS", "HDFCBANK", "trade_queue"]},
            metadata={"category": "morning_routine", "priority": "high"},
        ),
    ]


def create_price_alert_cases() -> list[Case]:
    """Create price alert test cases."""
    return [
        Case(
            name="price_alert_large_move_up",
            inputs={
                "context": "price_alert",
                "symbol": "RELIANCE",
                "price_move_pct": 5.5,
            },
            expected_output={"symbol": "RELIANCE", "contains": ["RELIANCE", "risk", "alert", "trim"]},
            metadata={"category": "price_alert", "priority": "high"},
        ),
        Case(
            name="price_alert_large_move_down",
            inputs={
                "context": "price_alert",
                "symbol": "TCS",
                "price_move_pct": -4.2,
            },
            expected_output={"symbol": "TCS", "contains": ["TCS", "risk", "alert", "review"]},
            metadata={"category": "price_alert", "priority": "high"},
        ),
        Case(
            name="price_alert_small_move",
            inputs={
                "context": "price_alert",
                "symbol": "INFY",
                "price_move_pct": 1.2,
            },
            expected_output={"symbol": "INFY", "contains": ["INFY", "monitor"]},
            metadata={"category": "price_alert", "priority": "medium"},
        ),
    ]


def create_risk_cases() -> list[Case]:
    """Create risk management test cases."""
    return [
        Case(
            name="risk_rejection_high_risk",
            inputs={
                "context": "manual",
                "trade": {
                    "symbol": "PENNY",
                    "action": "buy",
                    "quantity": 10000,
                    "price": 5.0,
                    "risk_score": 0.9,
                },
            },
            expected_output={"verdict": "reject", "contains": ["risk", "rejected", "high"]},
            metadata={"category": "risk_management", "priority": "high"},
        ),
        Case(
            name="risk_rejection_low_rr",
            inputs={
                "context": "manual",
                "trade": {
                    "symbol": "RELIANCE",
                    "action": "buy",
                    "quantity": 100,
                    "price": 2500,
                    "risk_reward": 1.2,
                    "conviction": 0.7,
                },
            },
            expected_output={"verdict": "reject", "contains": ["risk_reward", "below", "minimum"]},
            metadata={"category": "risk_management", "priority": "high"},
        ),
        Case(
            name="risk_rejection_low_conviction",
            inputs={
                "context": "manual",
                "trade": {
                    "symbol": "TCS",
                    "action": "buy",
                    "quantity": 50,
                    "price": 3500,
                    "risk_reward": 3.0,
                    "conviction": 0.3,
                },
            },
            expected_output={"verdict": "reject", "contains": ["conviction", "below", "minimum"]},
            metadata={"category": "risk_management", "priority": "high"},
        ),
        Case(
            name="risk_approval_good_trade",
            inputs={
                "context": "manual",
                "trade": {
                    "symbol": "RELIANCE",
                    "action": "buy",
                    "quantity": 100,
                    "price": 2500,
                    "risk_reward": 3.0,
                    "conviction": 0.7,
                },
            },
            expected_output={"verdict": "approve", "contains": ["approve", "RELIANCE"]},
            metadata={"category": "risk_management", "priority": "high"},
        ),
        Case(
            name="risk_conditional_sector_limit",
            inputs={
                "context": "manual",
                "trade": {
                    "symbol": "HDFCBANK",
                    "action": "buy",
                    "quantity": 200,
                    "price": 1600,
                    "risk_reward": 2.5,
                    "conviction": 0.6,
                },
                "sector_exposure": 0.35,  # Over 30% limit
            },
            expected_output={"verdict": "conditional", "contains": ["sector", "limit", "reduce"]},
            metadata={"category": "risk_management", "priority": "medium"},
        ),
    ]


def create_debate_cases() -> list[Case]:
    """Create debate test cases."""
    return [
        Case(
            name="debate_buy_decision_strong",
            inputs={
                "symbol": "RELIANCE",
                "analysis_data": {
                    "fundamental": {"health_score": 80, "recommendation": "buy"},
                    "technical": {"signal": "buy", "strength": 70},
                    "sentiment": {"sentiment_label": "bullish"},
                    "valuation": {"upside_pct": 20},
                },
            },
            expected_output={"decision": "buy", "contains": ["buy", "confidence"]},
            metadata={"category": "debate", "priority": "high"},
        ),
        Case(
            name="debate_sell_decision",
            inputs={
                "symbol": "PENNY",
                "analysis_data": {
                    "fundamental": {"health_score": 20, "recommendation": "sell", "debt_equity": 3.0},
                    "technical": {"signal": "sell", "strength": 30, "rsi": 80},
                    "sentiment": {"sentiment_label": "bearish"},
                },
            },
            expected_output={"decision": "sell", "contains": ["sell", "confidence"]},
            metadata={"category": "debate", "priority": "high"},
        ),
        Case(
            name="debate_hold_insufficient_evidence",
            inputs={
                "symbol": "MIDCAP",
                "analysis_data": {
                    "fundamental": {"health_score": 50, "recommendation": "hold"},
                    "technical": {"signal": "hold", "strength": 50},
                    "sentiment": {"sentiment_label": "neutral"},
                },
            },
            expected_output={"decision": "hold", "contains": ["hold", "insufficient"]},
            metadata={"category": "debate", "priority": "medium"},
        ),
    ]


def create_analysis_cases() -> list[Case]:
    """Create analysis test cases."""
    return [
        Case(
            name="analysis_strong_buy",
            inputs={
                "symbol": "RELIANCE",
                "depth": "standard",
            },
            expected_output={"recommendation": "buy", "contains": ["RELIANCE", "recommendation", "overall_score"]},
            metadata={"category": "analysis", "priority": "high"},
        ),
        Case(
            name="analysis_strong_sell",
            inputs={
                "symbol": "PENNY",
                "depth": "deep",
            },
            expected_output={"recommendation": "sell", "contains": ["PENNY", "sell", "weak"]},
            metadata={"category": "analysis", "priority": "high"},
        ),
        Case(
            name="analysis_hold_neutral",
            inputs={
                "symbol": "MIDCAP",
                "depth": "quick",
            },
            expected_output={"recommendation": "hold", "contains": ["MIDCAP", "hold", "neutral"]},
            metadata={"category": "analysis", "priority": "medium"},
        ),
    ]


def create_memory_cases() -> list[Case]:
    """Create memory test cases."""
    return [
        Case(
            name="memory_record_trade",
            inputs={
                "action": "record_trade",
                "trade_data": {
                    "symbol": "RELIANCE",
                    "action": "buy",
                    "quantity": 100,
                    "entry_price": 2500,
                    "exit_price": 2600,
                    "pnl_pct": 4.0,
                },
            },
            expected_output={"action": "recorded", "contains": ["recorded", "TRADE_OUTCOMES"]},
            metadata={"category": "memory", "priority": "medium"},
        ),
        Case(
            name="memory_get_calibration",
            inputs={
                "action": "get_calibration",
                "agent_name": "risk",
                "context": "morning_routine",
            },
            expected_output={"calibration": {"total_trades": 0}, "contains": ["calibration", "brier"]},
            metadata={"category": "memory", "priority": "medium"},
        ),
    ]


# ─── All Test Cases ───

ALL_TEST_CASES = (
    create_morning_routine_cases() +
    create_price_alert_cases() +
    create_risk_cases() +
    create_debate_cases() +
    create_analysis_cases() +
    create_memory_cases()
)


# ─── Built-in Evaluators ───

BUILTIN_EVALUATORS = [
    IsInstance(type_name="dict"),
    # EqualsExpected(expected_output=...),  # Requires exact match
    # Contains(value="required_field"),
    # MaxDuration(max_seconds=30.0),
]


# ─── Custom Financial Domain Evaluators ───

class TradeQualityEvaluator(Evaluator[dict, Any]):
    """Evaluate trade recommendation quality."""
    
    def evaluate(self, ctx: EvaluatorContext[dict, Any]) -> float:
        output = ctx.output
        
        # Must have required fields
        required = ["symbol", "action", "quantity", "risk_reward", "conviction"]
        if not all(field in output for field in required):
            return 0.0
        
        # Risk/reward >= 2.0
        rr = output.get("risk_reward", 0)
        if rr < 2.0:
            return 0.3
        
        # Conviction >= 0.5
        conv = output.get("conviction", 0)
        if conv < 0.5:
            return 0.4
        
        return 1.0


class RiskVetoEvaluator(Evaluator[dict, Any]):
    """Verify risk agent vetoes high-risk trades."""
    
    def evaluate(self, ctx: EvaluatorContext[dict, Any]) -> float:
        output = ctx.output
        verdict = output.get("verdict")
        
        if verdict == "reject":
            return 1.0  # Correctly rejected
        elif verdict == "conditional":
            return 0.8  # Acceptable with conditions
        elif verdict == "approve":
            # Check if it should have been rejected
            risk_factors = ctx.inputs.get("risk_factors", {})
            if risk_factors.get("high_risk"):
                return 0.0  # Should have rejected
            return 1.0
        else:
            return 0.5


class DebateQualityEvaluator(Evaluator[dict, Any]):
    """Evaluate debate output quality."""
    
    def evaluate(self, ctx: EvaluatorContext[dict, Any]) -> float:
        output = ctx.output
        
        required = ["bull_thesis", "bear_thesis", "decision", "confidence"]
        if not all(field in output for field in required):
            return 0.0
        
        # Both sides must have arguments
        if not output.get("bull_thesis") or not output.get("bear_thesis"):
            return 0.3
        
        # Confidence >= 0.5 for buy/sell
        decision = output.get("decision")
        confidence = output.get("confidence", 0)
        if decision in ("buy", "sell") and confidence < 0.5:
            return 0.4
        
        return 1.0


class AnalysisCompletenessEvaluator(Evaluator[dict, Any]):
    """Evaluate analysis report completeness."""
    
    def evaluate(self, ctx: EvaluatorContext[dict, Any]) -> float:
        output = ctx.output
        
        required_dims = ["fundamental", "technical", "sentiment", "recommendation", "confidence"]
        score = 0.0
        for dim in required_dims:
            if output.get(dim):
                score += 0.2
        
        return score


class PositionSizingEvaluator(Evaluator[dict, Any]):
    """Evaluate position sizing logic."""
    
    def evaluate(self, ctx: EvaluatorContext[dict, Any]) -> float:
        output = ctx.output
        
        if output.get("action") != "buy":
            return 1.0  # Not applicable for non-buy
        
        qty = output.get("quantity", 0)
        if qty <= 0:
            return 0.0
        
        # Verify position size is reasonable
        entry = output.get("entry_price", 0)
        stop = output.get("stop_loss", 0)
        if entry > 0 and stop > 0:
            risk_per_share = abs(entry - stop)
            if risk_per_share == 0:
                return 0.0
        
        return 1.0


# ─── LLM Judge Evaluators (Financial Domain) ───

FINANCIAL_LLM_JUDGES = [
    # Trade Recommendation Quality
    LLMJudge(
        rubric="""
        Evaluate the trade recommendation:
        1. Does it specify symbol, action, quantity, entry price, stop loss, target?
        2. Is risk/reward >= 2:1?
        3. Is conviction >= 0.5?
        4. Are stop loss and target reasonable (not too tight/wide)?
        5. Does it respect position size limits?
        """,
        include_input=True,
        assertion={"evaluation_name": "trade_quality", "include_reason": True},
        model_settings={"temperature": 0.0},
    ),
    
    # Risk Assessment Quality
    LLMJudge(
        rubric="""
        Evaluate the risk assessment:
        1. Does it check all required risk factors (position size, sector exposure, portfolio heat, correlation)?
        2. Is the verdict (approve/reject/conditional) justified?
        3. Are stop loss and position size calculated correctly?
        3. Does it reference specific analysis data?
        """,
        include_input=True,
        assertion={"evaluation_name": "risk_quality", "include_reason": True},
        model_settings={"temperature": 0.0},
    ),
    
    # Debate Quality
    LLMJudge(
        rubric="""
        Evaluate the debate output:
        1. Are both bull and bear theses present and substantive?
        2. Does each side use specific evidence from analysis_data?
        3. Is there a synthesis integrating both sides?
        4. Is the decision (buy/sell/hold/pass) consistent with the arguments?
        5. Is confidence >= 0.5 for buy/sell decisions?
        """,
        include_input=True,
        assertion={"evaluation_name": "debate_quality", "include_reason": True},
        model_settings={"temperature": 0.0},
    ),
    
    # Analysis Completeness
    LLMJudge(
        rubric="""
        Evaluate the analysis report:
        1. Are all dimensions covered (fundamental, technical, sentiment, valuation, peers, moat)?
        2. Is the overall score a weighted average of dimensions?
        3. Is the recommendation consistent with the score?
        4. Are key risks and catalysts identified?
        5. Is the confidence score justified?
        """,
        include_input=True,
        assertion={"evaluation_name": "analysis_quality", "include_reason": True},
        model_settings={"temperature": 0.0},
    ),
]


# ─── Span-Based Evaluators ───

SPAN_BASED_EVALUATORS = [
    # Verify sub-agent delegation
    HasMatchingSpan(
        query={"name_contains": "delegate_task"},
        evaluation_name="used_delegation",
    ),
    
    # Verify market_data was called
    HasMatchingSpan(
        query={"name_contains": "market_data"},
        evaluation_name="called_market_data",
    ),
    
    # Verify analysis was called
    HasMatchingSpan(
        query={"name_contains": "analyze"},
        evaluation_name="called_analysis",
    ),
    
    # Verify debate was called before trade
    HasMatchingSpan(
        query={"name_contains": "debate"},
        evaluation_name="called_debate_before_trade",
    ),
    
    # Verify risk check was called
    HasMatchingSpan(
        query={"name_contains": "risk"},
        evaluation_name="called_risk_check",
    ),
    
    # No errors in execution
    HasMatchingSpan(
        query={
            "not_": {"has_status": "error"},
            "no_descendant_has": {"has_status": "error"},
        },
        evaluation_name="no_errors",
    ),
    
    # Reasonable execution time
    HasMatchingSpan(
        query={"name_equals": "task_execution", "max_duration": 60.0},
        evaluation_name="within_time_budget",
    ),
    
    # No circular delegation
    HasMatchingSpan(
        query={
            "not_": {
                "and_": [
                    {"name_contains": "agent"},
                    {"some_descendant_has": {"name_contains": "agent"}},
                    {"some_ancestor_has": {"name_contains": "agent"}},
                ]
            }
        },
        evaluation_name="no_circular_delegation",
    ),
]


# ─── Custom Span Evaluators ───

class AgentBehaviorEvaluator(Evaluator[dict, Any]):
    """Custom evaluator for agent execution behavior using span tree."""
    
    def evaluate(self, ctx: EvaluatorContext[dict, Any]) -> dict[str, bool | int]:
        span_tree = ctx.span_tree
        
        # Find specific spans
        llm_spans = [n for n in span_tree if "llm" in n.name.lower()]
        tool_spans = [n for n in span_tree if "tool" in n.name.lower()]
        delegate_spans = [n for n in span_tree if "delegate" in n.name.lower()]
        
        # Calculate metrics
        total_llm_time = sum(s.duration.total_seconds() for s in llm_spans)
        total_tool_time = sum(s.duration.total_seconds() for s in tool_spans)
        
        return {
            "used_llm": len(llm_spans) > 0,
            "used_tools": len(tool_spans) > 0,
            "used_delegation": len(delegate_spans) > 0,
            "tool_count": len(tool_spans),
            "delegate_count": len(delegate_spans),
            "llm_fast": total_llm_time < 10.0,
            "total_llm_time_sec": total_llm_time,
            "total_tool_time_sec": total_tool_time,
        }


# ─── Dataset Builders ───

def create_autfin_dataset(
    include_llm_judges: bool = True,
    include_span_eval: bool = True,
) -> Dataset:
    """Create the complete AutFin evaluation dataset."""
    evaluators = list(BUILTIN_EVALUATORS)
    evaluators.extend([
        TradeQualityEvaluator(),
        RiskVetoEvaluator(),
        DebateQualityEvaluator(),
        AnalysisCompletenessEvaluator(),
        PositionSizingEvaluator(),
        AgentBehaviorEvaluator(),
    ])
    
    if include_llm_judges:
        evaluators.extend(FINANCIAL_LLM_JUDGES)
    
    if include_span_eval:
        evaluators.extend(SPAN_BASED_EVALUATORS)
    
    dataset = Dataset(
        name="autfin_v7",
        cases=ALL_TEST_CASES,
        evaluators=evaluators,
    )
    
    return dataset


def create_agent_specific_dataset(agent_name: str) -> Dataset:
    """Create dataset filtered for a specific agent."""
    category_map = {
        "market_data_agent": "morning_routine",
        "analysis_agent": "analysis",
        "risk_agent": "risk_management",
        "debate_agent": "debate",
        "memory_agent": "memory",
        "research_agent": "morning_routine",
        "conference_agent": "morning_routine",
    }
    
    category = category_map.get(agent_name, "")
    if not category:
        return create_autfin_dataset()
    
    filtered_cases = [c for c in ALL_TEST_CASES if c.metadata.get("category") == category]
    
    return Dataset(
        name=f"autfin_{agent_name}",
        cases=filtered_cases,
        evaluators=list(BUILTIN_EVALUATORS) + [
            TradeQualityEvaluator(),
            RiskVetoEvaluator(),
            DebateQualityEvaluator(),
            AnalysisCompletenessEvaluator(),
            PositionSizingEvaluator(),
            AgentBehaviorEvaluator(),
        ],
    )


# ─── Dataset Serialization ───

def save_dataset(dataset: Dataset, path: str, format: str = "json") -> None:
    """Save dataset to file."""
    import json
    
    if format == "json":
        data = {
            "name": dataset.name,
            "cases": [
                {
                    "name": case.name,
                    "inputs": case.inputs,
                    "expected_output": case.expected_output,
                    "metadata": case.metadata,
                    "evaluators": [str(e) for e in case.evaluators],
                }
                for case in dataset.cases
            ],
            "evaluators": [str(e) for e in dataset.evaluators],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    elif format == "yaml":
        import yaml
        data = {
            "name": dataset.name,
            "cases": [
                {
                    "name": case.name,
                    "inputs": case.inputs,
                    "expected_output": case.expected_output,
                    "metadata": case.metadata,
                }
                for case in dataset.cases
            ],
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


def load_dataset(path: str, format: str = "json") -> Dataset:
    """Load dataset from file."""
    import json
    
    if format == "json":
        with open(path, "r") as f:
            data = json.load(f)
        
        cases = []
        for case_data in data["cases"]:
            cases.append(Case(
                name=case_data["name"],
                inputs=case_data["inputs"],
                expected_output=case_data.get("expected_output"),
                metadata=case_data.get("metadata", {}),
            ))
        
        return Dataset(
            name=data["name"],
            cases=cases,
            evaluators=list(BUILTIN_EVALUATORS) + [
                TradeQualityEvaluator(),
                RiskVetoEvaluator(),
                DebateQualityEvaluator(),
                AnalysisCompletenessEvaluator(),
                PositionSizingEvaluator(),
                AgentBehaviorEvaluator(),
            ],
        )
    elif format == "yaml":
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        cases = []
        for case_data in data["cases"]:
            cases.append(Case(
                name=case_data["name"],
                inputs=case_data["inputs"],
                expected_output=case_data.get("expected_output"),
                metadata=case_data.get("metadata", {}),
            ))
        
        return Dataset(
            name=data["name"],
            cases=cases,
            evaluators=list(BUILTIN_EVALUATORS) + [
                TradeQualityEvaluator(),
                RiskVetoEvaluator(),
                DebateQualityEvaluator(),
                AnalysisCompletenessEvaluator(),
                PositionSizingEvaluator(),
                AgentBehaviorEvaluator(),
            ],
        )


# ─── Evaluation Runner ───

def create_test_deps(agent_name: str = "eval") -> AgentDeps:
    """Create test dependencies for evaluation."""
    return AgentDeps(
        user_id=f"eval_{agent_name}",
        capital=1_000_000.0,
        mode="paper",
    )


# ─── CLI Entry Point ───

async def main():
    """Main entry point for running evaluations."""
    configure_evals_logfire()
    
    # This would import actual agents
    # from autfin.agents import market_data_agent, analysis_agent, risk_agent, debate_agent, memory_agent
    
    # For now, just validate dataset
    dataset = create_autfin_dataset(
        include_llm_judges=False,  # Set True when API keys available
        include_span_eval=True,
    )
    print(f"Dataset: {dataset.name}")
    print(f"Cases: {len(dataset.cases)}")
    print(f"Evaluators: {len(dataset.evaluators)}")
    
    for case in dataset.cases:
        cat = case.metadata.get("category", "unknown")
        priority = case.metadata.get("priority", "medium")
        print(f"  - {case.name} ({cat}, {priority})")
    
    # Test serialization
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        save_dataset(dataset, f.name, "json")
        print(f"\nSaved dataset to {f.name}")
    
    print("\nPhase 5 Evaluation Framework Ready!")


if __name__ == "__main__":
    asyncio.run(main())