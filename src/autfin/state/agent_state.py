"""
AgentState - Main state shared across the v3.0 supervisor graph.

v3.0: Single supervisor ReAct loop with tool calling. No multi-graph.
v5.4: Added dict-merge reducers for parallel fan-out writes.
"""
from typing import TypedDict, Annotated, NotRequired
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
import operator


def dict_merge_reducer(existing: dict | None, update: dict | None) -> dict | None:
    """Merge two dicts for parallel fan-out writes.

    When multiple analyst nodes write to the same state key concurrently
    (via LangGraph Send fan-out), each writes {symbol: report}.  Without
    this reducer, last-writer-wins and earlier symbols are silently lost.

    With this reducer, updates are deep-merged:
        {"RELIANCE": report1} + {"TCS": report2}
        → {"RELIANCE": report1, "TCS": report2}

    If both write the SAME symbol, the later update wins (acceptable —
    the same symbol shouldn't be analyzed twice in parallel).
    """
    if existing is None:
        return update
    if update is None:
        return existing
    merged = dict(existing)
    merged.update(update)
    return merged


class AgentState(TypedDict):
    """
    v3.0 state — single graph, context-driven behavior.

    All tools read what they need and return partial updates.
    NEVER mutate state directly.
    """
    
    # Schema version for checkpoint migration safety
    schema_version: NotRequired[int]

    # Messages (accumulated across ReAct loop iterations)
    messages: Annotated[list[BaseMessage], add_messages]

    # ── Context — WHY was the supervisor invoked ──
    context: str                   # "morning_routine" | "price_alert" | "quarterly_review" | "manual"
    depth: NotRequired[str]        # "quick" | "standard" | "deep" (per-stock depth tier)
    alert_symbol: NotRequired[str]       # Set when context="price_alert"
    price_move_pct: NotRequired[float]   # Set when context="price_alert"
    
    # ── Capital (from Kite SDK — balance IS budget) ──
    available_cash: float
    utilized_margin: float
    available_margin: NotRequired[float]
    
    # ── Portfolio snapshot ──
    holdings: list[dict]            # [{tradingsymbol, qty, avg_price, ltp, pnl}]
    positions: list[dict]
    max_positions: NotRequired[int]

    # ── Screening output ──
    candidates: NotRequired[list[str] | None]        # Top 20 symbols from screening
    composite_ranks: NotRequired[dict[str, float] | None]  # {symbol: composite_score}
    
    # ── Research cache (keyed by symbol, populated concurrently by fan-out) ──
    market_snapshot: NotRequired[dict | None]
    macro_signals: NotRequired[dict | None]
    fundamental_reports: Annotated[dict | None, dict_merge_reducer]    # {symbol: {ratios, health, f_score, ...}}
    technical_reports: Annotated[dict | None, dict_merge_reducer]      # {symbol: {indicators, patterns, signal}}
    sentiment_reports: Annotated[dict | None, dict_merge_reducer]      # {symbol: {score, topics, catalysts}}
    valuation_reports: Annotated[dict | None, dict_merge_reducer]      # {symbol: {dcf_scenarios, reverse_dcf}}
    peer_comparisons: Annotated[dict | None, dict_merge_reducer]       # {symbol: {rankings vs peers}}
    moat_assessments: Annotated[dict | None, dict_merge_reducer]       # {symbol: {score, rating, sources}}

    # ── Synthesis ──
    research_debates: Annotated[dict | None, dict_merge_reducer]
    stock_pitches: Annotated[dict | None, dict_merge_reducer]          # {symbol: 6-element buy-side pitch}
    
    # ── Risk & Execution ──
    risk_assessments: Annotated[dict | None, dict_merge_reducer]
    trade_queue: NotRequired[list[dict] | None]
    execution_reports: NotRequired[list[dict] | None]

    # ── Learning context (retrieved BEFORE trade decisions) ──
    learning_context: NotRequired[dict | None]  # {similar_setups, calibration, patterns}

    # ── Routing ──
    next_agent: NotRequired[str | None]
    routing_history: NotRequired[list[str] | None]

    # ── Metadata ──
    thread_id: str
    run_mode: str
    errors: Annotated[list[str], operator.add]

    # ── Token budget ──
    total_tokens: int
    total_cost: float
    budget_limit: float

    # ── Graph iteration tracking ──
    iteration_count: int

    # ── Legacy guardian fields (still valid for v3.0 guardian track) ──
    guardian_sell_signals: NotRequired[list[dict] | None]
    freed_cash: NotRequired[float]
    guardian_sell_reports: NotRequired[list[dict] | None]
    thesis_check_results: Annotated[dict | None, dict_merge_reducer]

    # ── HITL ──
    pending_human_approval: NotRequired[dict | None]
    human_decision: NotRequired[str | None]

    # ── Validator / self-correction ──
    validation_count: int
    validator_verdict: NotRequired[str | None]
    validator_confidence: NotRequired[float]
    validator_issues: NotRequired[list[str] | None]
    validator_suggestions: NotRequired[list[str] | None]
    validator_note: NotRequired[str | None]
    should_reanalyze: NotRequired[bool]

    # ── Memory ──
    memory_context: NotRequired[dict | None]
    context_memory: NotRequired[dict | None]      # v5.3: cross-session memories from LangGraph Store
    memories_loaded: NotRequired[bool]             # v5.3: flag to load memories only once

    # ── Order management ──
    order_modifications: NotRequired[list[dict] | None]
    order_cancellations: NotRequired[list[dict] | None]
    pending_trade: NotRequired[dict | None]
    recent_trades: NotRequired[list[dict] | None]

    # ── Researcher output extras ──
    researcher_notes: NotRequired[list[str] | None]
    researcher_recommendation: NotRequired[str | None]
    catalyst_track: Annotated[dict | None, dict_merge_reducer]
    catalyst_impact_report: Annotated[dict | None, dict_merge_reducer]

    # ── Quarterly ──
    rebalancing_recommendations: NotRequired[list[dict] | None]
    research_reports: Annotated[dict | None, dict_merge_reducer]

    # ── v5.4 additions ──
    watchlist: NotRequired[list[str]]                                  # populated by Watchlist Curator
    distortion_reports: Annotated[dict | None, dict_merge_reducer]     # {symbol: distortion audit}
    regulatory_reports: Annotated[dict | None, dict_merge_reducer]     # {symbol: regulatory analysis}
    fii_dii_data: NotRequired[dict | None]                             # FII/DII daily data
    cache_invalidation_tokens: NotRequired[list[str]]                  # supervisor's invalidation directives
    news_velocity_reports: Annotated[dict | None, dict_merge_reducer]  # {symbol: news velocity analysis}

    # ── v5.6: 8-Agent System — Debate ──
    debate_results: Annotated[dict | None, dict_merge_reducer]        # {symbol: DebateResult}
    bull_arguments: Annotated[dict | None, dict_merge_reducer]        # {symbol: DebateArgument}
    bear_arguments: Annotated[dict | None, dict_merge_reducer]        # {symbol: DebateArgument}

    # ── v5.6: 8-Agent System — Conferences ──
    conference_results: NotRequired[list[dict] | None]                # list of ConferenceResult dicts
    active_conference: NotRequired[str | None]                        # "morning" | "trade_review" | "emergency" | "experience"
    market_regime: NotRequired[str]                                   # "normal" | "high_volatility" | "crash" | "rally" | "low_liquidity"

    # ── v5.6: 8-Agent System — Analyst Reports ──
    global_analyst_report: NotRequired[dict | None]                   # GlobalAnalyst output
    india_analyst_report: NotRequired[dict | None]                    # IndiaAnalyst output
    sentiment_analyst_report: NotRequired[dict | None]                # SentimentAnalyst output
    technical_analyst_report: NotRequired[dict | None]                # TechnicalAnalyst output

    # ── v5.6: 8-Agent System — Memory ──
    market_memory: NotRequired[list[dict] | None]                     # recent market events
    reflection_memory: NotRequired[list[dict] | None]                 # past trade outcomes
    experience_memory: NotRequired[list[dict] | None]                 # discovered patterns
    memory_stats: NotRequired[dict | None]                            # memory system statistics

    # ── v5.6: 8-Agent System — Risk Manager (veto power) ──
    risk_veto: NotRequired[bool]                                      # true if risk manager vetoed trade
    risk_veto_reason: NotRequired[str | None]                         # reason for veto
    position_limits: NotRequired[dict | None]                         # {symbol: limit_amount}
    sector_exposure: NotRequired[dict | None]                         # {sector: percentage}