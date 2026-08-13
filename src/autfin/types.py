"""AutFin v7.0 — Pydantic AI Domain Types & Agent State

Pydantic BaseModel for type-safe agents.
All agents share AgentDeps and produce structured output types.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any, Literal, NotRequired, TypedDict
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage as BaseMessage

# ═══════════════════════════════════════════════════════════════════════════
# Model Configuration
# ═══════════════════════════════════════════════════════════════════════════

class ModelConfig(BaseModel):
    """Model routing configuration.

    - nvidia/nemotron-3.5-lightning-30b-a3b (deep/reasoning)
    - nvidia/nemotron-3.5-lightning-30b-a3b (fast/medium/code)
    """
    deep: str = "nvidia/nemotron-3.5-lightning-30b-a3b"
    fast: str = "nvidia/nemotron-3.5-lightning-30b-a3b"
    reasoning: str = "nvidia/nemotron-3.5-lightning-30b-a3b"
    code: str = "nvidia/nemotron-3.5-lightning-30b-a3b"

MODEL_CONFIG = ModelConfig()


# ═══════════════════════════════════════════════════════════════════════════
# Domain Models — Portfolio, Quotes, Market Data
# ═══════════════════════════════════════════════════════════════════════════

class Holding(BaseModel):
    """Single portfolio holding."""
    model_config = ConfigDict(validate_assignment=True)

    tradingsymbol: str
    qty: int
    avg_price: Decimal
    ltp: Decimal
    pnl: Decimal
    sector: str = ""
    exchange: str = "NSE"

    @property
    def pnl_pct(self) -> float:
        if self.avg_price == 0:
            return 0.0
        return float((self.ltp - self.avg_price) / self.avg_price * 100)

    @property
    def market_value(self) -> Decimal:
        return self.ltp * self.qty

    @property
    def invested_value(self) -> Decimal:
        return self.avg_price * self.qty


class Quote(BaseModel):
    """Real-time quote for a symbol."""
    model_config = ConfigDict(validate_assignment=True)

    symbol: str
    ltp: Decimal
    change: Decimal
    change_pct: float
    volume: int
    oi: int = 0
    bid: Decimal | None = None
    ask: Decimal | None = None
    high: Decimal | None = None
    low: Decimal | None = None
    open: Decimal | None = None
    prev_close: Decimal | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class MarketStatus(BaseModel):
    """Market status information."""
    model_config = ConfigDict(validate_assignment=True)

    is_open: bool
    phase: Literal["pre_market", "open", "post_market", "closed", "holiday"]
    next_open: datetime | None = None
    next_close: datetime | None = None
    timezone: str = "Asia/Kolkata"


class MarketSnapshot(BaseModel):
    """Complete market snapshot at a point in time."""
    model_config = ConfigDict(validate_assignment=True)

    quotes: dict[str, Quote]
    portfolio: list[Holding]
    market_status: MarketStatus
    timestamp: datetime = Field(default_factory=datetime.now)
    risk_alerts: list[str] = Field(default_factory=list)
    next_refresh: str = ""  # ISO format

    @property
    def total_portfolio_value(self) -> Decimal:
        return sum(h.market_value for h in self.portfolio)

    @property
    def total_pnl(self) -> Decimal:
        return sum(h.pnl for h in self.portfolio)

    @property
    def total_invested(self) -> Decimal:
        return sum(h.invested_value for h in self.portfolio)


# ═══════════════════════════════════════════════════════════════════════════
# Runtime Dependencies — Injected into all agents
# ═══════════════════════════════════════════════════════════════════════════

class AgentDeps(BaseModel):
    """Runtime dependencies injected into all agents via dependency injection."""
    model_config = ConfigDict(validate_assignment=True)

    # User/session context
    user_id: str = "default"
    session_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    thread_id: str = Field(default_factory=lambda: str(uuid4())[:8])

    # Risk profile
    risk_profile: Literal["conservative", "moderate", "aggressive"] = "moderate"
    capital: Decimal = Decimal("1000000")
    max_position_pct: float = 0.3
    sector_limit_pct: float = 0.3
    max_drawdown_pct: float = 0.15

    # Execution mode
    mode: Literal["paper", "live", "backtest"] = "paper"

    # Optional runtime services (populated by runtime)
    kite_client: Any = Field(default=None, exclude=True)
    db_pool: Any = Field(default=None, exclude=True)
    vector_store: Any = Field(default=None, exclude=True)

    # Time travel for backtests
    as_of: datetime | None = None

    @property
    def max_position_size(self) -> Decimal:
        return self.capital * Decimal(str(self.max_position_pct))

    @property
    def max_sector_exposure(self) -> Decimal:
        return self.capital * Decimal(str(self.sector_limit_pct))


# ═══════════════════════════════════════════════════════════════════════════
# Agent Output Types — Structured outputs for each agent
# ═══════════════════════════════════════════════════════════════════════════

class MarketSnapshotOutput(BaseModel):
    """MarketDataAgent output."""
    quotes: dict[str, Quote]
    portfolio: list[Holding]
    market_status: MarketStatus
    risk_alerts: list[str] = Field(default_factory=list)
    next_refresh: str  # ISO format


class FundamentalReport(BaseModel):
    """Fundamental analysis report for a single symbol."""
    symbol: str
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    ev_ebitda: float | None = None
    roe: float | None = None
    roa: float | None = None
    net_margin: float | None = None
    operating_margin: float | None = None
    debt_equity: float | None = None
    revenue_growth: float | None = None
    eps_growth: float | None = None
    sector: str = ""
    market_cap: float | None = None
    health_score: int = 0  # 0-100
    moat_level: Literal["wide", "narrow", "none"] = "none"
    recommendation: Literal["strong_buy", "buy", "hold", "sell", "strong_sell"] = "hold"
    conviction: float = Field(ge=0, le=1, default=0.5)
    strength: int = Field(ge=0, le=100, default=50)
    risks: list[str] = Field(default_factory=list)
    catalysts: list[str] = Field(default_factory=list)
    summary: str = ""


class TechnicalReport(BaseModel):
    """Technical analysis report for a single symbol."""
    symbol: str
    sma_20: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    rsi: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    bollinger_upper: float | None = None
    bollinger_middle: float | None = None
    bollinger_lower: float | None = None
    atr: float | None = None
    volume_trend: Literal["increasing", "decreasing", "stable"] = "stable"
    candlestick_patterns: list[str] = Field(default_factory=list)
    signal: Literal["strong_buy", "buy", "hold", "sell", "strong_sell"] = "hold"
    strength: int = Field(ge=0, le=100, default=50)
    conviction: float = Field(ge=0, le=1, default=0.5)
    summary: str = ""


class SentimentReport(BaseModel):
    """Sentiment analysis report for a single symbol."""
    symbol: str
    sentiment_score: float = Field(ge=-1, le=1, default=0.0)
    sentiment_label: Literal["bullish", "bearish", "neutral"] = "neutral"
    news_sentiment: float = Field(ge=-1, le=1, default=0.0)
    announcement_sentiment: float = Field(ge=-1, le=1, default=0.0)
    catalysts: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    summary: str = ""


class ValuationReport(BaseModel):
    """Valuation report for a single symbol."""
    symbol: str
    dcf_base: Decimal | None = None
    dcf_bull: Decimal | None = None
    dcf_bear: Decimal | None = None
    reverse_dcf: Decimal | None = None
    fair_value_range: tuple[Decimal, Decimal] | None = None
    current_price: Decimal | None = None
    upside_pct: float | None = None
    key_assumptions: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""


class PeerComparison(BaseModel):
    """Peer comparison for a single symbol."""
    symbol: str
    peers: list[str] = Field(default_factory=list)
    rankings: dict[str, dict[str, float]] = Field(default_factory=dict)
    summary: str = ""


class MoatAssessment(BaseModel):
    """Moat assessment for a single symbol."""
    symbol: str
    score: int = Field(ge=0, le=100, default=0)
    rating: Literal["wide", "narrow", "none"] = "none"
    sources: list[str] = Field(default_factory=list)
    summary: str = ""


class AnalysisReport(BaseModel):
    """Complete analysis report combining all dimensions."""
    symbol: str
    fundamental: FundamentalReport | None = None
    technical: TechnicalReport | None = None
    sentiment: SentimentReport | None = None
    valuation: ValuationReport | None = None
    peer_comparison: PeerComparison | None = None
    moat: MoatAssessment | None = None
    overall_score: float = Field(ge=0, le=100, default=50)
    recommendation: Literal["strong_buy", "buy", "hold", "sell", "strong_sell"] = "hold"
    confidence: float = Field(ge=0, le=1, default=0.5)
    key_risks: list[str] = Field(default_factory=list)
    catalyst: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)


class ResearchReport(BaseModel):
    """Deep research report for a symbol."""
    symbol: str
    thesis: str
    bull_case: str = ""
    bear_case: str = ""
    catalysts: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    valuation_range: tuple[Decimal, Decimal] | None = None
    peer_comparison: dict[str, dict] = Field(default_factory=dict)
    conviction: float = Field(ge=0, le=1, default=0.5)
    target_price: Decimal | None = None
    rating: Literal["strong_buy", "buy", "hold", "sell", "strong_sell"] = "hold"
    portfolio_impact: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)


class RiskAssessment(BaseModel):
    """Risk assessment for a potential trade."""
    symbol: str
    position_size: int = 0
    max_loss: Decimal = Decimal("0")
    risk_reward: float = 0.0
    sector_exposure: float = 0.0
    portfolio_heat: float = 0.0
    verdict: Literal["approve", "reduce", "reject", "conditional"] = "conditional"
    conditions: list[str] = Field(default_factory=list)
    stop_loss: Decimal = Decimal("0")
    target: Decimal = Decimal("0")
    max_position_pct: float = 0.0
    correlation_risk: float = 0.0
    rationale: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)


class DebateArgument(BaseModel):
    """Single argument in a debate."""
    symbol: str
    side: Literal["bull", "bear"]
    argument: str
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1, default=0.5)


class DebateResult(BaseModel):
    """Result of bull/bear debate."""
    symbol: str
    bull_thesis: str
    bear_thesis: str
    bull_arguments: list[DebateArgument] = Field(default_factory=list)
    bear_arguments: list[DebateArgument] = Field(default_factory=list)
    rebuttal: str = ""
    synthesis: str = ""
    decision: Literal["buy", "sell", "hold", "pass"] = "pass"
    confidence: float = Field(ge=0, le=1, default=0.5)
    timestamp: datetime = Field(default_factory=datetime.now)


class ConferenceResult(BaseModel):
    """Result of multi-agent conference."""
    conference_type: Literal["morning", "trade_review", "emergency", "experience"]
    regime: Literal["normal", "high_volatility", "crash", "rally", "low_liquidity"]
    decisions: list[dict] = Field(default_factory=list)
    capital_allocation: dict[str, Decimal] = Field(default_factory=dict)
    risk_alerts: list[str] = Field(default_factory=list)
    next_meeting: datetime | None = None
    timestamp: datetime = Field(default_factory=datetime.now)


class MemoryResult(BaseModel):
    """Memory agent operation result."""
    action: Literal[
        "record_market", "record_trade", "record_pattern",
        "get_context", "get_calibration", "get_stats"
    ]
    result: dict
    files_updated: list[str] = Field(default_factory=list)


class ScreeningCandidate(BaseModel):
    """Single screening candidate."""
    symbol: str
    composite_score: float = 0.0
    fundamental_score: float = 0.0
    technical_score: float = 0.0
    sentiment_score: float = 0.0
    valuation_score: float = 0.0
    sector: str = ""
    market_cap: Decimal = Decimal("0")
    recommendation: Literal["strong_buy", "buy", "hold", "sell", "strong_sell"] = "hold"


class ScreeningResult(BaseModel):
    """Screening agent output."""
    candidates: list[ScreeningCandidate] = Field(default_factory=list)
    top_picks: list[str] = Field(default_factory=list)
    sector_distribution: dict[str, int] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class TradeOrder(BaseModel):
    """Trade order to be executed."""
    symbol: str
    action: Literal["buy", "sell"]
    quantity: int
    order_type: Literal["market", "limit", "sl", "slm"] = "market"
    price: Decimal | None = None
    trigger_price: Decimal | None = None
    product: Literal["CNC", "MIS", "NRML"] = "CNC"
    tag: str = ""
    risk_reward: float | None = None
    conviction: float | None = None


class ExecutionReport(BaseModel):
    """Trade execution report."""
    order: TradeOrder
    status: Literal["complete", "rejected", "partial", "pending"] = "pending"
    order_id: str | None = None
    filled_qty: int = 0
    avg_price: Decimal | None = None
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)


class SupervisorOutput(BaseModel):
    """Supervisor agent final output."""
    action: Literal["continue", "complete", "error", "await_approval"]
    message: str
    trade_queue: list[TradeOrder] = Field(default_factory=list)
    next_context: str | None = None
    metadata: dict = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Context & Routing Types
# ═══════════════════════════════════════════════════════════════════════════

class RunContextType(TypedDict):
    """Context for why the supervisor was invoked."""
    context: Literal["morning_routine", "price_alert", "quarterly_review", "manual", "distortion_check"]
    depth: NotRequired[Literal["quick", "standard", "deep"]]
    alert_symbol: NotRequired[str]
    price_move_pct: NotRequired[float]
    symbols: NotRequired[list[str]]


# ═══════════════════════════════════════════════════════════════════════════
# Agent State — Pydantic BaseModel
# ═══════════════════════════════════════════════════════════════════════════

class AgentState(BaseModel):
    """
    Pydantic-based agent state.

    Uses Pydantic validation and computed fields instead of reducers.
    For parallel fan-out, use asyncio.gather with agent.run() calls.
    """
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    # Messages (accumulated across ReAct loop iterations)
    messages: list[BaseMessage] = Field(default_factory=list)

    # ── Context — WHY was the supervisor invoked ──
    context: str = "manual"
    depth: Literal["quick", "standard", "deep"] = "standard"
    alert_symbol: str | None = None
    price_move_pct: float | None = None

    # ── Capital (from Kite SDK — balance IS budget) ──
    available_cash: Decimal = Decimal("0")
    utilized_margin: Decimal = Decimal("0")
    available_margin: Decimal = Decimal("0")

    # ── Portfolio snapshot ──
    holdings: list[Holding] = Field(default_factory=list)
    positions: list[dict] = Field(default_factory=list)

    # ── Screening output ──
    candidates: list[str] | None = None
    composite_ranks: dict[str, float] | None = None

    # ── Research cache (keyed by symbol, populated via asyncio.gather) ──
    market_snapshot: MarketSnapshot | None = None
    macro_signals: dict | None = None
    fundamental_reports: dict[str, FundamentalReport] = Field(default_factory=dict)
    technical_reports: dict[str, TechnicalReport] = Field(default_factory=dict)
    sentiment_reports: dict[str, SentimentReport] = Field(default_factory=dict)
    valuation_reports: dict[str, ValuationReport] = Field(default_factory=dict)
    peer_comparisons: dict[str, PeerComparison] = Field(default_factory=dict)
    moat_assessments: dict[str, MoatAssessment] = Field(default_factory=dict)

    # ── Synthesis ──
    research_debates: dict[str, DebateResult] = Field(default_factory=dict)
    research_reports: dict[str, ResearchReport] = Field(default_factory=dict)

    # ── Risk & Execution ──
    risk_assessments: dict[str, RiskAssessment] = Field(default_factory=dict)
    trade_queue: list[TradeOrder] = Field(default_factory=list)
    execution_reports: list[ExecutionReport] = Field(default_factory=list)

    # ── Learning context (retrieved BEFORE trade decisions) ──
    learning_context: dict | None = None

    # ── Routing ──
    next_agent: str | None = None
    routing_history: list[str] = Field(default_factory=list)

    # ── Metadata ──
    thread_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    run_mode: Literal["paper", "live", "backtest"] = "paper"
    errors: list[str] = Field(default_factory=list)

    # ── Token budget ──
    total_tokens: int = 0
    total_cost: float = 0.0
    budget_limit: float = 100.0

    # ── Graph iteration tracking ──
    iteration_count: int = 0
    llm_failure_count: int = 0
    remaining_steps: int = 50

    # ── HITL ──
    pending_human_approval: dict | None = None
    human_decision: str | None = None

    # ── Validator / self-correction ──
    validation_count: int = 0
    validator_verdict: str | None = None
    validator_confidence: float | None = None
    validator_issues: list[str] = Field(default_factory=list)
    validator_suggestions: list[str] = Field(default_factory=list)
    validator_note: str | None = None
    should_reanalyze: bool = False

    # ── Memory ──
    memory_context: dict | None = None
    context_memory: dict | None = None
    memories_loaded: bool = False

    # ── Order management ──
    order_modifications: list[dict] = Field(default_factory=list)
    order_cancellations: list[dict] = Field(default_factory=list)
    pending_trade: dict | None = None
    recent_trades: list[dict] = Field(default_factory=list)

    # ── Researcher output extras ──
    researcher_notes: list[str] = Field(default_factory=list)
    researcher_recommendation: str | None = None
    catalyst_track: dict = Field(default_factory=dict)
    catalyst_impact_report: dict = Field(default_factory=dict)

    # ── Quarterly ──
    rebalancing_recommendations: list[dict] = Field(default_factory=list)

    # ── v5.4 additions ──
    watchlist: list[str] = Field(default_factory=list)
    distortion_reports: dict = Field(default_factory=dict)
    regulatory_reports: dict = Field(default_factory=dict)
    fii_dii_data: dict | None = None
    news_velocity_reports: dict = Field(default_factory=dict)

    # ── v5.6: 8-Agent System — Debate ──
    debate_results: dict[str, DebateResult] = Field(default_factory=dict)
    bull_arguments: dict[str, DebateArgument] = Field(default_factory=dict)
    bear_arguments: dict[str, DebateArgument] = Field(default_factory=dict)

    # ── v5.6: 8-Agent System — Conferences ──
    conference_results: list[ConferenceResult] = Field(default_factory=list)
    active_conference: Literal["morning", "trade_review", "emergency", "experience"] | None = None
    market_regime: Literal["normal", "high_volatility", "crash", "rally", "low_liquidity"] = "normal"

    # ── v5.6: 8-Agent System — Analyst Reports ──
    global_analyst_report: dict | None = None
    india_analyst_report: dict | None = None
    sentiment_analyst_report: dict | None = None
    technical_analyst_report: dict | None = None

    # ── v5.6: 8-Agent System — Memory ──
    market_memory: list[dict] = Field(default_factory=list)
    reflection_memory: list[dict] = Field(default_factory=list)
    experience_memory: list[dict] = Field(default_factory=list)
    memory_stats: dict | None = None

    # ── v5.6: 8-Agent System — Risk Manager (veto power) ──
    risk_veto: bool = False
    risk_veto_reason: str | None = None
    position_limits: dict = Field(default_factory=dict)
    sector_exposure: dict = Field(default_factory=dict)

    # ── Phase 3.2: Point-in-time data fetching ──
    as_of: str | None = None

    # ── Phase 3.4: Emergency trigger ──
    emergency_pending: bool = False
    emergency_trigger: str | None = None


# ═══════════════════════════════════════════════════════════════════════════
# Runtime Context Dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RuntimeContext:
    """Runtime context injected into all nodes."""
    user_id: str = "default"
    mode: Literal["paper", "live", "backtest"] = "paper"
    store: Any = None
    db_pool: Any = None
    kite_client: Any = None
    vector_store: Any = None


# ═══════════════════════════════════════════════════════════════════════════
# Type Aliases for Agent Generics
# ═══════════════════════════════════════════════════════════════════════════

# All agents share AgentDeps as dependencies
AgentDepsT = AgentDeps

# Output types per agent
MarketDataOutput = MarketSnapshotOutput
AnalysisOutput = AnalysisReport
ResearchOutput = ResearchReport
RiskOutput = RiskAssessment
DebateOutput = DebateResult
ConferenceOutput = ConferenceResult
MemoryOutput = MemoryResult
ScreeningOutput = ScreeningResult
SupervisorOutputT = SupervisorOutput


# ═══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════

def create_agent_deps(
    user_id: str = "default",
    mode: Literal["paper", "live", "backtest"] = "paper",
    capital: Decimal = Decimal("1000000"),
    risk_profile: Literal["conservative", "moderate", "aggressive"] = "moderate",
    **kwargs
) -> AgentDeps:
    """Factory for creating AgentDeps with defaults."""
    return AgentDeps(
        user_id=user_id,
        mode=mode,
        capital=capital,
        risk_profile=risk_profile,
        **kwargs
    )


def merge_dicts(existing: dict | None, update: dict | None) -> dict:
    """Merge two dicts — ."""
    if existing is None:
        return update or {}
    if update is None:
        return existing
    merged = dict(existing)
    merged.update(update)
    return merged