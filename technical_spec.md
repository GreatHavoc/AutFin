# AutFin — Technical Specification v7.0

**Version**: 7.0.0 | **Date**: August 2026 | **Status**: Complete — Pydantic AI SubAgents + DynamicWorkflow, Model Tiers, Alpha Decay, Walk-Forward Backtesting, Logfire Observability

---

## 1. RESOLVED TECH STACK

| Layer | Technology | Version | Justification |
|-------|-----------|---------|---------------|
| **Orchestration** | Pydantic AI | `>=0.2.0` | MIT-licensed. SubAgents, DynamicWorkflow, Planning, Memory, CodeMode, Guardrails, SystemReminders. Atomic state + Command routing in one return. |
| **Kite/Zerodha** | KiteConnect SDK | `>=5.0.0` | Direct pykiteconnect SDK: auth, portfolio, orders, GTT, WebSocket. TOTP auto-login. |
| **LLM Client** | openai SDK | `>=1.0.0` | Direct connection to OpenAI-compatible endpoints (Modal vLLM, NVIDIA NIM, etc.). Parallel tool execution. |
| **LLM — Primary** | NVIDIA NIM | via openai SDK | `meta/llama-3.3-70b-instruct`, `deepseek-ai/deepseek-v4-flash` — 70B MoE, 1M context, free tier. |
| **LLM — Fallback** | Modal vLLM | via openai SDK | `qwen3.6-35b-a3b` — Qwen 3.6 35B, zero cost, no rate limits, cold starts. |
| **Database** | SQLite | via `sqlite-utils` | 5+ tables: trades, portfolio_snapshots, signals, balance_history, learning_journal. WAL mode. |
| **Vector Store** | ChromaDB | `>=0.5.0` | 3 collections: market_data, learning_journal, research_notes. Local-first. 384-dim embeddings. |
| **Observability** | Logfire | `>=0.1.0` | Auto-tracing via env vars, per-agent traces, prompt version control, RAGAS evaluation. Replaces LangSmith. |
| **Indian Stock Data** | NSE Public APIs | REST (no key) | Unlimited, real-time. Primary for Indian equity prices. |
| **Historical Data** | yfinance | `>=0.2.0` | Free, no key. `.NS`/`.BO` Indian tickers. OHLCV candles + fundamentals. |
| **Market Data Fallback** | yfinance | REST (no key) | Real-time NSE quotes as primary yfinance backup. |
| **News Sentiment** | Marketaux + Alpha Vantage | REST (free keys) | Marketaux: 100 articles/day NLP. Alpha Vantage: 200K+ tickers. NSE announcements as Indian fallback. |
| **Fundamentals** | yfinance (primary) + Screener.in | — | yfinance: ROIC, FCF yield, moat, health score. Screener.in: Firecrawl fallback for deep Indian fundamentals. |
| **Global Macro** | yfinance + FRED API | REST (free keys) | yfinance: S&P, NASDAQ, VIX, DXY, INR, crude, gold, US 10Y. FRED: GDP, CPI, Fed Funds with 24h cache. |
| **Technical Analysis** | pandas-ta-classic (pure Python) | `pandas-ta-classic>=0.6.0` | SMA 20/50/200, EMA, RSI, MACD, ADX, Bollinger, ATR, OBV. Multi-period + golden/death cross. |
| **Data** | Pandas + NumPy | `pandas>=2.2.0`, `numpy>=1.26.0` | All numerical computation. |
| **HTTP** | httpx | `>=0.27.0` | Async HTTP client. |
| **Config** | python-dotenv + settings.py | `python-dotenv>=1.0.0` | Environment + typed settings (Pydantic models). |
| **CLI** | Rich | `>=13.0.0` | Terminal output formatting. |
| **Validation** | Pydantic | `>=2.0.0` | All state schemas, config models. |
| **Capabilities** | pydantic-ai-harness | `>=0.1.0` | SubAgents, DynamicWorkflow, Planning, Memory, CodeMode, Guardrails, SystemReminders, Skills. |
| **Evals** | pydantic-evals | `>=0.1.0` | Structured evaluation suites, RAG quality assessment. |

---

## 2. PROJECT STRUCTURE

```
autfin/
├── pyproject.toml                         # Dependencies, project metadata
├── .env                                   # API keys, secrets (gitignored)
├── .env.example                           # Template for required env vars
├── README.md                              # Project overview, getting started
├── technical_spec.md                      # v7.0 Technical Specification (this file)
├── tests/                                 # 52 test files across 10 categories
│   ├── test_agents/       # Sub-agent tests (7 agents × parallel/modes)
│   ├── test_tools/        # Tool integration tests (Kite, yfinance, NSE, etc.)
│   ├── test_backtest/     # Walk-forward + alpha decay tests
│   ├── test_memory/       # Memory persistence & RAG eval tests
│   ├── test_learning/     # Online learning, calibration, critique tests
│   ├── test_risk/         # Risk management, Kelly, anomaly, regime tests
│   ├── test_tools/        # Sentiment, Reddit, NSE fix, screening tests
│   ├── test_llm/          # LLM client tests (model tiers, parallel exec)
│   └── test_validator.py  # Validation gates (8 gates)
├── technical_spec.md                      # v7.0 Technical Specification
├── .gitignore
└── uv.lock

src/autfin/
├── __init__.py
├── agents/            # 7 sub-agents + shared infrastructure
│   ├── __init__.py
│   ├── market_data.py      # Portfolio, quotes, market status, risk alerts
│   ├── analysis.py         # Technical + fundamental + sentiment + valuation
│   ├── research.py         # Deep research, screening, thesis generation
│   ├── risk.py             # Position sizing, stop-loss, correlation, portfolio heat
│   ├── debate.py           # Bull vs Bear adversarial debate (MANDATORY before trades)
│   ├── conference.py       # Multi-agent coordination conferences
│   ├── memory.py           # Cross-session memory: events, trades, patterns, calibration
│   ├── quant.py            # Kelly criterion with regime gates
│   ├── guardrails.py       # Input/output/tool guardrails per agent
│   ├── managed_prompts.py  # ManagedPrompt instances via Logfire (8 prompts)
│   ├── base.py             # Agent factory, create_agent(), create_subagent(), get_model()
│   ├── observability.py    # Logfire configuration, create_instrumentation(), etc.
│   └── tools/              # Agent tool implementations
│       ├── __init__.py                    # Lazy imports via __getattr__
│       ├── kite_tools.py                  # KiteConnect SDK wrappers (809 lines)
│       ├── fundamental_tools.py           # Ratio computation, DCF models
│       ├── sentiment_analyzer.py          # FinBERT + VADER + keyword hybrid
│       ├── sentiment_tools.py             # HybridSentimentAnalyzer + multi-source (339+ lines)
│       ├── reddit_sentiment.py            # Reddit PRAW integration
│       ├── technical_tools.py             # Indicator computation (pandas-ta-classic)
│       ├── scraper_tools.py              # NSE cookies, nselib, web scraping
│       ├── supervisor_tools.py            # 20 supervisor tools + callables
│       ├── screening.py                   # Magic Formula, F-Score, liquidity gate
│       ├── universe.py                    # Nifty 500 universe management
│       ├── market_feed.py                 # Real-time market feed
│       ├── memory_tools.py                # Store/Vector Store retrieval
│       └── reporting.py                   # Report generation
├── config/            # Settings management
│   ├── __init__.py
│   ├── settings.py                    # AutFinSettings Pydantic model (modal_vllm_url, modal_vllm_model, nvidia_nim_api_key, etc.)
│   └── capabilities.py                # Data source capabilities registry (auto-updated, graceful degradation)
├── llm/               # LLM client abstraction
│   ├── __init__.py
│   ├── client.py                      # LLMClient: generate(), generate_json(), generate_with_tools()
│   │                                 # openai SDK direct (NVIDIA NIM + Modal vLLM, parallel tool execution)
│   ├── langchain_model.py             # ChatOpenAI wrapper for LangGraph (legacy import compatibility)
│   └── model_tiers.py                 # ModelTier, MODEL_REGISTRY, AGENT_TIERS (deep/medium/fast per-agent)
├── memory/            # Long-term memory layer
│   ├── __init__.py
│   ├── setup.py                       # Async checkpointer/store initialization
│   ├── store.py                       # Memory store operations (SQLite + ChromaDB + managed prompts)
│   ├── vector_store.py                # ChromaDB collections (325 lines, AutFinEmbeddingFunction)
│   ├── database.py                    # SQLite schema + CRUD (128 lines)
│   ├── connection_pool.py             # Thread-safe SQLite connection pool
│   ├── retrieval.py                   # Multi-scope memory retrieval (688 lines)
│   └── rag_eval.py                    # RAG quality evaluation
├── state/             # State type definitions
│   ├── __init__.py
│   └── agent_state.py                 # AgentState TypedDict (errors reducer, schema_version, run_mode)
├── learning/          # Self-learning modules
│   ├── __init__.py
│   ├── judge.py                       # LLM-as-Judge
│   ├── critique.py                    # Self-critique loop
│   ├── calibrator.py                  # Conviction calibration
│   ├── online.py                      # O(1) incremental online learning
│   └── feedback_loop.py               # Trade outcome feedback
├── backtest/          # Historical backtesting
│   ├── __init__.py
│   ├── engine.py                      # BacktestEngine (303 lines)
│   ├── strategies.py                  # 5 strategies (546 lines)
│   └── metrics.py                     # 18+ metrics (181 lines)
├── backtesting/       # Walk-forward + alpha decay (v7.0)
│   ├── __init__.py
│   ├── walk_forward.py                # WalkForwardConfig, fetch_price_data, walk_forward_with_symbol
│   └── alpha_decay.py                 # AlphaDecayTracker, FoldSnapshot, DecayReport
├── risk/              # Risk management modules
│   ├── __init__.py
│   ├── kelly.py                       # Kelly criterion with regime gates
│   ├── anomaly.py                     # Rolling z-score anomaly detection
│   ├── regime.py                      # VIX/volatility regime detector
│   └── vol_targeting.py               # Volatility-targeting overlay
├── cli/               # CLI interface
│   ├── __init__.py
│   └── main.py                        # 7 commands: status|analyze|scan|start|backtest|history|config (916 lines)
├── prompts/           # Agent system prompts (all 8 created via ManagedPrompt)
│   ├── __init__.py
│   ├── supervisor.py
│   ├── market_data.py
│   ├── fundamentals.py
│   ├── technical.py
│   ├── sentiment.py
│   ├── global_macro.py
│   ├── researcher.py
│   └── risk_manager.py
└── universe/          # Stock universe management
    └── curator.py
```

### Key v7.0 Changes vs v6.0

| Aspect | v6.0 | v7.0 |
|---|---|---|
| **Orchestration** | LangGraph StateGraph, `interrupt()`, `Command` | Pydantic AI SubAgents + DynamicWorkflow, `Command`-based routing |
| **LLM** | Modal vLLM primary, NVIDIA NIM fallback | NVIDIA NIM primary, Modal vLLM fallback |
| **Observability** | LangSmith (cloud) | Logfire (auto-tracing, per-agent, prompt versioning) |
| **Agents** | 8 LangGraph sub-agents + compiled StateGraph | 7 Pydantic AI SubAgents + DynamicWorkflow |
| **Workflow** | LangGraph `interrupt()` + HITL + conditional edges | Pydantic AI `DynamicWorkflow` + `Command`-based routing |
| **Memory** | LangGraph Store API | SQLite + ChromaDB + ManagedPrompts (Logfire) |
| **Validation** | Manual/partial | 8 automated gates (type_checking, unit_tests, eval_suite, guardrails, memory_persistence, plan_persistence, load_test, durable_execution) |
| **Prompts** | Static files in `prompts/` | Logfire ManagedPrompt (version-controlled) |
| **Capabilities** | Manual dict in `config/capabilities.py` | Pydantic AI harness capabilities (SubAgents, DynamicWorkflow, etc.) |

---

## 3. STATE SCHEMAS

### 3.1 AgentState (main shared state)

```python
from typing import TypedDict, Annotated, Literal, NotRequired
from typing_extensions import NotRequired
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # Messages (accumulated across nodes)
    messages: Annotated[list[BaseMessage], add_messages]

    # Capital (from KiteConnect SDK — balance IS budget)
    available_cash: float           # ₹ available from get_margins()
    utilized_margin: float          # Margin tied up in positions

    # Portfolio snapshot
    holdings: list[dict]            # Name, qty, avg_price, ltp, pnl
    positions: list[dict]           # Intraday + carry-forward positions

    # Research outputs (populated by Layer 1 agents)
    macro_signals: dict | None      # Global macro composite signal
    market_snapshot: dict | None    # Live prices, indices, volumes
    fundamental_reports: dict | None  # Per-ticker ratios, DCF, moat
    technical_reports: dict | None  # Per-ticker indicators, patterns
    sentiment_reports: dict | None  # Per-ticker NLP scores

    # Analysis output (Layer 2)
    research_debates: dict | None   # {ticker: {bull_case, bear_case, synthesis}}

    # Risk & portfolio decisions (Layer 3-4)
    risk_assessments: dict | None   # {ticker: {checks_passed, severity, recommendation}}
    trade_queue: list[dict] | None  # Prioritized trade proposals
    execution_reports: list[dict] | None  # Order confirmations

    # Routing
    next_agent: str | None          # Which agent to route to next

    # Metadata
    thread_id: str                  # LangGraph thread identifier (for compat)
    run_mode: str                   # "daily_morning" | "adrenaline" | "quarterly"
    errors: list[str]               # Accumulated error messages
```

### 3.2 Decision Record (stored in ManagedPrompt/Logfire)

```python
class DecisionRecord(TypedDict):
    decision_id: str                # UUID v4
    timestamp: str                  # ISO 8601 UTC
    symbol: str                     # Ticker symbol (e.g., "RELIANCE", "TCS")
    side: Literal["buy", "sell"]    # Trade side
    quantity: int                   # Number of shares
    price: float                    # Execution price (LTP)
    order_id: str                   # Kite order ID
    status: Literal["pending", "filled", "cancelled", "rejected"]
    conviction: float               # Agent conviction score (0.0–1.0)
    rationale: str                  # Free-text rationale from agent
    timestamp_created: str          # ISO 8601 when record was created
```

### 3.3 ModelConfig (model tier registry)

```python
from typing import Literal

class ModelTier(str, Literal["deep", "medium", "fast"]):
    """Model tier for per-agent selection."""
    pass

MODEL_REGISTRY = {
    "deep": "deepseek-ai/deepseek-v4-flash",      # 284B MoE, 1M context, complex reasoning
    "medium": "meta/llama-3.3-70b-instruct",      # 70B, reliable, balanced
    "fast": "meta/llama-3.1-8b-instruct",          # 8B, low latency, time-sensitive
}

# Agent → tier mapping
AGENT_TIERS = {
    "supervisor": "deep",           # Debate, research synthesis
    "market_data": "fast",          # Quotes, status checks
    "analysis": "medium",           # TA, fundamentals, sentiment
    "research": "deep",             # Screening, thesis generation
    "risk": "medium",               # Position sizing, risk checks
    "debate": "deep",               # Mandatory before any trade
    "conference": "medium",         # Coordination, rebalancing
    "memory": "fast",               # Pattern lookup, calibration
}
```

---

## 4. CAPABILITIES REGISTRY

### 4.1 Data Source Capabilities

```python
CAPABILITIES: dict[str, dict[str, Any]] = {
    # ── Kite SDK (Personal tier ≠ Connect tier) ──
    "kite": {
        "auto_login": True,
        "holdings": True,
        "positions": True,
        "margins": True,
        "orders": True,
        "order_place": True,
        "order_modify": True,
        "order_cancel": True,
        "trades": True,
        "profile": True,
        "ltp": False,               # BLOCKED on Personal — needs Connect (₹500/mo)
        "quotes": False,            # BLOCKED on Personal — needs Connect (₹500/mo)
        "ohlc": False,              # BLOCKED on Personal
        "historical_data": False,   # BLOCKED on Personal
        "gtt": True,
        "detail": "Personal tier: ltp/quote/ohlc blocked. Use holdings LTP or yfinance fallback.",
    },
    # ── yfinance ──
    "yfinance": {
        "ohlcv": True,
        "ratios": True,             # PE, PB, ROE, D/E, market cap (reliable)
        "financials": "partial",    # Balance sheet/income/cashflow: structure exists,
                                    # values can be 0 for some Indian stocks
        "global_indices": True,
        "forex": True,
        "commodities": True,
        "detail": "No API key needed. 15-min delay for NSE. Financial statements partially populated.",
    },
    # ── LLM Providers ──
    "nvidia_nim": {
        "available": True,
        "rate_limited": False,      # Separately quota per model
        "best_models": [
            "deepseek-ai/deepseek-v4-flash",     # 284B MoE, 1M context
            "meta/llama-3.3-70b-instruct",        # 70B, reliable
            "meta/llama-3.1-8b-instruct",         # 8B, fast
        ],
        "detail": "NVIDIA-hosted, free tier. Better reliability than OpenRouter.",
    },
    "openrouter": {
        "available": True,
        "rate_limited": "often",    # 50 free/day shared across all users
        "best_model": "google/gemma-4-31b-it:free",
        "detail": "Rate-limited often. Use NVIDIA NIM as primary instead.",
    },
    # ── News / Macro / Scraping ──
    "marketaux": {
        "available": True,
        "rate_limit": "100 articles/day",
        "detail": "Working. 5000+ news sources.",
    },
    "fred": {
        "available": True,
        "rate_limit": "120 requests/min",
        "detail": "Working. US macro indicators only.",
    },
    "firecrawl": {
        "available": True,
        "rate_limit": "500 credits/month",
        "detail": "Used for Screener.in, NSE, Moneycontrol scraping.",
    },
    # ── Observability ──
    "langsmith": {
        "available": False,         # LangSmith API key not configured in v7.0
        "configured": False,
        "detail": "LangSmith tracing disabled. Use Logfire instead.",
    },
    "logfire": {
        "available": True,
        "configured": True,         # Logfire token present
        "detail": "Logfire tracing with auto-tracing for Pydantic AI.",
    },
}
```

### 4.2 Capability Functions

```python
def can_use(source: str, capability: str) -> bool:
    """Check if a capability is available. Returns True/False/partial."""
    cap = CAPABILITIES.get(source, {}).get(capability)
    if cap is True:
        return True
    if cap == "partial":
        return True  # Partial still works, caller should handle empty results
    return False


def get_fallback(source: str, capability: str) -> str | None:
    """Return the recommended fallback for a blocked capability."""
    if source == "kite" and capability in ("ltp", "quotes", "ohlc"):
        return "Use holdings.last_price for owned stocks, yfinance for others"
    if source == "kite" and capability == "historical_data":
        return "Use yfinance.history() for OHLCV"
    if source == "nsepython" and capability in ("quotes", "ohlcv"):
        return "Use yfinance (15-min delayed) or Kite holdings LTP"
    if source == "nvidia_nim" and capability == "available":
        return "Use Modal vLLM fallback"
    return None


def mark_failed(source: str, capability: str) -> None:
    """Mark a capability as failed so agents skip it next time."""
    CAPABILITIES.setdefault(source, {})[capability] = False
    logger.warning("Capability %s.%s marked as failed", source, capability)


def mark_available(source: str, capability: str) -> None:
    """Mark a capability as available after successful use."""
    CAPABILITIES.setdefault(source, {})[capability] = True


def get_status() -> dict:
    """Return a summary of all capabilities for dashboard/system page."""
    return {
        source: {
            cap: val for cap, val in caps.items() if cap != "detail"
        }
        for source, caps in CAPABILITIES.items()
    }


def print_capabilities():
    """Print human-readable capability summary."""
    print("\nData Source Capabilities:")
    for source, caps in sorted(CAPABILITIES.items()):
        detail = caps.pop("detail", "")
        failed = [c for c, v in caps.items() if v is False]
        partial = [c for c, v in caps.items() if v == "partial"]
        ok = [c for c, v in caps.items() if v is True]
        
        status = "OK" if not failed else f"BLOCKED: {', '.join(failed)}"
        print(f"  {source:15s} [{status}]")
        if partial:
            print(f"    partial: {', '.join(partial)}")
        if detail:
            print(f"    {detail}")
    print()
```

---

## 5. AGENT ARCHITECTURE

### 5.1 SubAgents Capability

```python
from pydantic_ai_harness.subagents import SubAgents, SubAgent
from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow, WorkflowAgent
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.memory import Memory
from pydantic_ai_harness.system_reminders import SystemReminders

subagents_capability = SubAgents(
    agents=[
        SubAgent(market_data_agent, name="market_data", ...),
        SubAgent(analysis_agent, name="analyze", ...),
        SubAgent(research_agent, name="research", ...),
        SubAgent(risk_agent, name="risk", ...),
        SubAgent(debate_agent, name="debate", ...),
        SubAgent(conference_agent, name="conference", ...),
        SubAgent(memory_agent, name="memory", ...),
    ],
    models={
        "fast": get_model("fast"),
        "standard": get_model("deep"),
        "deep": get_model("reasoning"),
    },
    forward_usage=True,
    inherit_tools=False,
    shared_capabilities=[],
    tool_retries=2,
    contain_errors=True,
)
```

### 5.2 DynamicWorkflow

```python
# Morning routine workflow
morning_workflow = DynamicWorkflow(
    agents=[
        WorkflowAgent(market_data_agent, name="market_data"),
        WorkflowAgent(analysis_agent, name="analyze"),
        WorkflowAgent(research_agent, name="research"),
        WorkflowAgent(debate_agent, name="debate"),
        WorkflowAgent(risk_agent, name="risk"),
        WorkflowAgent(conference_agent, name="conference"),
    ],
    max_agent_calls=50,
    sub_agent_usage_limits=UsageLimits(request_limit=20, total_tokens_limit=50000),
    forward_usage=True,
    resource_limits={"max_duration_secs": 300},
    inherit_model=False,
    id="morning_workflow",
    defer_loading=True,
    description="Full morning routine workflow",
    tool_name="run_morning_workflow",
)

# Quarterly review workflow
quarterly_workflow = DynamicWorkflow(
    agents=[
        WorkflowAgent(market_data_agent, name="market_data"),
        WorkflowAgent(analysis_agent, name="analyze"),
        WorkflowAgent(research_agent, name="research"),
        WorkflowAgent(debate_agent, name="debate"),
        WorkflowAgent(risk_agent, name="risk"),
        WorkflowAgent(conference_agent, name="conference"),
    ],
    max_agent_calls=100,
    sub_agent_usage_limits=UsageLimits(request_limit=50, total_tokens_limit=200000),
    forward_usage=True,
    resource_limits={"max_duration_secs": 600},
    inherit_model=False,
    id="quarterly_workflow",
    defer_loading=True,
    description="Quarterly portfolio review workflow",
    tool_name="run_quarterly_workflow",
)
```

### 5.3 Planning Capability

```python
planning_capability = Planning(
    store=None,  # In-memory by default, can use SqlitePlanStore
    enable_subtasks=True,
    inject=True,
    cache_ttl="5m",
)
```

### 5.4 Memory Capability

```python
memory_capability = Memory(
    FileStore("./data/memory"),
    namespace=lambda ctx: ctx.deps.user_id,
    max_tokens=5000,
    max_lines=500,
    inject_memory=True,
)
```

### 5.4 System Reminders

```python
system_reminders_capability = SystemReminders(
    reminders=[
        "Stay focused on risk management. Check risk/reward >= 2:1 before any trade.",
        "Verify conviction >= 0.5 before any trade execution.",
        "ALWAYS run debate before any trade decision. No exceptions.",
        "Batch sub-agent calls in a single response for speed.",
        "Use dynamic workflow for complex multi-step orchestration.",
    ],
    dynamic_reminders=[],
    cache_ttl="5m",
)
```

### 5.4 Supervisor Agent

```python
from autfin.agents.base import create_agent, get_model
from autfin.agents.managed_prompts import supervisor_instructions
from autfin.agents.observability import create_debug_instrumentation
from autfin.types import AgentDeps, SupervisorOutput, MarketSnapshotOutput, ...

supervisor_agent = create_agent(
    name="supervisor",
    output_type=SupervisorOutput,
    model_type="deep",
    instructions=supervisor_instructions.default,
    capabilities=[
        subagents_capability,
        morning_workflow,
        quarterly_workflow,
        planning_capability,
        memory_capability,
        system_reminders_capability,
        Thinking(effort="high"),
        WebSearch(local="duckduckgo"),
        create_debug_instrumentation(),
    ],
    model_settings=ModelSettings(temperature=0.0, max_tokens=8192),
)
```

### 5.5 Supervisor Tools

```python
@supervisor_agent.tool
async def run_morning_routine(ctx: RunContext[AgentDeps]) -> SupervisorOutput:
    """Run the full morning routine workflow."""
    ...
```

---

## 6. MANAGED PROMPTS (Logfire)

### 6.1 Prompt Structure

All 8 agent prompts are managed via Logfire `ManagedPrompt`, providing:

- **Version control**: Each prompt has immutable template versions
- **Rollout**: Labels (production/canary/staging) control which version is served
- **Template inputs**: Handlebars `{{field}}` placeholders filled from per-request inputs
- **Composition**: `@{other_variable}@` references splice in other managed variable values

### 6.2 Prompt Instances

```python
from pydantic_ai_harness.logfire import ManagedPrompt

supervisor_instructions = ManagedPrompt(
    name="supervisor_instructions",
    default="""You are the AutFin v7.0 Supervisor — a PM-level AI coordinating a 7-agent financial team.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR TEAM (7 Sub-Agents via SubAgents):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. market_data  → Portfolio, quotes, market status, risk alerts
2. analyze      → Technical + fundamental + sentiment + valuation
3. research     → Deep research, screening, thesis generation
4. risk         → Position sizing, stop-loss, correlation, portfolio heat
5. debate       → Bull/Bear adversarial debate (MANDATORY before trades)
6. conference   → Multi-agent coordination conferences
7. memory       → Cross-session learning & calibration

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ORCHESTRATION RULES (CRITICAL):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚡ BATCHING (SPEED):
- ALWAYS batch sub-agent calls in ONE response
- 3 parallel calls = ~5s, 3 sequential = ~15s
- WRONG:  Response 1 → analyze("RELIANCE")
- CORRECT:  Response 1 → [analyze("RELIANCE"), analyze("TCS"), analyze("HDFC")]

📋 CONTEXT ROUTING:
- "morning_routine"  → Full portfolio scan + screening + research + debate + trade
- "price_alert"      → Quick 3-agent fan-out on one symbol (market_data, analyze, risk)
- "quarterly_review" → Deep dive all holdings with DCF + debate + rebalancing
- "manual"           → User-requested single analysis
- "distortion_check" → Quick distortion audit on symbol

🎯 TRADING RULES (NON-NEGOTIABLE):
- ALWAYS run_debate BEFORE any trade. No exceptions.
- Risk/reward MUST be >= 2:1. Conviction >= 0.5.
- run_conference("emergency") if intraday move > 3%.
- analyze("distortion") BEFORE recommending trade.
- Risk agent has VETO power. Respect it.
""",
)
```

### 6.3 Prompt Versioning

```
prompt_create_version(supervisor_instructions, version=1, template=...)
prompt_create_version(supervisor_instructions, version=2, template=...)
# Edit rollout via variable_assign_label / variable_update_rollout
```

### 6.4 Prompt Resolution

```
prompt_preview(supervisor_instructions, inputs={"agent_count": 7})
# Returns rendered template with Handlebars filled + @{...} composition resolved
```

---

## 7. LLM TIER SELECTION

### 7.1 Model Configuration

```python
from autfin.agents.base import get_model, MODEL_REGISTRY, AGENT_TIERS

# Get model by tier
fast_model = get_model("fast")      # Llama 3.1 8B via NVIDIA NIM
deep_model = get_model("deep")      # Deepseek v4-flash via NVIDIA NIM
medium_model = get_model("medium")  # Llama 3.3 70B via NVIDIA NIM

# Per-agent tier
tier = AGENT_TIERS["risk"]  # Returns "medium"
model = get_model(tier)
```

### 7.2 Agent Model Assignment

| Agent | Tier | Model | Purpose |
|---|---|---|---|
| supervisor | deep | deepseek-v4-flash | Complex reasoning, debate oversight |
| market_data | fast | Llama 3.1 8B | Quick quotes, status checks |
| analysis | medium | Llama 3.3 70B | TA, fundamentals, sentiment |
| research | deep | deepseek-v4-flash | Screening, thesis generation |
| risk | medium | Llama 3.3 70B | Position sizing, correlation |
| debate | deep | deepseek-v4-flash | Mandatory adversarial debate |
| conference | medium | Llama 3.3 70B | Coordination, rebalancing |
| memory | fast | Llama 3.1 8B | Pattern lookup, calibration |

### 7.3 Model Tiers Summary

- **deep**: Highest capability, 1M+ context, slowest, most expensive (free tier via NVIDIA NIM). Used for: debate, research synthesis, supervisor oversight.
- **medium**: Balanced capability and speed, 70B parameter, reliable. Used for: technical analysis, fundamentals, risk assessment, conference coordination.
- **fast**: Lowest latency, 8B parameter, fastest responses. Used for: market data, memory pattern lookup, quick checks.

### 7.4 Model Switching

```python
# Switch agent to different tier dynamically
async def switch_agent_tier(agent_name: str, new_tier: str):
    tier = AGENT_TIERS.get(agent_name, "deep")
    if new_tier != tier:
        # Update the agent's model type
        await agent.override(model_type=new_tier)
```

---

## 8. WALK-FORCE BACKTESTING

### 8.1 WalkForwardConfig

```python
class WalkForwardConfig(TypedDict):
    symbol: str          # Ticker symbol
    interval: str        # "1D", "1H", "30MIN", etc.
    walk_forward_len: int    # In-sample length (days)
    walk_forward_step: int   # Step size between folds (days)
    walk_forward_total: int  # Total number of folds
    regime_filter: bool     # Whether to apply VIX regime filter
    min_sharpe: float       # Minimum Sharpe for a fold to pass
    max_drawdown: float     # Maximum acceptable drawdown
```

### 8.2 WalkForwardEngine

```python
class WalkForwardEngine:
    def __init__(self, config: WalkForwardConfig):
        self.config = config
    
    def run(self) -> WalkForwardReport:
        """Run walk-forward backtesting across all folds."""
        ...
    
    def _run_fold(self, fold_idx: int) -> FoldSnapshot:
        """Run a single walk-forward fold."""
        ...
    
    def _calculate_metrics(self, prices: pd.DataFrame) -> dict:
        """Calculate performance metrics for a fold."""
        ...
```

### 8.3 AlphaDecayTracker

```python
class AlphaDecayTracker:
    def __init__(self, config: dict):
        self.fold_sharpe_history = []  # Sharpe per fold
        self.fold_number = 0
        self.floor = config.get("sharpe_floor", 0.5)
        self.decay_threshold = config.get("decay_threshold", 0.3)
    
    def record_fold(self, sharpe: float, fold_number: int):
        """Record Sharpe ratio for a completed fold."""
        self.fold_sharpe_history.append((fold_number, sharpe))
        self.fold_number = fold_number + 1
    
    def check_decay(self) -> bool:
        """Check if alpha decay exceeds threshold."""
        if len(self.fold_sharpe_history) < 2:
            return False
        
        recent_sharpe = self.fold_sharpe_history[-1][1]
        historical_sharpe = sum(s for _, s in self.fold_sharpe_history[:-1]) / (len(self.fold_sharpe_history) - 1)
        
        return (historical_sharpe - recent_sharpe) > self.decay_threshold
    
    def should_alert(self) -> bool:
        """Check if Sharpe has fallen below floor."""
        if not self.fold_sharpe_history:
            return False
        latest_sharpe = self.fold_sharpe_history[-1][1]
        return latest_sharpe < self.floor
```

### 8.4 DecayReport

```python
class DecayReport(TypedDict):
    fold_count: int
    sharpe_history: list[tuple[int, float]]  # (fold_number, sharpe)
    current_sharpe: float
    floor_sharpe: float
    decay_detected: bool
    alerts: list[str]  # Human-readable alert messages
```

### 8.5 Walk-Forward Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                Walk-Forward Backtesting Pipeline                │
├─────────────────────────────────────────────────────────────────┤
│  1. Load historical price data for all portfolio symbols         │
│  2. Split into in-sample / out-of-sample periods per config      │
│  3. For each fold:                                     │
│     a. Train strategy on in-sample data                        │
│     b. Execute on out-of-sample data                           │
│     c. Calculate metrics: Sharpe, Sortino, Max DD, Win Rate     │
│     d. Record results in FoldSnapshot                          │
│  4. Compile all fold results into DecayReport                    │
│  5. Check alpha decay: if detected → alert via Logfire           │
│  6. Return report + recommendations (rebalance, walk-forward step)│
│  └────────────────────────────────────────────────────────────────┘
```

---

## 9. EVALUATION SUITE

### 9.1 AutFin Dataset

```python
from autfin.evals import create_autfin_dataset, run_evaluations

dataset = create_autfin_dataset(
    include_llm_judges=False,
    include_span_eval=True,
)

agents = {
    "market_data_agent": market_data_agent,
    "analysis_agent": analysis_agent,
    "research_agent": research_agent,
    "risk_agent": risk_agent,
    "debate_agent": debate_agent,
    "memory_agent": memory_agent,
}

# Run evaluations with concurrency limit
results = await run_evaluations(agents, max_concurrency=3)

# Calculate overall pass rate
total_cases = 0
passed_cases = 0

for agent_name, result in results.items():
    if "case_results" in result:
        for cr in result["case_results"]:
            total_cases += 1
            if cr.get("passed"):
                passed_cases += 1

pass_rate = passed_cases / total_cases if total_cases > 0 else 0.0
```

### 9.2 Evaluation Types

- **Span evaluation**: LLM judges span quality (coverage, relevance)
- **RAG evaluation**: Retrieval quality, relevance scoring
- **Tool use evaluation**: Correct tool selection, parameter passing
- **Output format validation**: Pydantic output type compliance
- **Model tier selection**: Correct tier for given task

### 9.3 Pass Rate Threshold

```
Target: > 80% across all evaluation categories
Warning: 70–80% — review eval suite
Fail: < 70% — require model/tier/prompt updates
```

---

## 10. VALIDATION GATES (8 Gates)

### 10.1 Type Checking (pyright)

```
Run: pyright --outputjson
Pass: 0 errors (or only pre-existing suppressible warnings)
Fail: Any type errors blocking build
```

### 10.2 Unit Tests

```
Run: pytest -x -v --tb=short
Pass: All tests pass (52 files, 5400+ tests)
Fail: Any test failure
```

### 10.3 Evaluation Suite

```
Run: python -m pytest tests/ -k "eval" -v
Pass: Overall pass rate > 80%
Fail: Pass rate ≤ 80%
```

### 10.4 Guardrails

```
Run: python -c "from test_validation import test; asyncio.run(test())"
Pass: All guardrail checks pass (input, output, tool guardrails)
Fail: Any guardrail violation
```

### 10.5 Memory Persistence

```
Run: python -c "from autfin.validation import validate_memory_persistence; ..."
Pass: Memory records persist across runs (SQLite + ChromaDB)
Fail: Memory loss detected
```

### 10.6 Plan Persistence

```
Run: python -c "from autfin.validation import validate_plan_persistence; ..."
Pass: Decision records persist across runs (ManagedPrompt versions)
Fail: Plan history lost
```

### 10.7 Load Test

```
Run: python -c "from test_validation import validate_load_test; ..."
Pass: System handles N concurrent requests without degradation
Fail: Performance threshold not met
```

### 10.8 Durable Execution

```
Run: python -c "from test_validation import validate_durable_execution; ..."
Pass: Tool retries, error recovery, checkpointing all functional
Fail: Any durable execution failure
```

### 10.9 Validation Report

```
python -m pytest --validation-report
# Generates HTML report with:
# - Gate results (pass/fail per gate)
# - Duration per gate
# - Detailed error messages on failures
# - Trend tracking (previous run comparisons)
```

---

## 11. OBSERVABILITY (Logfire)

### 11.1 Auto-Tracing

```python
from autfin.observability import configure_logfire, trace_operation, trace_agent_run

configure_logfire()  # Auto-injects from LOGFIRE_TOKEN env var

@trace_operation("daily_morning_routine")
async def run_morning_routine(ctx: RunContext[AgentDeps]) -> SupervisorOutput:
    """Trace the full morning routine."""
    ...
```

### 11.2 Per-Agent Traces

```python
@trace_agent_run("market_data")
async def market_data_trace(ctx: RunContext[AgentDeps]) -> MarketSnapshotOutput:
    """Trace market data agent specifically."""
    ...

### 11.3 Prompt Version Tracking

```python
# View which prompt version is currently served
prompt_version = logfire.prompt_get_version("supervisor_instructions", version=1)
# View rollout configuration
rollout = logfire.variable_get("supervisor_prompt_rollout")
# View assignment history
history = logfire.variable_label_history("production", "supervisor_instructions")
```

### 11.4 RORO (Return on Research Output)

```python
from autfin.observability import roro

roro_score = roro.calculate(
    research_output=research_result,
    trade_outcome=actual_trade_result,
    conviction=conviction_score,
)
```

### 11.5 Metrics Export

```
# Export to Prometheus / Grafana
logfire.metrics.export(
    metrics=["agent_calls", "token_usage", "latency", "error_rate"],
    labels={"agent": "supervisor", "tier": "deep"},
)
```

---

## 12. DATA SOURCE INTEGRATIONS

### 12.1 Kite SDK (Zerodha)

```python
from autfin.tools.kite_tools import kite_tools

# Get portfolio
holdings = await kite_tools.get_holdings()
positions = await kite_tools.get_positions()
margins = await kite_tools.get_margins()

# Place order
order_result = await kite_tools.place_order(
    variety="regular",
    exchange="NSE",
    tradingsymbol="RELIANCE",
    quantity=10,
    order_type="MARKET",
    product="CNC",
    price=None,
    validity="DAY",
    trigger_price=None,
)

# Modify order
order_result = await kite_tools.modify_order(
    order_id="OA123456",
    quantity=15,
    price=2500.0,
)

# Cancel order
result = await kite_tools.cancel_order(order_id="OA123456")
```

### 12.2 yfinance

```python
import yfinance as yf

# Get historical data
ticker = yf.Ticker("RELIANCE.NS")
data = ticker.history(period="1y", interval="1D")

# Get fundamentals
info = ticker.info
ratios = {
    "pe_ratio": info.get("trailingPE"),
    "pb_ratio": info.get("priceToBook"),
    "roe": info.get("returnOnEquity"),
    "d_e_ratio": info.get("debtToEquity"),
    "market_cap": info.get("marketCap"),
}

# Get financial statements
financials = ticker.financials
balance_sheet = ticker.balance_sheet
cash_flow = ticker.cashflow
```

### 12.3 Marketaux (News Sentiment)

```python
from autfin.tools.sentiment_tools import HybridSentimentAnalyzer

analyzer = HybridSentimentAnalyzer()
news = analyzer.get_recent_news(days=1, symbols=["RELIANCE"])
sentiment = analyzer.analyze_sentiment(news)

# Result: {"compound": -0.2, "positive": 0.1, "negative": 0.7, "neutral": 0.2}
```

### 12.4 FRED (Macro Economic Data)

```python
from autfin.tools.market_aux import fetch_fred_data

# Get GDP
gdp = fetch_fred_data("GDP", start_date="2020-01-01", end_date="2026-08-01")

# Get CPI
cpi = fetch_fred_data("CPIAUCSL", start_date="2020-01-01", end_date="2026-08-01")

# Get Fed Funds Rate
fed funds = fetch_fred_data("FEDFUNDS", start_date="2020-01-01", end_date="2026-08-01")
```

### 12.5 Screener.in (Fundamentals Fallback)

```python
from autfin.tools.fundamental_tools import fetch_screener_fundamentals

fundamentals = fetch_screener_fundamentals("RELIANCE")
# Returns: {"roic": 0.15, "fcf_yield": 0.08, "moat": "strong", ...}
```

---

## 13. ERROR HANDLING & RECOVERY

### 13.1 RetryPolicy

```python
from pydantic_ai import ToolRetries

# Default: 2 retries with exponential backoff
tool_retries = ToolRetries(
    limit=2,
    backend="fixed",  # or "exponential"
)
```

### 13.2 TimeoutPolicy

```python
from pydantic_ai.settings import ModelSettings

model_settings = ModelSettings(
    timeout=30.0,  # seconds per agent call
    max_tokens=8192,
    temperature=0.0,
)
```

### 13.2 Error Recovery Patterns

```python
# Supervisor error handling with Command-based routing
try:
    result = await agent_call(...)
except Exception as e:
    return Command(
        update={"errors": [str(e)]},
        goto="human_review",  # HITL node
    )

# Graceful degradation — skip blocked capability
if not can_use("kite", "historical_data"):
    # Fall back to yfinance
    data = yfinance_download(...)
else:
    data = kite_historical(...)
```

### 13.3 Circuit Breaker

```python
# Mark capability as failed after N consecutive failures
mark_failed("kite", "historical_data")
# Agent will skip this capability on next call
# Manual reset via mark_available()
```

---

## 14. CL INTERFACE

### 14.1 CLI Commands

```
autfin status          # Show system status, agent health, capabilities
autfin analyze <symbol>  # Run full analysis on a ticker
autfin scan <universe>   # Screen universe using formula criteria
autfin start           # Start daily morning routine workflow
autfin backtest        # Run walk-forward backtest
autfin history         # Show trade history & performance
autfin config          # Show configuration, API keys status
autfin config reset    # Reset all API keys to .env.example defaults
```

### 14.2 main.py Entry Point

```python
#!/usr/bin/env python3
"""AutFin CLI — 7 commands for daily operation."""

import sys
import argparse
from autfin.cli.main import cli

if __name__ == "__main__":
    cli()
```

### 14.2 CLI Command Handlers

```python
@cli.command()
@click.pass_context
def status(ctx):
    """Show system status."""
    from autfin.observability import check_logfire_health
    from autfin.config.capabilities import print_capabilities
    
    print("=== AutFin v7.0 Status ===")
    print_capabilities()
    check_logfire_health()
    # Show agent health, memory status, etc.

@cli.command()
@click.argument("symbol")
def analyze(ctx, symbol):
    """Run full analysis on a ticker."""
    from autfin.agents import supervisor_agent
    result = supervisor_agent.run(
        f"analyze {symbol} with full technical + fundamental + sentiment",
        model_type="medium",
    )
    print(result.output)

@cli.command()
def start():
    """Start the daily morning routine."""
    from autfin.agents import run_morning_routine
    result = run_morning_routine()
    print(result.output if hasattr(result, 'output') else result)
```

---

## 15. MIGRATION FROM v6.0 → v7.0

### 15.1 Breaking Changes

| v6.0 Feature | v7.0 Replacement | Migration Effort |
|---|---|---|
| LangGraph StateGraph | Pydantic AI SubAgents + DynamicWorkflow | High — rewrite agent orchestration |
| LangSmith tracing | Logfire observability | Medium — update env vars, API keys |
| LangGraph Store | SQLite + ChromaDB + ManagedPrompts | Medium — data migration scripts exist |
| 8 LangGraph sub-agents | 7 Pydantic AI SubAgents | Medium — consolidate 1 agent (conferences subsumed into DynamicWorkflow) |
| Static prompt files | Logfire ManagedPrompt | Low — prompt content transfers, versioning added |
| langgraph.json + langgraph dev | pyproject.toml + pip install -e . | Low — dependency changes only |
| langsmith API key | NVIDIA NIM API key + Marketaux/FRED keys | Medium — new free APIs, no credit card |

### 15.2 Migration Steps

1. **Update dependencies**: Replace `langgraph>=1.0.0` with `pydantic-ai[temporal]>=0.1.0`, `pydantic-ai-harness[codemode,dynamic-workflow]>=0.1.0`
2. **Update .env**: Replace LangSmith key with NVIDIA_NIM_API_KEY, add Marketaux/FRED keys
3. **Migrate prompts**: Copy static prompt content into Logfire ManagedPrompt instances
4. **Migrate data**: Run `python -m autfin.migration.convert_langgraph_store` to migrate LangGraph checkpoints to SQLite
5. **Update imports**: Replace `from langgraph.graph import ...` with `from pydantic_ai import ...`, `from pydantic_ai_harness import ...`
6. **Update CLI**: Replace `langgraph dev` with `autfin start` / `autfin analyze`
7. **Run validation**: Execute all 8 validation gates, fix any failures
8. **Deploy**: Deploy to production with updated configuration

### 15.3 Post-Migration Checklist

- [ ] All 8 validation gates pass
- [ ] 7 sub-agents functional (market_data, analyze, research, risk, debate, conference, memory)
- [ ] Dynamic workflows (morning, quarterly) functional
- [ ] Model tier selection working per agent
- [ ] Logfire observability active (traces appearing)
- [ ] ManagedPrompt versioning operational
- [ ] Kite SDK integration working (paper trading first)
- [ ] yfinance, Marketaux, FRED data flows
- [ ] Backtesting pipeline functional
- [ ] CLI commands operational

---

## 16. SECURITY & COMPLIANCE

### 16.1 API Key Management

- All API keys stored in `.env` (gitignored)
- Never commit `.env` to version control
- Rotate NVIDIA NIM keys quarterly
- Marketaux: 100 req/day free tier — monitor usage
- FRED: 120 req/min, no daily limit — cache aggressively

### 16.2 Data Privacy

- User account data (Zerodha) never leaves local environment except via secure SDK
- Market data (yfinance, NSE public) is publicly available
- News sentiment (Marketaux) aggregates anonymized headlines only
- No PII stored in ChromaDB or SQLite (anonymize user_id before storage)

### 16.3 Compliance Notes

- SEBI regulations: No personalized investment advice without disclaimer
- RBI guidelines: Position sizing respects account margin limits
- Pattern Day Trader: US accounts >3 day trades/5-day window — monitor and warn
- Indian tax: Capital gains calculated per Income Tax Act 1961, Section 111/112

---

## 17. VERSION HISTORY

| Version | Date | Orchestration | LLM | Observability | Key Changes |
|---------|------|--------------|-----|---------------|-------------|
| v5.0 | Early 2025 | LangGraph StateGraph | Modal vLLM primary | LangSmith | Initial release, 8 sub-agents |
| v6.0 | Mid 2025 | LangGraph StateGraph + `interrupt()` | Modal vLLM + NVIDIA NIM fallback | LangSmith | HITL, Command routing, alpha decay |
| v7.0 | Aug 2026 | Pydantic AI SubAgents + DynamicWorkflow | NVIDIA NIM primary, Modal vLLM fallback | Logfire | SubAgents, ManagedPrompts, 8 validation gates, model tiers |

---

## 17. APPENDIX: Quick Reference

### 17.1 Environment Variables (`.env`)

```env
# === REQUIRED ===
NVIDIA_NIM_API_KEY=your_nvidia_nim_api_key    # Primary LLM (free tier)
MARKETAUX_API_KEY=your_marketaux_api_key     # News sentiment (free tier)
FRED_API_KEY=your_fred_api_key               # Macro data (free tier)

# === BROWSER MCP (Obscura — REQUIRED) ===
BROWSER_MCP_TRANSPORT=streamable_http
BROWSER_MCP_HOST=127.0.0.1
BROWSER_MCP_PORT=9333
BROWSER_MCP_ENDPOINT=http://127.0.0.1:9333/mcp
BROWSER_MCP_AUTOSTART=true

# === ZERODHA KITE (Personal Tier — free) ===
KITE_API_KEY=your_kite_api_key
KITE_API_SECRET=your_kite_api_secret
KITE_USER_ID=your_zerodha_user_id
KITE_PASSWORD=your_zerodha_password
KITE_TOTP_SECRET=your_totp_secret

# === OPTIONAL: LLM FALLBACK ===
MODAL_VLLM_URL=http://127.0.0.1:8000  # Self-hosted vLLM endpoint (optional)
# MODAL_API_KEY=your_modal_api_key       # If self-hosted Modal

# === OBSERVABILITY ===
LOGFIRE_TOKEN=your_logfire_token           # Auto-tracing (free tier)

# === SOCIAL SENTIMENT (optional) ===
# REDDIT_CLIENT_ID=your_reddit_client_id
# REDDIT_CLIENT_SECRET=your_reddit_client_secret
# REDDIT_USER_AGENT=AutFin-SentimentAgent/1.0

# === OPTIONAL NOTIFICATIONS ===
# PUSHOVER_USER_KEY=your_pushover_user_key
# PUSHOVER_API_TOKEN=your_pushover_api_token
```

### 17.2 Model Tier Quick Reference

```
deep  → deepseek-v4-flash  (1M ctx, complex reasoning)
      → Used by: supervisor, research, debate

medium → llama-3.3-70b-instruct  (70B, balanced)
         → Used by: analysis, risk, conference

fast   → llama-3.1-8b-instruct  (8B, low latency)
       → Used by: market_data, memory
```

### 17.3 Validation Gates Summary

| Gate | Tool | Target |
|------|------|--------|
| 1 | pyright --strict | Type safety |
| 2 | pytest -x -v | Unit tests (52 files) |
| 3 | evals suite | LLM judge quality (>80% pass) |
| 4 | guardrails | Input/tool/output safety |
| 5 | memory persistence | SQLite + ChromaDB |
| 6 | plan persistence | ManagedPrompt versions |
| 6 | load test | Concurrency without degradation |
| 8 | durable execution | Retries, timeouts, checkpointing |

### 17.4 Agent Quick Reference

```
supervisor_agent  → Deep tier, coordinates all 7 sub-agents
market_data_agent → Fast tier, quotes + portfolio + market status
analysis_agent    → Medium tier, 7 analysis types (TA, fundamentals, etc.)
research_agent    → Deep tier, screening + thesis generation
risk_agent        → Medium tier, position sizing + correlation + VaR
debate_agent      → Deep tier, mandatory Bull/Bear before trades
conference_agent  → Medium tier, coordination + rebalancing
memory_agent      → Fast tier, cross-session patterns + calibration
```

### 17.5 Workflow Quick Reference

```
morning_workflow  → Full pipeline: market_data → analyze → research → debate → risk → conference → memory
quarterly_workflow→ Deep dive: all holdings + DCF + debate + rebalancing + alpha decay check
```

---

*End of Technical Specification v7.0*