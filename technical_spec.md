# AutFin v7.0 — Technical Specification

**Version**: 7.0 | **Date**: August 2026 | **Status**: Complete — Pydantic AI architecture, SubAgents, DynamicWorkflow, ManagedPrompts, 6 validation gates.

---

## 1. RESOLVED TECH STACK

| Layer | Technology | Version | Justification |
|-------|-----------|---------|---------------|
| **Orchestration** | Pydantic AI | `>=0.1.0` | Native SubAgents, DynamicWorkflow, Planning, Memory, CodeMode, Guardrails, SystemReminders |
| **Harness** | pydantic-ai-harness | `>=0.1.0` | SubAgents, DynamicWorkflow, Planning, Memory, CodeMode, Skills, StackOne, ManagedPrompt |
| **Evals** | pydantic-evals | `>=0.1.0` | Evaluation datasets, LLM judges, span evals |
| **Kite/Zerodha** | KiteConnect SDK | `>=5.0.0` | Direct pykiteconnect: auth, portfolio, orders, GTT, WebSocket. TOTP auto-login. |
| **LLM Client** | openai SDK | `>=1.0.0` | Direct connection to OpenAI-compatible endpoints (NVIDIA NIM, OpenRouter, etc.) |
| **LLM — Primary** | NVIDIA NIM | via openai SDK | deepseek-ai/deepseek-v4-flash (284B MoE, 1M ctx), meta/llama-3.3-70b-instruct, meta/llama-3.1-8b-instruct |
| **Database** | SQLite | via `sqlite-utils` | 5+ tables: trades, portfolio_snapshots, signals, balance_history, learning_journal. WAL mode. |
| **Vector Store** | ChromaDB | `>=0.5.0` | 3 collections: market_data, learning_journal, research_notes. Local-first. 384-dim embeddings. |
| **Long-Term Memory** | Logfire Store / FileStore | `logfire>=0.1.0` | Managed variables, prompts, cross-session learning. FileStore for local dev. |
| **Observability** | Logfire | `logfire[variables]>=0.1.0` | Auto-tracing, metrics, managed variables/prompts, A/B thesis, RAGAS evaluation |
| **Indian Stock Data** | NSE Public APIs | REST (no key) | Unlimited, real-time. Primary for Indian equity prices. |
| **Historical Data** | yfinance | `>=0.2.0` | Free, no key. `.NS`/`.BO` Indian tickers. OHLCV candles + fundamentals. |
| **News Sentiment** | Marketaux + Alpha Vantage | REST (free keys) | Marketaux: 100 req/day NLP. Alpha Vantage: 200K+ tickers. NSE announcements fallback. |
| **Fundamentals** | yfinance (primary) + Screener.in | — | yfinance: ROIC, FCF yield, moat, health score. Screener.in: Firecrawl fallback for deep Indian fundamentals. |
| **Global Macro** | yfinance + FRED API | REST (free keys) | yfinance: S&P, NASDAQ, VIX, DXY, INR, crude, gold, US 10Y. FRED: GDP, CPI, Fed Funds with 24h cache. |
| **Technical Analysis** | pandas-ta-classic (pure Python) | `>=0.6.0` | SMA 20/50/200, EMA, RSI, MACD, ADX, Bollinger, ATR, OBV. Multi-period + golden/death cross. |
| **Data** | Pandas + NumPy | `pandas>=2.2.0`, `numpy>=1.26.0` | All numerical computation |
| **HTTP** | httpx | `>=0.27.0` | Async HTTP client |
| **Config** | python-dotenv + settings.py | `python-dotenv>=1.0.0` | Environment + typed settings |
| **CLI** | Rich | `>=13.0.0` | Terminal output formatting |
| **Validation** | Pydantic | `>=2.0.0` | All state schemas, config models |

---

## 2. PROJECT STRUCTURE

```
autfin/
├── pyproject.toml                         # Dependencies, project metadata
├── .env                                   # API keys, secrets (gitignored)
├── .env.example                           # Template for required env vars
├── README.md                              # Project overview, getting started
├── technical_spec.md                      # v7.0 Technical Specification (this file)
│
├── src/autfin/
│   ├── __init__.py
│   ├── types.py                           # Shared type definitions (AgentDeps, outputs, MODEL_CONFIG)
│   ├── validation.py                      # 6 validation gates (type check, tests, evals, guardrails, memory, plan, load, durable)
│   │
│   ├── agents/                            # 7 SubAgents + Supervisor + capabilities
│   │   ├── __init__.py                    # Re-exports all agents + capabilities
│   │   ├── base.py                        # Agent factory, NVIDIA provider, model config
│   │   ├── market_data.py                 # Portfolio, quotes, market status, risk alerts
│   │   ├── analysis.py                    # Technical + fundamental + sentiment + valuation
│   │   ├── research.py                    # Deep research, screening, thesis generation
│   │   ├── risk.py                        # Position sizing, stop-loss, correlation, veto
│   │   ├── debate.py                      # Bull vs Bear adversarial debate
│   │   ├── conference.py                  # Multi-agent coordination conferences
│   │   ├── memory.py                      # Cross-session learning & calibration
│   │   ├── quant.py                       # Quantitative analysis agent
│   │   ├── supervisor.py                  # Supervisor + SubAgents + DynamicWorkflow
│   │   ├── managed_prompts.py             # Logfire ManagedPrompt instances
│   │   ├── guardrails.py                  # Input/output/tool guardrails
│   │   ├── observability.py               # Logfire instrumentation
│   │   ├── durable.py                     # Durable execution backends (Temporal/DBOS/Preft)
│   │   ├── schemas.py                     # Output schemas for all agents
│   │   ├── agent_tools.py                 # 10 deterministic tool wrappers
│   │   ├── yfinance_utils.py              # fetch_history(as_of=), normalize_symbol()
│   │   ├── technical.py                   # SMA 20/50/200, indicators, signals
│   │   ├── fundamentals.py                # Ratios, health score, moat
│   │   ├── sentiment.py                   # Marketaux + NSE fallback
│   │   ├── global_macro.py                # yfinance + FRED + 16 indices + sector matrix
│   │   ├── researcher.py                  # Weighted synthesis + LLM recommendation
│   │   ├── risk_manager.py                # 14 checks: VaR, CVaR, news risk
│   │   ├── portfolio_manager.py           # Capital allocation, tax, top-up
│   │   ├── executor.py                    # Kite SDK order execution, 3 modes
│   │   ├── valuation.py                   # DCF valuation (deterministic)
│   │   ├── distortion.py                  # Price-distortion detection (deterministic)
│   │   ├── regulatory.py                  # SEBI/RBI regulatory scanner (deterministic)
│   │   ├── peer_compare.py                # Peer comparison agent
│   │   ├── moat.py                        # Moat assessment agent
│   │   ├── auto_execute.py                # Automated execution orchestration
│   │   ├── conferences.py                 # Morning/TradeReview/Emergency/Experience
│   │   ├── memory_system.py               # 3-type persistent memory
│   │   └── subagent_states.py             # Per-agent TypedDict schemas
│   │
│   ├── evals/                             # Evaluation framework
│   │   ├── __init__.py
│   │   └── dataset.py                     # Pydantic eval datasets + runners
│   │
│   ├── memory/                            # Long-term memory layer
│   │   ├── __init__.py
│   │   ├── database.py                    # SQLite schema + CRUD
│   │   ├── vector_store.py                # ChromaDB collections (AutFinEmbeddingFunction)
│   │   ├── retrieval.py                   # Multi-scope memory retrieval
│   │   ├── consolidation.py               # Memory consolidation
│   │   ├── episodic.py                    # Episodic memory
│   │   ├── performance.py                 # Performance tracking
│   │   ├── rag_eval.py                    # RAG quality evaluation
│   │   └── connection_pool.py             # Thread-safe SQLite connection pool
│   │
│   ├── tools/                             # External integrations
│   │   ├── __init__.py                    # Lazy imports via __getattr__
│   │   ├── kite_tools.py                  # KiteConnect SDK wrappers
│   │   ├── fundamental_tools.py           # Ratio computation, DCF models
│   │   ├── sentiment_tools.py             # HybridSentimentAnalyzer + multi-source
│   │   ├── sentiment_analyzer.py          # FinBERT + VADER + keyword hybrid
│   │   ├── reddit_sentiment.py            # Reddit PRAW integration
│   │   ├── technical_tools.py             # Indicator computation (pandas-ta-classic)
│   │   ├── scraper_tools.py               # NSE cookies, nselib, web scraping
│   │   ├── supervisor_tools.py            # Supervisor tool implementations
│   │   ├── screening.py                   # Magic Formula, F-Score, liquidity gate
│   │   ├── universe.py                    # Nifty 500 universe management
│   │   └── memory_tools.py                # Store/Vector Store retrieval
│   │
│   ├── config/                            # Configuration
│   │   ├── __init__.py
│   │   ├── settings.py                    # AutFinSettings Pydantic model
│   │   └── capabilities.py                # Data source capability registry
│   │
│   ├── llm/                               # LLM client abstraction
│   │   ├── __init__.py
│   │   ├── client.py                      # LLMClient: generate(), generate_json(), generate_with_tools()
│   │   ├── importlib_patch.py             # Importlib patches
│   │   └── model_tiers.py                 # ModelTier, MODEL_REGISTRY, AGENT_TIERS
│   │
│   ├── state/                             # State type definitions
│   │   ├── __init__.py
│   │   ├── agent_state.py                 # AgentState (messages, portfolio, research, risk, routing)
│   │   └── portfolio_state.py             # PortfolioState
│   │
│   ├── learning/                          # Self-learning modules
│   │   ├── __init__.py
│   │   ├── judge.py                       # LLM-as-Judge
│   │   ├── critique.py                    # Self-critique loop
│   │   ├── calibrator.py                  # Conviction calibration
│   │   ├── online.py                      # O(1) incremental online learning
│   │   ├── feedback_loop.py               # Trade outcome feedback
│   │   ├── ab_testing.py                  # A/B thesis testing
│   │   └── retriever.py                   # Retrieval evaluation
│   │
│   ├── backtesting/                       # Walk-forward + alpha decay
│   │   ├── __init__.py
│   │   ├── walk_forward.py                # WalkForwardConfig, fetch_price_data, walk_forward_with_symbol
│   │   └── alpha_decay.py                 # AlphaDecayTracker, FoldSnapshot, DecayReport
│   │
│   ├── backtest/                          # Historical backtesting
│   │   ├── __init__.py
│   │   ├── engine.py                      # BacktestEngine
│   │   ├── strategies.py                  # 5 strategies
│   │   └── metrics.py                     # 18+ metrics
│   │
│   ├── risk/                              # Risk management modules
│   │   ├── __init__.py
│   │   ├── kelly.py                       # Kelly criterion with regime gates
│   │   ├── anomaly.py                     # Rolling z-score anomaly detection
│   │   ├── regime.py                      # VIX/volatility regime detector
│   │   └── vol_targeting.py               # Volatility-targeting overlay
│   │
│   ├── portfolio/                         # Portfolio management
│   │   ├── __init__.py
│   │   └── tax_engine.py                  # Tax calculation engine
│   │
│   ├── prompts/                           # Agent system prompts (legacy, replaced by ManagedPrompts)
│   │   ├── __init__.py
│   │   ├── supervisor.py
│   │   ├── market_data.py
│   │   ├── fundamentals.py
│   │   ├── technical.py
│   │   ├── sentiment.py
│   │   ├── global_macro.py
│   │   ├── researcher.py
│   │   └── risk_manager.py
│   │
│   ├── compat/                            # Compatibility shims
│   │   └── metadata_patch.py              # Metadata patches
│   │
│   ├── observability/                     # Logfire tracing + metrics
│   │   ├── __init__.py
│   │   ├── handler.py                     # Logfire auto-tracing handler
│   │   ├── tracer.py                      # @traceable decorator, SQLite persistence
│   │   ├── metrics.py                     # Custom metrics
│   │   ├── roro.py                        # Return on Research Output
│   │   └── thesis_ab.py                   # A/B thesis framework
│   │
│   ├── dashboard/                         # Streamlit web UI
│   │   ├── __init__.py
│   │   ├── app.py                         # 7-page dashboard with auto-refresh
│   │   └── data.py                        # Real-time SQLite queries
│   │
│   ├── paper_trading/                     # Simulated trading
│   │   ├── __init__.py
│   │   ├── engine.py                      # Fills, P&L, lifecycle
│   │   └── reconciliation.py              # Paper-live reconciliation
│   │
│   ├── mcp/                               # MCP (Model Context Protocol) integration
│   │   ├── __init__.py                    # Re-exports public surface
│   │   └── browser_client.py              # Obscura HTTP MCP client (singleton, autostart)
│   │
│   ├── universe/                          # Stock universe management
│   │   ├── __init__.py
│   │   └── curator.py                     # Nifty 500 universe curation
│   │
│   └── cli/                               # CLI interface
│       ├── __init__.py
│       └── main.py                        # 7 commands: status|analyze|scan|start|backtest|history|config
│
├── tests/                                 # 52 test files, 5,400+ tests
│   ├── conftest.py
│   ├── test_metrics.py
│   ├── test_validation.py
│   ├── test_agents/                       # Agent unit + integration tests
│   ├── test_tools/                        # Tool tests
│   ├── test_memory/                       # Memory layer tests
│   ├── test_learning/                     # Learning module tests
│   ├── test_llm/                          # LLM client tests
│   ├── test_risk/                         # Risk module tests
│   ├── test_backtest/                     # Backtest engine tests
│   ├── test_backtesting/                  # Walk-forward + alpha decay tests
│   ├── test_graphs/                       # Graph/Workflow tests
│   └── test_portfolio/                    # Portfolio tests
│
├── data/                                  # Local data storage (gitignored)
│   ├── autfin.db                          # Structured data: trades, signals, portfolio
│   ├── autfin_store.db                    # Logfire Store SQLite
│   ├── checkpoints.db                     # Legacy checkpoints
│   ├── paper_trading.db                   # Paper trading data
│   ├── test_checkpointer.db               # Test checkpoints
│   └── memory/                            # FileStore directory
│
└── notebooks/                             # Exploration & backtesting
    └── backtesting.ipynb
```

---

## 3. AGENT ARCHITECTURE

### 3.1 Agent Factory (`base.py`)

```python
def create_agent[OutputT](
    *,
    name: str,
    output_type: type[OutputT],
    instructions: str,
    model_type: Literal["deep", "fast", "reasoning", "code"] = "deep",
    tools: list[Callable] | None = None,
    toolsets: list[AbstractToolset[AgentDeps]] | None = None,
    capabilities: list[Any] | None = None,
    output_validators: list[Callable] | None = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    retries: int = 2,
) -> Agent[AgentDeps, OutputT]
```

- **NVIDIAProvider**: OpenAI-compatible client for NVIDIA NIM endpoints
- **Model Tiers**: deep (deepseek-v4-flash), standard (llama-3.3-70b), fast (llama-3.1-8b), reasoning, code
- **Shared Config**: temperature=0.0, max_tokens=8192, retries=2

### 3.2 SubAgents Capability (`supervisor.py`)

```python
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
    models={"fast": get_model("fast"), "standard": get_model("deep"), "deep": get_model("reasoning")},
    forward_usage=True,
    inherit_tools=False,
    contain_errors=True,
)
```

**7 SubAgents** with isolated context, parallel execution, usage limits, timeouts, max calls.

### 3.3 DynamicWorkflow (`supervisor.py`)

```python
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
    resource_limits={"max_duration_secs": 300},
    tool_name="run_morning_workflow",
)
```

**Two Workflows**:
- `morning_workflow`: 5 min timeout, 50 agent calls, 50K tokens
- `quarterly_workflow`: 10 min timeout, 100 agent calls, 200K tokens

### 3.4 Capabilities

| Capability | Config | Purpose |
|---|---|---|
| **Planning** | `enable_subtasks=True, inject=True, cache_ttl="5m"` | Task decomposition, subtask execution |
| **Memory** | `FileStore("./data/memory"), namespace=lambda ctx: ctx.deps.user_id` | Cross-session learning |
| **SystemReminders** | 6 persistent reminders, `cache_ttl="5m"` | Behavioral guardrails |
| **Thinking** | `effort="high"` | Extended reasoning |
| **WebSearch** | `local="duckduckgo"` | Real-time web search |

### 3.5 Agent Outputs (`schemas.py`)

Each agent returns typed Pydantic models:
- `MarketSnapshotOutput` — portfolio, quotes, indices, risk alerts
- `AnalysisReport` — technical + fundamental + sentiment + valuation
- `ResearchReport` — screening results, deep research, thesis
- `RiskAssessment` — 14 checks, position sizing, correlation matrix
- `DebateResult` — bull_case, bear_case, synthesis, recommendation
- `ConferenceResult` — coordination decisions, action items
- `MemoryResult` — events, patterns, calibration updates
- `SupervisorOutput` — routing decision, trade queue, final decision

---

## 4. STATE SCHEMAS

### 4.1 AgentState (main shared state)

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    available_cash: float
    utilized_margin: float
    holdings: list[dict]
    positions: list[dict]
    macro_signals: dict | None
    market_snapshot: dict | None
    fundamental_reports: dict | None
    technical_reports: dict | None
    sentiment_reports: dict | None
    research_debates: dict | None
    risk_assessments: dict | None
    trade_queue: list[dict] | None
    execution_reports: list[dict] | None
    next_agent: str | None
    thread_id: str
    run_mode: str  # "daily_morning" | "adrenaline" | "quarterly"
    errors: list[str]
```

### 4.2 Decision Record (Logfire Store)

```python
class DecisionRecord(TypedDict):
    decision_id: str
    timestamp: str
    symbol: str
    decision_type: str  # "buy" | "sell" | "hold" | "stop_loss" | "profit_book"
    conviction: float
    risk_reward: float
    debate_summary: str
    outcome: str | None  # Filled after trade closes
    pnl_pct: float | None
```

---

## 5. VALIDATION GATES (Phase 6)

Run via `python test_validation.py`:

| Gate | Command | Pass Criteria |
|------|---------|---------------|
| **Type Checking** | `pyright --strict` | 0 errors |
| **Unit Tests** | `pytest -x -v` | All pass |
| **Evaluation Suite** | `run_evaluations()` | >80% pass rate |
| **Guardrails** | `validate_guardrails()` | All pass |
| **Memory Persistence** | `validate_memory_persistence()` | FileStore round-trip |
| **Plan Persistence** | `validate_plan_persistence()` | Planning state recovery |
| **Load Test** | `validate_load_test()` | 10 concurrent agents < 30s |
| **Durable Execution** | `validate_durable_execution()` | Temporal/DBOS/Preft backend works |

---

## 6. OBSERVABILITY (Logfire)

### 6.1 Auto-Instrumentation

```python
from autfin.agents.observability import configure_logfire
configure_logfire()  # Call once at startup
```

- Auto-traces all Pydantic AI agent runs
- Captures tool calls, model calls, token usage
- Structured logging to Logfire cloud

### 6.2 ManagedPrompts

```python
from autfin.agents.managed_prompts import supervisor_instructions

supervisor_instructions = ManagedPrompt(
    name="supervisor_instructions",
    default="You are the AutFin v7.0 Supervisor...",
)
```

- Version-controlled in Logfire UI
- Template inputs with JSON schema validation
- Rollout/targeting (canary, production, per-user)
- Instant updates without code deploy

### 6.3 Metrics & A/B Thesis

```python
from autfin.agents.observability import record_trade_metric, record_risk_metric

record_trade_metric("trade_executed", {"symbol": "RELIANCE", "pnl": 2500})
record_risk_metric("portfolio_heat", 0.72)
```

- Custom metrics for business KPIs
- A/B thesis framework for strategy comparison
- Return on Research Output (RORO) tracking

---

## 7. DATA SOURCES & CAPABILITIES

`capabilities.py` registry — agents check before calling external APIs:

```python
CAPABILITIES = {
    "kite": {"holdings": True, "positions": True, "ltp": False, "quotes": False, ...},
    "yfinance": {"ohlcv": True, "ratios": True, "financials": "partial", ...},
    "nvidia_nim": {"available": True, "rate_limited": False, ...},
    "marketaux": {"available": True, "rate_limit": "100 articles/day"},
    "fred": {"available": True, "rate_limit": "120 requests/min"},
    "nsepython": {"symbols": True, "quotes": False, "ohlcv": False},
    "logfire": {"available": True, "configured": True},
}
```

Auto-fallback logic:
- Kite LTP/quotes blocked on Personal tier → use holdings.last_price or yfinance
- NSE quotes blocked → use yfinance (15-min delay)
- OpenRouter rate-limited → use NVIDIA NIM

---

## 8. MEMORY ARCHITECTURE

### 8.1 Three-Layer Memory

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Structured** | SQLite (autfin.db) | Trades, signals, portfolio snapshots, balance history |
| **Vector** | ChromaDB (3 collections) | Semantic search for research, learning journal |
| **Managed** | Logfire Store / FileStore | Cross-session learning, managed variables, prompts |

### 8.2 Collections

- `market_data` — OHLCV, technical indicators, news embeddings
- `learning_journal` — Trade decisions, outcomes, patterns
- `research_notes` — Deep research reports, thesis drafts

---

## 9. DAILY PIPELINE

```
09:00 IST ──► Morning Workflow Triggered
    │
    ├─► market_data (parallel)
    │     ├─ get_portfolio()
    │     ├─ get_quotes(symbols)
    │     ├─ get_market_status()
    │     └─ risk_alerts()
    │
    ├─► analyze (parallel, per holding)
    │     ├─ technical_analysis()
    │     ├─ fundamental_analysis()
    │     ├─ sentiment_analysis()
    │     └─ valuation_dcf()
    │
    ├─► research (parallel)
    │     ├─ screen_universe()
    │     ├─ deep_research(candidates)
    │     └─ generate_thesis()
    │
    ├─► debate (MANDATORY for trades)
    │     ├─ bull_case()
    │     ├─ bear_case()
    │     └─ synthesis()
    │
    ├─► risk
    │     ├─ position_sizing()
    │     ├─ correlation_check()
    │     ├─ portfolio_heat()
    │     └─ veto_if_needed()
    │
    └─► execute (if approved)
          ├─ place_orders()
          ├─ set_gtt()
          └─ log_decisions()
```

---

## 10. FREE LLM STACK

| Provider | Models | Context | Cost | Use Case |
|----------|--------|---------|------|----------|
| **NVIDIA NIM** | deepseek-v4-flash (284B) | 1M | Free tier | Supervisor, Debate, Research |
| **NVIDIA NIM** | llama-3.3-70b-instruct | 1M | Free tier | Analysis, Risk, Technical |
| **NVIDIA NIM** | llama-3.1-8b-instruct | 1M | Free tier | Market Data, Quant, Fast |
| **OpenRouter** | gemma-4-31b-it:free | 32K | 50/day shared | Fallback only |

---

## 11. DEPLOYMENT

### Local Development

```bash
pip install -e .
python -c "from autfin.agents import supervisor_agent; ..."
```

### Production Options

| Backend | Package | Config |
|---------|---------|--------|
| **Temporal** | `temporalio>=1.0.0` | Durable execution, retries, visibility |
| **DBOS** | `dbos>=0.1.0` | Transactional workflows, time travel |
| **Prefect** | `prefect>=3.0.0` | Flow orchestration, scheduling |

Enable via `pip install -e .[durable]` and configure in `settings.py`.

---

## 12. CONFIGURATION REFERENCE

Key `AutFinSettings` fields (`config/settings.py`):

```python
class AutFinSettings(BaseSettings):
    # NVIDIA NIM
    nvidia_nim_api_key: str
    nvidia_nim_model_deep: str = "deepseek-ai/deepseek-v4-flash"
    nvidia_nim_model_standard: str = "meta/llama-3.3-70b-instruct"
    nvidia_nim_model_fast: str = "meta/llama-3.1-8b-instruct"
    
    # Data APIs
    marketaux_api_key: str
    fred_api_key: str
    
    # Zerodha
    kite_api_key: str
    kite_api_secret: str
    kite_user_id: str
    kite_password: SecretStr
    kite_totp_secret: str
    
    # Obscura MCP
    browser_mcp_transport: str = "streamable_http"
    browser_mcp_host: str = "127.0.0.1"
    browser_mcp_port: int = 9333
    browser_mcp_autostart: bool = True
    
    # Logfire
    logfire_token: str | None = None
    
    # Database
    database_url: str = "sqlite+aiosqlite:///data/autfin.db"
    vector_store_path: str = "data/chroma"
    
    # Runtime
    run_mode: str = "daily_morning"
    log_level: str = "INFO"
```

---

## 13. TESTING STRATEGY

| Category | Files | Focus |
|----------|-------|-------|
| **Agent Tests** | `tests/test_agents/` | Unit + integration for 7 subagents + supervisor |
| **Tool Tests** | `tests/test_tools/` | Kite, sentiment, screening, universe, scraper |
| **Memory Tests** | `tests/test_memory/` | SQLite, ChromaDB, retrieval, consolidation |
| **Learning Tests** | `tests/test_learning/` | Calibrator, critique, online, feedback, A/B |
| **LLM Tests** | `tests/test_llm/` | Client, validation, model tiers |
| **Risk Tests** | `tests/test_risk/` | Kelly, anomaly, regime, vol targeting |
| **Backtest Tests** | `tests/test_backtest/`, `test_backtesting/` | Engine, walk-forward, alpha decay |
| **Graph Tests** | `tests/test_graphs/` | Supervisor, workflow integration |
| **Portfolio Tests** | `tests/test_portfolio/` | Tax engine, portfolio manager |

Run: `pytest tests/ -v --cov=autfin`

---

## 14. LICENSE

MIT — Personal project. See LICENSE file.