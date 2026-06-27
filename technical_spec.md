# AutFin — Technical Specification

**Version**: 5.1 | **Date**: June 2026 | **Status**: Complete — 31 improvements across 5 waves: critical bug fixes, architecture improvements, new capabilities (Kelly criterion, anomaly detection, episodic memory, LLM-as-Judge, reconciliation), production polish, and research infrastructure (walk-forward backtesting, regime detection, online learning, Kelly executor integration). 885+ tests passing.

---

## 1. RESOLVED TECH STACK

| Layer | Technology | Version | Justification |
|-------|-----------|---------|---------------|
| **Orchestration** | LangGraph | `>=1.0.0` | MIT. Stateful multi-agent workflows, checkpointing, Store API, HITL interrupts |
| **Kite/Zerodha** | KiteConnect SDK | `>=5.0.0` | Direct pykiteconnect SDK: auth, portfolio, orders, GTT, WebSocket. TOTP auto-login. |
| **LLM Router** | LiteLLM | `>=1.0.0` | Single entry point for all providers. Handles retries, rate-limit detection, model routing. |
| **LLM — Deep** | OpenRouter (free) | via LiteLLM | `llama-3.3-70b-instruct:free`, `deepseek-r1:free`. 50 req/day free. |
| **LLM — Fast** | Groq | via LiteLLM | `llama-3.1-8b-instant` (320 tps, 30 req/min), `llama-3.1-70b` (30 req/min). |
| **LLM — Long Context** | Google Gemini | `google-generativeai>=0.8.0` | `gemini-2.5-flash` (1,500 RPD, 1M context). Full research reports. |
| **LLM — Batch/Local** | Ollama | N/A (local) | Any GGUF model. Unlimited, $0, 100% privacy. Fallback for offline use. |
| **LLM — Burst** | Cerebras | via LiteLLM | `llama-3.3-70b`. 1M tokens/day, no credit card. DCF models, RAG docs. |
| **Database** | SQLite | via `sqlite-utils` | 5 tables: trades, portfolio_snapshots, signals, balance_history, learning_journal. WAL mode. |
| **Checkpointer** | SqliteSaver | `langgraph-checkpoint-sqlite>=2.0.0` | Node-level checkpointing, crash recovery, thread-level state isolation |
| **Vector Store** | ChromaDB | `>=0.5.0` | 3 collections: market_data, learning_journal, research_notes. Local-first. |
| **Long-Term Memory** | LangGraph Store API | `langgraph>=1.0.0` | Multi-scope (user/agent/run/app). Episodic + semantic + procedural. Async reads. |
| **Indian Stock Data** | NSE Public APIs | REST (no key) | Unlimited, real-time. Primary for Indian equity prices. |
| **Historical Data** | yfinance | `>=0.2.0` | Free, no key. `.NS`/`.BO` Indian tickers. OHLCV candles + fundamentals. |
| **Market Data Fallback** | NSE Public API | REST (no key) | Real-time NSE quotes as primary yfinance backup |
| **News Sentiment** | Marketaux + Alpha Vantage | REST (free keys) | Marketaux: 100 req/day NLP. Alpha Vantage: 200K+ tickers, per-article + per-ticker sentiment. NSE announcements as Indian fallback. |
| **Fundamentals** | yfinance (primary) + Screener.in | — | yfinance: ROIC, FCF yield, moat, health score. Screener.in: Firecrawl fallback for deep Indian fundamentals. |
| **Global Macro** | yfinance + FRED API | REST (free keys) | yfinance: S&P, NASDAQ, VIX, DXY, INR, crude, gold, US 10Y. FRED: GDP, CPI, Fed Funds with 24h cache. |
| **Web Scraping** | Firecrawl | `firecrawl-py>=1.0.0` | 500 credits/mo free tier. Screener.in + Moneycontrol fallback. |
| **Technical Analysis** | pandas-ta-classic (pure Python) | `pandas-ta-classic>=0.6.0` (optional TA-Lib) | SMA 20/50/200, EMA, RSI, MACD, ADX, Bollinger, ATR, OBV. Multi-period + golden/death cross. |
| **Scheduling** | APScheduler | `>=3.10.0` | Cron-style job scheduling |
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
├── autfin_prd.md                          # v2.0 Product Requirements Document
├── technical_spec.md                      # v2.0 Technical Specification (this file)
│
├── src/autfin/
│   ├── __init__.py
│   │
│   ├── graph/                             # LangGraph workflow definitions
│   │   ├── __init__.py
│   │   ├── supervisor.py                  # LLM-based dynamic routing + rule-based fallback (350 lines) ✅
│   │   ├── hitl.py                        # Human-in-the-loop trade gate. 294 lines.
│   │   └── error_handler.py               # Retry policy + circuit breaker. 548 lines.
│   │
│   ├── agents/                            # 13 specialized worker agents
│   │   ├── __init__.py
│   │   ├── market_data.py                 # Balance, Price, Market Data (404 lines)
│   │   ├── fundamentals.py                # Ratios, Health Score, Moat (1191 lines)
│   │   ├── technical.py                   # SMA 20/50/200, Indicators, Signals (777 lines)
│   │   ├── sentiment.py                   # Marketaux + NSE fallback (499 lines)
│   │   ├── global_macro.py                # yfinance + FRED API + 16 Indices + Sector Matrix
│   │   ├── researcher.py                  # Weighted synthesis + LLM recommendation (951 lines)
│   │   ├── risk_manager.py                # 14 checks: VaR, CVaR, news risk (1005 lines)
│   │   ├── portfolio_manager.py           # Capital allocation, tax, top-up (613 lines)
│   │   ├── executor.py                    # Kite SDK order execution, 3 modes (421 lines)
│   │   ├── valuation.py                   # DCF valuation agent
│   │   ├── peer_compare.py                # Peer comparison agent
│   │   ├── moat.py                        # Moat assessment agent
│   │   └── auto_execute.py                # Automated execution orchestration
│   │
│   ├── state/                        # State type definitions
│   │   ├── __init__.py
│   │   ├── agent_state.py           # Main AgentState TypedDict (shared across graphs)
│   │   ├── market_state.py          # Market snapshot: prices, volumes, indices
│   │   ├── portfolio_state.py       # Holdings, P&L, allocation
│   │   └── macro_state.py           # Global indicators, composite signals
│   │
│   ├── tools/                             # Agent tool implementations
│   │   ├── __init__.py
│   │   ├── kite_tools.py                   # KiteConnect SDK wrappers (809 lines)
│   │   ├── fundamental_tools.py            # Ratio computation, DCF models
│   │   ├── sentiment_tools.py              # HybridSentimentAnalyzer + multi-source (339+ lines)
│   │   ├── sentiment_analyzer.py           # FinBERT + VADER + keyword hybrid
│   │   ├── reddit_sentiment.py             # Reddit PRAW integration
│   │   ├── technical_tools.py              # Indicator computation (pandas-ta-classic)
│   │   ├── scraper_tools.py                # NSE cookies, nselib, web scraping (315+ lines)
│   │   ├── supervisor_tools.py             # 20 supervisor tools + callables
│   │   ├── screening.py                    # Magic Formula, F-Score, liquidity gate
│   │   ├── universe.py                     # Nifty 500 universe management
│   │   ├── market_feed.py                  # Real-time market feed
│   │   ├── memory_tools.py                 # Store/Vector Store retrieval
│   │   └── reporting.py                    # Report generation
│   │
│   ├── memory/                             # Long-term memory layer
│   │   ├── __init__.py
│   │   ├── setup.py                         # Initialize checkpointer, store, vector store
│   │   ├── store.py                         # LangGraph Store operations (271 lines)
│   │   ├── vector_store.py                  # ChromaDB collections (325 lines)
│   │   ├── database.py                      # SQLite schema + CRUD (128 lines)
│   │   ├── connection_pool.py               # Thread-safe SQLite connection pool
│   │   └── retrieval.py                     # Multi-scope memory retrieval (688 lines)
│   │
│   ├── llm/                                 # LLM client abstraction
│   │   ├── __init__.py
│   │   └── client.py                        # LLMClient: generate(), generate_json() (243 lines)
│   │
│   ├── prompts/                             # Agent system prompts (✅ all 8 created)
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
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py                      # AutFinSettings Pydantic model (151 lines)
│   │
│   ├── scheduler/
│   │   ├── __init__.py
│   │   └── jobs.py                          # APScheduler: 4 cron jobs (393 lines)
│   │
│   ├── dashboard/                           # Streamlit web UI
│   │   ├── __init__.py
│   │   ├── app.py                           # 7-page dashboard with auto-refresh (465 lines)
│   │   └── data.py                          # Real-time SQLite queries (471 lines)
│   │
│   ├── observability/                      # Langfuse tracing + metrics (✅ Phase F)
│   │   ├── __init__.py
│   │   ├── tracer.py
│   │   ├── handler.py
│   │   ├── metrics.py
│   │   ├── roro.py                          # Return on Research Output
│   │   └── thesis_ab.py                     # A/B thesis framework
│   │
│   ├── paper_trading/                       # Simulated trading (✅ Phase F)
│   │   ├── __init__.py
│   │   └── engine.py                        # 372 lines — fills, P&L, lifecycle
│   │
│   ├── backtest/                            # Historical backtesting
│   │   ├── __init__.py
│   │   ├── engine.py                        # BacktestEngine (303 lines)
│   │   ├── strategies.py                    # 5 strategies (546 lines)
│   │   └── metrics.py                       # 18+ metrics (181 lines)
│   │
│   ├── alerts/                              # Alert notification system
│   │   ├── __init__.py
│   │   └── notifier.py                      # 4 backends: Console, Desktop, Pushover, Email (509 lines)
│   │
│   ├── state/                               # State type definitions
│   │   ├── __init__.py
│   │   └── agent_state.py                   # AgentState + DecisionRecord (74 lines)
│   │
│   └── cli/                                 # CLI interface
│       ├── __init__.py
│       └── main.py                          # 7 commands: status|analyze|scan|start|backtest|history|config (916 lines)
│
├── tests/                               # 914+ tests, 0 failures
│   ├── test_agents/
│   │   ├── test_market_data.py           # 6 tests ✅
│   │   ├── test_global_macro.py          # 8 tests ✅
│   │   ├── test_global_macro_v4.py       # 13 tests ✅ (v4.0)
│   │   ├── test_fundamentals.py          # 6 tests ✅
│   │   ├── test_technical.py             # 6 tests ✅
│   │   ├── test_sentiment.py             # 6 tests ✅
│   │   ├── test_researcher.py            # 9 tests ✅
│   │   ├── test_risk_manager.py          # 29 tests ✅
│   │   ├── test_portfolio_manager.py
│   │   ├── test_executor.py
│   │   └── test_auto_execute.py
│   ├── test_tools/
│   │   ├── test_kite_tools.py
│   │   ├── test_sentiment_analyzer.py    # 23 tests ✅ (v4.0)
│   │   ├── test_reddit_and_multi_source.py # 17 tests ✅ (v4.0)
│   │   ├── test_nse_fix.py              # 10 tests ✅ (v4.0)
│   │   ├── test_paper_trading.py         # 43 tests ✅ (v4.1)
│   │   └── test_universe.py
│   ├── test_graphs/
│   │   ├── test_v3_integration.py
│   │   ├── test_v4_integration.py        # 15 tests ✅ (v4.0)
│   │   ├── test_hitl.py
│   │   ├── test_error_handler.py
│   │   └── test_supervisor.py            # 21 tests ✅ (v4.1)
│   ├── test_memory/
│   │   ├── test_store.py
│   │   ├── test_vector_store.py
│   │   ├── test_database.py
│   │   ├── test_performance.py
│   │   └── test_memory_integration.py
│   ├── test_learning/
│   │   ├── test_calibrator.py
│   │   ├── test_retriever.py
│   │   └── test_feedback_loop.py
│   ├── test_dashboard/
│   │   ├── test_phase3.py               # 37 tests ✅
│   │   └── test_phase4.py              # 13 tests ✅ (v4.0)
│   ├── test_llm/
│   │   └── test_client.py
│   ├── test_backtest/
│   ├── test_alerts/
│   ├── test_cli/
│   └── test_scheduler/
│
├── data/                             # Local data storage (gitignored)
│   ├── checkpoints.db                # LangGraph checkpoints (SQLite WAL)
│   ├── autfin.db                     # Structured data: trades, signals, portfolio
│   └── chroma/                       # ChromaDB persistence directory
│
└── notebooks/                        # Exploration & backtesting
    └── backtesting.ipynb
```

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
    thread_id: str                  # LangGraph thread identifier
    run_mode: str                   # "daily_morning" | "adrenaline" | "quarterly"
    errors: list[str]               # Accumulated error messages
```

### 3.2 Decision Record (stored in Store)

```python
class DecisionRecord(TypedDict):
    decision_id: str                # UUID
    timestamp: str                  # ISO 8601
    symbol: str
    action: str                     # "BUY" | "SELL" | "HOLD" | "REDUCE"
    conviction: float               # 0.0 - 1.0

    # Input context
    fundamentals_score: float       # 0-1
    technical_score: float          # 0-1
    sentiment_score: float          # -1 to +1
    macro_score: float              # -1 to +1
    risk_score: float               # 0-1

    # Output
    rationale: str                  # LLM-generated explanation
    position_size: float | None     # ₹ amount
    price: float | None             # Execution price
    budget_deployed: float | None   # ₹ from balance used

    # Outcome (filled later)
    outcome_pnl: float | None       # Realized P&L
    outcome_date: str | None        # When position closed
    was_correct: bool | None        # Retrospective accuracy
```

---

## 4. AGENT INTERFACES

### 4.1 All agents follow this pattern

Every agent node is an async function with this signature:

```python
async def agent_name(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """
    All agents:
    1. Read what they need from state (NOT the entire state)
    2. Call their tools / LLM
    3. Return a partial state update (only the keys they changed)
    4. NEVER modify state directly — always return
    """
    # 1. Extract relevant context
    symbols = [h["trading_symbol"] for h in state.get("holdings", [])]

    # 2. Do the work (LLM call, tool call, computation)
    result = await do_analysis(symbols, state)

    # 3. Return ONLY what this agent produced
    return {
        "fundamental_reports": result,
        "errors": []  # Clear any prior errors
    }
```

### 4.2 Example: Market Data Agent

```python
# agents/market_data.py
async def market_data_agent(state: AgentState, config: RunnableConfig) -> dict:
    """Reads balances, holdings, positions, live prices from KiteConnect SDK."""
    from .tools import kite_tools

    # Fetch ALL Kite data in parallel
    margins, holdings, positions = await asyncio.gather(
        kite_tools.get_margins(),
        kite_tools.get_holdings(),
        kite_tools.get_positions()
    )

    available_cash = margins.get("available", {}).get("cash", 0)
    utilized = margins.get("utilised", {}).get("debits", 0)

    return {
        "available_cash": available_cash,
        "utilized_margin": utilized,
        "holdings": holdings if isinstance(holdings, list) else [],
        "positions": positions if isinstance(positions, dict) else {},
        "errors": [],
    }
```

### 4.3 Example: Fundamentals Agent

```python
# agents/fundamentals.py
async def fundamentals_agent(state: AgentState, config: RunnableConfig) -> dict:
    """
    Computes fundamental analysis for holdings + watchlist.
    Primary: Screener.in via Firecrawl. Fallback: yfinance fundamentals.
    """
    from .tools import fundamental_tools
    from ..memory import retrieval

    symbols = _extract_symbols(state)

    # 1. Check memory — do we have recent reports?
    cached = await retrieval.get_latest_research(symbols, "fundamentals", max_age_days=7)

    needs_refresh = [s for s in symbols if s not in cached]
    if not needs_refresh:
        return {"fundamental_reports": cached}

    # 2. Scrape Screener.in for stocks needing refresh
    reports = {}
    for symbol in needs_refresh:
        try:
            report = await fundamental_tools.scrape_screener(symbol)
            reports[symbol] = report
        except Exception as e:
            reports[symbol] = {"error": str(e), "source": "failed"}
            # Fallback: try yfinance
            try:
                fallback = await fundamental_tools.yfinance_fundamentals(symbol + ".NS")
                reports[symbol] = {**reports[symbol], "fallback": fallback}
            except:
                pass

    # 3. Compute ratios + DCF via LLM
    analyzed = await _analyze_with_llm(reports, state)

    # 4. Cache in memory
    await retrieval.store_research(symbols, "fundamentals", analyzed)

    return {
        "fundamental_reports": {**cached, **analyzed},
        "errors": [],
    }
```

### 4.4 Example: Researcher Agent (Bull/Bear debate)

```python
# agents/researcher.py
async def researcher_agent(state: AgentState, config: RunnableConfig) -> dict:
    """
    Takes Layer 1 research outputs, runs structured Bull/Bear debate,
    and produces balanced synthesis for each stock.
    """
    symbols = _get_analyzed_symbols(state)

    debates = {}
    for symbol in symbols:
        # Build context from all research agents
        context = _build_stock_context(symbol, state)

        # Bull case
        bull_case = await _generate_case(context, stance="bull")

        # Bear case
        bear_case = await _generate_case(context, stance="bear")

        # Synthesis (compare bull vs bear, decide weighted recommendation)
        synthesis = await _synthesize(bull_case, bear_case, context)

        debates[symbol] = {
            "bull_case": bull_case,
            "bear_case": bear_case,
            "synthesis": synthesis,
            "rating": synthesis["rating"],
            "conviction": synthesis["conviction"],
        }

    return {
        "research_debates": debates,
        "errors": [],
    }
```

### 4.5 Example: Risk Manager Agent (veto power)

```python
# agents/risk_manager.py
async def risk_manager_agent(state: AgentState, config: RunnableConfig) -> dict:
    """
    Gatekeeper. Evaluates every trade proposal. Has VETO power.
    Checks: cash availability, position sizing, sector exposure,
            drawdown, volatility, liquidity, pacing.
    """
    available_cash = state["available_cash"]
    proposals = state.get("research_debates", {})
    settings = _load_settings(config)

    assessments = {}
    for symbol, debate in proposals.items():
        checks = []

        # 1. Cash available?
        checks.append(_check_cash(available_cash, settings.min_trade_amount))

        # 2. Position size within limits?
        proposed_size = _calculate_position_size(debate, available_cash)
        checks.append(_check_position_size(proposed_size, available_cash, settings))

        # 3. Sector exposure?
        checks.append(_check_sector_exposure(symbol, state, proposed_size, settings))

        # 4. Drawdown halt?
        checks.append(_check_drawdown(state, settings))

        # 5. Liquidity?
        checks.append(_check_liquidity(symbol, proposed_size, state))

        # 6. Pacing (deploying too fast?)
        checks.append(_check_pacing(state, settings))

        # 7. Volatility regime?
        checks.append(_check_volatility_regime(state, settings))

        # Aggregate
        passed = all(c["passed"] for c in checks)
        critical_blocks = any(c["severity"] == "critical" for c in checks if not c["passed"])

        if critical_blocks:
            verdict = "BLOCKED"
        elif not passed:
            verdict = "SIZE_REDUCED"
        else:
            verdict = "APPROVED"

        assessments[symbol] = {
            "checks": checks,
            "verdict": verdict,
            "approved_size": proposed_size if verdict != "BLOCKED" else 0,
            "rationale": "; ".join(c["message"] for c in checks if not c["passed"]),
        }

    return {
        "risk_assessments": assessments,
        "errors": [],
    }
```

---

## 5. GRAPH DESIGNS

> **v3.0 NOTE:** This section documents the legacy multi-graph architecture (v2.x). The v3.0 redesign collapses all graphs into a single supervisor running a ReAct loop with 13 domain tools. Context in state determines behavior instead of separate graphs. See `autfin_prd.md` Section 3 for the v3.0 architecture. Legacy graphs remain functional until the v3.0 supervisor is built.

### 5.0 v3.0 Research Engine — Fan-Out Architecture

The v3.0 research engine uses a **Fan-Out/Fan-In (scatter-gather) pattern**: screen 500+ → select 20 → research all 20 in parallel with 6 agents each → converge to top 5. This turns an 8.5-hour sequential analysis into a 25-minute parallel scan.

```
SCREEN (60s)        FAN-OUT (20 min)         CONVERGE (5 min)
──────────          ───────────────          ──────────────
Nifty 500         ┌─ fundamentals ─┐
  │  Filter 1     │  (10 min)     │
  │  Liquidity    │               │
  ▼               ├─ technical ────┤
420 stocks         │  (3 min)      │      ┌──────────────┐
  │  Filter 2     │               │      │  SUPERVISOR  │
  │  Magic Formula├─ sentiment ───┤      │  ranks 20 → 5│
  ▼               │  (5 min)      ├─────▶│  builds pitch │
85 stocks          │               │      │  decides      │
  │  Filter 3     ├─ DCF (deep) ──┤      └──────────────┘
  │  F-Score+     │  (20 min)     │
  │  Red flags    │               │
  ▼               ├─ peer compare─┤
20 candidates      │  (8 min)      │
                   │               │
                   └─ moat assess ─┘
                   ALL 20 CONCURRENTLY
                   120 analyses / 20 min
```

**Screening Pipeline (3 deterministic filters):**
1. **Liquidity Gate**: Avg daily volume ≥ ₹1 Cr, Market Cap ≥ ₹500 Cr
2. **Magic Formula** (Greenblatt): Rank by Earnings Yield + ROIC → top 85
3. **Piotroski F-Score ≥ 6** + Indian red-flag auto-reject (promoter pledge >20%, promoter holding declining, SEBI penalties, related-party transactions, negative OCF streak, auditor qualifications)

**6 Fan-Out Research Agents:**
| Agent | Time | Key Insight |
|-------|------|-------------|
| fundamentals | 10 min | F-Score, ratios, health, growth trends |
| technical | 3 min | Multi-TF indicators, S/R, patterns, signal |
| sentiment | 5 min | Hard/soft catalysts with dates, insider activity |
| valuation | 20 min | 3-scenario DCF + Reverse DCF (implied growth) |
| peer_compare | 8 min | 8-metric ranking vs sector peers |
| moat | 5 min | Morningstar 5-source moat score (0-100) |

**Conviction Weighting**: Fundamentals(0.30) + Valuation Upside%(0.25) + Sentiment(0.15) + Technical(0.10) + Moat(0.10) + Peer Position(0.10)

**Decision Gate:** Conviction ≥ 0.5 + Risk/Reward ≥ 2:1 + Sector < 30% conc. + Cash sufficient → trade queue

**Sector Rotation:** Macro overlay dynamically adjusts screening weights — IT/Pharma weights ↑ when INR weakens, NBFC weights ↓ when RBI hawkish, defensive sectors ↑ during global recession signals.

### 5.1 Supervisor Graph (top-level router)

```python
# graph/supervisor.py
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.types import Command
from typing import Literal

# Define routing types
class Router(TypedDict):
    next: Literal[
        "market_data",
        "global_macro",
        "fundamentals",
        "technical",
        "sentiment",
        "researcher",
        "risk_manager",
        "portfolio_manager",
        "executor",
        "human_review",
        "__end__",
    ]

def supervisor_node(state: AgentState) -> Command[Router]:
    """
    LLM-based supervisor. Reads state, decides next agent.
    Uses structured output to guarantee valid routing.
    """
    system = """You are the AutFin Supervisor. Route tasks to specialist agents.

    AGENTS AVAILABLE:
    - market_data: Fetch live quotes, margins, holdings from Kite SDK
    - global_macro: Fetch global indices, macro indicators, compute impact signal
    - fundamentals: Financial statement analysis, ratios, DCF
    - technical: Price pattern detection, indicators, signals
    - sentiment: News NLP sentiment scores
    - researcher: Bull/Bear debate and synthesis
    - risk_manager: Pre-trade risk checks (CAN VETO trades)
    - portfolio_manager: Capital allocation, trade prioritization
    - executor: Place/modify/cancel orders via Kite SDK
    - human_review: Pause for user approval

    ROUTING RULES:
    - Morning scan → market_data first, then parallel research
    - Trade signal → researcher → risk_manager → portfolio_manager → executor
    - FULL scan → market_data → parallel(global_macro, fundamentals, technical, sentiment) → researcher → risk_manager
    - Risk fail → stop, do NOT proceed to executor
    - All done → __end__
    """

    messages = [SystemMessage(content=system)] + state["messages"]
    response = model.with_structured_output(Router).invoke(messages)
    return Command(goto=response["next"])


def build_supervisor_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    # Add all agent nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("market_data", market_data_agent)
    builder.add_node("global_macro", global_macro_agent)
    builder.add_node("fundamentals", fundamentals_agent)
    builder.add_node("technical", technical_agent)
    builder.add_node("sentiment", sentiment_agent)
    builder.add_node("researcher", researcher_agent)
    builder.add_node("risk_manager", risk_manager_agent)
    builder.add_node("portfolio_manager", portfolio_manager_agent)
    builder.add_node("executor", executor_agent)
    builder.add_node("human_review", human_review_node)

    # Start → supervisor
    builder.add_edge(START, "supervisor")

    # All agents route back to supervisor (dynamic routing)
    builder.add_edge("market_data", "supervisor")
    builder.add_edge("global_macro", "supervisor")
    builder.add_edge("fundamentals", "supervisor")
    builder.add_edge("technical", "supervisor")
    builder.add_edge("sentiment", "supervisor")
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("risk_manager", "supervisor")
    builder.add_edge("portfolio_manager", "supervisor")
    builder.add_edge("executor", "supervisor")
    builder.add_edge("human_review", "supervisor")

    return builder
```

### 5.2 Daily Morning Graph (structured pipeline)

```python
# [REMOVED] graph/daily_morning.py — retired, logic absorbed into supervisor
# def build_daily_morning_graph() -> StateGraph:
#     """
    Deterministic morning routine — no LLM routing overhead.
    Two parallel tracks:
      Track A: New buys (if cash > min_trade_amount)
      Track B: Portfolio guardian (ALWAYS runs)
    """
    builder = StateGraph(AgentState)

    # --- Phase 0: Capital Check ---
    builder.add_node("capital_check", capital_check_node)
    builder.add_node("route_decision", route_decision_node)  # conditional edge

    # --- Track A: New Buys ---
    builder.add_node("parallel_research", parallel_research_node)  # Send API to 4 agents
    builder.add_node("researcher", researcher_agent)
    builder.add_node("risk_manager", risk_manager_agent)
    builder.add_node("portfolio_manager", portfolio_manager_agent)
    builder.add_node("executor", executor_agent)

    # --- Track B: Portfolio Guardian ---
    builder.add_node("position_monitor", position_monitor_node)
    builder.add_node("sell_executor", sell_executor_node)

    # --- Report ---
    builder.add_node("daily_report", daily_report_node)

    # Edges
    builder.add_edge(START, "capital_check")
    builder.add_conditional_edges(
        "capital_check",
        route_decision,
        {
            "scan_and_guard": "parallel_research",  # Cash available → Track A
            "guard_only": "position_monitor",        # No cash → Track B only
        }
    )

    # Track A: Buys
    builder.add_edge("parallel_research", "researcher")
    builder.add_edge("researcher", "risk_manager")
    builder.add_edge("risk_manager", "portfolio_manager")
    builder.add_edge("portfolio_manager", "executor")
    builder.add_edge("executor", "daily_report")

    # Track B: Guardian (always runs after capital check)
    builder.add_edge("position_monitor", "sell_executor")
    builder.add_edge("sell_executor", "daily_report")

    builder.add_edge("daily_report", END)

    return builder
```

### 5.3 Adrenaline Graph (real-time alert)

```python
# [REMOVED] graph/adrenaline.py — retired, logic absorbed into supervisor
# def build_adrenaline_graph() -> StateGraph:
#     """Triggered by significant price move (>2%). Budget-aware."""
    builder = StateGraph(AgentState)

    builder.add_node("price_alert", price_alert_node)       # Verify alert
    builder.add_node("quick_technical", technical_agent)      # Re-analyze
    builder.add_node("quick_research", researcher_agent)      # Re-evaluate signal
    builder.add_node("quick_risk", risk_manager_agent)        # Quick risk check
    builder.add_node("quick_execute", executor_agent)         # Act if approved

    builder.add_edge(START, "price_alert")
    builder.add_conditional_edges(
        "price_alert",
        lambda s: "analyze" if s["price_move_pct"] > 2.0 else "ignore",
        {"analyze": "quick_technical", "ignore": END}
    )
    builder.add_edge("quick_technical", "quick_research")
    builder.add_edge("quick_research", "quick_risk")
    builder.add_edge("quick_risk", "quick_execute")
    builder.add_edge("quick_execute", END)

    return builder
```

---

## 6. COMPILED GRAPH WITH ALL BACKENDS

```python
# graph/__init__.py
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore  # Phase 1
# from langgraph.store.sqlite import SqliteStore   # Phase 4+
import chromadb

def create_autfin_graph():
    """
    Compiles the full AutFin graph with all backends attached.
    Returns: (compiled_graph, store, vector_store)
    """
    # 1. Checkpointer (durability)
    checkpointer = SqliteSaver.from_conn_string(
        "data/checkpoints.db",
        wal=True  # Write-ahead logging for crash safety
    )

    # 2. Store (cross-thread long-term memory)
    store = InMemoryStore()
    # Phase 4 upgrade: SqliteStore for cross-process persistence

    # 3. Vector Store (semantic search)
    chroma_client = chromadb.PersistentClient(path="data/chroma")
    vector_store = chroma_client

    # 4. Build & compile
    from .supervisor import build_supervisor_graph
    graph = build_supervisor_graph()
    compiled = graph.compile(
        checkpointer=checkpointer,
        store=store
    )

    return compiled, store, vector_store
```

---

## 7. DATABASE SCHEMA (SQLite)

```sql
-- Table: trades
-- Every order placed, every order filled
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,              -- ISO 8601
    symbol TEXT NOT NULL,
    exchange TEXT NOT NULL DEFAULT 'NSE',
    action TEXT NOT NULL,                 -- 'BUY' | 'SELL'
    quantity INTEGER NOT NULL,
    price REAL NOT NULL,
    total_value REAL NOT NULL,
    order_id TEXT,                        -- Kite order ID
    order_status TEXT,                    -- 'COMPLETE' | 'PENDING' | 'REJECTED'
    decision_id TEXT,                     -- Links to DecisionRecord in Store
    rationale_summary TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);

-- Table: portfolio_snapshots
-- Daily snapshot of account state at market close
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,                   -- 'YYYY-MM-DD'
    available_cash REAL NOT NULL,
    utilized_margin REAL NOT NULL,
    total_holdings_value REAL NOT NULL,
    total_pnl REAL NOT NULL,
    total_pnl_pct REAL NOT NULL,
    holdings_json TEXT NOT NULL,          -- Full holdings as JSON
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_snapshots_date ON portfolio_snapshots(date);

-- Table: signals
-- Every buy/sell/hold signal generated
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    signal_type TEXT NOT NULL,            -- 'BUY' | 'SELL' | 'HOLD'
    strength REAL NOT NULL,               -- 0.0 - 1.0
    fundamentals_score REAL,
    technical_score REAL,
    sentiment_score REAL,
    macro_score REAL,
    risk_score REAL,
    rationale TEXT,
    was_acted_on BOOLEAN DEFAULT FALSE,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_acted ON signals(was_acted_on);

-- Table: balance_history
-- Daily balance snapshot from KiteConnect SDK
CREATE TABLE IF NOT EXISTS balance_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    available_cash REAL NOT NULL,
    utilized_margin REAL NOT NULL,
    source TEXT DEFAULT 'kite_sdk'
);

-- Table: learning_journal (v3.0 schema)
-- Post-trade reflections stored by the system
CREATE TABLE IF NOT EXISTS learning_journal (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL DEFAULT 'BUY',
    conviction REAL NOT NULL DEFAULT 0.5,
    pnl_pct REAL,
    thesis_summary TEXT,
    outcome TEXT DEFAULT 'open',
    sector TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_learning_symbol ON learning_journal(symbol);
CREATE INDEX IF NOT EXISTS idx_learning_sector ON learning_journal(sector);
CREATE INDEX IF NOT EXISTS idx_learning_outcome ON learning_journal(outcome);

-- Table: stock_pitches (v3.0)
-- 6-element buy-side pitches rendered by dashboard
CREATE TABLE IF NOT EXISTS stock_pitches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    context TEXT NOT NULL DEFAULT 'manual',
    investment_thesis TEXT,
    catalysts_json TEXT,
    risks_json TEXT,
    valuation_summary TEXT,
    competitive_moat TEXT,
    price_target_base REAL,
    price_target_upside REAL,
    price_target_downside REAL,
    price_target_rationale TEXT,
    raw_json TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_pitches_symbol ON stock_pitches(symbol);
CREATE INDEX IF NOT EXISTS idx_pitches_created ON stock_pitches(created_at);
CREATE INDEX IF NOT EXISTS idx_pitches_context ON stock_pitches(context);

-- Table: fanout_progress (v3.0)
-- Live status of concurrent research per symbol
CREATE TABLE IF NOT EXISTS fanout_progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    context TEXT NOT NULL,
    symbol TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    started_at TEXT,
    finished_at TEXT,
    error TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_fanout_session ON fanout_progress(session_id);
CREATE INDEX IF NOT EXISTS idx_fanout_status ON fanout_progress(status);
CREATE INDEX IF NOT EXISTS idx_fanout_created ON fanout_progress(created_at);

-- Table: thesis_check_ab
-- A/B testing framework for LLM vs rule-based thesis checks
CREATE TABLE IF NOT EXISTS thesis_check_ab (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    llm_verdict TEXT,
    rule_verdict TEXT NOT NULL,
    llm_reasoning TEXT,
    price_change_pct REAL DEFAULT 0,
    days_held INTEGER DEFAULT 0,
    subsequent_outcome TEXT,
    subsequent_pnl_pct REAL,
    timestamp TEXT NOT NULL
);

-- Table: research_output
-- RORO tracking: research depth vs trade outcomes
CREATE TABLE IF NOT EXISTS research_output (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    depth TEXT NOT NULL,
    research_weight INTEGER NOT NULL,
    fundamentals_score REAL,
    technical_score REAL,
    sentiment_score REAL,
    macro_score REAL,
    valuation_score REAL,
    research_duration_ms REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Table: trace_history (v4.1)
-- Observability traces persisted to SQLite
CREATE TABLE IF NOT EXISTS trace_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT NOT NULL,
    name TEXT NOT NULL,
    model TEXT,
    total_tokens INTEGER DEFAULT 0,
    total_cost REAL DEFAULT 0.0,
    errors_json TEXT,
    metadata_json TEXT,
    duration_ms REAL DEFAULT 0.0,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_trace_id ON trace_history(trace_id);
CREATE INDEX IF NOT EXISTS idx_trace_created ON trace_history(created_at);
```

---

## 8. MEMORY RETRIEVAL STRATEGIES

```python
# memory/retrieval.py

async def build_research_context(
    symbol: str,
    store: BaseStore,
    vector_store,
    thread_id: str
) -> dict:
    """
    When analyzing a stock, automatically:
    1. Retrieves past research on this stock (Store)
    2. Retrieves similar past trade rationales (Vector similarity)
    3. Retrieves sector knowledge (Vector Store)
    4. Includes relevant past outcomes
    """

    # 1. Cross-thread memory via Store
    past_reports = await store.aget(("research_reports", symbol))
    past_trades = await store.aget(("trade_history", symbol))
    past_decisions = await store.aget(("decisions", symbol))

    # 2. Semantic memory via vector store
    similar_trades = vector_store.get(
        collection="strategy_learnings",
        where={"symbol": symbol},
        limit=5
    )

    sector_info = vector_store.get(
        collection="company_knowledge",
        where={"sector": _get_sector(symbol)},
        limit=5
    )

    # 3. Recent performance
    recent_pnl = await _get_recent_pnl(symbol, days=90)

    return {
        "past_research": past_reports,
        "trade_history": past_trades,
        "past_decisions": past_decisions,
        "similar_trades_learnings": similar_trades,
        "sector_context": sector_info,
        "recent_90d_pnl": recent_pnl,
    }
```

---

## 9. CONFIGURATION

### 9.1 Environment Variables (`.env`)

```bash
# ===== REQUIRED =====
# NVIDIA NIM (primary LLM — free, 1M context)
NVIDIA_NIM_API_KEY=...

# Marketaux (news sentiment — free tier, 100 req/day)
MARKETAUX_API_KEY=...

# FRED (macro data — free tier)
FRED_API_KEY=...

# ===== OPTIONAL: Zerodha KiteConnect SDK (Personal tier) =====
KITE_API_KEY=your_kite_api_key
KITE_API_SECRET=your_kite_api_secret
KITE_USER_ID=your_zerodha_user_id
KITE_PASSWORD=your_zerodha_password
KITE_TOTP_SECRET=your_totp_secret_from_zerodha_console

# ===== OPTIONAL: Observability =====
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com

# ===== OPTIONAL: Alternative LLM providers (via litellm) =====
OPENROUTER_API_KEY=sk-or-v1-...
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=...
CEREBRAS_API_KEY=...

# ===== OPTIONAL: Social sentiment =====
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT=AutFin/1.0
```

### 9.2 Settings (typed config)

```python
# config/settings.py
from pydantic import Field
from pydantic_settings import BaseSettings

class AutFinSettings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # ── LLM ──────────────────────────────────────────
    deep_model: str = Field(
        default="nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
        description="Primary model — Nemotron 3 Super 120B (free, 1M context)"
    )
    fast_model: str = Field(
        default="nvidia_nim/meta/llama-3.1-8b-instruct",
        description="Fast model for data extraction, routing"
    )
    model_temperature: float = Field(default=0.0)

    # ── Zerodha ──────────────────────────────────────
    kite_user_id: str = Field(default="", description="Zerodha user ID for programmatic login")
    kite_password: str = Field(default="", description="Zerodha password for programmatic login")
    kite_totp_secret: str = Field(default="", description="TOTP secret from Zerodha console for auto-login")

    # ── Capital Management ──────────────────────────
    min_trade_amount: float = Field(default=1000.0)
    cash_buffer_pct: float = Field(default=0.10)
    deployment_pacing: str = Field(default="spread")

    # ── Risk Parameters ─────────────────────────────
    max_single_position_pct: float = Field(default=0.30)
    max_sector_exposure_pct: float = Field(default=0.30)
    default_stop_loss_pct: float = Field(default=0.08)
    trailing_stop_loss_pct: float = Field(default=0.05)
    profit_target_pct: float = Field(default=0.15)
    portfolio_drawdown_halt_pct: float = Field(default=0.10)
    max_daily_trades: int = Field(default=5)
    min_market_cap_crore: float = Field(default=500.0)
    min_avg_volume_lakhs: float = Field(default=5.0)

    # ── Execution ───────────────────────────────────
    auto_execute_max_amount: float = Field(default=5000.0)
    execution_mode: str = Field(default="recommendation")

    # ── Memory ──────────────────────────────────────
    vector_store_type: str = Field(default="chromadb")
    trade_history_retention_days: int = Field(default=3650)
    research_freshness_days: int = Field(default=7)
    checkpoint_durability: str = Field(default="wal")

    # ── Scheduling ──────────────────────────────────
    daily_scan_time: str = Field(default="09:00")
    market_open: str = Field(default="09:15")
    market_close: str = Field(default="15:30")
    data_refresh_interval_min: int = Field(default=5)

    # ── API Keys ────────────────────────────────────
    openrouter_api_key: str = Field(default="")
    gemini_api_key: str = Field(default="")
    groq_api_key: str = Field(default="")
    marketaux_api_key: str = Field(default="")
    fred_api_key: str = Field(default="")
    firecrawl_api_key: str = Field(default="")
```

---

## 10. SCHEDULED JOB DEFINITIONS

```python
# scheduler/jobs.py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

def register_jobs(scheduler: AsyncIOScheduler, graph, store, settings, checkpointer=None):
    """Register all scheduled jobs.

    Args:
        checkpointer: Optional LangGraph checkpointer for crash recovery.
                     If None, jobs run without persistence.
    """

    # Daily morning routine (Mon-Fri, 9:00 AM IST)
    scheduler.add_job(
        daily_morning_routine,
        CronTrigger(day_of_week="mon-fri", hour=9, minute=0, timezone="Asia/Kolkata"),
        args=[graph, store, settings, checkpointer],
        id="daily_morning",
        name="Daily Market Scan + Portfolio Guardian",
    )

    # Mid-day portfolio check (Mon-Fri, 1:00 PM IST)
    scheduler.add_job(
        portfolio_guardian_check,
        CronTrigger(day_of_week="mon-fri", hour=13, minute=0, timezone="Asia/Kolkata"),
        args=[graph, store, settings, checkpointer],
        id="midday_guardian",
        name="Mid-Day Portfolio Health Check",
    )

    # Weekly review (Sunday, 10:00 AM IST)
    scheduler.add_job(
        weekly_portfolio_review,
        CronTrigger(day_of_week="sun", hour=10, minute=0, timezone="Asia/Kolkata"),
        args=[graph, store, settings, checkpointer],
        id="weekly_review",
        name="Weekly Portfolio Review + Watchlist Refresh",
    )

    # Monthly deep dive (1st of month, 8:00 AM IST)
    scheduler.add_job(
        monthly_deep_dive,
        CronTrigger(day=1, hour=8, minute=0, timezone="Asia/Kolkata"),
        args=[graph, store, settings, checkpointer],
        id="monthly_deep_dive",
        name="Monthly Strategy Review + Full Fundamental Refresh",
    )


async def daily_morning_routine(graph, store, settings):
    """The main daily routine."""
    thread_id = f"morning-{datetime.now().strftime('%Y-%m-%d')}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "thread_id": thread_id,
        "run_mode": "daily_morning",
        "messages": [HumanMessage(content="Run the daily morning routine")],
    }

    async for event in graph.astream(initial_state, config, stream_mode="values"):
        # Log each state update
        logger.info(f"[{thread_id}] State update: {list(event.keys())}")

    logger.info(f"[{thread_id}] Morning routine complete")


async def portfolio_guardian_check(graph, store, settings):
    """Mid-day position health check — always runs regardless of cash."""
    thread_id = f"guardian-{datetime.now().strftime('%Y-%m-%d-%H')}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "thread_id": thread_id,
        "run_mode": "adrenaline",
        "messages": [HumanMessage(content="Run portfolio guardian check")],
    }

    async for event in graph.astream(initial_state, config, stream_mode="values"):
        pass
```

---

## 11. ERROR HANDLING MATRIX

| Error Type | Example | Handler | Recovery |
|-----------|---------|---------|----------|
| **Transient** | API timeout, rate limit | `RetryPolicy(max_attempts=3, initial_interval=1.0, backoff_factor=2)` | Auto-retry |
| **LLM Recoverable** | Bad JSON parse, wrong format | Loop back node with error in state, retry with stricter prompt | Auto-fix |
| **User Fixable** | Auth expired, balance changed | `interrupt()` → notify user → wait for `Command(resume=)` | HITL resume |
| **Graceful Degradation** | LLM API all down, Kite API down | `error_handler()` → use cached data, mark as stale, alert user | Run degraded |
| **Critical** | Database corruption, unrecoverable | Bubble up, log, alert user, attempt last-good-checkpoint restore | Manual + auto-checkpoint |

---

## 12. COMMAND-LINE INTERFACE

```bash
# Check status
autfin status
# → "Available: ₹48,230 | Positions: ₹1,770 | P&L: +₹1,240 (+2.5%)"

# Analyze a stock
autfin analyze RELIANCE
# → Full research report with Bull/Bear debate, scores, recommendation

# Full daily scan (manual trigger)
autfin scan
# → Runs the morning routine immediately

# Start as daemon (scheduled autonomous mode)
autfin start
# → Starts APScheduler, runs daily at 9:00 AM, runs forever

# Backtest a strategy
autfin backtest --strategy momentum --period 1y
# → Tests strategy on historical data, reports win rate, max drawdown, Sharpe

# View trade history
autfin history --symbol RELIANCE --days 90
# → Shows all trades, P&L, rationale for RELIANCE

# Set execution mode
autfin config set execution_mode recommendation  # Human approval needed
autfin config set execution_mode auto_small       # Auto-execute under ₹5K
autfin config set execution_mode paper            # Paper trading only

# Set pacing
autfin config set deployment_pacing aggressive
```

---

## 13. IMPLEMENTATION STATUS

```
✅ PHASE A-D — v2.x Foundation (COMPLETE)
  ✅ All 10 agents implemented (market_data through executor)
  ✅ 3 graphs: daily_morning, adrenaline, quarterly_deep_dive (all retired)
  ✅ 4 tools: kite, technical, fundamental, sentiment, scraper
  ✅ 8 prompts, dashboard (7 pages), scheduler, CLI, alerts
  ✅ 683 tests passing, 0 failures

✅ PHASE E — Research Depth (COMPLETE)
  ✅ DCF Valuation Agent (3-scenario + Reverse DCF)
  ✅ Peer Compare Agent (8-metric sector ranking)
  ✅ Moat Agent (Morningstar 5-source framework)
  ✅ Catalyst Engine (hard + soft catalyst extraction)

✅ PHASE F — Autonomy (COMPLETE)
  ✅ Langfuse observability (tracer, handler, metrics, RORO, thesis A/B)
  ✅ Paper trading engine (order lifecycle, P&L tracking)
  ✅ Backtest engine (5 strategies, 18+ metrics, Indian market costs)
  ✅ Learning loop (feedback_loop, calibrator, retriever)
  ✅ Alerts (Console, Desktop, Pushover, Email backends)

✅ v3.0 Redesign (COMPLETE)
  ✅ Single supervisor ReAct graph (graph/supervisor.py)
  ✅ 20 tools in supervisor_tools.py (1700+ lines)
  ✅ Fan-out research: screen → 20 candidates × 6 agents → converge
  ✅ Stock pitches (6-element buy-side standard)
  ✅ HITL gate (graph/hitl.py)
  ✅ Error handler with retry + circuit breaker
  ✅ 746+ tests passing

✅ v4.0 Global Markets & Sentiment (COMPLETE)
  ✅ 16 macro indices (+5: Shanghai, Hang Seng, DAX, CAC 40, Euro Stoxx 50)
  ✅ India 10Y G-Sec yield via FRED (INDIRLTLT01STM)
  ✅ Bank Nifty tracked separately from Nifty
  ✅ 14-sector impact matrix (was 8)
  ✅ HybridSentimentAnalyzer: FinBERT (news) + VADER (social) + keyword fallback
  ✅ Reddit PRAW integration (r/IndianStockMarket, r/IndiaInvestments)
  ✅ Multi-source sentiment: Marketaux (0.4) + MC (0.2) + ET (0.2) + Reddit (0.2)
  ✅ NSE announcements fix (cookie-based session)
  ✅ nselib integration (corporate actions + event calendar)
  ✅ pandas-ta-classic replaces custom numpy indicators + optional TA-Lib
  ✅ asyncio.get_event_loop() deprecation fix (4 files)
  ✅ 841+ tests passing

✅ v4.1 Reliability & Test Coverage (COMPLETE)
  ✅ Wired SqliteSaver checkpointer into scheduler (crash recovery)
  ✅ Refactored tools/__init__.py to lazy imports (reduces startup time)
  ✅ Fixed lookback_days unused param in thesis_ab.py and roro.py
  ✅ Fixed _persist_position bug in paper_trading/engine.py (6→7 placeholders)
  ✅ Fixed _get_trade_pnl_for_decision → _get_trade_pnl_for_symbol bug in roro.py
  ✅ Added SQLite connection pool (memory/connection_pool.py)
  ✅ Added trace persistence to SQLite (observability/tracer.py)
  ✅ Added 43 paper trading engine tests
  ✅ Added 21 supervisor graph unit tests
  ✅ 914+ tests passing

TEST COVERAGE: 914+ tests, all passing
COST: $0/month (all free-tier tools)
```

---

*This specification reflects the **v3.0 redesign** of AutFin. Sections 1-4 and 6-12 document the existing v2.x codebase (functional, 683 tests). Section 5 (v3.0 redesign) and Section 13 (implementation status) document the target architecture. The canonical PRD is `autfin_prd.md` v3.0.*
