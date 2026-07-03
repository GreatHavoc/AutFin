# AutFin v5.4

**Autonomous AI-Powered Financial Research & Portfolio Management System**

A personal multi-agentic system that runs daily autonomous financial research, manages a Zerodha portfolio, and learns from every trade — using LangGraph tools pattern, Modal vLLM (Qwen 3.6 35B), and free data APIs.

## Features

- **LangGraph Tools Architecture** — Each tool (fundamentals, technical, sentiment, etc.) runs deterministically with context isolation, parallel execution, and full LangSmith visibility
- **10 Specialized Tools** — 9 deterministic (fundamentals, technical, sentiment, macro, valuation, distortion, regulatory, risk, portfolio) + 1 LLM-driven (research synthesis)
- **Modal vLLM (Primary LLM)** — Self-hosted Qwen 3.6 35B via Modal, zero cost, no rate limits, connects directly via openai SDK
- **NVIDIA NIM (Fallback LLM)** — MiniMax M3 via NVIDIA NIM when Modal is unavailable
- **LangSmith Tracing** — Full observability with auto-tracing, Studio visual debugging, per-tool traces
- **Daily Autonomous Operation** — Runs at 9:00 AM IST via APScheduler, manages existing positions even with ₹0 cash
- **Balance-is-Budget** — Your Zerodha account balance IS the deployable pool, no separate config needed
- **Long-Term Memory** — SQLite + ChromaDB + LangGraph Store for trade history, lessons learned, and cross-session learning
- **HITL with Interrupts** — LangGraph `interrupt()` for proper human-in-the-loop trade approval
- **Error Recovery** — Circuit breakers, retry policies, and graceful degradation for every tool
- **Backtesting** — Walk-forward backtesting with regime detection and online learning
- **Streamlit Dashboard** — Visualize portfolio, P&L, signals, and tool performance
- **Alert System** — Console, desktop, Pushover, and email notifications for key events
- **Full CLI** — `autfin status`, `autfin analyze RELIANCE`, `autfin scan`, `autfin start`

## Quick Start

```
autfin status                  # Check Kite connection & balance
autfin analyze RELIANCE        # Full research report on a stock
autfin scan                    # Run daily morning routine
autfin start                   # Run as daemon with scheduler
autfin history --symbol RELIANCE --days 90
autfin backtest --strategy momentum --period 1y
autfin config set KEY VALUE    # Update configuration
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    CLI / Dashboard                        │
├──────────────────────────────────────────────────────────┤
│              Scheduler (APScheduler)                      │
│         Daily 9AM │ Midday │ Weekly │ Monthly             │
├──────────────────────────────────────────────────────────┤
│              LangGraph Supervisor (ReAct Loop)            │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Calls tools directly (deterministic):             │  │
│  │  ├── fundamentals_tool  → Piotroski, ratios, DCF   │  │
│  │  ├── technical_tool     → RSI, MACD, patterns      │  │
│  │  ├── sentiment_tool     → FinBERT + VADER scoring  │  │
│  │  ├── macro_tool         → Global indices, FII/DII  │  │
│  │  ├── valuation_tool     → DCF scenarios            │  │
│  │  ├── distortion_tool    → Fake breakouts, pumps    │  │
│  │  ├── regulatory_tool    → SEBI/RBI announcements   │  │
│  │  ├── risk_tool          → 14 risk checks           │  │
│  │  ├── portfolio_tool     → Kite SDK holdings        │  │
│  │  └── research_tool      → LLM Bull/Bear debate     │  │
│  │  LangSmith traces each tool individually           │  │
│  └────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────┤
│              Memory Layer                                 │
│  SQLite (trades, signals, snapshots)                      │
│  ChromaDB (vector search for past research)               │
│  LangGraph Store (cross-session learning)                 │
├──────────────────────────────────────────────────────────┤
│           LLM Stack (openai SDK)                          │
│  Modal vLLM (Qwen 3.6 35B) │ NVIDIA NIM (MiniMax M3)    │
├──────────────────────────────────────────────────────────┤
│           Observability                                   │
│  LangSmith (tracing, Studio, evaluation)                  │
├──────────────────────────────────────────────────────────┤
│              Data Sources (₹0/month)                      │
│  Kite SDK │ yfinance │ Marketaux │ FRED │ Screener       │
└──────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.11+
- A Zerodha account (for live portfolio data via Kite SDK)
- Modal account (for self-hosted vLLM — free tier, cold starts after inactivity)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/autfin.git
cd autfin

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -e .

# Install dashboard (optional)
pip install -e ".[dashboard]"
```

## Configuration

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Get free API keys (no credit card required for any):
   - **Modal**: [modal.com](https://modal.com) — Self-hosted vLLM (Qwen 3.6 35B), free tier
   - **NVIDIA NIM**: [build.nvidia.com](https://build.nvidia.com) — MiniMax M3 fallback (free, 1M context)
   - **LangSmith**: [smith.langchain.com](https://smith.langchain.com) — Tracing, evaluation (free Developer tier)
   - **Marketaux**: [marketaux.com](https://marketaux.com) — News sentiment
   - **FRED**: [fred.stlouisfed.org](https://fred.stlouisfed.org) — Macro economic data

3. Fill in your `.env` file with the API keys.

4. For Zerodha integration, register your app at [developers.kite.trade](https://developers.kite.trade) and fill in the `KITE_*` variables in `.env`

## Usage

### CLI Commands

```bash
# Check Kite connection and account balance
autfin status

# Run full research on a stock (all 9 agents)
autfin analyze RELIANCE

# Run daily morning scan (portfolio + watchlist)
autfin scan

# Start as background daemon with scheduler
autfin start

# View trade history
autfin history --symbol RELIANCE --days 90
autfin history --days 30

# Backtest a strategy
autfin backtest --strategy momentum --period 1y --symbol RELIANCE
autfin backtest --strategy mean_reversion --period 6m
autfin backtest --strategy trend_following --period 2y

# Update configuration
autfin config set alerts_enabled true
autfin config set auto_execute_max_amount 10000
autfin config show
```

### Dashboard

```bash
# Launch Streamlit dashboard
streamlit run -m autfin.dashboard.app
```

Dashboard pages:
- **Overview** — Portfolio KPIs, holdings, equity curve
- **Trades** — Filterable trade history
- **Signals** — Agent signal log with strength indicators
- **Performance** — Win rate, profit factor, drawdown analysis
- **Learnings** — Lessons learned from past trades

## Project Structure

```
autfin/
├── src/autfin/
│   ├── agents/           # 10 specialized tools (9 deterministic + 1 LLM)
│   │   ├── market_data.py
│   │   ├── technical.py
│   │   ├── fundamentals.py
│   │   ├── sentiment.py
│   │   ├── global_macro.py
│   │   ├── researcher.py
│   │   ├── risk_manager.py
│   │   ├── portfolio_manager.py
│   │   ├── executor.py
│   │   ├── valuation.py          # DCF valuation (deterministic)
│   │   ├── distortion.py         # Price-distortion detection (deterministic)
│   │   ├── regulatory.py         # SEBI/RBI regulatory scanner (deterministic)
│   │   └── agent_tools.py        # Tool wrappers (v5.4 deterministic-first)
│   ├── alerts/           # Notification system
│   │   └── notifier.py
│   ├── backtest/         # Backtesting engine
│   │   ├── engine.py
│   │   ├── strategies.py
│   │   └── metrics.py
│   ├── cli/              # Command-line interface
│   │   └── main.py
│   ├── config/           # Settings management
│   │   └── settings.py
│   ├── dashboard/        # Streamlit dashboard
│   │   ├── app.py
│   │   └── data.py
│   ├── graph/            # LangGraph pipelines
│   │   ├── supervisor.py       # Supervisor with tools (v5.4)
│   │   ├── builder.py          # Graph builder with store support
│   │   └── error_handler.py
│   ├── memory/           # Long-term memory
│   │   ├── setup.py            # Async checkpointer/store init
│   │   ├── store.py            # LangGraph Store (cross-session learning)
│   │   ├── vector_store.py     # ChromaDB collections
│   │   ├── database.py
│   │   ├── connection_pool.py
│   │   └── retrieval.py
│   ├── llm/              # LLM client abstraction
│   │   └── client.py           # openai SDK (Modal vLLM + NVIDIA NIM)
│   ├── observability/    # LangSmith tracing + metrics
│   │   ├── handler.py
│   │   └── tracer.py
│   ├── prompts/          # Agent system prompts
│   ├── scheduler/        # APScheduler jobs
│   │   └── jobs.py
│   ├── state/            # State schemas
│   │   └── agent_state.py
│   ├── tools/            # External integrations
│   │   ├── supervisor_tools.py
│   │   ├── fii_dii.py          # FII/DII data fetcher
│   │   ├── news_velocity.py    # News velocity analyzer
│   │   └── ...
│   └── universe/         # Stock universe management
│       └── curator.py          # Watchlist Curator
├── agent.py              # LangGraph dev entry point
├── langgraph.json        # LangGraph dev config
├── tests/                # 113+ tests, all passing
├── .env.example          # Environment template
├── autfin_prd.md         # Product requirements
├── technical_spec.md     # Technical specification
└── pyproject.toml        # Project configuration
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_agents/ -v
python -m pytest tests/test_backtest/ -v
python -m pytest tests/test_cli/ -v

# Run with coverage
python -m pytest tests/ --cov=autfin --cov-report=term-missing
```

## How It Works

### Daily Pipeline

Every day at 9:00 AM IST, AutFin runs its morning pipeline:

1. **Position Monitor** — Checks all existing positions for stop-losses and profit targets
2. **Guardian Executor** — Executes stop-losses and profit bookings immediately
3. **Parallel Research** — Supervisor calls tools (fundamentals, technical, sentiment, macro, distortion, regulatory) — each runs deterministically in parallel via asyncio.gather
4. **Researcher** — Synthesizes all research into a Bull/Bear debate (LLM-driven)
5. **Risk Manager** — Validates trade proposals against 14 risk checks
6. **Portfolio Manager** — Makes final allocation decisions
7. **Executor** — Recommends or executes trades

### Balance-is-Budget Model

Your Zerodha account balance IS the budget. No separate config:
- ₹50,000 cash → AutFin manages with ₹50,000 deployable
- ₹0 cash → System still manages existing positions (stop-losses, profit booking)
- New money added to Zerodha → Automatically becomes available

### Free LLM Stack

All AI runs on free models via openai SDK (direct connection to OpenAI-compatible endpoints):
- **Primary**: Qwen 3.6 35B via Modal vLLM (self-hosted, no rate limits, no cost)
- **Fallback**: MiniMax M3 via NVIDIA NIM (1M context, free tier)
- **Speed**: Llama 3.1 8B via NVIDIA NIM (fast responses for time-sensitive operations)

## Contributing

This is a personal project. For issues or suggestions, open a GitHub issue.

## License

MIT
