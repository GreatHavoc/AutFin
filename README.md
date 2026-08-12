# AutFin v7.0

**Autonomous AI-Powered Financial Research & Portfolio Management System**

A personal multi-agentic system that runs daily autonomous financial research,
manages a Zerodha portfolio, and learns from every trade — using **Pydantic AI**
SubAgents + DynamicWorkflow, NVIDIA NIM ThinkingMachines Inkling (primary), and
free data APIs.

## Features

- **SubAgents Architecture** — 7 specialized sub-agents (MarketData, Analysis,
  Research, Risk, Debate, Conference, MemorySystem) coordinated via
  SubAgents/DynamicWorkflow. Context isolation, parallel execution, adversarial
  debate, coordination mechanisms, persistent memory.
- **Command-Based Routing** — All dynamic routing via `Command(update=..., goto=...)` —
  atomic state updates + routing in one return. No conditional edges, no separate
  routing functions.
- **Runtime[Context]** — Store access via `Runtime[Context]` pattern (Pydantic AI
  best practice). No legacy `config["store"]` paths.
- **HITL with Interrupts** — `interrupt()` + `Command(resume=...)` for proper
  human-in-the-loop trade approval. End-to-end tested.
- **Deterministic-First Analysis** — 9 deterministic tools (fundamentals,
  technical, sentiment, macro, valuation, distortion, regulatory, risk, portfolio)
  + 1 LLM-driven (research synthesis). 3x faster than LLM-per-tool.
- **Model Tiers** — Per-agent model selection (deep/medium/fast) via `ModelTier`
  registry. Debate uses deep tier, market data uses fast tier.
- **Walk-Forward Backtesting** — WalkForwardConfig with regime-adaptive windows,
  online learning, and overfit detection.
- **Alpha Decay Tracking** — AlphaDecayTracker monitors strategy degradation across
  walk-forward folds. Alerts when Sharpe drops below floor.
- **NVIDIA NIM (Primary LLM)** — ThinkingMachines Inkling via NVIDIA NIM, 1M
  context, free tier
- **Logfire Observability** — Full observability with auto-tracing, per-agent
  traces, evaluation, prompt version control. Replaces LangSmith.
- **Balance-is-Budget** — Your Zerodha account balance IS the deployable pool, no
  separate config needed.
- **Long-Term Memory** — SQLite + ChromaDB + managed prompts for trade history,
  lessons learned, and cross-session learning.
- **Error Recovery** — Native RetryPolicy/TimeoutPolicy and graceful degradation
  for every node.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run with default settings
autfin

# Or use the supervisor agent directly:
python -c "from autfin.agents import supervisor_agent; print(supervisor_agent.name)"
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                Pydantic AI Supervisor Agent (v7.0)                  │
├─────────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  7 Sub-Agents via SubAgents Capability:                       │  │
│  │  ├── market_data → Portfolio, quotes, market status, risk alerts│  │
│  │  ├── analyze     → Technical + fundamental + sentiment + valuation│  │
│  │  ├── research    → Deep research, screening, thesis generation   │  │
│  │  ├── risk        → Position sizing, stop-loss, correlation, portfolio heat│  │
│  │  ├── debate      → Bull vs Bear adversarial debate (MANDATORY before trades)│  │
│  │  ├── conference  → Multi-agent coordination conferences         │  │
│  │  └── memory      → Cross-session learning & calibration          │  │
│  │  Each sub-agent runs in parallel via asyncio.gather              │  │
│  │  Model tiers: deep (reasoning), medium (balanced), fast (low latency)│  │
│  │  DynamicWorkflow: morning_workflow, quarterly_workflow           │  │
│  └────────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│              Memory Layer                                             │
│  SQLite (trades, signals, snapshots)                                 │
│  ChromaDB (vector search for past research)                          │
│  Managed prompts (version-controlled agent instructions)              │
├─────────────────────────────────────────────────────────────────────┤
│              LLM Stack (openai SDK)                                   │
│  NVIDIA NIM (ThinkingMachines Inkling, primary)                      │
│  Modal vLLM (fallback — Llama 3.1 8B, self-hosted)                    │
├─────────────────────────────────────────────────────────────────────┤
│              Observability                                            │
│  Logfire (auto-tracing, per-agent traces, evaluation, prompt versioning)│
├─────────────────────────────────────────────────────────────────────┤
│              Data Sources (₹0/month)                                  │
│  Kite SDK │ yfinance │ Marketaux │ FRED │ Screener                        │
└─────────────────────────────────────────────────────────────────────┘

## Prerequisites

- Python 3.11+
- A Zerodha account (for live portfolio data via Kite SDK)
- **Obscura** (free, on-device Rust browser for MCP) — required for NSE/Screener/MC/ET scraping.
  Download the release `.zip`/`.tar.gz` for your OS from
  <https://github.com/h4ckf0r0day/obscura/releases/latest>, extract, and put the
  folder on `PATH`. AutFin starts it as a long-lived HTTP MCP server.
- API keys (free, no credit card required):
  - **NVIDIA NIM**: [build.nvidia.com](https://build.nvidia.com) — primary LLM
  - **Marketaux**: [marketaux.com](https://marketaux.com) — news sentiment
  - **FRED**: [fred.stlouisfed.org](https://fred.stlouisfed.org) — macro economic data
  - **Modal**: [modal.com](https://modal.com) — self-hosted vLLM fallback

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/autfin.git
cd autfin

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies (includes mcp + langchain-mcp-adapters)
pip install -e .
```

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```
2. Verify Obscura is reachable: `obscura --version` should print a version.
3. (Optional) Start Obscura as a long-lived HTTP MCP server:
   ```bash
   obscura mcp --http --port 8080
   ```
   AutFin will autostart it on first browser-mcp call if you skip this and
   `BROWSER_MCP_AUTOSTART=true` (the default).
4. Fill in your `.env` file with the API keys (see .env.example).
5. For Zerodha integration, register your app at [developers.kite.trade](https://developers.kite.trade) and fill in the `KITE_*` variables in `.env`.

## Usage

AutFin runs as a Pydantic AI supervisor agent. Start it via:

```bash
autfin
```

Interact through the CLI, or import `supervisor_agent` from `autfin.agents`:

```python
from autfin.agents import supervisor_agent, run_morning_routine

# Run the daily morning routine
result = run_morning_routine()
```

## Project Structure

```
autfin/
├── src/autfin/
│   ├── __init__.py
│   ├── agents/           # Sub-agents + tool definitions
│   │   ├── __init__.py
│   │   ├── subagents.py        # 7 sub-agents + ANALYSIS_DISPATCH
│   │   ├── subagent_states.py  # Per-agent TypedDict schemas
│   │   ├── debate.py           # Bull vs Bear adversarial research
│   │   ├── conferences.py      # Morning/TradeReview/Emergency/Experience
│   │   ├── memory_system.py    # 3-type persistent memory
│   │   ├── agent_tools.py      # 10 deterministic tool wrappers
│   │   ├── yfinance_utils.py   # fetch_history(as_of=), normalize_symbol()
│   │   ├── market_data.py
│   │   ├── fundamentals.py
│   │   ├── technical.py
│   │   ├── sentiment.py
│   │   ├── global_macro.py
│   │   ├── researcher.py
│   │   ├── risk_manager.py
│   │   ├── portfolio_manager.py
│   │   ├── executor.py
│   │   ├── valuation.py
│   │   ├── distortion.py
│   │   ├── regulatory.py
│   │   ├── peer_compare.py
│   │   ├── moat.py
│   │   └── auto_execute.py
│   ├── llm/              # LLM client abstraction
│   │   ├── client.py           # openai SDK (NVIDIA NIM + Modal vLLM)
│   │   └── model_tiers.py      # ModelTier, MODEL_REGISTRY, AGENT_TIERS
│   ├── config/           # Settings management
│   │   └── settings.py
│   ├── observability/    # Logfire tracing + metrics
│   │   ├── handler.py
│   │   └── tracer.py
│   ├── prompts/          # Agent system prompts (all 8 created via ManagedPrompt)
│   ├── tools/            # External integrations
│   │   ├── supervisor_tools.py
│   │   └── ...
│   ├── state/            # State type definitions
│   │   └── agent_state.py      # AgentState TypedDict
│   ├── learning/         # Self-learning modules
│   │   ├── __init__.py
│   │   ├── judge.py
│   │   ├── critique.py
│   │   ├── calibrator.py
│   │   ├── online.py
│   │   └── feedback_loop.py
│   └── mcp/              # MCP (Model Context Protocol) integration
│       └── browser_client.py              # Obscura HTTP MCP client
├── pyproject.toml                         # Dependencies, project metadata
├── .env                                   # API keys, secrets (gitignored)
├── .env.example                           # Environment template
├── technical_spec.md                      # v7.0 Technical Specification
├── tests/                                 # 52 test files, all passing
│   ├── test_agents/       # Sub-agent tests
│   ├── test_tools/        # Tool integration tests
│   ├── test_backtest/     # Backtesting tests
│   ├── test_memory/       # Memory persistence tests
│   ├── test_learning/     # Learning & calibration tests
│   ├── test_risk/         # Risk management tests
│   ├── test_tools/        # Tool integration tests
│   ├── test_llm/          # LLM client tests
│   └── test_validator.py  # Validation gates
├── .gitignore
└── uv.lock
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific categories
python -m pytest tests/test_agents/ -v        # Sub-agent tests
python -m pytest tests/test_tools/ -v         # Tool integration
python -m pytest tests/test_backtest/ -v      # Backtesting
python -m pytest tests/test_memory/ -v        # Memory persistence
python -m pytest tests/test_learning/ -v      # Learning & calibration
python -m pytest tests/test_risk/ -v          # Risk management

# Run with coverage
python -m pytest tests/ --cov=autfin --cov-report=term-missing
```

## How It Works

### Daily Pipeline

Every day at 9:00 AM IST, AutFin runs its morning pipeline:

1. **Position Monitor** — Checks all existing positions for stop-losses and profit targets
2. **Guardian Executor** — Executes stop-losses and profit bookings immediately
3. **Parallel Research** — Supervisor calls sub-agents in parallel:
   - `get_market_data("all")` — fetches portfolio, quotes, market status
   - `analyze_stock(symbol, "full")` — runs all 7 analysis types per holding
   - `research_stock("research_batch", symbols)` — screens and researches candidates
4. **Risk Validation** — `execute_trade("risk_check")` validates all trade proposals
5. **Execution** — `execute_trade("execute")` places approved orders
6. **Memory** — `manage_memory("log")` records decisions, `manage_memory("calibrate")` updates conviction scores

### Command-Based Routing

All routing decisions are made inside node returns via `Command(update=..., goto=...)`:

```python
# Supervisor routes to END when session is finalized
return Command(update={...}, goto=END)
```

No conditional edges. No separate routing functions. Atomic state update + routing in one return.

### Model Tiers

Each agent uses the optimal model tier:

- **Deep** (ThinkingMachines Inkling): Debate, Research Synthesis — complex reasoning
- **Medium** (Llama 3.3 70B): Technical, Fundamentals — balanced
- **Fast** (Llama 3.1 8B): Market Data, Risk Check — low latency

### Balance-is-Budget Model

Your Zerodha account balance IS the budget. No separate config:

- ₹50,000 cash → AutFin manages with ₹50,000 deployable
- ₹0 cash → System still manages existing positions (stop-losses, profit booking)
- New money added to Zerodha → Automatically becomes available

### Free LLM Stack

All AI runs on free models via openai SDK (direct connection to OpenAI-compatible endpoints):

- **Primary**: ThinkingMachines Inkling via NVIDIA NIM (1M context, free tier)
- **Fallback**: Llama 3.1 8B via Modal vLLM (self-hosted, no rate limits, no cost)
- **Speed**: Llama 3.1 8B via NVIDIA NIM (fast responses for time-sensitive operations)

## Contributing

This is a personal project. For issues or suggestions, open a GitHub issue.

## License

MIT