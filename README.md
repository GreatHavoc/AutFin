# AutFin v7.0

**Autonomous AI-Powered Financial Research & Portfolio Management System**

A personal multi-agentic system built with **Pydantic AI** that runs daily autonomous financial research, manages a Zerodha portfolio, and learns from every trade — using 7 SubAgents, DynamicWorkflow orchestration, ManagedPrompts via Logfire, and free data APIs.

## Features

- **7 SubAgents via SubAgents Capability** — market_data, analyze, research, risk, debate, conference, memory — coordinated by a Supervisor with context isolation, parallel execution, adversarial debate, coordination mechanisms, and persistent memory.
- **DynamicWorkflow Orchestration** — `morning_workflow` and `quarterly_workflow` for multi-step routines with resource limits, usage tracking, and graceful error containment.
- **ManagedPrompts via Logfire** — Version-controlled agent instructions with template inputs, rollout/targeting, and instant updates without code changes.
- **Pydantic AI Capabilities** — Planning (task decomposition), Memory (FileStore cross-session), CodeMode (Python tool execution), Guardrails (input/output/tool validation), SystemReminders (persistent behavioral rules).
- **Model Tiers** — Per-agent selection via NVIDIA NIM: deep (deepseek-v4-flash 284B), standard (llama-3.3-70b), fast (llama-3.1-8b).
- **Walk-Forward Backtesting** — Regime-adaptive windows, online learning, overfit detection.
- **Alpha Decay Tracking** — Monitors strategy degradation across folds; alerts when Sharpe drops below floor.
- **NVIDIA NIM (Primary LLM)** — OpenAI-compatible endpoint, free tier, 1M context.
- **Logfire Observability** — Auto-tracing, metrics, managed variables/prompts, A/B thesis framework.
- **Balance-is-Budget** — Your Zerodha account balance IS the deployable pool, no separate config.
- **Long-Term Memory** — SQLite + ChromaDB + Logfire Store for trade history, lessons learned, cross-session learning.
- **Deterministic-First Analysis** — 9 deterministic tools (fundamentals, technical, sentiment, macro, valuation, distortion, regulatory, risk, portfolio) + 1 LLM-driven (research synthesis).

## Quick Start

```bash
# Install dependencies (using uv)
uv sync

# Run the supervisor agent (interactive)
python -c "
import asyncio
from autfin.agents import supervisor_agent
from autfin.types import AgentDeps

async def main():
    deps = AgentDeps(user_id='default')
    result = await supervisor_agent.run('Run morning routine', deps=deps)
    print(result.output)

asyncio.run(main())
"
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│              Supervisor Agent (Pydantic AI)               │
│  ┌────────────────────────────────────────────────────┐  │
│  │  SubAgents Capability (7 agents):                   │  │
│  │  ├── market_data  → Portfolio, quotes, status      │  │
│  │  ├── analyze      → 4 analysis types (TF/SV/VA)    │  │
│  │  ├── research     → Screen, research, pitch        │  │
│  │  ├── risk         → Position sizing, heat, veto    │  │
│  │  ├── debate       → Bull vs Bear adversarial       │  │
│  │  ├── conference   → Multi-agent coordination       │  │
│  │  └── memory       → Cross-session learning         │  │
│  │  Batched parallel execution, context isolation     │  │
│  └────────────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────┐  │
│  │  DynamicWorkflow: morning_workflow, quarterly_wf   │  │
│  │  Resource limits, usage tracking, error containment│  │
│  └────────────────────────────────────────────────────┘  │
│  Capabilities: Planning, Memory(FileStore), CodeMode,    │
│  Guardrails, SystemReminders, Thinking, WebSearch        │
├──────────────────────────────────────────────────────────┤
│              Memory Layer                                 │
│  SQLite (trades, signals, snapshots)                     │
│  ChromaDB (vector search for past research)              │
│  Logfire Store (cross-session learning, managed vars)    │
├──────────────────────────────────────────────────────────┤
│           LLM Stack (openai SDK)                          │
│  NVIDIA NIM (primary): deepseek-v4-flash / llama-3.3-70b│
│  Fallback: llama-3.1-8b (fast), openrouter (optional)    │
├──────────────────────────────────────────────────────────┤
│           Observability                                   │
│  Logfire (tracing, metrics, managed prompts, evals)      │
├──────────────────────────────────────────────────────────┤
│              Data Sources (₹0/month)                      │
│  Kite SDK │ yfinance │ Marketaux │ FRED │ NSE Public API │
└──────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.11+
- A Zerodha account (for live portfolio data via Kite SDK)
- **Obscura** (free, on-device Rust browser for MCP) — required for NSE/Screener/MC/ET scraping.
  Download the release `.zip`/`.tar.gz` for your OS from
  <https://github.com/h4ckf0r0day/obscura/releases/latest>, extract, and put the
  folder on `PATH`. AutFin starts it as a long-lived HTTP MCP server.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/autfin.git
cd autfin

# Install dependencies with uv (creates virtual environment automatically)
uv sync

# Or manually:
# uv venv
# source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate     # Windows
# uv pip install -e .
```

## Configuration

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```
2. Verify Obscura is reachable:  `obscura --version` should print a version.
3. (Optional) Start Obscura as a long-lived HTTP MCP server:
   ```bash
   obscura mcp --http --port 8080
   ```
   AutFin will autostart it on first browser-mcp call if you skip this and
   `BROWSER_MCP_AUTOSTART=true` (the default).
4. Get free API keys (no credit card required for any):
   - **NVIDIA NIM**: [build.nvidia.com](https://build.nvidia.com) — deepseek-v4-flash, llama-3.3-70b (free tier)
   - **Marketaux**: [marketaux.com](https://marketaux.com) — News sentiment
   - **FRED**: [fred.stlouisfed.org](https://fred.stlouisfed.org) — Macro economic data
5. Fill in your `.env` file with the API keys.
6. For Zerodha integration, register your app at [developers.kite.trade](https://developers.kite.trade) and fill in the `KITE_*` variables in `.env`

## Usage

AutFin runs as a Pydantic AI agent graph. The supervisor agent coordinates
7 sub-agents via the SubAgents capability and DynamicWorkflow orchestration.

```python
from autfin.agents import supervisor_agent, morning_workflow
from autfin.types import AgentDeps

deps = AgentDeps(user_id="my_user")

# Run morning routine
result = await supervisor_agent.run("Run morning routine", deps=deps)

# Or invoke workflow directly
result = await morning_workflow.run("Full portfolio review", deps=deps)
```

## Project Structure

```
autfin/
├── src/autfin/
│   ├── agents/              # 7 SubAgents + Supervisor + capabilities
│   │   ├── base.py              # Agent factory, NVIDIA provider
│   │   ├── market_data.py       # Portfolio, quotes, market status
│   │   ├── analysis.py          # Technical + fundamental + sentiment + valuation
│   │   ├── research.py          # Deep research, screening, thesis
│   │   ├── risk.py              # Position sizing, stop-loss, correlation
│   │   ├── debate.py            # Bull vs Bear adversarial
│   │   ├── conference.py        # Multi-agent coordination
│   │   ├── memory.py            # Cross-session learning
│   │   ├── quant.py             # Quantitative analysis
│   │   ├── supervisor.py        # Supervisor + SubAgents + Workflows
│   │   ├── managed_prompts.py   # Logfire ManagedPrompt instances
│   │   ├── guardrails.py        # Input/output/tool guardrails
│   │   ├── observability.py     # Logfire instrumentation
│   │   └── ...                  # Other agents (technical, fundamentals, etc.)
│   ├── validation.py        # 6 validation gates (type check, tests, evals, etc.)
│   ├── evals/               # Evaluation datasets + runners
│   ├── memory/              # SQLite, ChromaDB, retrieval, consolidation
│   ├── tools/               # External integrations (Kite, scrapers, etc.)
│   ├── config/              # Settings, capabilities registry
│   ├── llm/                 # LLM client abstraction
│   ├── state/               # Pydantic state models
│   ├── learning/            # Online learning, calibration, critique
│   ├── backtesting/         # Walk-forward + alpha decay
│   ├── risk/                # Kelly, anomaly, regime, vol targeting
│   └── types.py             # Shared type definitions
├── tests/                   # 52 test files, 5,400+ tests
├── data/                    # Local data (gitignored)
├── .env.example             # Environment template
├── technical_spec.md        # Technical specification (v7.0)
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test module
uv run pytest tests/test_agents/ -v
uv run pytest tests/test_tools/ -v
uv run pytest tests/test_learning/ -v

# Run with coverage
uv run pytest tests/ --cov=autfin --cov-report=term-missing

# Run validation gates
uv run python test_validation.py
```

## Validation Gates (Phase 6)

Run `uv run python test_validation.py` to execute all 6 gates:
1. **Type Checking** — `uv run pyright --strict`
2. **Unit Tests** — Full pytest suite
3. **Evaluation Suite** — >80% pass rate on agent evals
4. **Guardrails** — Input/output/tool validation tests
5. **Memory Persistence** — FileStore cross-session validation
6. **Plan Persistence** — Planning capability state validation
7. **Load Test** — Concurrent agent execution stress test
8. **Durable Execution** — Temporal/DBOS/Preft backend validation

## How It Works

### Morning Routine

Every day at 9:00 AM IST, AutFin runs its morning pipeline:

1. **Market Data** — Portfolio, quotes, market status, risk alerts
2. **Analysis** — Technical + fundamental + sentiment + valuation per holding
3. **Research** — Screening, deep research, thesis generation
4. **Debate** — Bull/Bear adversarial on trade candidates (MANDATORY)
5. **Risk** — Position sizing, correlation, portfolio heat, veto power
6. **Execution** — Kite SDK order placement for approved trades
7. **Memory** — Log decisions, calibrate conviction, update patterns

### DynamicWorkflow

Complex multi-step orchestration via `morning_workflow` and `quarterly_workflow`:

```python
from autfin.agents import morning_workflow

result = await morning_workflow.run(
    "Full morning routine with deep research",
    deps=AgentDeps(user_id="default")
)
```

Workflows handle:
- Resource limits (duration, token usage, agent calls)
- Usage tracking and forwarding
- Error containment (failures don't crash supervisor)
- Deferred loading for fast startup

### ManagedPrompts

Agent instructions are version-controlled via Logfire:

```python
from autfin.agents.managed_prompts import supervisor_instructions

# Update prompt in Logfire UI — instantly served to agents
# No code changes, no restarts
```

### Model Tiers

Each agent uses the optimal NVIDIA NIM model:
- **Deep** (deepseek-v4-flash 284B): Supervisor, Debate, Research Synthesis
- **Standard** (llama-3.3-70b): Analysis, Risk, Technical, Fundamentals
- **Fast** (llama-3.1-8b): Market Data, Quant, quick checks

### Balance-is-Budget Model

Your Zerodha account balance IS the budget. No separate config:
- ₹50,000 cash → AutFin manages with ₹50,000 deployable
- ₹0 cash → System still manages existing positions (stop-losses, profit booking)
- New money added to Zerodha → Automatically becomes available

### Free LLM Stack

All AI runs on free models via openai SDK (OpenAI-compatible endpoints):
- **Primary**: NVIDIA NIM (deepseek-v4-flash 284B, llama-3.3-70b, llama-3.1-8b)
- **Optional fallback**: OpenRouter (rate-limited)

## Contributing

This is a personal project. For issues or suggestions, open a GitHub issue.

## License

MIT