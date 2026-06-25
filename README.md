# AutFin

**Autonomous AI-Powered Financial Research & Portfolio Management System**

A personal multi-agentic system that runs daily autonomous financial research, manages a Zerodha portfolio, and learns from every trade — using only free LLM models and free data APIs.

## Features

- **9 Specialized AI Agents** — Market Data, Technical Analysis, Fundamentals, Sentiment, Global Macro, Researcher (Bull/Bear debate), Risk Manager, Portfolio Manager, Execution
- **Daily Autonomous Operation** — Runs at 9:00 AM IST via APScheduler, manages existing positions even with ₹0 cash
- **Balance-is-Budget** — Your Zerodha account balance IS the deployable pool, no separate config needed
- **Long-Term Memory** — SQLite + ChromaDB stores trade history, lessons learned, and research reports
- **Error Recovery** — Circuit breakers, retry policies, and graceful degradation for every agent
- **Backtesting** — Test strategies on historical data with realistic Indian market transaction costs
- **Streamlit Dashboard** — Visualize portfolio, P&L, signals, and agent performance
- **Alert System** — Console, desktop, Pushover, and email notifications for key events
- **Full CLI** — `autfin status`, `autfin analyze RELIANCE`, `autfin scan`, `autfin start`

## Architecture

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
┌─────────────────────────────────────────────────────┐
│                    CLI / Dashboard                   │
├─────────────────────────────────────────────────────┤
│              Scheduler (APScheduler)                 │
│         Daily 9AM │ Midday │ Weekly │ Monthly        │
├─────────────────────────────────────────────────────┤
│              Daily Morning Pipeline                  │
│  ┌─────────┐  ┌──────────────────────────────────┐  │
│  │ Track B │  │           Track A                 │  │
│  │Guardian │  │  Research → Debate → Risk → PM    │  │
│  └─────────┘  └──────────────────────────────────┘  │
├─────────────────────────────────────────────────────┤
│                   9 AI Agents                        │
│  Market│Tech│Fund│Sent│Macro│Research│Risk│PM│Exec   │
├─────────────────────────────────────────────────────┤
│              Memory Layer                            │
│  SQLite (trades, signals, snapshots)                 │
│  ChromaDB (vector search for past research)          │
│  Performance Tracker (win rate, Sharpe, drawdown)    │
├─────────────────────────────────────────────────────┤
│           Free LLM Stack (via LiteLLM)              │
│  OpenRouter (Llama 3.3 70B:free) │ Groq │ Gemini    │
├─────────────────────────────────────────────────────┤
│              Data Sources (₹0/month)                 │
│  Kite SDK │ yfinance │ Marketaux │ FRED │ Screener  │
└─────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.11+
- [TA-Lib](https://ta-lib.org/) C library installed on your system
- A Zerodha account (for live portfolio data via Kite SDK)

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
   - **OpenRouter**: [openrouter.ai](https://openrouter.ai) — Free LLM models (Llama 3.3 70B, etc.)
   - **Groq**: [console.groq.com](https://console.groq.com) — Fast inference
   - **Gemini**: [aistudio.google.com](https://aistudio.google.com) — Google's free tier
   - **Marketaux**: [marketaux.com](https://marketaux.com) — News sentiment
   - **FRED**: [fred.stlouisfed.org](https://fred.stlouisfed.org) — Macro economic data
   - **Firecrawl**: [firecrawl.dev](https://firecrawl.dev) — Web scraping (500 free credits/mo)

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
│   ├── agents/           # 9 specialized AI agents
│   │   ├── market_data.py
│   │   ├── technical.py
│   │   ├── fundamentals.py
│   │   ├── sentiment.py
│   │   ├── global_macro.py
│   │   ├── researcher.py
│   │   ├── risk_manager.py
│   │   ├── portfolio_manager.py
│   │   └── executor.py
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
│   │   ├── error_handler.py
│   │   └── state.py
│   ├── memory/           # Long-term memory
│   │   ├── database.py
│   │   ├── retrieval.py
│   │   └── performance.py
│   ├── prompts/          # Agent system prompts
│   ├── scheduler/        # APScheduler jobs
│   │   └── jobs.py
│   ├── state/            # State schemas
│   │   └── schemas.py
│   └── tools/            # External integrations
│       ├── kite_tools.py
│       ├── fundamental_tools.py
│       ├── sentiment_tools.py
│       ├── scraper_tools.py
│       └── memory_tools.py
├── tests/                # 613 tests, 100% passing
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
3. **Parallel Research** — Runs 5 research agents simultaneously (Market Data, Technical, Fundamentals, Sentiment, Macro)
4. **Researcher** — Synthesizes all research into a Bull/Bear debate
5. **Risk Manager** — Validates trade proposals against 9 risk checks
6. **Portfolio Manager** — Makes final allocation decisions
7. **Executor** — Recommends or executes trades

### Balance-is-Budget Model

Your Zerodha account balance IS the budget. No separate config:
- ₹50,000 cash → AutFin manages with ₹50,000 deployable
- ₹0 cash → System still manages existing positions (stop-losses, profit booking)
- New money added to Zerodha → Automatically becomes available

### Free LLM Stack

All AI runs on free models via LiteLLM router:
- **Primary**: Llama 3.3 70B via OpenRouter (free, high quality)
- **Speed**: Groq inference (fast responses for time-sensitive operations)
- **Fallback**: Gemini 2.5 Flash (Google's free tier, high volume)

## Contributing

This is a personal project. For issues or suggestions, open a GitHub issue.

## License

MIT
