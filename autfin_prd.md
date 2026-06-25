# AutFin — Personal AI-Powered Financial Research & Portfolio Management System

## Product Requirements Document (PRD)

**Version**: 3.0  
**Date**: June 2026  
**Author**: Prakhar  
**Status**: Redesign Phase — Single agentic supervisor, 13-tool architecture, self-learning loop. 683 tests passing.

---

## 1. EXECUTIVE SUMMARY

### 1.1 What Is AutFin?

AutFin is an **agentic-first AI portfolio management system**. A single LangGraph supervisor runs a ReAct loop — thinking, calling research tools, verifying risks, and executing trades — with per-stock depth decisions (Quick/Standard/Deep), 13 domain tools (market data through order execution), self-learning from every trade outcome, and zero operating cost via free-tier LLMs and data APIs.

### 1.2 How It Works (User's Perspective)

```
┌─────────────────────────────────────────────────────────┐
│  1. USER transfers money to Zerodha account               │
│     (e.g. ₹50,000 at the start of the month)              │
│  2. AutFin reads the actual balance via Kite SDK           │
│     → That ₹50,000 IS the month's deployable pool          │
│  3. AutFin runs EVERY DAY at 9:00 AM automatically         │
│     → Single supervisor wakes up, plans morning routine    │
│  4. Supervisor thinks: "What's the portfolio state?        │
│     Any thesis breaks? Worth scanning watchlist?"          │
│  5. Calls tools as needed — market data, fundamentals,     │
│     technical, sentiment, macro, risk, execution           │
│  6. Learns from every outcome: "Last 3 similar setups      │
│     won 2/3. Proceed with caution if macro is bearish."    │
│  7. When cash runs out → stops trading until more is added │
│  8. USER just adds money when ready — AutFin handles it    │
│  9. No separate "budget config" — the balance speaks       │
└─────────────────────────────────────────────────────────┘
```

### 1.3 Core Value Proposition

| Problem | AutFin Solution |
|---------|----------------|
| Information overload — thousands of stocks, indicators, news | Multi-agent team each specialized in one domain, synthesizing insights |
| Emotional trading decisions | Systematic agent framework with risk guardrails and debate mechanism |
| Zerodha integration is manual | Kite SDK provides full read/write account access via OAuth |
| No persistent financial context | 3-type memory (episodic, semantic, procedural) stores portfolio history, decisions, learnings |
| Analysis siloed (fundamental OR technical) | All analysis types converge into unified decision framework |
| "How much should I invest this month?" | Balance-is-budget — actual Zerodha balance IS the deployable pool |
| "Did I miss an opportunity?" | Runs every day automatically — no manual monitoring needed |
| "Is my thesis still valid?" | PM-style daily thesis checks — validates every holding against original research |
| Analysis is one-dimensional | DCF valuation with Base/Upside/Downside scenarios, catalyst tracking, counter-consensus theses |

### 1.4 Key Differentiators

- **Agentic, not scripted**: LLM-powered Supervisor dynamically routes agents — not a fixed pipeline
- **Hybrid architecture**: 70% agentic (routing, depth, thesis), 30% deterministic (execution, risk limits, tax)
- **Balance-is-budget**: Actual Zerodha balance IS the deployable pool — no separate config needed
- **Learning system**: Procedural memory builds a corpus of trade patterns — gets smarter with every trade
- **PM-level research quality**: DCF valuation, catalyst tracking, variant views — not just ratio screens
- **Autonomous daily operation**: Runs every morning at 9 AM, scans markets, deploys capital, manages positions
- **Multi-agent architecture** inspired by real trading firms (TradingAgents paper pattern) and professional PM workflows
- **Long-running persistence** — survives reboots, learns from past decisions across months
- **Risk-first design**: Risk Manager with 14 checks including CVaR, news risk, sector exposure — veto power
- **Full Zerodha integration** via official Kite SDK (OAuth, TOTP auto-login, no password storage)
- **Built on LangGraph** for durable, cyclical, observable agent workflows with HITL gates
- **$0/month operating cost**: Entire stack uses free tiers (OpenRouter, Groq, Gemini, Cerebras, Ollama) and free data APIs (NSE, yfinance, Finnhub, Alpha Vantage, FRED)

---

## 2. SYSTEM VISION & GOALS

### 2.1 The Balance-Is-Budget Model

AutFin operates on one simple rule: **the money in your Zerodha account IS the budget. Nothing else.**

```
YOUR ROLE                              AutFin's ROLE
───────────                            ──────────────
1. Transfer money to Zerodha           1. Reads balance via Kite SDK get_margins()
   (once a month, or whenever)         2. That balance IS its deployable pool
2. That's it.                          3. Runs every day at 9:00 AM automatically
                                       4. Screens markets for opportunities

                          ┌─ CASH AVAILABLE ──────────────────────┐
                          │ 5a. Finds stocks to buy                │
                          │ 6a. Decides position sizes             │
                          │ 7a. Paces deployments across days      │
                          └───────────────────────────────────────┘

                          ALWAYS RUNNING (even with ₹0 cash):
                          ┌───────────────────────────────────────┐
                          │ 5b. Monitors existing positions        │
                          │ 6b. Executes stop-losses (-8%)         │
                          │ 7b. Books profits at targets           │
                          │ 8b. Sells weak positions (thesis dead) │
                          │ 9b. Reallocates freed-up cash          │
                          │ 10b. Risk checks: drawdown, exposure   │
                          │ 11b. Daily P&L, sector allocation      │
                          └───────────────────────────────────────┘

                                       12. Tracks P&L, learns from outcomes
                                       13. Reports what it did and why
                                       14. When you add more money → new buys resume
```

**No separate budget config.** The Zerodha balance is the single source of truth.
- Add ₹50,000 → AutFin has ₹50,000 to deploy
- Add ₹20,000 mid-month → AutFin sees ₹20,000 extra and can deploy it
- Balance hits near-zero → AutFin stops trading automatically

**The intelligence is in pacing, not in cap.** AutFin decides HOW to spend, not how much to spend:
- Should I deploy ₹10,000 today or wait for a better opportunity?
- Keep ₹5,000 cash buffer for averaging down?
- Sell a holding to free up cash for a stronger signal?
- Spread deployments across the month or go aggressive early?

### 2.2 How AutFin Reads Its Capital

```
Every Morning (9:00 AM):
  ┌─────────────────────────────────────────────────────────┐
  │ 1. Kite SDK: get_margins()                              │
  │    → available_cash: from Zerodha                        │
  │    → utilized_margin: tied up in positions               │
  │                                                          │
  │ 2. TWO OPERATING MODES (runs BOTH every day):             │
  │                                                          │
  │    MODE A — NEW BUYS (cash > ₹1,000):                     │
  │    ├── Scan markets for buy opportunities                │
  │    ├── Pace deployments across remaining days            │
  │    └── Deploy only with available cash                   │
  │                                                          │
  │    MODE B — PORTFOLIO GUARDIAN (ALWAYS runs):             │
  │    ├── Check every open position against stop-loss       │
  │    ├── Check every open position against profit target   │
  │    ├── Re-evaluate thesis for each holding               │
  │    ├── Sell if thesis broken or allocation exceeded      │
  │    ├── Freed-up cash → reallocated to best signal        │
  │    ├── Risk scan: drawdown, sector exposure, VaR         │
  │    └── Alert user if action needed                       │
  └─────────────────────────────────────────────────────────┘
```

**Critical**: "No cash" only pauses **Mode A** (new buying). **Mode B** runs every single day regardless — protecting existing capital, booking profits, cutting losses, reallocating freed cash.

### 2.3 Long-Term Goals (1+ Year Horizon)

| Goal | Description | Measurable Target |
|------|-------------|-------------------|
| Wealth compounding | Consistent risk-adjusted returns | Beat Nifty 50 by 3-5% annually |
| Capital discipline | Never exceed monthly budget | Zero budget overruns |
| Knowledge accumulation | Build personal financial knowledge graph | Track every decision with rationale |
| Strategy refinement | Improve capital deployment efficiency | Higher win rate, better entry timing |
| Tax optimization | Tax-aware trading decisions | Minimize short-term capital gains |
| Multi-asset support | Expand beyond equities | Mutual funds, ETFs, bonds, gold |

### 2.4 Short-Term Goals (90-Day Horizon)

| Goal | Description | Measurable Target |
|------|-------------|-------------------|
| Zerodha connectivity | Full read/write via Kite SDK | Place/modify/cancel orders programmatically |
| Budget tracking | Monthly budget set + enforced | System never exceeds budget |
| Daily market scan | Screen top opportunities each morning | Generate ranked watchlist within budget |
| Smart capital deployment | Deploy capital across the month | Budget fully utilized by month-end, not day 1 |
| Portfolio health checks | Monitor existing positions | Risk exposure, P&L, allocation alerts |
| Decision support | Generate buy/sell/hold recommendations | Explainable rationale for each signal |
| Learning engine | Store outcomes and improve | Win/loss ratio tracking by strategy type |

### 2.5 Core Principles

1. **Explainability over opacity** — Every signal must come with natural language rationale
2. **Risk first, returns second** — The Risk Manager can veto any decision (14 checks)
3. **Capital discipline** — Monthly budget is a hard limit; never exceed it
4. **Smart pacing** — Don't deploy entire budget on day 1; spread across the month based on trading days
5. **Human-in-the-loop for execution** — AI recommends, human approves (configurable auto-execute)
6. **Incremental improvement** — Start lean, add agents over time
7. **Local-first** — Data lives on your machine, no cloud dependency for core functions
8. **Kite SDK** — Use KiteConnect SDK directly (Personal tier, no MCP needed)
9. **Agentic over scripted** — LLM decides what to do and at what depth; agents self-reflect and course-correct
10. **Learning over repeating** — Procedural memory extracts patterns from every trade instead of starting fresh
11. **Professional-grade research** — DCF valuation, variant views, catalyst tracking — not just ratio screens
12. **Free-tier sustainability** — Entire stack runs at $0/month operating cost

---

## 3. AGENT ARCHITECTURE & DIVISION

### 3.1 Design Philosophy — Agentic-First, Single Supervisor

AutFin v3.0 abandons the multi-graph workflow approach. There is **one LangGraph graph** with **one supervisor** that runs a **ReAct loop**: think → call tools → observe → re-assess → decide next action. The supervisor is a PM-in-code — it plans its own workflow rather than following a pre-defined pipeline.

**70% agentic / 30% deterministic** — same split, but now the agentic portion IS the orchestration itself. The supervisor decides what to research, at what depth, for which stocks, in what order. The 30% that stays deterministic: risk checks, position sizing math, order execution, tax tracking.

```
┌──────────────────────────────────────────────────────────────┐
│                    SUPERVISOR (ReAct Loop)                     │
│                                                                │
│  system_prompt = "You are a PM running AutFin v3.0..."        │
│  while not done:                                               │
│    think → call_tools → observe → re-assess → decide           │
│                                                                │
│  Tools available (13):                                         │
│  ┌──────────────────┐  ┌──────────────────┐                   │
│  │ DATA LAYER       │  │ RESEARCH LAYER   │                   │
│  │ get_portfolio_state│  │ analyze_fundamentals│              │
│  │ get_market_quotes  │  │ analyze_technical   │              │
│  │                    │  │ analyze_sentiment   │              │
│  │                    │  │ analyze_macro       │              │
│  └──────────────────┘  └──────────────────┘                   │
│  ┌──────────────────┐  ┌──────────────────┐                   │
│  │ ANALYSIS LAYER   │  │ DECISION LAYER    │                   │
│  │ research_synthesis│  │ run_risk_checks    │                  │
│  │ validate_thesis   │  │ allocate_capital   │                  │
│  │ retrieve_learnings│  │ execute_order      │                  │
│  └──────────────────┘  │ log_decision       │                  │
│                         └──────────────────┘                   │
│  ┌──────────────────┐                                          │
│  │ HITL             │                                          │
│  │ request_human_approval│                                      │
│  └──────────────────┘                                          │
└──────────────────────────────────────────────────────────────┘

Context in state determines behavior (not different graphs):
  context="morning_routine" → Full scan, check all holdings
  context="price_alert"    → Investigate one symbol
  context="quarterly_review"→ Deep dive with DCF on everything
```

### 3.2 Supervisor — PM-in-Code (not a router)

**Role**: The supervisor IS AutFin. It does NOT "route to the next node in a fixed pipeline." It thinks, plans, calls tools, observes results, and decides what to do next — exactly like a fund manager.

**Entry points (all invoke the SAME graph):**

| Context | Trigger | Supervisor behavior |
|---------|---------|-------------------|
| `morning_routine` | APScheduler 9:00 AM | Check balances → verify thesis on all holdings → scan watchlist → research promising candidates → risk check → allocate → execute |
| `price_alert` | Kite price trigger / yfinance poll | "Something moved. Investigate this symbol. Is this noise or thesis-relevant?" → check technicals + sentiment → decide |
| `quarterly_review` | APScheduler quarterly | Deep dive on every holding: full DCF, catalyst review, thesis re-validation, rebalancing recommendation |
| `manual` | CLI command | "User asked me to analyze INFY. Let me research it." |

**Per-stock depth decision:**
- **Quick** (5 min): market data + technical only → for existing holdings with intact thesis
- **Standard** (15 min): all 4 data agents + research synthesis → for new candidates
- **Deep** (1+ hour): Standard + DCF valuation + variant view + peer comparison → for high-conviction new positions, quarterly reviews

**Self-learning loop:** Before every trade decision, the supervisor calls `retrieve_learnings(query)` — a vector search over the learning journal that returns:
- Similar past setups and their outcomes (win rate, avg return)
- Conviction calibration data ("your last 10 'high conviction' calls had 40% win rate — temper by 0.15")
- Pattern alerts ("this setup matches INFY Jun 15 — LOSS due to weakening INR + earnings optimism")

This closes the feedback loop: every trade outcome feeds into the next trade's context. The system gets smarter with every trade.

**Triggers**: User request, scheduled task (daily scan, hourly check), alert event, conviction change.

```
StateGraph with LLM-powered Supervisor routing:
  - "New stock candidate" → routes to Research Layer at appropriate depth
  - "Portfolio holding review" → routes to Daily Thesis Check → Analysis Layer if thesis broken
  - "Quick morning scan" → routes to Market Data + Technical only (Quick Screen tier)
  - "Deep dive request" → routes through ALL agents including DCF Valuation + Catalyst Engine
  - Daily scan → parallel trigger to all research agents, depth per stock
```

**Responsibilities**:
- Task decomposition and routing to appropriate specialist agents
- Workflow lifecycle management (start → research → analyze → risk-check → execute → log)
- User interaction (HITL pausing before critical actions)
- Agent handoff coordination (research outputs → analysis inputs)
- Error recovery (agent failure rerouting)

### 3.3 Research Engine — Fan-Out Discovery & Analysis Pipeline

The research engine operates in three phases: **Screen → Fan-Out → Converge.** All phases run on the same supervisor, just different tools called.

#### Phase 1: Screening — 500+ to 20 Candidates (60 seconds, deterministic)

Three sequential filters reduce the Nifty 500+ universe to 15-25 high-quality candidates. No LLM calls — pure math.

**Filter 1 — Liquidity Gate** (eliminates illiquid and micro-cap stocks):
- Average daily volume ≥ ₹1 Cr (ensures exit liquidity)
- Market Cap ≥ ₹500 Cr
- Source: yfinance `fast_info.marketCap` + 5-day volume average

**Filter 2 — Magic Formula** (Joel Greenblatt, back-tested 20+ years):
- Rank 1: Earnings Yield = EBIT / Enterprise Value (higher = cheaper)
- Rank 2: ROIC = EBIT / (Net Fixed Assets + Working Capital) (higher = better business)
- Combined Rank = Rank_EY + Rank_ROIC → select top 85
- Why it works: Screens for cheap + high-quality businesses simultaneously. Outperforms S&P 500 by 4-6% CAGR.

**Filter 3 — Piotroski F-Score + Indian Red Flags** (quality gate):
- 9-point checklist (score ≥ 6 = fundamentally improving business):
  - Profitability: +ROA, +CFO, ROA improving, accrual quality (4 pts)
  - Health: Deleveraging, liquidity improving, no dilution (3 pts)
  - Efficiency: Expanding margins, improving asset turnover (2 pts)
- **Indian-market auto-reject red flags:**
  - Promoter holding declined >5% in 4 quarters
  - Promoter pledge >20% of holdings
  - Auditor resignation/qualification in 2 years
  - Related-party transactions >10% of revenue
  - Negative OCF for 3+ consecutive quarters
  - SEBI investigation/penalty in 12 months

**Result: 15-25 candidates from 500+ universe in under 60 seconds.**

#### Phase 2: Fan-Out — 6 Parallel Agents per Stock (15-25 minutes)

Every candidate gets 6 independent research agents running concurrently via `asyncio.gather`. No agent waits for another. All 20 candidates run simultaneously — 120 agent invocations completing in the time of the slowest single agent.

```
               STOCK: RELIANCE
  ┌──────────────────────────────────────────┐
  │  asyncio.gather(                          │
  │    fundamentals_agent()   → 10 min       │
  │    technical_agent()      →  3 min       │
  │    sentiment_agent()      →  5 min       │
  │    valuation_agent()      → 20 min  ← slowest
  │    peer_compare_agent()   →  8 min       │
  │    moat_agent()           →  5 min       │
  │  )                                        │
  │  Wall time: 20 min (not 51 min)           │
  └──────────────────────────────────────────┘

  ALL 20 CANDIDATES concurrently = 20 min total
  (120 analyses in 20 min vs 8.5 hours sequential — 24× speedup)
```

**The 6 Research Agents:**

| Agent | Source | Output | Time |
|-------|--------|--------|------|
| `fundamentals_agent` | yfinance 3yr financials | Ratios, health score, growth trends, red flags, F-Score | 10 min |
| `technical_agent` | yfinance 6mo OHLCV | Multi-timeframe indicators, patterns, signal + conviction | 3 min |
| `sentiment_agent` | Marketaux 50+ articles | Score, topics, hard/soft catalysts with dates, insider activity | 5 min |
| `valuation_agent` | FCF, WACC, sector template | 3-scenario DCF + **Reverse DCF** (implied growth vs historical) | 20 min |
| `peer_compare_agent` | yfinance peer group | 8-metric sector ranking: PE, ROE, FCF yield, growth, margin, debt, moat score | 8 min |
| `moat_agent` | Morningstar 5-source framework | Intangibles, switching costs, network effect, cost advantage, efficient scale → score 0-100 | 5 min |

**Reverse DCF** is the key differentiator. Instead of guessing future growth rates, it:
1. Backs out what growth the current stock price IMPLIES
2. Compares implied growth to historical + industry + consensus
3. Quantifies the gap: "At ₹2,500, RELIANCE is priced for 6.2% CAGR. It delivered 9.1% historically. If it returns to 8%, fair value is ₹3,100 (+24%)."

#### Phase 3: Converge — Supervisor Ranks & Selects (5 minutes)

After all 20 candidates complete, the supervisor:
1. **Aggregates** 6 agent outputs per stock
2. **Ranks** by weighted composite: Fundamentals(0.30) + Valuation Upside%(0.25) + Sentiment(0.15) + Technical(0.10) + Moat(0.10) + Peer Positioning(0.10)
3. **Selects top 5** for full 6-element stock pitch generation
4. **Self-audits**: cross-verify the weakest assumptions in each pitch
5. **Decision gate**: Conviction ≥ 0.5 + Risk/Reward ≥ 2:1 + Sector exposure < 30% → trade queue
6. Presents top 5 with full conviction pitches, or auto-executes if confidence > 0.85

**6-Element Stock Pitch** (buy-side standard, replaces aggregate bull/bear):
1. **Recommendation** — Action, price target, conviction, risk/reward ratio, 3 reasons
2. **Company Background** — Revenue, segments, market cap, promoter holding (summarized, not copied)
3. **Investment Thesis** — VARIANT PERCEPTION: what 80% of investors don't see, dollar-per-share impact
4. **Catalysts** — Hard (dated) + Soft (possible), each with expected per-share impact
5. **Valuation** — 3-scenario DCF table + Reverse DCF insight + peer multiple comparison
6. **Risk Factors & Mitigants** — Company-specific risks, what happens if catalysts fail, stop-loss level

#### Guardian Track — Always On (runs parallel to discovery)

Holding monitoring never pauses. While discovery runs for new candidates, all existing holdings are continuously checked:
- Stop-loss hit? → immediate SELL
- Target hit? → SELL with profit booking
- Thesis BROKEN? → trigger immediate full re-analysis
- Catalyst date approaching? → flag for pre-event re-analysis
- Governance red flag? → SELL immediately (no debate)

#### Quick Mode — Adrenaline Response

When a price alert fires (3%+ move), the supervisor shifts context to `price_alert`:
- Skips screening (not looking for new candidates)
- Fan-out 3 agents for the ALERTED stock only: technical + sentiment + thesis validation
- 5-minute wall time
- Supervisor decides: noise (ignore) / weak signal (full 6-agent research) / broken thesis (sell now)

#### Sector Rotation Awareness

Macro context feeds into screening weights dynamically:
- INR weakening → increase IT, Pharma weights
- Crude rising → reduce OMC, Aviation; increase upstream oil
- RBI hawkish → reduce NBFC; increase Banks
- Global recession signal → shift toward defensive sectors (Consumer Staples, Pharma)

#### The Data Flywheel — Research Gets Better

Every trade outcome feeds back into calibration:
- Conviction calibration: "My high-conviction retail picks win 71%. Metals: 33%." → adjust weights
- Catalyst accuracy: "I predicted earnings beat 14 times, correct 9 times (64%)." → adjust impact scoring
- Thesis durability: "IT services theses last 8 months avg. Real estate: 3 months." → factor into time horizon
- Pattern detection: "4 of 5 losses had: promoter pledge >20% + declining promoter holding" → auto-flag

---

<!-- Agent descriptions that follow remain valid as TOOL implementations the supervisor calls -->

### 3.4 Layer 1 — Research Agents (Information Gathering)

These agents run **in parallel** every morning or on demand. They gather raw data, not opinions.

#### 3.3.1 Market Data Agent (`market_data_agent`)

| Aspect | Detail |
|--------|--------|
| **Role** | Raw data ingestor — prices, volumes, purchase power |
| **Sources** | Kite SDK (get_quotes, get_ltp, get_ohlc, get_historical_data, get_margins, get_holdings, get_positions) |
| **Fallback** | yfinance (LTP via `fast_info.last_price` when Kite quotes fail) |
| **Output** | Structured market snapshot dict with available capital, live prices per symbol, holdings, positions |
| **Cache** | 60-second TTL per symbol to respect Kite rate limits |
| **Schedule** | Every morning 9:00 AM IST, every 5 minutes during market hours |

#### 3.3.2 Fundamentals Agent (`fundamentals_agent`)

| Aspect | Detail |
|--------|--------|
| **Role** | Financial statement analysis, ratio computation, health scoring, moat assessment |
| **Sources** | yfinance (primary: financial statements, ratios), Screener.in (fallback), company filings |
| **Output** | Company fundamental report with ratios, health score, moat level (wide/narrow/none), ROIC, FCF yield |
| **Schedule** | On quarterly earnings, on new stock screening, on user request |
|| **LLM** | Uses LLM for narrative generation explaining ratios and health |

**What it computes**:
- P/E, PEG, P/B, P/S, EV/EBITDA vs peers and 5-year history
- ROE, ROIC (NOPAT/Invested Capital), FCF yield (FCF/Market Cap), dividend yield
- Debt-to-Equity, interest coverage, current/quick ratio
- Revenue/earnings growth trends (3yr, 5yr CAGR)
- Margin trends (gross, operating, net)
- **Moat Assessment**: Scores 0-100 based on ROE, operating margin, net margin, revenue growth, debt level
- **Health Score**: Weighted composite (profitability 30%, growth 25%, financial health 25%, efficiency 20%)
- **DCF Valuation** (future agent): Base/Upside/Downside scenarios with per-share price targets

#### 3.3.3 Technical Agent (`technical_agent`)

| Aspect | Detail |
|--------|--------|
| **Role** | Chart pattern recognition, indicator computation, trend/breakout detection |
| **Sources** | yfinance (primary: OHLCV candles via `history()`), Kite SDK (fallback when yfinance unavailable) |
| **Output** | Technical analysis report with multi-period indicators, patterns, composite signal score |
| **Schedule** | Daily end-of-day scan, on stock screening request |
| **Tools** | yfinance, Kite SDK: get_ohlc, get_historical_data; Python: pandas, numpy, ta-lib |

**What it computes**:
- Trend indicators: SMA 20/50/200 (multi-period), EMA crossovers, ADX, Ichimoku
- Golden Cross/Death Cross detection (SMA50 crossing SMA200)
- Momentum: RSI, MACD, Stochastic, MFI
- Volatility: Bollinger Bands, ATR
- Volume: OBV, A/D Line, Volume Profile
- Patterns: Support/resistance, trendlines, chart patterns (reversal + continuation)
- Signal strength: Composite score across all indicators (trend 30%, momentum 25%, volume 15%, patterns 15%, volatility 15%)

#### 3.3.4 Sentiment Agent (`sentiment_agent`)

| Aspect | Detail |
|--------|--------|
| **Role** | News sentiment, social media mood, corporate announcement analysis |
| **Sources** | Marketaux (primary: NLP sentiment via API), NSE Announcements (fallback: keyword-based scoring of corp announcements), Financial news APIs |
| **Output** | Sentiment score (-1.0 to +1.0) with source breakdown |
| **Schedule** | Daily, triggered by major news events |
| **Tools** | SentimentTools (Marketaux wrapper), httpx (NSE API calls), Firecrawl |

#### 3.3.5 Global Macro Agent (`global_macro_agent`)

| Aspect | Detail |
|--------|--------|
| **Role** | Global macro-to-India impact analysis — how world markets affect Indian equities |
| **Sources** | yfinance (global indices, commodities, currencies, bonds), FRED API (US GDP, CPI, Fed Funds Rate) |
| **Output** | India impact signal (-1.0 to +1.0) + sector impact matrix + LLM-generated macro narrative |
| **Schedule** | Daily 8:45 AM IST (before market scan) |
| **Tools** | yfinance, FRED API (httpx), LLM client for narrative |
| **Cache** | 5-minute TTL for macro indicators; 24-hour TTL for FRED economic data |

**What it computes**:
- **8 Daily Indicators**: US 10Y yield, Fed Funds Rate (FRED), USD/INR, Crude Oil (WTI), VIX, DXY, S&P 500 direction, Gold
- **3 US Economic Data (FRED API)**: GDP, CPI (inflation), Fed Funds Rate
- **5 Global Indices**: S&P 500, NASDAQ, Nikkei, Hang Seng, FTSE overnight moves
- **Composite Impact Signal**: Weighted score of all indicators → India market impact (-1 bearish to +1 bullish)
- **Sector Impact Matrix**: How each macro factor affects specific Indian sectors (IT positive on USD/INR rise, OMCs negative on crude rise, Banking positive on rising rates, etc.)
- **LLM Narrative**: 2-3 sentence plain-English macro summary, with rule-based fallback when LLM unavailable

### 3.5 Layer 2 — Analysis Agents (Synthesis & Debate)

#### 3.4.1 Researcher Agent (Bull/Bear) (`research_agents`)

| Aspect | Detail |
|--------|--------|
| **Role** | Debate framework — synthesizes research inputs into balanced thesis |
| **Inputs** | Reports from all 5 Layer 1 agents |
| **Output** | Bull case, Bear case, and Neutral thesis for each stock |
| **Method** | Structured debate (inspired by TradingAgents paper) |

**Debate Process**:
1. **Bull Researcher**: Builds the optimistic case — growth catalysts, undervaluation signals, positive momentum
2. **Bear Researcher**: Builds the pessimistic case — risks, overvaluation signals, negative indicators
3. **Synthesis**: Both cases presented side-by-side; LLM synthesizes a balanced recommendation

```
Debate Output Format:
{
  "stock": "RELIANCE",
  "bull_case": {
    "thesis": "Strong refining margins + Jio growth...",
    "catalysts": ["Q3 earnings beat", "5G rollout completion"],
    "target_upside": "+15%",
    "confidence": 0.7
  },
  "bear_case": {
    "thesis": "Debt levels elevated, petchem cycle peaking...",
    "risks": ["Crude price volatility", "Regulatory headwinds"],
    "target_downside": "-10%",
    "confidence": 0.4
  },
  "synthesis_rating": "MODERATE BUY",
  "rationale": "Bull case weighted more heavily due to..."
}
```

### 3.6 Layer 3 — Risk Management Agents

#### 3.5.1 Risk Manager Agent (`risk_manager`)

| Aspect | Detail |
|--------|--------|
| **Role** | **Gatekeeper with veto power** — evaluates every trade against risk framework |
| **Inputs** | Trade proposals from Researcher, portfolio state, market conditions |
| **Output** | Risk assessment: APPROVED / SIZE-REDUCED / BLOCKED + rationale |
| **Schedule** | Every pre-trade check, daily portfolio risk scan |

**Risk Checks** (14 checks implemented):
1. **Cash availability**: Available balance (from Kite SDK) must cover trade | If ₹0 → BLOCK ALL BUYS
2. **Position sizing**: Single stock max % of available cash (configurable, default 30%)
3. **Sector exposure**: Max sector allocation (default 30%)
4. **Drawdown stop**: If portfolio down X% in a day → halt new entries
5. **Correlation check**: New position vs existing — avoid over-concentration
6. **Liquidity check**: Average daily volume must support position size
7. **VaR**: Parametric Value at Risk (95% confidence, max 2% of portfolio)
8. **CVaR (Conditional VaR)**: Expected Shortfall — what happens BEYOND VaR (95% confidence, max 5%)
9. **Volatility regime**: ATR-based volatility check — high ATR = reduce position sizes
10. **News risk**: Sentiment score check — BLOCK buys if sentiment < -0.3
11. **Margin usage**: Margin utilization must be under threshold
12. **Stock concentration**: No single stock > N% of portfolio
13. **Stop-loss distance**: Validates stop-loss not too tight (<1%) or too loose (>10%)
14. **Risk-reward ratio**: Minimum 1:2 risk/reward for new trades
15. **Monthly pacing**: Budget used this month / days elapsed — warn if deploying too fast
16. **Max positions**: Hard limit on total open positions
17. **Correlation matrix**: Full portfolio correlation check
18. **Liquidity depth**: Average daily volume must cover position

**Risk Manager has FINAL VETO on all trades.** No trade executes without its approval.

### 3.7 Layer 4 — Execution Agents

#### 3.6.1 Portfolio Manager Agent (`portfolio_manager`)

| Aspect | Detail |
|--------|--------|
| **Role** | Capital allocation, trade prioritization, portfolio balancing |
| **Inputs** | Risk-approved trade proposals, current portfolio state, available capital |
| **Output** | Prioritized trade queue with position sizes |
| **Schedule** | On every trade decision |

**Responsibilities**:
- **Balance awareness**: Read available cash from Kite SDK — this IS the deployable pool
- **Capital deployment pacing**: Spread deployments across days — don't buy everything at once
- **Capital allocation**: How much to deploy on each opportunity (from available cash)
- **Trade prioritization**: Which signal to act on first when cash is constrained
- **Cash buffer**: Maintain ~10% idle cash for averaging down / opportunities
- **Portfolio rebalancing**: Periodic rebalancing to target weights
- **Tax awareness**: Hold period tracking (LT capital gains vs ST)
- **Auto-detect top-ups**: When user adds money to Zerodha, detect the cash increase and resume deploying

#### 3.6.2 Execution Agent (`execution_agent`)

| Aspect | Detail |
|--------|--------|
| **Role** | Order placement, modification, cancellation via Kite SDK |
| **Inputs** | Portfolio Manager's trade queue |
| **Output** | Order confirmations, execution reports |
| **Schedule** | Market hours, on Portfolio Manager output |
| **Tools** | Kite SDK: place_order, modify_order, cancel_order, get_orders, get_trades |

**Execution Modes** (configurable per agent):
- **Recommendation Only** (default): Generates recommendation, waits for human approval
- **Small Auto-Execute**: Auto-execute trades under configurable ₹ threshold
- **Paper Trading**: Simulate execution without real money (development/testing mode)

### 3.8 Agent Interaction Flow

```
MORNING ROUTINE (Scheduled 9:00 AM IST):
┌────────────────────────────────────────────────────────────┐
│ 0. Market Data Agent checks balance (Kite SDK get_margins)  │
│    → Two parallel tracks start:                             │
│                                                             │
│    TRACK A — NEW BUYS (only if cash > ₹1,000):               │
│    ┌──────────────────────────────────────────────────┐    │
│    │ 1. Supervisor triggers "Daily Market Scan"        │    │
│    │ 2. PARALLEL: Market Data + Fundamentals +         │    │
│    │    Technical + Sentiment agents run               │    │
│    │ 3. Researcher → Bull/Bear debate                  │    │
│    │ 4. Risk Manager evaluates each proposal            │    │
│    │    → Cash check: position fits in balance?         │    │
│    │    → Pacing check: deploying too fast?            │    │
│    │ 5. Portfolio Manager allocates from available cash │    │
│    │ 6. Execution Agent places orders                   │    │
│    └──────────────────────────────────────────────────┘    │
│                                                             │
│    TRACK B — PORTFOLIO GUARDIAN (ALWAYS runs, even ₹0 cash)  │
│    ┌──────────────────────────────────────────────────┐    │
│    │ 1. Load all open positions from Kite SDK           │    │
│    │ 2. For EACH position:                              │    │
│    │    a. Check stop-loss: price < -8% → SELL          │    │
│    │    b. Check profit target: price > +15% → SELL     │    │
│    │    c. Re-evaluate thesis: score dropped? → SELL    │    │
│    │    d. Check allocation: >30% of cash? → REDUCE     │    │
│    │ 3. If positions sold → freed cash available         │    │
│    │ 4. Freed cash → routed back to Track A (buy scan)   │    │
│    │ 5. Risk scan: drawdown > 10%? sector > 30%?         │    │
│    │ 6. Alert user if any action taken                   │    │
│    └──────────────────────────────────────────────────┘    │
│                                                             │
│ 7. Everything logged to memory store                        │
└────────────────────────────────────────────────────────────┘

ADRENALINE FLOW (Real-time, during market hours):
┌────────────────────────────────────────────────────────────┐
│ Price alert triggers → Market Data Agent checks →           │
│ If significant move (>2%) → Technical Agent re-analyzes →   │
│ If signal change → Researcher re-evaluates →                │
│ If rating changes → Trader proposes → Risk Manager checks → │
│ Execution (or alert to user)                                │
└────────────────────────────────────────────────────────────┘

QUARTERLY DEEP DIVE (On earnings / scheduled):
┌────────────────────────────────────────────────────────────┐
│ Fundamentals Agent does full 5-stage due diligence →         │
│ Full financial ratio suite → DCF valuation →                 │
│ Researcher long-term bull/bear thesis →                     │
│ Portfolio review → Rebalancing recommendations              │
└────────────────────────────────────────────────────────────┘
```

---

## 4. LONG-TERM MEMORY ARCHITECTURE

> Finance is a long-term game. Decisions made today affect outcomes months later. The system MUST remember.

### 4.1 Why Long-Term Memory Matters for Finance

| Memory Type | Financial Example | Retention |
|-------------|-------------------|-----------|
| **Episodic Memory** | "On Jan 15 I bought RELIANCE at 2450 because..." | Forever |
| **Semantic Memory** | "RELIANCE has strong refining margins, high debt" | Updated quarterly |
| **Procedural Memory** | "My momentum strategy works well in trending markets" | Forever, with decay |
| **Short-term/Buffer** | "Today's price action for watchlist stocks" | 30 days rolling |

Memory uses a **multi-scope** model composable at retrieval time:

```
user_id:  Prakhar         ← Portfolio owner
agent_id: researcher      ← Which agent learned this
run_id:  2026-06-23-a     ← Which morning run
app_id:  autfin           ← The AutFin system
```

### 4.2 Memory Implementation (LangGraph Store + Vector DB)

```
┌──────────────────────────────────────────────────────────┐
│                    MEMORY SYSTEM                           │
├──────────────────────────────────────────────────────────┤
│                                                            │
│  LangGraph Store (cross-thread, long-term persistence)     │
│  ┌────────────────────────────────────────────────────┐   │
│  │ Namespace: capital_state                            │   │
│  │  - available_cash snapshot (from Kite SDK)          │   │
│  │  - deployment_log: [{date, amount, symbol, reason}] │   │
│  │ Namespace: portfolio_state                          │   │
│  │  - Current positions, allocation, P&L, margins     │   │
│  │ Namespace: trade_history/{stock}                    │   │
│  │  - Every trade: entry, exit, rationale, outcome     │   │
│  │ Namespace: research_reports/{stock}                 │   │
│  │  - Latest fundamental, technical, sentiment reports │   │
│  │ Namespace: decisions/{decision_id}                  │   │
│  │  - Full decision trace: input → debate → risk →     │   │
│  │    budget_check → execution → outcome → reflection  │   │
│  └────────────────────────────────────────────────────┘   │
│                                                            │
│  Vector Store (ChromaDB / LanceDB — local-first)           │
│  ┌────────────────────────────────────────────────────┐   │
│  │ Collection: market_data                         │   │
│  │  - Embedded summaries of all research reports       │   │
│  │  - Enables: "What were my reasons for buying X?"    │   │
│  │ Collection: learning_journal                       │   │
│  │  - Embedded reflections on past trades              │   │
│  │  - Enables: "What patterns preceded my winning      │   │
│  │    trades?"                                         │   │
│  │ Collection: research_notes                        │   │
│  │  - Embedded company profiles, moats, business model │   │
│  │  - Enables: "What do I know about this sector?"     │   │
│  └────────────────────────────────────────────────────┘   │
│                                                            │
│  SQLite Database (structured financial data)               │
│  ┌────────────────────────────────────────────────────┐   │
│  │ Table: trades (timestamp, symbol, action, price,    │   │
│  │         quantity, rationale, decision_id, outcome)  │   │
│  │ Table: portfolio_snapshots (daily P&L, allocation,  │   │
│  │         available_cash, deployed_value)              │   │
│  │ Table: signals (timestamp, symbol, signal_type,     │   │
│  │         strength, rationale, was_acted_on)          │   │
│  │ Table: balance_history (timestamp, available_cash,  │   │
│  │         utilized_margin — daily snapshot from Kite) │   │
│  └────────────────────────────────────────────────────┘   │
│                                                            │
└──────────────────────────────────────────────────────────┘
```

### 4.3 Memory Retrieval Strategies

```python
# When analyzing a new stock, the system automatically:
# 1. Retrieves past research on this stock (Store)
# 2. Retrieves similar past trade rationales (Vector Store similarity search)
# 3. Retrieves sector knowledge (Vector Store)
# 4. Includes relevant past outcomes in the agent's context window

def build_research_context(stock, thread_id):
    # Cross-thread memory via Store
    past_reports = store.get(("research_reports", stock))
    past_trades = store.get(("trade_history", stock))

    # Semantic memory via vector store
    similar_decisions = vector_store.similarity_search(
        f"What were the factors behind past {stock} decisions?",
        collection="strategy_learnings",
        k=5
    )

    # Combined context injected into agent
    return {
        "past_research": past_reports,
        "trade_history": past_trades,
        "relevant_learnings": similar_decisions
    }
```

### 4.4 Durability & Crash Recovery

- **LangGraph Checkpointer** (SqliteSaver with WAL): Every node execution is checkpointed
- **Crash recovery**: System resumes from last completed node, not from start
- **Durability mode**: Write-ahead logging (WAL) for SQLite backend
- **Periodic snapshots**: Full state serialized daily to backup
- **Agent timeout handling**: Long-running analysis operations capped at configurable timeout (default 5 min)
- **Interrupted trades**: If execution interrupted mid-order, system checks order status on restart

---

## 5. DATA PIPELINE & SOURCES

> Full research: `data_sources_research.md` — detailed API specs, cost analysis, scraping strategies

### 5.1 Agent-to-Source Mapping (Concrete)

Each research agent has a **primary** and **fallback** data source:

| Agent | Primary Source | What It Gets | Fallback | Schedule |
|-------|---------------|-------------|----------|----------|
| **Market Data** | Kite SDK (primary), yfinance (fallback LTP) | Live prices, margins, holdings, positions | — | Every 5 min (market hrs), 9 AM daily |
| **Technical** | yfinance `.NS` / `.BO` historical OHLCV | 2yr daily data → 20+ indicators (multi-SMA, EMA, RSI, MACD, BB, OBV, ATR) | Kite SDK `get_historical_data` | Daily EOD + on-demand |
| **Fundamentals** | yfinance (primary: financials, ratios, ROIC, FCF yield) | P/E, ROE, ROIC, FCF Yield, D/E, margins, growth + moat assessment | Screener.in (Firecrawl fallback) | Weekly refresh, quarterly deep dive |
| **Sentiment** | Marketaux API (NLP sentiment) | Ticker-level sentiment -1.0 to +1.0 | NSE Announcements (keyword-based scoring) | Daily 8 AM + event-triggered |
| **Global Macro** | yfinance (indices, commodities, currencies) + FRED API | S&P 500, VIX, DXY, USD/INR, Crude, Gold, US 10Y, GDP, CPI, Fed Funds + LLM narrative | N/A | Daily 8:45 AM (before market scan) |

### 5.2 Tiered Data Architecture

```
TIER 1 — ALWAYS-ON (real-time APIs, no scraping):
┌────────────────────────────────────────────────────────┐
│ Kite SDK      → Live prices, margins, portfolio, orders │
│ Marketaux     → News sentiment per ticker (NLP)         │
│ yfinance      → Historical OHLCV, fundamentals, macros   │
│ NSE APIs      → Corporate announcements (fallback)      │
│ FRED API      → US GDP, CPI, Fed Funds Rate (24h cache) │
└────────────────────────────────────────────────────────┘

TIER 2 — SCHEDULED SCRAPES (Firecrawl, weekly):
┌────────────────────────────────────────────────────────┐
│ Screener.in   → Financials, ratios (fallback)           │
│ NSE/BSE       → Bulk/block deals, corp announcements    │
│ RBI DBIE      → GDP, inflation, interest rates (monthly)│
│ Moneycontrol  → Headlines, results, board meetings      │
└────────────────────────────────────────────────────────┘

TIER 3 — ON-DEMAND (per-analysis research):
┌────────────────────────────────────────────────────────┐
│ Trendlyne     → FII/DII flows, insider trades, SWOT     │
│ TradingView   → Charts, community sentiment             │
│ SEBI/EDGAR    → Regulatory filings, prospectus          │
│ World Bank    → Macro context (quarterly)               │
└────────────────────────────────────────────────────────┘
```

### 5.3 Concrete Data Sources

#### 5.3.1 Market Data (Real-time + Historical)

| Source | Type | Cost | Key Details |
|--------|------|------|-------------|
| **Kite SDK** | KiteConnect SDK (Personal) | Free (Zerodha account) | Portfolio, orders, margins, GTT. LTP, quotes, holdings, positions. |
| **yfinance** | Python library | Free, no API key | `.NS` / `.BO` suffix for Indian stocks, 2yr+ daily OHLCV, fundamentals, global macros |
| **NSE Public APIs** | Free REST endpoints | Free, no API key | Corporate announcements, bhav copies, indices, bulk/block deals |

#### 5.3.2 Fundamental Data

| Source | Type | Cost | Key Details |
|--------|------|------|-------------|
| **Screener.in** | Web platform | Free (+₹4,999/yr Premium) | **Best free Indian fundamentals**. 10yr P&L, BS, CF, 30+ ratios, peer comparison, shareholding, stock screener. No official API → scrape via Firecrawl |
| **Moneycontrol** | Web platform | Free | Financials, news, announcements, technical charts. Scrape via Firecrawl |
| **FinEdge API** | REST API | Paid | Structured Indian financial data via API. Clean but paid |
| **Financial Modeling Prep** | REST API | Free tier | Global financial statements. Limited India coverage vs Screener.in |

#### 5.3.3 News & Sentiment

| Source | Type | Cost | Key Details |
|--------|------|------|-------------|
| **Marketaux** | REST API | Free, no credit card | 5000+ sources, 200K+ entities tracked, NLP sentiment per ticker (0-1 score), Indian sources verified. `entities[].{symbol, sentiment_score, match_score, highlights}` |
| **StockNewsAPI** | REST API | Free trial → $29/mo | Ticker-filtered news, sentiment, upgrades/downgrades, historical to 2019 |
| **NSE Corporate Announcements** | Web | Free | Direct from exchange → scrape with Firecrawl |
| **Moneycontrol News** | Web | Free | Company-specific news and board announcements |

#### 5.3.4 Special Data (Deals, FII/DII, Insider)

| Source | Type | Cost | What It Provides |
|--------|------|------|-----------------|
| **NSE Bulk/Block Deals** | Web | Free | Daily institutional trade disclosures |
| **Trendlyne** | Web platform | Free tier | FII/DII daily flows, insider trades, SWOT analysis, technical scans |
| **RBI DBIE** | Official portal | Free | GDP, CPI, WPI, repo rates, monetary policy, banking stats |
| **World Bank / IMF** | Open data APIs | Free | Macro indicators, development data |

#### 5.3.5 Global Macro — Free Global-to-India Impact Analysis

AutFin tracks how outside world events impact Indian markets. A composite **India impact signal** (-1.0 to +1.0) is computed daily and injected into every agent's context.

**Data sources (all free)**: yfinance (indices, commodities, currencies, bonds, VIX — no API key) + FRED API (US 10Y, Crude, USD/INR, CPI, Fed Rate, Unemployment — free API key)

**Key correlation framework**:

| Global Factor | India Impact | Who Wins | Who Loses |
|--------------|-------------|----------|-----------|
| S&P 500 / NASDAQ | Overnight gap effect (0.6-0.7 correlation) | — | — |
| USD/INR ↑ (rupee weak) | Import bill pressure | IT, Pharma, Exporters | Aviation, OMCs, Importers |
| Crude Oil ↑ | 85% import share of India | ONGC, Reliance, Oil India | OMCs, Paints, Tyres, Airlines |
| US 10Y / Fed Rate ↑ | FII outflow from India | Domestic-consumption stocks | High-FII-holding stocks |
| VIX ↑ (>20) | Risk-off: FII selling EMs | Defensives (FMCG, Pharma) | Cyclicals, High-Beta |
| DXY ↑ (strong dollar) | EM headwind | — | Broad market |
| Gold ↑ | Risk-off proxy | Gold NBFCs, SGBs | Equity sentiment broadly |
| China stimulus | FII rotates to China | Metals, Commodities | Domestic India |

**Implementation**: Global Macro Agent runs at 8:45 AM daily, computes `india_impact_signal` via weighted scoring of all factors, injects into `state["macro_signals"]`. All other agents read this signal alongside their domain-specific data.

### 5.4 Data Flow — Daily Startup

```
09:00 AM — DAILY STARTUP:
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  Kite SDK: get_margins()                                   │
│    → available_cash = deployable pool                      │
│                                                            │
│  Kite SDK: get_holdings(), get_positions()                  │
│    → Current portfolio state                               │
│                                                            │
│  PARALLEL DATA GATHERING (simultaneous):                    │
│  ┌──────────────────────────────────────────────────┐     │
│  │ yfinance: .history(period="1y") for each holding  │     │
│  │          + watchlist candidates                    │     │
│  │                                                    │     │
│  │ Marketaux: GET /v1/news/all?entities=RELIANCE.NS, │     │
│  │           TCS.NS,...&filter_entities=true           │     │
│  │           → Ticker-level sentiment scores           │     │
│  │                                                    │     │
│  │ Firecrawl (if scheduled):                           │     │
│  │  ├─ Screener.in → quarterly financials for holdings│     │
│  │  ├─ NSE → bulk/block deals from last session       │     │
│  │  └─ Moneycontrol → corporate announcements         │     │
│  └──────────────────────────────────────────────────┘     │
│                                                            │
│  Agents receive structured data → begin analysis            │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 5.5 Firecrawl Scraping Strategy

For Screener.in and other web-based sources:

```python
# Weekly fundamentals refresh (runs Sunday)
from firecrawl import FirecrawlApp

app = FirecrawlApp()

# 1. Scrape financial tables from Screener.in
for stock in watchlist:
    result = app.scrape(
        f"https://www.screener.in/company/{stock}/consolidated/",
        formats=["json"],
        jsonOptions={
            "prompt": "Extract: P&L table (revenue, expenses, net profit last 5 years), "
                      "Balance Sheet (assets, liabilities, equity), "
                      "Ratios (PE, ROE, ROCE, Debt/Equity, EPS, Book Value), "
                      "Shareholding pattern (promoter %, FII %, DII %, public %), "
                      "Quarterly results (last 4 quarters)"
        }
    )
    store.put(("research_reports", stock), f"fundamentals_{date}", result)

# 2. Bulk/Block deals from NSE
deals = app.scrape(
    "https://www.nseindia.com/report-detail/display-bulk-and-block-deals",
    formats=["json"],
    jsonOptions={
        "prompt": "Extract all bulk and block deals: date, stock symbol, "
                  "buyer name, seller name, quantity, price, deal type"
    }
)

# 3. Corporate announcements
announcements = app.scrape(
    "https://www.nseindia.com/companies-listing/corporate-filings-announcements",
    formats=["json"],
    jsonOptions={
        "prompt": "Extract latest announcements: company, date, subject, details"
    }
)
```

### 5.6 Marketaux Sentiment Integration

```python
import httpx

MARKETAUX_TOKEN = "your_free_api_key"

async def get_ticker_sentiment(symbols: list[str]) -> dict:
    """Get NLP sentiment scores for Indian stocks."""
    entities = ",".join(f"{s}.NS" for s in symbols)

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.marketaux.com/v1/news/all",
            params={
                "api_token": MARKETAUX_TOKEN,
                "entities": entities,
                "filter_entities": "true",
                "language": "en",
                "limit": 20
            }
        )
        data = resp.json()

    # Aggregate sentiment per ticker
    sentiment = {}
    for article in data.get("data", []):
        for entity in article.get("entities", []):
            symbol = entity["symbol"]
            if symbol not in sentiment:
                sentiment[symbol] = {"scores": [], "mentions": 0}
            sentiment[symbol]["scores"].append(entity["sentiment_score"])
            sentiment[symbol]["mentions"] += 1

    # Average sentiment per ticker
    return {
        s: {
            "avg_sentiment": sum(v["scores"]) / len(v["scores"]),
            "mention_count": v["mentions"],
            "range": (min(v["scores"]), max(v["scores"]))
        }
        for s, v in sentiment.items()
    }
```

### 5.7 Cost Analysis — Entire Data Stack

| Component | Provider | Monthly Cost |
|-----------|----------|-------------|
| Real-time + portfolio | Kite SDK (Zerodha) | ₹0 |
| Historical OHLC | yfinance | ₹0 |
| News sentiment | Marketaux | ₹0 (free tier) |
| Quick stock lookup | Indian-Stock-Market-API | ₹0 |
| Fundamental data | Screener.in via Firecrawl | ₹0 (free tier: 500 credits) |
| Bulk deals, announcements | NSE/BSE via Firecrawl | Included above |
| Macro data | RBI DBIE, World Bank | ₹0 |
| Web scraping infra | Firecrawl | ₹0 (Hobby $19/mo if needed) |
| **TOTAL** | | **₹0** |

> Firecrawl free tier (500 credits/mo) covers ~400 credits for AutFin's usage. Upgrade to Hobby ($19/mo, 3000 credits) only if needed.

### 5.8 KiteConnect SDK Integration

```json
{
  "zerodha": {
    "type": "kiteconnect_sdk",
    "tier": "personal",
    "auth": "programmatic_totp"
  }
}
```

**Available Methods (KiteConnect SDK)**:
- Auth: `login_url()`, `generate_session()`, `set_access_token()`, `profile()`, `margins()`
- Portfolio: `holdings()`, `positions()`, `position_convert()`
- Orders: `place_order()`, `modify_order()`, `cancel_order()`, `orders()`, `order_history()`, `trades()`
- GTT: `place_gtt_order()`, `modify_gtt_order()`, `cancel_gtt_order()`, `gtt_triggers()`
- MF: `place_mf_order()`, `cancel_mf_order()`, `mf_orders()`, `mf_instruments()`, `mf_sips()`

**Personal Tier Limitations**:
- ❌ `quote()`, `ohlc()`, `ltp()`, `historical_data()` — NOT available on Personal tier
- ✅ All portfolio, order, and GTT methods work on Personal tier
- Market data (prices, OHLCV) sourced from yfinance (free, no API key)

**Programmatic Login Flow (No Browser Required)**:
1. Initialize `KiteConnect(api_key=api_key)`
2. `POST https://kite.zerodha.com/api/login` with user_id + password → request_id
3. `POST https://kite.zerodha.com/api/twofa` with request_id + TOTP → session cookie
4. Follow redirect to get request_token
5. `kite.generate_session(request_token, api_secret)` → access_token
6. Token valid until 6:00 AM IST next day (no refresh token for Personal tier)
7. Daily re-authentication required before market open

**Setup**:
- Get API key + secret from https://developers.kite.trade
- Enable TOTP in Zerodha console → copy TOTP secret
- Set `KITE_API_KEY`, `KITE_API_SECRET`, `KITE_USER_ID`, `KITE_PASSWORD`, `KITE_TOTP_SECRET` in `.env`
- AutFin handles daily token refresh automatically

---

## 6. DECISION FRAMEWORK

### 6.1 Capital-Aware Decision Architecture

Before any analysis begins, AutFin first checks its bank balance:

```
EVERY MORNING — CAPITAL CHECK (runs before all analysis):
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  Kite SDK: get_margins()                                      │
│    → available_cash: ₹48,230  ← The budget, directly from     │
│    → utilized_margin: ₹1,770     Zerodha. No config needed.   │
│                                                               │
│  Deployable = ₹48,230                                          │
│  (The cash IS the budget. When it's gone, trading stops.)     │
│                                                               │
│  IF deployable < min_trade_amount (₹1000):                    │
│     → SKIP scanning, just monitor existing positions           │
│     → Log: "Cash depleted. Add funds to resume trading."       │
│  ELSE:                                                        │
│     → Proceed to full daily scan                               │
│     → Pace deployments — make this cash last                    │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 Capital Deployment Pacing Strategy

The Portfolio Manager manages the available cash intelligently:

```
CAPITAL DEPLOYMENT STRATEGY:
─────────────────────────────
Rule 1 — NEVER exceed available cash (hard limit from Zerodha)
Rule 2 — Spread across days (don't buy everything on day 1)
Rule 3 — Max 30% of available cash on any single stock
Rule 4 — Keep 10% cash buffer for averaging down
Rule 5 — Priority to highest-conviction signals when cash is tight
Rule 6 — Selling frees up cash for better opportunities
Rule 7 — When user adds more money → deploy again

PACING GUIDELINE (not enforced, just guidance):
  ideal_deployed = available_cash_at_month_start × (day / total_trading_days)

  If deployed_too_fast → tighten filters (only A+ signals)
  If deployed_too_slow → loosen slightly (consider B+ signals)
  If cash ≈ 0 → HALT. Wait for user top-up.
```

### 6.3 How AutFin Decides: Buy / Sell / Hold

The system uses a **weighted scoring model** that combines outputs from all agents:

```
FINAL_SCORE = (
    fundamentals_score * 0.35 +
    technical_score * 0.25 +
    sentiment_score * 0.15 +
    risk_score * 0.20 +
    portfolio_fit_score * 0.05
)

Decision Matrix:
  Score > 0.70  → STRONG BUY
  Score 0.55-0.70 → BUY
  Score 0.40-0.55 → HOLD / WATCH
  Score 0.25-0.40 → WEAK HOLD / REDUCE
  Score < 0.25  → SELL

Risk Override:
  Risk Manager VETO overrides any BUY signal
  Portfolio Manager allocation caps limit position size
```

### 6.4 Scoring Details

#### Fundamentals Score (0-1)
- P/E vs sector median: below median = positive
- PEG ratio: < 1 = positive
- ROE trend: improving = positive
- FCF yield > risk-free rate = positive
- D/E in safe range = positive
- Revenue/earnings growth > sector avg = positive
- DCF intrinsic value vs current price (margin of safety)

#### Technical Score (0-1)
- Price vs SMA(50) and SMA(200): above = positive
- RSI in 30-70 range: healthy; < 30 = oversold opportunity
- MACD signal: bullish crossover = positive
- ADX > 20: trending (good for trend strategies)
- Volume confirmation: breakout with volume = positive
- Chart pattern: bullish continuation/reversal = positive

#### Sentiment Score (-1 to +1)
- News sentiment: aggregated NLP score
- Analyst consensus: upgrades vs downgrades
- Social media momentum: unusual activity detection
- Insider buying/selling activity

#### Risk Score (0-1)
- Volatility (ATR): lower = higher score
- Beta vs market: not extreme = higher score
- Max drawdown history: acceptable = higher score
- Liquidity (avg daily volume): sufficient = higher score
- Correlation with existing portfolio: low correlation = higher score

#### Portfolio Fit Score (0-1)
- Sector allocation gap: underrepresented sector = higher score
- Diversification benefit: uncorrelated = higher score
- Capital availability: has room for position = higher score

### 6.5 Sell Triggers (Automatic)

| Trigger | Action |
|---------|--------|
| Stop-loss hit (configurable %, default -8%) | Auto-sell or alert |
| Fundamentals Score drops below 0.3 | Alert → human review |
| Technical breakdown (price < SMA(200) + death cross) | Alert → recommend reduce |
| Thesis broken (earnings miss, management change, competitive threat) | Recommend sell |
| Better opportunity found (reallocate capital) | Recommend swap |
| Portfolio risk limit exceeded | Reduce largest positions |

---

## 7. TECHNICAL ARCHITECTURE

### 7.1 Technology Stack

```
┌─────────────────────────────────────────────────────────┐
│                     TECH STACK                            │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ORCHESTRATION: LangGraph (Python)                        │
│  ─────────────────────────────────                       │
│  • StateGraph for multi-agent workflows                   │
│  • SqliteSaver for checkpointing (WAL mode)               │
│  • Store API for cross-thread long-term memory            │
│  • interrupt() for human-in-the-loop                      │
│  • Command for dynamic agent routing                      │
│                                                           │
│  MCP INTEGRATION: Removed (using KiteConnect SDK directly)    │
│  ─────────────────────────────                                │
│  • AutFin connects to Zerodha via KiteConnect SDK (Personal)  │
│  • No MCP server or client needed                             │
│  • Market data sourced from yfinance (free, no API key)        │
│                                                           │
│  AI/LLM — FREE MODEL STRATEGY:                              │
│  ─────────────────────────────────                           │
│  • Router: LiteLLM (multi-provider failover)                 │
│  • Primary: OpenRouter (20+ free models, single API key)     │
│  • Secondary: Groq (speed — 320 tps on Llama 70B)            │
│  • Fallback: Google Gemini (1M context, 1,500 req/day)       │
│  • Optional $10 OpenRouter top-up → 1,000 RPD (from 200)     │
│                                                           │
│  Tier 1 — DEEP ANALYSIS (Bull/Bear, Risk, Research):         │
│  • Llama 3.3 70B:free (OpenRouter) — solid reasoning         │
│  • DeepSeek R1:free (OpenRouter) — best for math/reasoning   │
│  • Gemini 2.5 Flash (Google) — 1M context for filings        │
│                                                           │
│  Tier 2 — FAST/ROUTING (Supervisor, extraction, scoring):    │
│  • Gemma 4 26B:free (OpenRouter) — balanced speed/quality    │
│  • Llama 3.1 8B (Groq) — 14,400 req/day, ultra-fast          │
│                                                           │
│  Rate Limits (combined across providers):                    │
│  • ~200-1,500 RPD free (sufficient for ~30-50 AutFin calls)  │
│  • 20-30 RPM, 6K-30K TPM                                   │
│  • No credit card required for any provider listed            │
│                                                           │
│  MEMORY & STORAGE:                                        │
│  ─────────────────                                        │
│  • SQLite: Structured data (trades, signals, portfolio)   │
│  • ChromaDB/LanceDB: Vector embeddings (local-first)      │
│  • Redis (optional): Real-time market data buffer          │
│                                                           │
│  DATA PIPELINE:                                           │
│  ──────────────                                           │
│  • Firecrawl: Web scraping for financial data             │
│  • Pandas/NumPy: Technical indicator computation          │
│  • ta-lib: Specialized technical analysis                 │
│  • Schedule/APScheduler: Job scheduling                   │
│                                                           │
│  UI (Future Phase):                                       │
│  ──────────────────                                       │
│  • Streamlit / Gradio dashboard                           │
│  • CLI interface (initial phase)                          │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### 7.2 Project Structure

```
autfin/
├── pyproject.toml              # Project config, dependencies
├── README.md
│
├── src/autfin/
│   ├── __init__.py
│   │
│   ├── graph/                  # LangGraph workflow definitions
│   │   ├── __init__.py
│   │   ├── supervisor.py       # Supervisor agent + routing
│   │   (daily_morning, adrenaline, quarterly_deep_dive — retired ✅)
│   │
│   ├── agents/                 # Individual agent implementations
│   │   ├── __init__.py
│   │   ├── supervisor.py       # Supervisor agent logic
│   │   ├── market_data.py      # Market data ingestion agent
│   │   ├── fundamentals.py     # Fundamental analysis agent
│   │   ├── technical.py        # Technical analysis agent
│   │   ├── sentiment.py        # Sentiment/News analysis agent
│   │   ├── global_macro.py     # Global macro → India impact analysis
│   │   ├── researcher.py       # Bull/Bear research debate
│   │   ├── risk_manager.py     # Risk evaluation + veto
│   │   ├── portfolio_manager.py # Capital allocation
│   │   └── executor.py         # Trade execution via Kite SDK
│   │
│   ├── state/                  # State definitions
│   │   ├── __init__.py
│   │   ├── agent_state.py      # Shared StateGraph state schema
│   │   ├── market_state.py     # Market data snapshots
│   │   ├── macro_state.py      # Global macro signals schema
│   │   └── portfolio_state.py  # Portfolio tracking state
│   │
│   ├── tools/                  # Agent tools
│   │   ├── __init__.py
│   │   ├── kite_tools.py       # Kite SDK tool wrappers
│   │   ├── technical_tools.py  # Technical indicator computation
│   │   ├── fundamental_tools.py# Ratio computation, DCF models
│   │   ├── sentiment_tools.py  # Marketaux API + news scoring
│   │   ├── macro_tools.py      # FRED API + yfinance global indices
│   │   ├── scraper_tools.py    # Firecrawl wrappers (Screener.in, NSE, etc.)
│   │   └── memory_tools.py     # Long-term memory access tools
│   │
│   ├── memory/                 # Memory management
│   │   ├── __init__.py
│   │   ├── store.py            # LangGraph Store setup + operations
│   │   ├── vector_store.py     # ChromaDB/LanceDB management
│   │   ├── database.py         # SQLite schema + operations
│   │   └── retrieval.py        # Memory retrieval strategies
│   │
│   ├── prompts/                # Agent prompts
│   │   ├── __init__.py
│   │   ├── supervisor.py
│   │   ├── market_data.py
│   │   ├── fundamentals.py
│   │   ├── technical.py
│   │   ├── sentiment.py
│   │   ├── global_macro.py
│   │   ├── researcher.py
│   │   ├── risk_manager.py
│   │   └── portfolio_manager.py
│   │
│   ├── config/                 # Configuration
│   │   ├── __init__.py
│   │   ├── settings.py         # All configurable parameters
│   │   └── model_config.py     # LLM model routing config
│   │
│   ├── scheduler/              # Job scheduling
│   │   ├── __init__.py
│   │   └── jobs.py             # Scheduled tasks (daily scan, etc.)
│   │
│   └── cli/                          # CLI interface
│
├── tests/
│   ├── test_agents/
│   ├── test_graphs/
│   └── test_memory/
│
├── data/                       # Local data storage
│   ├── database/               # SQLite databases
│   └── vectors/                # ChromaDB persistence
│
└── notebooks/                  # Research/exploration notebooks
    └── backtesting.ipynb
```

### 7.3 Dependencies

```toml
[project]
name = "autfin"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    # Orchestration
    "langgraph>=1.0.0",
    "langchain-core>=0.3.0",
    "langgraph-checkpoint-sqlite>=2.0.0",

    # Kite SDK
    "kiteconnect>=5.0.0",
    "pyotp>=2.6.0",
    "requests>=2.31.0",

    # Data & Analysis
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "yfinance>=0.2.0",
    "TA-Lib>=0.6.0",

    # Storage
    "chromadb>=0.5.0",
    "sqlite-utils>=3.0.0",

    # Web Scraping
    "firecrawl-py>=1.0.0",

    # Scheduling
    "apscheduler>=3.10.0",

    # LLM
    "litellm>=1.0.0",       # Multi-provider routing + failover
    "openai>=1.0.0",        # OpenAI-compatible client for OpenRouter/Groq
    "google-generativeai>=0.8.0",  # Google Gemini (optional direct)

    # Utilities
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
    "rich>=13.0.0",  # CLI output
    "httpx>=0.27.0",
]
```

### 7.4 Configuration (settings.py) Overview

```python
class AutFinSettings:
    # LLM — Free Model Strategy (via litellm)
    deep_model: str = "nvidia_nim/nvidia/nemotron-3-super-120b-a12b"  # NVIDIA NIM (free, 1M ctx)
    fast_model: str = "nvidia_nim/meta/llama-3.1-8b-instruct"        # NVIDIA NIM (free, fast)
    litellm_failover: bool = True              # litellm handles retries + rate-limit detection
    nvidia_nim_api_key: str = ""               # From build.nvidia.com

    # Zerodha
    # Zerodha (KiteConnect SDK — Personal tier)
    kite_user_id: str = "your_zerodha_user_id"
    kite_password: str = "your_zerodha_password"
    kite_totp_secret: str = "your_totp_secret"

    # Capital Management (balance = budget, from Kite SDK)
    min_trade_amount: float = 1000.0           # Don't trade if less than ₹1000 available
    cash_buffer_pct: float = 0.10              # Keep 10% cash for averaging down
    deployment_pacing: str = "spread"          # "spread" | "aggressive" | "conservative"

    # Risk Parameters
    max_single_position_pct: float = 0.30     # 30% max of available cash in one stock
    max_sector_exposure_pct: float = 0.30     # 30% max in one sector
    default_stop_loss_pct: float = 0.08       # -8% stop loss
    trailing_stop_loss_pct: float = 0.05      # -5% trailing stop from recent high
    profit_target_pct: float = 0.15           # +15% take-profit target
    portfolio_drawdown_halt_pct: float = 0.10 # Halt new entries if -10% in a week
    max_daily_trades: int = 5                 # Max trades per day
    min_market_cap_crore: float = 500.0       # Minimum market cap filter (₹500 Cr)
    min_avg_volume_lakhs: float = 5.0         # Minimum daily volume (₹5 Lakhs)

    # Execution
    auto_execute_max_amount: float = 5000.0   # Auto-execute trades under ₹5000
    execution_mode: str = "recommendation"    # "recommendation" | "auto_small" | "paper"

    # Memory
    vector_store_type: str = "chromadb"       # "chromadb" | "lancedb"
    trade_history_retention_days: int = 3650  # ~10 years
    checkpoint_durability: str = "wal"        # WAL for crash recovery
    research_freshness_days: int = 7          # Re-scrape fundamentals after 7 days

    # Scheduling
    daily_scan_time: str = "09:00"            # IST
    market_open: str = "09:15"
    market_close: str = "15:30"
    data_refresh_interval_min: int = 5        # During market hours
```

---

## 8. LONG-RUNNING AGENT SYSTEM DESIGN

### 8.1 Why LangGraph for Long-Running Systems

| Requirement | LangGraph Solution |
|-------------|-------------------|
| **Durability** | Checkpointing after every node execution |
| **Crash Recovery** | Resume from last checkpointed node |
| **Cross-session memory** | Store API for cross-thread persistence |
| **Human-in-the-loop** | `interrupt()` + `Command(resume=)` |
| **Streaming visibility** | `stream_events` v3 for real-time progress |
| **Background tasks** | LangGraph + APScheduler for scheduled runs |
| **Error recovery** | RetryPolicy + conditional routing on failure |
| **Long-running analysis** | Async nodes, timeout handling, progress updates |

### 8.2 Checkpointing & Recovery Pattern

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# WAL mode for write durability
checkpointer = SqliteSaver.from_conn_string(
    "data/database/checkpoints.db",
    wal=True
)

graph = workflow.compile(checkpointer=checkpointer)

# Every run is identified by thread_id
config = {"configurable": {"thread_id": "daily-scan-2026-06-21"}}

# If crash mid-workflow, resume:
# graph.invoke(None, config)  # Resumes from last checkpoint
```

### 8.3 Memory Persistence Across Sessions

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()  # In production: SqliteStore or PostgresStore

# Research reports persist across sessions
store.put(("research_reports", "RELIANCE"), "report_2026_q1", {
    "date": "2026-06-21",
    "fundamentals_score": 0.72,
    "technical_score": 0.65,
    "recommendation": "BUY"
})

# Retrieval later:
report = store.get(("research_reports", "RELIANCE"), "report_2026_q1")
```

### 8.4 Scheduled Agent Lifecycle (Budget-Aware)

```
STARTUP (System boot — once):
  1. Worker process starts
  2. Connect to Kite SDK (OAuth flow if needed)
  3. Connect to LangGraph checkpointer (SQLite WAL)
  4. Load Store and Vector Store
  5. Read available cash from Kite SDK → this is the deployable pool
  6. Register scheduled jobs with APScheduler

DAILY CYCLE (every day):
  09:00 → 1. Check balance (Kite SDK: get_margins)
         2. IF available_cash < min_trade_amount:
              → Log "Cash depleted. Monitoring only."
              → Skip to monitoring, wait for user top-up
         3. Morning Scan workflow launches
         4. All agents run with available_cash injected into state
  09:15 → Market Data agent starts continuous feed
         → Every 5 min: price update → check alerts
         → If price alert triggers → Adrenaline workflow
  15:30 → Market closes → End-of-day analysis
  16:00 → Daily portfolio snapshot saved
  19:00 → Evening report: "Available: ₹X | Deployed: ₹Y | P&L: ₹Z"

WEEKLY (Sunday):
  → Full portfolio review
  → Watchlist refresh + new screener candidates
  → Pacing check: are we deploying too fast given the remaining cash?

MONTHLY (1st of month — guidance, not enforcement):
  → Review last month: deployments, P&L, win rate
  → Note: user may top up mid-month — system auto-detects cash increase
  → Deep fundamental review of all holdings
  → Rebalancing recommendations
  → Strategy performance review
  → Knowledge consolidation (vector store summary generation)
```

### 8.5 Error Handling & Resilience

```
ERROR TIERS (from LangGraph research):
  1. Transient (API timeout) → RetryPolicy(max_attempts=3, backoff=exponential)
  2. LLM-recoverable (bad parse) → Loop back with error in state, retry prompt
  3. User-fixable (auth expired) → interrupt() → notify user → wait for resume
  4. Recoverable after retries → error_handler() → fallback path
  5. Unexpected/critical → Bubble up, log, notify, graceful degradation

GRACEFUL DEGRADATION:
  - Kite SDK down → Switch to cached data, alert user
  - LLM API down → Use cached analysis, mark as "stale"
  - Web scraper blocked → Skip sentiment, note missing data
  - Database corrupted → Restore from backup, resume from last good checkpoint
```

---

## 9. RISK MANAGEMENT FRAMEWORK

### 9.1 Multi-Layer Risk Approach

```
PRE-TRADE CHECKS (Risk Manager Agent):
  ├── Position Size Check (< max_single_position_pct)
  ├── Sector Exposure Check (< max_sector_exposure_pct)
  ├── Liquidity Check (avg daily volume × position size)
  ├── Correlation Check (vs existing portfolio)
  ├── News Risk Check (earnings, events pending)
  └── Volatility Check (ATR, beta)

IN-TRADE MONITORING (Market Data Agent):
  ├── Stop-loss monitoring (price < stop_loss → alert/auto-sell)
  ├── Take-profit monitoring (price > target → recommend exit)
  └── Trailing stop updates

POST-TRADE (Daily):
  ├── Portfolio drawdown check
  ├── Sector allocation rebalancing alerts
  ├── Single-stock concentration alerts
  └── Overall VaR computation
```

### 9.2 Configurable Risk Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_single_position_pct` | 30% | Max of available cash in one stock |
| `max_sector_exposure_pct` | 30% | Max allocation to one sector |
| `default_stop_loss_pct` | -8% | Stop loss trigger |
| `trailing_stop_loss_pct` | -5% | From recent high |
| `portfolio_drawdown_halt_pct` | -10% | Halt new entries if weekly drawdown |
| `max_daily_trades` | 5 | Max trades per day |
| `min_market_cap_crore` | 500 | Minimum market cap filter (₹500 Cr) |
| `min_avg_volume_lakhs` | 5 | Minimum daily volume (₹5 Lakhs) |
| `available_cash` | From Kite SDK | Cash in Zerodha = deployable pool (read-only, no config) |

---

## 10. IMPLEMENTATION ROADMAP

### Current Status: Phase A-D Complete

| Phase | Status | Key Deliverables |
|-------|--------|-----------------|
| Phase A (Foundation) | ✅ Complete | LLM client (LiteLLM), memory (Store + ChromaDB + SQLite), yfinance data, checkpointing |
| Phase B (Core Agents) | ✅ Complete | All 10 agents, decision weights, risk checks (including CVaR + news risk), order management |
| Phase C (Polish) | ✅ Complete | Moat assessment, sector impact matrix, FRED API, NSE fallback, dashboard, macro narrative |
| Phase D (Agentic) | ✅ Complete | Supervisor LLM routing, multi-SMA 20/50/200, ROIC + FCF yield, HITL gate, auto-refresh |
| **Phase E (Research Depth)** | 🔵 Current | DCF Valuation Agent, Catalyst Engine, PM daily check loop, procedural memory |
| **Phase F (Autonomy)** | ⚪ Future | Agentic risk decisions, portfolio conviction tracking, Langfuse observability |

### Phase E: Research Depth (Current)

**Goal**: Professional-grade analysis — DCF, catalysts, daily thesis validation

| # | Item | Effort | Impact |
|---|------|--------|--------|
| E.1 | **DCF Valuation Agent** — Base/Upside/Downside, per-share price targets | Large | #1 professional gap closed |
| E.2 | **Catalyst Engine** — Extract + classify + alert catalysts (hard + soft) | Medium | Forward-looking, not just historical |
| E.3 | **PM Daily Check Loop** — Morning thesis validation for every holding | Medium | Daily guardrail, catches thesis breaks |
| E.4 | **Procedural Memory (Learning Journal)** — Write after each trade, retrieve before decisions | Medium | System gets smarter with every trade |

### Phase F: Autonomy (Future)

**Goal**: Full agentic decision-making with observability

| # | Item | Effort | Impact |
|---|------|--------|--------|
| F.1 | **Agentic Risk Assessment** — Risk Manager decides severity + action (not just pass/fail) | Medium | Adaptive risk, not binary |
| F.2 | **Portfolio-Level Conviction Tracking** — All positions ranked by conviction daily | Small | PM-level portfolio view |
| F.3 | **Langfuse Observability** — Per-agent token costs, latency, success rates | Small | System visibility, cost tracking |
| F.4 | **Integration Testing** — End-to-end autonomous run with 20-stock portfolio | Large | Validates full agentic pipeline
- [ ] Community: Share anonymized learnings (optional)
- [ ] AutFin MCP server: Expose as MCP server for other clients
- [ ] Mobile notifications integration

---

## 11. OPEN QUESTIONS & DECISIONS NEEDED

| # | Question | Options | Recommended |
|---|----------|---------|-------------|
| 1 | Monthly budget amount | Any ₹ amount | **Start with ₹30,000-50,000 (manageable for learning)** |
| 2 | Budget pacing style | Spread evenly, aggressive early, conservative | **Spread evenly (deploy ~1/20 of budget per trading day)** |
| 3 | LLM Provider (free) | OpenRouter + Groq + Gemini (free tiers) | **OpenRouter primary ($0), Groq secondary, $10 top-up optional** |
| 4 | Deep analysis model | Llama 3.3 70B:free, DeepSeek R1:free, Gemini 2.5 Flash | **Llama 3.3 70B:free (default), DeepSeek R1:free (reasoning)** |
| 5 | Fast/routing model | Gemma 4 26B:free, Llama 3.1 8B (Groq) | **Gemma 4 26B:free (routing), Llama 3.1 8B (extraction)** |
| 6 | Vector Store | ChromaDB, LanceDB, Qdrant | **ChromaDB (simplest, local-first)** |
| 5 | Checkpointer | SqliteSaver (current), PostgresSaver (future upgrade) | **SqliteSaver (zero-setup, WAL durable)** |
| 6 | Execution Mode | Recommendation only, auto-small, full-auto, paper | **Start: Recommendation only + paper trading** |
| 7 | How many stocks in budget? | 1-3 concentrated, 5-10 diversified | **Start: 3-5 (stays within budget, manageable)** |
| 8 | Trading strategy style | Momentum, value, swing, multi-strategy | **Multi-strategy (agents adapt to regime)** |
| 9 | Dashboard | Streamlit, Gradio, CLI-only initially | **CLI first, Streamlit in Phase 5** |
| 10 | Self-hosted LLM or API? | Local (Ollama/vLLM), cloud API | **Cloud API initially (quality), local later** |

---

## 12. KEY REFERENCES

### Research Documents (in workspace)
- `kite_connect_sdk_research.md` — Kite SDK: auth flow, all 30+ methods, 5 KiteTicker modes, order lifecycle, rate limits
- `free_tools_power_agentic_research.md` — Free LLM stack ($0/month), free market data APIs, free news/sentiment APIs
- `agentic_architecture_research.md` — Agentic AI patterns, memory systems, PM workflows, DCF + catalysts + variant views
- `stock_market_research.md` — Technical + fundamental analysis, due diligence checklists, decision frameworks
- `langgraph_research.md` — Graph API, 6 workflow patterns, 5 multi-agent patterns, HITL, persistence, Store API

### External References
- TradingAgents paper: Multi-Agent LLM Financial Trading Framework (Xiao et al., arXiv 2412.20138, 2024)
- Mem0: "State of AI Agent Memory 2026" — benchmarks, memory types, production gaps (mem0.ai, June 2026)
- QuantInsti: "Build an AI Agent Portfolio Manager" — LangGraph + Alpaca (2025)
- Couchbase: "Agentic Workflows vs AI Agents" distinction (2026)
- Alpha Theory: "Portfolio Optimization" — position sizing framework (2026)
- Financial Edge: "Day in the Life of a Portfolio Manager" (2026)
- Mergers & Inquisitions: Stock Pitch Guide + Equity Research Career Path (2026)
- Klymentiev: "Top Free LLMs" comparison (June 2026)
- PromptingGuide.ai: Agentic Design Patterns (2026)
- Kite SDK official: github.com/zerodha/pykiteconnect
- LangGraph official: docs.langchain.com/oss/python/langgraph

---

## 13. GLOSSARY

| Term | Definition |
|------|-----------|
| **AutFin** | Autonomous Finance — the system name |
| **Monthly Budget** | Deprecated concept — replaced by "available cash = budget" (Zerodha balance IS the budget) |
| **Deployable Cash** | Available cash from Kite SDK `get_margins()` — actual rupees in Zerodha ready to deploy |
| **Cash Buffer** | Percentage of available balance kept uninvested for averaging down / emergencies |
| **Balance-Is-Budget** | Core philosophy: no separate config parameter; the Zerodha account balance is the single source of truth |
| **KiteConnect** | Kite SDK — direct Python wrapper for Zerodha API |
| **LangGraph** | Stateful multi-agent orchestration framework |
| **Store API** | LangGraph cross-thread persistent memory |
| **Checkpointer** | LangGraph node-level execution state saver |
| **HITL** | Human-in-the-loop — pause for human approval |
| **VaR** | Value at Risk — maximum expected loss |
| **CVaR** | Conditional VaR — expected loss in worst-case scenarios |
| **DCF** | Discounted Cash Flow — intrinsic valuation model |
| **FCF** | Free Cash Flow — cash available after operations+investment |
| **GTT** | Good Till Triggered — Zerodha order type |
| **SMA** | Simple Moving Average — technical indicator |
| **RSI** | Relative Strength Index — momentum oscillator |
| **ADX** | Average Directional Index — trend strength indicator |
| **OAuth** | Open Authorization — secure delegated access protocol |

---

*This PRD is a living document. It should be updated as the system evolves and new discoveries are made through implementation. The architecture described here is the target state — phases will build incrementally toward it.*
