# AutFin v5.4 Implementation Plan

> **Goal:** Fix speed bottleneck (5-7 LLM calls → 2-4), add cross-domain correlation
> intelligence, add price-distortion detection, and properly design the Supervisor's role.

---

## Part A — What Changed and Why

### The Problem (v5.3)

Every "agent tool" in `agent_tools.py` wraps a **deterministic Python agent** (FundamentalsAgent, TechnicalAgent, etc.) with an **extra LLM call** that re-summarizes already-structured JSON output. This is pure overhead:

```
Supervisor LLM call #1 (routing)
  → fundamentals_tool  → FundamentalsAgent.analyze() [deterministic, 3s]
                       → LLM re-summarizes the JSON [10-20s]  ← WASTE
  → technical_tool     → TechnicalAgent.analyze() [deterministic, 2s]
                       → LLM re-summarizes the JSON [10-20s]  ← WASTE
  → sentiment_tool     → SentimentAgent (FinBERT) [deterministic, 5s]
                       → LLM re-summarizes [10-20s]           ← WASTE
Supervisor LLM call #2 (continues reasoning)
  ... more LLM calls ...
Supervisor LLM call #N (final)
```

**Result:** 5-7 LLM calls per analysis = 60-120s wall time. Most of the LLM calls are redundant.

### The Fix (v5.4)

Remove the inner LLM wrappers from agent tools. Call deterministic agents directly. Keep LLM only where it genuinely adds value (routing, synthesis, narrative, debate).

```
Supervisor LLM call #1 (routing — structured output)
  → fundamentals_analyst → FundamentalsAgent.analyze() [deterministic, 3s]
  → technical_analyst    → TechnicalAgent.analyze()    [deterministic, 2s]
  → sentiment_analyst    → SentimentAgent (FinBERT)    [deterministic, 5s]
  → macro_analyst        → GlobalMacroAgent            [deterministic, 3s]
  → valuation_analyst    → ValuationAgent.compute_dcf() [deterministic, 2s]
  (all 5 run in parallel via asyncio.gather — ~5s total)
Supervisor LLM call #2 (synthesis + pitch — only if intent = invest/trade)
  → build_stock_pitch / research_synthesis (Bull/Bear debate)
```

**Result:** 2-3 LLM calls = 20-40s wall time. **3x faster.**

---

## Part B — Architecture (v5.4)

### Agent Inventory — What Needs LLM vs What Doesn't

#### Tier 1: Deterministic Analysts (NO LLM — parallel via asyncio.gather)

| Agent | What it actually does | Speed | LLM? |
|-------|----------------------|-------|------|
| `FundamentalsAgent.analyze()` | yfinance scrape → calculate ratios → Piotroski F-score → moat assessment | ~3s | NO |
| `TechnicalAgent.analyze()` | OHLCV DataFrame → pandas_ta_classic indicators (RSI, MACD, BB, ATR, Stochastic) → signal | ~2s | NO |
| `SentimentAgent.analyze_news()` + `aggregate_sentiment()` | Marketaux news → FinBERT scoring → weighted aggregation. Reddit PRAW included | ~5s | NO |
| `GlobalMacroAgent.analyze_impact()` + `fetch_indicators()` | yfinance macro tickers → deterministic India impact scoring → sector tilt | ~3s | NO |
| `ValuationAgent` (module) | `compute_dcf()`, `reverse_dcf()`, `build_scenarios()`, `compute_wacc()` | ~2s | NO |
| `market_data_agent()` | Kite SDK: balance_fetcher + price_fetcher (LTP via yfinance fallback) | ~1s | NO |
| `RiskManagerAgent` | 14 deterministic risk checks + `apply_veto()` | ~0.5s | NO |
| `DistortionAnalyst` (NEW) | Price-distortion detection: fake breakouts, pump patterns, distribution | ~2s | NO |
| `RegulatoryAnalyst` (NEW) | SEBI/RBI/DPCO announcement scanning + classification | ~3s | NO |

#### Tier 2: LLM Tools (genuinely need reasoning)

| Tool | Why LLM is needed | When invoked |
|------|-------------------|-------------|
| `supervisor_decide(state)` | Routing decision — which agents, what depth, for which symbols | Every session, 1 call |
| `research_synthesis(symbols)` | Bull/Bear debate — adversarial multi-round reasoning | Only when intent = invest/trade |
| `build_stock_pitch(symbol, reports)` | 6-element buy-side pitch narrative | Only on final report |
| `validate_thesis(symbol, holding)` | Qualitative thesis validation against current state | Only for existing holdings |
| `risk_narrative(checks)` (optional) | Prose explanation of risk verdict | Only when user requests narrative |

#### Tier 3: Supervisor Meta-Work (LLM or hybrid)

| Job | Nature | Frequency |
|-----|--------|-----------|
| Routing decision | 1 LLM call, structured output | Every session |
| State management (read/write) | Deterministic state manipulation | Every iteration |
| Failure recovery & circuit breaking | Deterministic rules | On failure |
| Context building for downstream | Deterministic + optional LLM narrative | Per symbol |
| Cache invalidation (macro regime change) | Deterministic trigger detection | On macro change |
| Memory & belief updates | Deterministic write + optional LLM summary | End of session |
| Watchlist curation triggers | Deterministic (time/price/news/catalyst) | Continuous |

### New Components

#### 1. Watchlist Curator (`universe/curator.py`)

Deterministic Python class. Runs BEFORE supervisor. Produces `state["watchlist"]`.

Triggers that ADD to watchlist:
- Holdings + manual user watchlist (Kite SDK + user input)
- Pre-earnings within 14 days (NSE/BSE calendar)
- Large moves >3% intraday (LTP scan)
- News-driven first-time mention (Marketaux velocity)
- Macro shock → broaden to ALL holdings
- New IPO within 30 days (NSE/BSE IPO feed)
- Earnings surprise >5% EPS delta (Screener.in)
- Top movers by volume (NSE data)

#### 2. Distortion Analyst (`agents/distortion.py`)

Deterministic Python. 5-8 small detection functions:

| Detector | What it catches | Key signals |
|----------|----------------|-------------|
| `detect_fake_breakout()` | Price breaks 20D high without volume confirmation | Price vs volume-price correlation |
| `detect_distribution()` | Near 52w high + OBV falling + MFI diverging | Institutional selling into retail buying |
| `detect_pump_pattern()` | Volume spike >5x avg + no news + social chatter spike | Coordinated buying signal |
| `detect_circuit_suspicion()` | Upper circuit hit multiple sessions without catalyst | Thin-float momentum trading |
| `detect_sector_relative_weakness()` | Stock up 3% when sector up 7% | Underperformance despite rally |
| `check_promoter_pledge()` | Promoter pledge % increased | Distribution / stress signal |
| `check_bulk_deals()` | NSE bulk/block deal filings | Insider selling visible |
| `check_index_reweight()` | Index composition change driving mechanical flows | Not organic growth |

Each returns `{flag: bool, severity: "low"/"medium"/"high", reason: str}`.
Composite output: conviction adjustment `-0.1` per active flag.

#### 3. Regulatory Analyst (`agents/regulatory.py`)

Deterministic Python + optional LLM classification for ambiguous cases.

Sources:
- NSE corporate announcements API
- SEBI orders page (scrape via Firecrawl)
- RBI circulars page
- DPCO price notifications (pharma)

Classification:
- `sector_impact`: which sectors affected
- `severity`: low / medium / high / critical
- `direction`: positive / negative / neutral
- `urgency`: immediate / 1-day / 1-week

#### 4. FII/DII Data Fetcher

Single function in `tools/scraper_tools.py`:
```python
async def fetch_fii_dii_daily() -> dict:
    """Fetch daily FII/DII provisional data from NSE."""
```

Adds to macro_signals. Used by supervisor for:
- FII net-negative 5 days → invalidate small/mid-cap theses
- DII buying > FII selling → support signal
- FII/DII divergence → regime detection input

#### 5. News Velocity Analyzer

Extends `sentiment_tools.py`:
- Compare news rate vs sector baseline (30-day average)
- Detect low-karma social signals (Reddit)
- Flag "manufactured news flow" (4+ positive news in 2 hours, no social chatter)

---

## Part C — Implementation Steps

### Step 1: Dict-merge reducers for AgentState (BLOCKING FIRST)

**File:** `src/autfin/state/agent_state.py`

**Problem:** 12+ dict-typed fields (`fundamental_reports`, `technical_reports`, etc.) have NO reducer. Parallel writes via `Send` fan-out silently lose data.

**Fix:** Add `Annotated[dict | None, dict_merge_reducer]` to all dict-typed report fields.

**Fields to add reducer:**
- `fundamental_reports`
- `technical_reports`
- `sentiment_reports`
- `valuation_reports`
- `peer_comparisons`
- `moat_assessments`
- `research_debates`
- `stock_pitches`
- `risk_assessments`
- `thesis_check_results`
- `catalyst_track`
- `catalyst_impact_report`
- `research_reports`

**Behavior:** `{SYM1: {...}}` written by analyst 1 + `{SYM2: {...}}` written by analyst 2 → merged dict with both keys.

---

### Step 2: Strip LLM from agent_tools.py (BIGGEST SPEED WIN)

**File:** `src/autfin/agents/agent_tools.py`

**Change:** For each of the 7 agent tool wrappers, replace the inner `llm.generate_with_tools(...)` + JSON parser loop with a direct call to the underlying deterministic agent.

**Before (fundamentals_tool):**
```python
raw = await llm.generate_with_tools(
    system_prompt=system_prompt,
    tools=FUNDAMENTALS_TOOLS,
    tool_callables=FUNDAMENTALS_CALLABLES,
    model=llm.deep_model,
    max_tokens=4096,
    max_iterations=15,
)
result = extract_json(raw)
```

**After (fundamentals_analyst):**
```python
agent = FundamentalsAgent()
report = await agent.analyze(symbol)
if include_narrative:
    narrative = await agent.generate_llm_narrative(symbol, report)
    report["llm_narrative"] = narrative
return json.dumps(report, default=str)
```

**Apply to all 7:**
1. `fundamentals_tool` → `fundamentals_analyst` — calls `FundamentalsAgent.analyze()` directly
2. `technical_tool` → `technical_analyst` — calls `TechnicalAgent.analyze()` directly
3. `sentiment_tool` → `sentiment_analyst` — calls `SentimentAgent.analyze_news()` + `aggregate_sentiment()`
4. `macro_tool` → `macro_analyst` — calls `GlobalMacroAgent.fetch_indicators()` + `analyze_impact()`
5. `portfolio_tool` → `portfolio_state` — calls `market_data_agent()` directly
6. `risk_tool` → split into `run_risk_checks` (deterministic, already exists) + optional `risk_narrative` (LLM)
7. `research_tool` → KEEP as LLM (Bull/Bear debate is genuinely LLM-value-added)

**Add flag:** `include_narrative: bool = False` on all tool wrappers. Default fast path skips LLM. Supervisor prompt can request `include_narrative=True` when it wants prose for the final pitch.

**Keep `@_traceable` decorators** so LangSmith still traces them (now they show as fast `chain` runs, no LLM cost).

---

### Step 3: Add distortion_analyst tool

**New files:**
- `src/autfin/agents/distortion.py` — 5-8 detector functions
- Add tool wrapper in `agent_tools.py`

**Detector implementations:**

```python
# agents/distortion.py

def detect_fake_breakout(price_data: pd.DataFrame, volume_data: pd.Series) -> dict:
    """Price breaks 20D high + volume < 1.2x avg = suspicious breakout."""
    ...

def detect_distribution(price_data: pd.DataFrame, obv: pd.Series, mfi: pd.Series) -> dict:
    """Near 52w high + OBV falling + MFI diverging = institutional distribution."""
    ...

def detect_pump_pattern(volume_data: pd.Series, social_velocity: float, has_news: bool) -> dict:
    """Volume spike >5x avg + no news + social chatter spike = pump."""
    ...

def detect_circuit_suspicion(price_data: pd.DataFrame, circuit_hits: int) -> dict:
    """Upper circuit hit multiple sessions without catalyst = thin-float momentum."""
    ...

def detect_sector_relative_weakness(stock_return: float, sector_return: float) -> dict:
    """Stock up 3% when sector up 7% = laggard."""
    ...

def check_promoter_pledge(current_pledge_pct: float, prior_pledge_pct: float) -> dict:
    """Promoter pledge % increased = distribution/stress signal."""
    ...

def check_bulk_deals(symbol: str) -> dict:
    """NSE bulk/block deal filings — insider selling visible."""
    ...

def distortion_audit(symbol: str, depth: str = "standard") -> dict:
    """Run all distortion detectors, return composite result."""
    ...
```

**Tool wrapper in `agent_tools.py`:**
```python
@_traceable(name="distortion_audit", run_type="chain", ...)
async def distortion_tool(symbol: str, depth: str = "standard") -> str:
    """Run price-distortion detection on a stock."""
    from src.autfin.agents.distortion import distortion_audit
    result = await distortion_audit(symbol, depth)
    return json.dumps(result, default=str)
```

**Add to AGENT_TOOL_DEFINITIONS and AGENT_TOOL_CALLABLES.**

---

### Step 4: Add regulatory_analyst tool

**New file:** `src/autfin/agents/regulatory.py`

```python
class RegulatoryAnalyst:
    async def scan_announcements(self, symbol: str = None) -> dict:
        """Scan NSE/BSE corporate announcements for regulatory actions."""
        ...

    async def classify_impact(self, announcement: dict) -> dict:
        """Classify regulatory announcement impact (deterministic rules + optional LLM)."""
        ...

    async def analyze(self, symbol: str) -> dict:
        """Full regulatory analysis for a symbol."""
        ...
```

**Tool wrapper in `agent_tools.py`:**
```python
@_traceable(name="regulatory_analysis", run_type="chain", ...)
async def regulatory_tool(symbol: str) -> str:
    from src.autfin.agents.regulatory import RegulatoryAnalyst
    analyst = RegulatoryAnalyst()
    result = await analyst.analyze(symbol)
    return json.dumps(result, default=str)
```

---

### Step 5: Add FII/DII data fetcher

**File:** `src/autfin/tools/scraper_tools.py`

```python
async def fetch_fii_dii_daily() -> dict:
    """Fetch daily FII/DII provisional buy/sell data from NSE.

    Returns:
        {
            "date": "2026-07-01",
            "fii_buy": float, "fii_sell": float, "fii_net": float,
            "dii_buy": float, "dii_sell": float, "dii_net": float,
            "fii_net_5d_avg": float, "dii_net_5d_avg": float,
            "signal": "risk_on" | "risk_off" | "neutral"
        }
    """
```

**Wire into:** `GlobalMacroAgent.fetch_indicators()` as a new indicator.
**Used by:** Supervisor for macro-regime invalidation triggers.

---

### Step 6: Add news velocity analyzer

**File:** `src/autfin/tools/sentiment_tools.py` (extend)

```python
async def analyze_news_velocity(symbol: str, hours: int = 24) -> dict:
    """Compare news rate vs sector baseline + detect anomalies.

    Returns:
        {
            "news_count_24h": int,
            "sector_avg_24h": float,
            "velocity_ratio": float,  # news_count / sector_avg
            "social_velocity_ratio": float,  # reddit mentions / baseline
            "has_news_catalyst": bool,
            "anomaly_flags": ["high_velocity_no_catalyst", "low_karma_surge"],
            "signal": "normal" | "elevated" | "suspicious"
        }
    """
```

**Wire into:** Distortion analyst's `detect_pump_pattern()` as a dependency.

---

### Step 7: Add Watchlist Curator

**New file:** `src/autfin/universe/curator.py`

```python
class WatchlistCurator:
    """Deterministic watchlist builder. Runs BEFORE supervisor."""

    async def curate(self, state: AgentState) -> list[str]:
        """Build today's watchlist from multiple sources.

        Sources:
        1. Holdings (from Kite SDK)
        2. Manual watchlist (from user input / CLI)
        3. Pre-earnings (within 14 days, NSE calendar)
        4. Large moves >3% (LTP scan)
        5. News-driven first-time mention (Marketaux velocity)
        6. Macro shock → broaden to ALL holdings
        7. New IPO within 30 days
        8. Earnings surprise >5% EPS delta
        9. Top movers by volume (NSE data)
        """
        ...
```

**Add to AgentState:**
```python
watchlist: list[str]  # populated by Watchlist Curator, consumed by Supervisor
```

**Wire into graph:** `WatchlistCurator` node runs before supervisor node.

---

### Step 7b: Add Stock Discovery Engine (`universe/discovery.py`)

**Problem:** The `WatchlistCurator` only works with **known universe** stocks (Nifty 500, holdings, manual watchlist). It cannot find a stock it's never heard of before.

**Example scenarios the system must handle:**
- A new IPO lists on NSE — system should detect it and evaluate
- A stock is trending on Reddit/News but isn't in any index
- A small-cap stock reports a massive earnings surprise
- A stock is being accumulated by FIIs (visible in bulk deals)
- A sector is rallying and there are laggard stocks not in universe
- An insider buying signal appears on a stock not tracked

**Solution:** A **Stock Discovery Engine** that runs BEFORE the Watchlist Curator. It finds NEW symbols from external signals, validates them (is it real? is it liquid?), and feeds them into the watchlist.

```python
# universe/discovery.py

class StockDiscoveryEngine:
    """Discovers NEW stocks not in the existing universe.

    Unlike WatchlistCurator (which works with known stocks),
    this engine finds stocks the system has never seen before.
    """

    async def discover(self, state: AgentState) -> list[dict]:
        """Run all discovery sources, return candidate stocks.

        Returns:
            [
                {
                    "symbol": "NEWSTOCK",
                    "source": "news" | "social" | "ipo" | "earnings" |
                              "sector_laggard" | "insider_buy" | "index_inclusion",
                    "confidence": float,  # 0.0-1.0
                    "reason": str,  # why this stock was discovered
                    "priority": "high" | "medium" | "low",
                },
                ...
            ]
        """
        candidates = []

        # Source 1: News-driven discovery
        news_discoveries = await self._discover_from_news()
        candidates.extend(news_discoveries)

        # Source 2: Social-driven discovery (Reddit, Twitter)
        social_discoveries = await self._discover_from_social()
        candidates.extend(social_discoveries)

        # Source 3: IPO discovery (NSE/BSE new listings)
        ipo_discoveries = await self._discover_ipos()
        candidates.extend(ipo_discoveries)

        # Source 4: Earnings surprise discovery
        earnings_discoveries = await self._discover_earnings_surprises()
        candidates.extend(earnings_discoveries)

        # Source 5: Sector momentum laggards
        sector_discoveries = await self._discover_sector_laggards()
        candidates.extend(sector_discoveries)

        # Source 6: Insider buying signals (bulk/block deals)
        insider_discoveries = await self._discover_insider_signals()
        candidates.extend(insider_discoveries)

        # Source 7: Index inclusion announcements
        index_discoveries = await self._discover_index_inclusions()
        candidates.extend(index_discoveries)

        # Deduplicate and rank
        return self._rank_and_dedupe(candidates)
```

**Discovery Sources — How Each Works:**

| Source | How it finds NEW stocks | Signal quality |
|--------|------------------------|----------------|
| **News discovery** | Marketaux returns `symbol` field for each news article. If symbol is NOT in Nifty 500 + NOT in holdings + NOT in manual watchlist → it's a NEW stock. Filter: news must be within 24h, must have positive/negative sentiment (not neutral). | High |
| **Social discovery** | Reddit Indian finance subs (r/IndianStreetBets, r/IndianStockMarket, r/StockMarketIndia). Extract stock mentions via regex (NSE symbol patterns). If mentioned 3+ times in 24h by different users AND not in universe → NEW stock. | Medium |
| **IPO discovery** | NSE corporate announcements API — filter for "listing" / "IPO" type. BSE new listings page. Scrape via Firecrawl. Filter: must be equity (not bonds/ETFs), must be listed within 30 days. | High |
| **Earnings surprise** | Screener.in — scan for EPS > consensus by >5% OR EPS < consensus by >10%. If symbol not in universe → NEW stock with earnings catalyst. | High |
| **Sector laggard** | When sector is up >3% in a day, find stocks in that sector that are up <1% (laggards). Filter: must have avg volume >₹1Cr, must not be suspended/delisted. | Medium |
| **Insider buying** | NSE bulk/block deal filings — if a bulk deal is a BUY (not sell) on a stock not in universe → potential accumulation signal. Filter: deal value >₹1Cr. | High |
| **Index inclusion** | NSE announces quarterly re-balancing of Nifty indices. Stocks being ADDED to Nifty 500/100/Next 50 → mechanical buying pressure. Scrape NSE announcements. | Very High |

**Validation Layer (before adding to watchlist):**

Every discovered stock must pass a lightweight validation:
```python
async def _validate_candidate(self, symbol: str) -> bool:
    """Check if a discovered stock is worth analyzing.

    Returns False if:
    - Stock is suspended / delisted
    - Avg daily volume < ₹50 Lakhs (illiquid)
    - Stock is a shell company / penny stock (< ₹10)
    - Stock is in a restricted sector (tobacco, gambling — user preference)
    - Stock was already analyzed in last 7 days (avoid re-work)
    """
```

**Integration with Watchlist Curator:**

```python
# In WatchlistCurator.curate():
async def curate(self, state: AgentState) -> list[str]:
    # Step 1: Known universe (existing logic)
    known = self._get_known_stocks(state)

    # Step 2: Discovery engine (NEW)
    discovery = StockDiscoveryEngine()
    discovered = await discovery.discover(state)

    # Step 3: Merge — discovered stocks get "discovered" tag
    watchlist = known
    for d in discovered:
        if d["symbol"] not in watchlist:
            watchlist.append(d["symbol"])
            # Tag for supervisor: "this is a NEW discovery, do quick analysis first"
            state.setdefault("discovery_tags", {})[d["symbol"]] = {
                "source": d["source"],
                "confidence": d["confidence"],
                "reason": d["reason"],
            }

    return watchlist
```

**How Supervisor handles discovered stocks:**

The supervisor sees `state["discovery_tags"]` and adjusts behavior:
- **High confidence** (IPO, index inclusion, earnings surprise) → run `quick` depth analysis (fundamentals + technical + sentiment, no DCF)
- **Medium confidence** (social mention, sector laggard) → run `quick` depth on fundamentals + sentiment only
- **Low confidence** (news mention without catalyst) → add to watchlist but don't analyze yet; wait for second signal

This is the answer to "how does it find stocks it's never heard of?" — the Discovery Engine is the **scout** that explores outside the known universe.

---

### Step 8: Refactor supervisor routing

**File:** `src/autfin/graph/supervisor.py`

**Change:** Replace the long ReAct loop (`generate_with_tools` with 20 iterations) with a structured-output LLM call.

**Before:**
```python
result = await llm.generate_with_tools(
    system_prompt=system_prompt,
    tools=all_tools,
    tool_callables=all_callables,
    model=llm.deep_model,
    max_tokens=4096,
    max_iterations=20,
    state=state,
)
```

**After:**
```python
# One structured-output call
decision = await llm.generate(
    prompt=build_routing_prompt(state),
    system_prompt=SUPERVISOR_DECISION_PROMPT,
    model=llm.deep_model,
    max_tokens=2048,
    response_format=SupervisorDecision,
)
```

**SupervisorDecision schema:**
```python
class SupervisorDecision(TypedDict):
    route: str  # "morning_routine" | "price_alert" | "quarterly_review" | "manual"
    analysis_targets: list[dict]  # [{symbol, depth, agents: ["fundamentals","technical",...]}]
    include_narrative: bool  # whether to request LLM narrative on final reports
    generate_pitch: bool  # whether to run build_stock_pitch
    escalate_to_human: list[dict]  # trades needing HITL
    cache_invalidation: list[str]  # symbols whose cached reports are stale
```

**After parsing the decision**, the graph uses deterministic nodes + `asyncio.gather` to fan out to the requested agents.

---

### Step 9: Add valuation_analyst tool wrapper

**File:** `src/autfin/agents/agent_tools.py`

**Change:** Wrap existing `valuation` module functions as a tool.

```python
@_traceable(name="valuation_analysis", run_type="chain", ...)
async def valuation_tool(symbol: str, depth: str = "standard") -> str:
    from src.autfin.agents.valuation import fetch_financials, compute_dcf, reverse_dcf, build_scenarios
    fin_data = await fetch_financials(symbol)
    scenarios = build_scenarios(fin_data)
    # ... compute DCF for each scenario ...
    return json.dumps(result, default=str)
```

**Add to AGENT_TOOL_DEFINITIONS and AGENT_TOOL_CALLABLES.**

---

### Step 10: Update supervisor prompt for v5.4

**File:** `src/autfin/graph/supervisor.py` — `_build_context_prompt()`

**Changes:**
1. Update tool list (now 9 deterministic analysts + 3 LLM tools)
2. Update strategy section with new batching patterns
3. Add distortion_analyst to morning routine flow
4. Add regulatory_analyst to quarterly review flow
5. Add cache invalidation instructions

---

### Step 11: Add state fields for new components

**File:** `src/autfin/state/agent_state.py`

**New fields:**
```python
# v5.4 additions
watchlist: NotRequired[list[str]]  # populated by Watchlist Curator
distortion_reports: NotRequired[dict | None]  # distortion audit results
regulatory_reports: NotRequired[dict | None]  # regulatory analysis results
fii_dii_data: NotRequired[dict | None]  # FII/DII daily data
cache_invalidation_tokens: NotRequired[list[str]]  # supervisor's invalidation directives
news_velocity_reports: NotRequired[dict | None]  # news velocity analysis
```

---

### Step 12: Tests

**New test files:**
- `tests/test_graphs/test_v5_4_integration.py` — new tool tests, distortion tests, regulatory tests
- `tests/test_agents/test_distortion.py` — distortion detector unit tests
- `tests/test_agents/test_regulatory.py` — regulatory analyst tests

**Update existing:**
- `tests/test_graphs/test_supervisor.py` — update for new tool count
- `tests/test_graphs/test_v5_3_integration.py` — update for renamed tools

---

### Step 13: Benchmark

Before/after wall-time on `morning_routine` for 3 symbols.

| Metric | v5.3 | v5.4 Target |
|--------|------|-------------|
| LLM calls per morning routine | 5-7 | 2-3 |
| Wall time (3 symbols) | 60-120s | 20-40s |
| Token usage per run | ~100k | ~15k |
| Modal vLLM seconds | 60-120s | 20-40s |

---

## Part D — Sequencing

| Step | What | Depends on | Risk |
|------|------|-----------|------|
| 1 | Dict-merge reducers | Nothing | Low — behavior change but fixes silent bug |
| 2 | Strip LLM from agent_tools | Step 1 | Low — biggest speed win, most code already deterministic |
| 3 | Distortion analyst | Nothing | Low — new file, no existing code disturbed |
| 4 | Regulatory analyst | Nothing | Medium — NSE scraping may need iteration |
| 5 | FII/DII fetcher | Nothing | Low — single function |
| 6 | News velocity analyzer | Nothing | Low — extends existing sentiment_tools |
| 7 | Watchlist Curator | Step 1 (needs `watchlist` state field) | Low — new file |
| 8 | Supervisor refactor | Steps 1+2 | Medium — largest code change |
| 9 | Valuation tool wrapper | Nothing | Low — existing module, just needs wrapper |
| 10 | Update supervisor prompt | Steps 2-9 | Low — prompt text change |
| 11 | Add state fields | Step 1 | Low — additive only |
| 12 | Tests | Steps 1-11 | Low |
| 13 | Benchmark | Step 12 | None |

**Recommended commit sequence:**
1. Steps 1+2+11+12 (speed fix — biggest impact, commit together)
2. Steps 3+5+6+12 (intelligence additions — distortion + FII + news velocity)
3. Steps 4+7+12 (regulatory + watchlist curator)
4. Steps 8+9+10+12+13 (supervisor refactor + valuation + benchmark)

---

## Part E — Supervisor's Role (Why It Still Needs LLM)

Even with deterministic sub-agents, the Supervisor is the ONLY place that does:

1. **Routing decisions** — which agents for this context, what depth, which symbols
2. **State management** — reads all `*_reports`, writes `messages[]`, `trade_queue`, `stock_pitches`, `composite_ranks`
3. **Failure recovery** — retry vs escalate, continue vs halt, which partial result is good enough
4. **Context building** — 6-element pitch, composite scoring, conviction → Kelly sizing
5. **Conflict resolution** — Technical says BUY, Fundamentals says SELL → Supervisor picks
6. **Cache invalidation** — macro regime change → invalidate stale sub-agent reports
7. **Memory & belief updates** — cross-session learning, conviction calibration

**The Supervisor's LLM call is the one place where ambiguous, multi-source, conflicting information becomes a single decision.** Deterministic agents provide the facts; the Supervisor provides the judgment.

---

## Part F — New Agent Architecture (Visual)

```
                       USER REQUEST / SCHEDULED TRIGGER
                            │
                            ▼
                ┌───────────────────────┐
                │  WATCHLIST CURATOR    │  ◄── deterministic Python
                │  (runs BEFORE super)  │      produces state["watchlist"]
                │  Holdings + triggers  │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │   SUPERVISOR NODE     │  ◄── single LLM call (1)
                │   structured output   │      routing + synthesis decision
                │   SupervisorDecision  │
                └───────────┬───────────┘
                            │
        ┌───────────────────┼─────────────────────────────────┐
        │           ROUTER (pure conditional, no LLM)        │
        │  Based on route: morning / alert / manual / review  │
        └───┬─────┬─────┬─────┬─────┬─────┬─────────────────┘
            │     │     │     │     │     │
            ▼     ▼     ▼     ▼     ▼     ▼
        ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐
        │FUNDAM││TECH  ││SENTI ││MACRO ││VALUA ││DISTOR│
        │DET.  ││DET.  ││DET.  ││DET.  ││DET.  ││DET.  │
        └──┬───┘└──┬───┘└──┬───┘└──┬───┘└──┬───┘└──┬───┘
           │       │       │       │       │       │
           └───────┴───────┴───┬───┴───────┴───────┘
                               ▼
                    ┌─────────────────────┐
                    │  REGULATORY (DET.)  │  ◄── SEBI/RBI/DPCO scan
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  RESEARCHER (LLM)   │  ◄── LLM call (2)
                    │  Bull + Bear Debate │      only if intent = invest
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  RISK GATE (DET.)   │  ◄── 14 deterministic rules
                    │  veto → loop back   │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  COMPOSE_PITCH (LLM)│  ◄── LLM call (3)
                    │  6-element pitch    │      only on final report
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  DISTORTION AUDIT   │  ◄── deterministic
                    │  conviction adj.    │      -0.1 per active flag
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  EXECUTOR (DET.)    │  ◄── tax-aware sizing + order
                    └──────────┬──────────┘
                               ▼
                             END
```

**LLM calls:** 2-3 per session (supervisor routing + optional pitch + optional debate)
**Deterministic calls:** 6-9 per session (parallel via asyncio.gather)
**Wall time target:** 20-40s (vs 60-120s in v5.3)
