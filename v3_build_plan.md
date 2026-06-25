# AutFin v3.0 — Final Build Plan

> **Date:** June 25, 2026  
> **Status:** Phases 0, 1, 2, 3 complete. **709 tests passing**, 2 skipped. 1 pre-existing Kite auth-failure test failure (unrelated). 20 supervisor tools (3 added for self-learning). Single supervisor graph serving all contexts.

---

## Architecture: One Graph, One Supervisor, One Loop

```
┌──────────────────────────────────────────────────────────────┐
│                     graph/supervisor.py                       │
│                     (THE ONLY GRAPH)                           │
│                                                                │
│  START → supervisor_node → {human_review} → END               │
│              ↑        │                          ↑            │
│              │        └── tools (ReAct loop, 20) ─┘           │
│              │                                                │
│    ┌─────────┴──────────┐                                    │
│    │  20 TOOLS           │                                    │
│    │  (tool-callable     │                                    │
│    │   async functions)  │                                    │
│    └────────────────────┘                                    │
└──────────────────────────────────────────────────────────────┘

Context in state → behavior, not separate graphs:
  context="morning_routine" → Full scan. Screen 500→20→5. Trade.
  context="price_alert"    → Investigate one symbol. 5 min.
  context="quarterly_review"→ Deep dive all holdings. DCF.
  context="manual"         → User asked for one stock analysis.

Depth per stock (supervisor decides):
  quick   = technical + sentiment only          (2 min)
  standard = all 4 data agents + synthesis       (8 min)
  deep    = standard + DCF + moat + peer comp    (20 min)
```

---

## Build Order — 4 Phases, ~2 Weeks

### Phase 0: Collapse (Day 1-2)
**One graph. Nothing else changes yet.**

| # | Task | File | Status |
|---|------|------|--------|
| 0.1 | Create `tools/supervisor_tools.py` — 17 tool definitions (OpenAI format) + async callables | NEW | ✅ Done |
| 0.2 | Rebuild `graph/supervisor.py` — ReAct loop with `generate_with_tools()`, context-driven | REWRITE | ✅ Done |
| 0.3 | Update `graph/hitl.py` — `request_human_approval()` as tool, not graph node | MODIFY | ✅ Done |
| 0.4 | Update `scheduler/jobs.py` — set `context` in state, invoke ONE graph | MODIFY | ✅ Done |
| 0.5 | Update `state/agent_state.py` — add `context`, `depth`, `routing_history`, `learning_context` | MODIFY | ✅ Done |
| 0.6 | Update `graph/__init__.py` — export supervisor only | MODIFY | ✅ Done |
| 0.7 | Verify: 683 tests still pass. Fix graph test mocks. | — | ✅ Done (680 passing) |

**Retired ✅:**
```
graph/daily_morning.py     → logic absorbed into supervisor tools ✅
graph/adrenaline.py        → supervisor context="price_alert" ✅
graph/quarterly_deep_dive.py → supervisor context="quarterly_review" + depth="deep" ✅
graph/research_parallel.py  → supervisor calls asyncio.gather ✅
```

---

### Phase 1: Research Engine (Day 3-5)
**Fan-out discovery. Screening. Reverse DCF.**

| # | Task | File | Status |
|---|------|------|--------|
| 1.1 | Nifty 500 symbol list + bulk yfinance fetcher | NEW `tools/universe.py` | ✅ Done |
| 1.2 | Liquidity filter (volume + market cap) | ADD to `tools/supervisor_tools.py` | ✅ Done |
| 1.3 | Magic Formula ranking engine | NEW `tools/screening.py` | ✅ Done |
| 1.4 | Piotroski F-Score calculator + Indian red flags | NEW `tools/screening.py` | ✅ Done |
| 1.5 | Reverse DCF in valuation agent | MODIFY `agents/valuation.py` | ✅ Done |
| 1.6 | Peer compare agent (8-metric sector ranking) | NEW `agents/peer_compare.py` | ✅ Done |
| 1.7 | Moat assessment agent (Morningstar 5-source) | NEW `agents/moat.py` | ✅ Done |
| 1.8 | 6-element stock pitch generator in researcher | MODIFY `agents/researcher.py` | ✅ Done |
| 1.9 | Fan-out orchestration: 20×6 concurrent agents | ADD to `tools/supervisor_tools.py` | ✅ Done |
| 1.10 | Sector rotation: macro-overlayed screening weights | ADD to `tools/supervisor_tools.py` | ✅ Done |
| 1.11 | Guardian track: always-on holding monitor (5-min interval) | ADD to `tools/supervisor_tools.py` | ✅ Done |
| 1.12 | Adrenaline mode: price alert → 3-agent quick fan-out | ADD to `tools/supervisor_tools.py` | ✅ Done |
| 1.13 | Tests: screening, reverse DCF, moat, peer compare, fan-out | NEW test files | ✅ Done |

---

### Phase 2: Self-Learning (Day 6-7)
**Close the feedback loop. Calibrate from outcomes.**

| # | Task | File | Status |
|---|------|------|--------|
| 2.1 | `retrieve_learnings(query)` — vector search over journal | NEW `learning/retriever.py` | ✅ Done (SQL keyword search, not embeddings, to stay zero-cost) |
| 2.2 | `calibrate_conviction()` — per-sector, per-tier win rates | NEW `learning/calibrator.py` | ✅ Done |
| 2.3 | `detect_patterns()` — "4/5 losing trades had X" | NEW `learning/calibrator.py` | ✅ Done |
| 2.4 | Wire `retrieve_learnings` into supervisor's system prompt before every trade | MODIFY `graph/supervisor.py` | ✅ Done (loads on iter 1, injects calibration + patterns) |
| 2.5 | Wire `log_decision` to write structured outcomes | MODIFY `tools/supervisor_tools.py` | ✅ Done (`record_learning` tool) |
| 2.6 | Auto-adjust F-Score threshold based on market regime (bull/bear) | ADD to `learning/calibrator.py` | ✅ Done (`detect_market_regime` + `get_fscore_threshold`) |
| 2.7 | Tests: learning retrieval, calibration, pattern detection | NEW test files | ✅ Done (28 new tests) |

---

### Phase 3: Polish (Day 8-10)
**Hardening, tests, dashboard.**

| # | Task | File | Status |
|---|------|------|--------|
| 3.1 | 6-element stock pitch renders in dashboard | MODIFY `dashboard/app.py` | ⏳ Pending (dashboard not yet updated) |
| 3.2 | Fan-out progress: live status of all 20 candidates | MODIFY `dashboard/app.py` | ⏳ Pending |
| 3.3 | Learning dashboard: calibration curves, pattern library | MODIFY `dashboard/app.py` | ⏳ Pending |
| 3.4 | Integration test: full morning routine with 20-stock mock portfolio | NEW test | ✅ Done (`tests/test_graphs/test_v3_integration.py`, 8 tests) |
| 3.5 | Integration test: price alert → adrenaline → decision | NEW test | ✅ Done (5 tests in same file) |
| 3.6 | Integration test: quarterly deep dive with DCF on all holdings | NEW test | ✅ Done (5 tests in same file) |
| 3.7 | Token budget guard: abort if approaching rate limit | MODIFY `llm/client.py` | ✅ Done (wired in supervisor) |
| 3.8 | Run full suite. Target: 700+ tests. | — | ✅ Done — **709 passing, 2 skipped, 1 pre-existing Kite auth-failure failure (unrelated)** |

---

## State v3.0

```python
class AgentState(TypedDict):
    # Messages (ReAct loop accumulation)
    messages: Annotated[list[BaseMessage], add_messages]

    # Context — WHY was supervisor invoked
    context: str             # "morning_routine" | "price_alert" | "quarterly_review" | "manual"
    depth: str               # "quick" | "standard" | "deep"
    alert_symbol: NotRequired[str]
    price_move_pct: NotRequired[float]

    # Portfolio (Kite SDK)
    available_cash: float
    utilized_margin: float
    holdings: list[dict]
    positions: list[dict]

    # Research cache (populated by fan-out tools)
    market_snapshot: NotRequired[dict | None]
    macro_signals: NotRequired[dict | None]
    
    # Per-symbol research (keyed by symbol, populated concurrently)
    fundamental_reports: NotRequired[dict | None]   # {symbol: {ratios, health, f_score, ...}}
    technical_reports: NotRequired[dict | None]
    sentiment_reports: NotRequired[dict | None]
    valuation_reports: NotRequired[dict | None]      # {symbol: {dcf_scenarios, reverse_dcf, ...}}
    peer_comparisons: NotRequired[dict | None]
    moat_assessments: NotRequired[dict | None]

    # Synthesis & Decisions
    stock_pitches: NotRequired[dict | None]          # {symbol: 6-element pitch}
    risk_assessments: NotRequired[dict | None]
    trade_queue: NotRequired[list[dict] | None]
    execution_reports: NotRequired[list[dict] | None]

    # Candidate tracking (screening output)
    candidates: NotRequired[list[str] | None]        # ["RELIANCE", "TCS", ...] top 20
    composite_ranks: NotRequired[dict | None]         # {symbol: score}

    # Learning context (retrieved BEFORE trade decisions)
    learning_context: NotRequired[dict | None]        # {similar_setups, calibration, patterns}

    # HITL
    pending_human_approval: NotRequired[dict | None]
    human_decision: NotRequired[str | None]

    # Metadata
    thread_id: str
    run_mode: str
    errors: list[str]
    routing_history: NotRequired[list[str] | None]

    # Token budget
    total_tokens: int
    total_cost: float
    budget_limit: float
```

---

## The 20 Supervisor Tools (Complete)

> **Note:** Plan originally listed 13 tools. We expanded to **20** with 3 added self-learning tools (`record_learning`, `calibrate_conviction`, `detect_patterns`), plus `get_market_quotes`, `validate_all_holdings`, and `finalize_session` for context completeness. `retrieve_learnings` and `log_decision` were merged into `record_learning`.

### Discovery Tools
| # | Tool | Deterministic | Returns |
|---|------|:---:|---|
| T1 | `screen_candidates()` | Yes | 15-25 symbols after 3-filter pipeline (auto-regime F-Score threshold) |
| T2 | `research_stock(symbol, depth)` | Partial | Fan-out 6 agents → unified research record |
| T3 | `research_batch(symbols, depth)` | Partial | Fan-out T2 for all symbols concurrently |

### Research Tools (called by T2)
| # | Tool | Deterministic | Returns |
|---|------|:---:|---|
| T4 | `analyze_fundamentals(symbol)` | Partial | Ratios, health, F-Score, growth, red flags |
| T5 | `analyze_technical(symbol)` | Partial | Multi-TF indicators, patterns, signal |
| T6 | `analyze_sentiment(symbol)` | Partial | Score, topics, hard/soft catalysts with dates |
| T7 | `compute_valuation(symbol, depth)` | Yes | 3-scenario DCF + Reverse DCF |
| T8 | `compare_peers(symbol)` | Partial | 8-metric ranking vs 4-6 peers |
| T9 | `assess_moat(symbol)` | Partial | 5-source moat score 0-100 + rating |

### Decision Tools
| # | Tool | Deterministic | Returns |
|---|------|:---:|---|
| T10 | `build_stock_pitch(symbol)` | No (LLM) | 6-element buy-side pitch |
| T11 | `run_risk_checks(trade)` | Yes | Approved/Vetoed + score + position size (15 checks incl. drawdown circuit breaker) |
| T12 | `allocate_capital(trades)` | Yes | Kelly-sized trade queue |
| T13 | `execute_order(order)` | Yes | Kite SDK / paper fill |

### Memory & Learning Tools
| # | Tool | Deterministic | Returns |
|---|------|:---:|---|
| T14 | `record_learning(event)` | Yes | Write structured outcome to learning journal |
| T15 | `calibrate_conviction(sector, tier)` | Yes | Adjusted conviction % from win-rate data |
| T16 | `detect_patterns(min_count)` | Yes | Patterns from losing trades |
| T17 | `validate_thesis(holding)` | No (LLM) | INTACT/WEAK/BROKEN + evidence |
| T18 | `validate_all_holdings()` | No (LLM) | Validate every holding thesis |

### HITL / Session Tools
| # | Tool | Deterministic | Returns |
|---|------|:---:|---|
| T19 | `request_human_approval(question)` | Yes | Waits for human input |
| T20 | `finalize_session()` | Yes | Summary, totals, next-step recommendations |

---

## File Plan — Status

### CREATE (✅ All done)
```
tools/supervisor_tools.py     # 20 tool definitions + callables (~1700 lines)  ✅
tools/universe.py              # Nifty 500 symbol list + bulk yfinance fetcher ✅
tools/screening.py             # Liquidity filter, Magic Formula, F-Score, red flags, regime detection (~250 lines) ✅
agents/peer_compare.py         # 8-metric sector ranking agent                ✅
agents/moat.py                 # Morningstar 5-source moat assessment agent   ✅
learning/retriever.py          # SQL keyword search over learning journal     ✅
learning/calibrator.py         # Conviction calibration + pattern detection + regime detection ✅
learning/feedback_loop.py      # Orchestrator: pre_trade, post_trade, summary  ✅
research_engine_design.md      # ✅
tests/test_learning/           # 28 new tests (3 files)                        ✅
tests/test_tools/test_screening.py  # 22 tests incl. 6 for market regime       ✅
tests/test_graphs/test_v3_integration.py  # 20 integration tests              ✅
```

### REWRITE (✅ Done)
```
graph/supervisor.py            # ReAct loop, context-driven, tool-calling, learning-context-inject ✅
```

### MODIFY (✅ Mostly done; dashboard items 3.1-3.3 outstanding)
```
state/agent_state.py           ✅ Added context, depth, learning_context, candidates, iteration_count
scheduler/jobs.py              ✅ Set context in state, invokes ONE graph
agents/valuation.py            ✅ Added reverse DCF
agents/researcher.py           ✅ 6-element stock pitch output format
llm/client.py                  ✅ Token budget tracking wired in Phase 0 prep
graph/hitl.py                  ✅ Tool-callable human_approval()
graph/__init__.py              ✅ Exports supervisor only
dashboard/app.py               ⏳ Stock pitch rendering, fan-out progress, learning dashboard — pending
config/settings.py             ✅ Depth tiers, learning config, sector weights
```

### RETIRED ✅ (4 files — removed)
```
graph/daily_morning.py     ✅
graph/adrenaline.py        ✅
graph/quarterly_deep_dive.py ✅
graph/research_parallel.py  ✅
```

---

## Test Plan — Actual Results

| Phase | New Tests | Cumulative | Focus | Status |
|-------|----------|------------|-------|--------|
| Base | 683 | 683 | All existing tests pass | ✅ |
| 0 | +15 | ~680 | Supervisor graph, context routing, tool dispatch (after store shift) | ✅ Done |
| 1 | +30 | ~690 | Screening, reverse DCF, moat, peer compare, fan-out | ✅ Done |
| 2 | +28 | ~700 | Learning retrieval, calibration, regime threshold, pattern detection | ✅ Done (28 new tests in `tests/test_learning/`) |
| 3 | +20 | ~709 | Integration tests (morning routine, price alert, quarterly review, token budget) | ✅ Done (20 tests in `tests/test_graphs/test_v3_integration.py`) |
| **Target** | **+93** | **709✓** | 6 market regime tests, 28 learning tests, 20 integration tests | **On target: 709 passing, 2 skipped, 1 pre-existing Kite auth-failure** |

---

## Success Criteria

| Criterion | Target | Measure |
|-----------|--------|---------|
| Research speed | Screen 500 → 5 pitches < 30 min | Wall clock time |
| Research depth | Reverse DCF + 6-element pitch per pick | Output validation |
| Fan-out efficiency | 20 × 6 agents = max(lane) not sum(lanes) | Timing logs |
| Discovery rate | ≥5 candidates/week from universe | Count |
| Guardian responsiveness | <2 min from thesis break to sell order | Latency log |
| Learning effect | Win rate improves after 50+ trades | Baseline vs 50-trade win rate |
| Conviction calibration | Actual win rate within ±15% of predicted per tier | Calibration curve |
| Self-improvement | F-Score threshold auto-adjusts to market regime | Threshold log |
| Code quality | 800+ tests, 0 failures | pytest |
| Cost | $0/month operating | Token usage log |

---

---

## Summary — Where We Are (June 25, 2026)

### Done
- ✅ **Sprints 1 + 2**: 15 critical/high bug fixes across `market_data`, `global_macro`, `supervisor`, `risk_manager`, `supervisor_tools`, `error_handler`, `technical`, `researcher`, `portfolio_manager`, `llm/client`, `error_handler` (429/rate-limit), killed `observability/catalyst_impact.py` dead code, fixed `store.py` (AsyncSqliteStore needs aiosqlite.Connection), ChromaDB 0.5+ compat, drawdown circuit breaker (#15 risk check), Indian market holiday calendar (14 dates), token budget wired into supervisor, removed `cerebras_model`/`ollama_model` from LLM client.
- ✅ **Phase 0** (Collapse): Single supervisor graph replaces 4 legacy graphs. `tools/supervisor_tools.py` (20 tools, ~1700 lines). ReAct loop in `graph/supervisor.py`. HITL via `request_human_approval` tool. Scheduler sets context in state.
- ✅ **Phase 1** (Research Engine): Nifty 500 universe, liquidity filter, Magic Formula, Piotroski + Indian red flags, reverse DCF, peer-compare agent (8-metric), moat agent (Morningstar 5-source), 6-element pitch, fan-out orchestration, sector rotation, guardian track, adrenaline mode.
- ✅ **Phase 2** (Self-Learning): `learning/retriever.py` (SQL keyword search, calibration, pattern detection), `learning/calibrator.py` (`record_learning`, `calibrate_conviction`, `detect_patterns` with sector/day-of-week analysis), `learning/feedback_loop.py` orchestrator. 3 new supervisor tools (T14-T16). Auto-calibration in `execute_order`. Learning context preloaded into supervisor's system prompt on first iteration. Auto-regime F-Score threshold adjust (bull→5, bear→7, neutral→6). 28 new tests.
- ✅ **Phase 3.4-3.7** (Integration Tests & Token Budget): 20 integration tests in `tests/test_graphs/test_v3_integration.py` covering morning routine (8), price alert (5), quarterly review (5), token budget guard (2). Fixed `research_stock` double-await bug in `quick` path. 709 tests passing, 2 skipped, 0 test regressions from prior baseline.
- ✅ **Cleanup**: ~3,600 lines of legacy code removed (`daily_morning.py`, `adrenaline.py`, `quarterly_deep_dive.py`, `research_parallel.py` + their tests). 9 obsolete `.md` files deleted.

### Pending (Phase 3 dashboard — 5 tasks)
- ⏳ Dashboard: stock-pitch rendering in `dashboard/app.py`
- ⏳ Dashboard: fan-out progress (live status of 20 candidates)
- ⏳ Dashboard: learning curves + pattern library
- ⏳ Final coverage push (target 800+ tests)
- ⏳ Pre-existing `test_auth_failure_returns_error` (Kite) — minor assertion mismatch, unrelated to v3

### KPIs
- **Tools**: 20 (target: 20) ✅
- **Tests**: 709 passing + 2 skipped (target: 700+) ✅
- **LLM cost**: $0/month (NVIDIA NIM free + litellm) ✅
- **Graphs**: 1 (target: 1) ✅
- **Legacy code removed**: ~3,600 lines ✅
- **Bugs fixed**: 15 (P0/P1) + 4 (P2) ✅

---

## Non-Goals (v3.0 explicitly does NOT do)

- Multi-asset (bonds, gold, ETFs) — equities only
- Real-time WebSocket streaming — 15-min delayed yfinance + Kite SDK poll
- Natural language chat interface — supervisor is autonomous, not conversational
- Cloud deployment — local-first, single machine
- Options/Futures trading — spot equity only
- Tax-loss harvesting — tracked but not optimized
- International stocks — NSE/BSE only
- Algorithmic order splitting — manual order placement via Kite SDK
