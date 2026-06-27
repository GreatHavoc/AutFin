# AutFin v5.0 — Improvement Plan

> **Date:** June 27, 2026
> **Base:** v4.2 (972+ tests, 10 reliability improvements done)
> **Sources:** 2026 production AI patterns, Indian market research, deep code analysis, academic papers

---

## Executive Summary

This plan identifies **50 improvements** across 9 categories, sourced from:
1. **2026 agentic AI patterns** — MIT/Google multi-agent research, MLflow production guide, LangChain state-of-agent-engineering
2. **Academic finance** — Caner-Capponi precision-matrix portfolio (2026), regime-aware trading, walk-forward backtesting
3. **Deep code analysis** — 45 concrete issues with file:line references
4. **Indian market specifics** — SEBI Apr 2026 algo rules, F&O lot sizes, tax engine, nsepython API

**Top 10 improvements by impact:**
1. Fix screen_candidates 60s sequential scan (supervisor_tools.py:293)
2. Fix researcher ignoring catalysts (researcher.py:346)
3. Wire peak_portfolio_value into check_drawdown (risk_manager.py:622)
4. Fix check_guardian hardcoded stop-loss (supervisor_tools.py:1308)
5. Add market-hours check to executor (executor.py:59)
6. Fix researcher nested/flat schema mismatch (researcher.py:286)
7. Add Indian tax engine (STCG/LTCG/XIRR)
8. Add vol-targeting overlay (risk_manager.py)
9. Add episodic memory (learning/)
10. Add regime detection (agents/global_macro.py)

---

## Improvement Inventory by Category

---

### A) RESEARCH QUALITY (10 items)

| # | Improvement | File | Issue | Effort | Impact |
|---|-------------|------|-------|--------|--------|
| A1 | Fix researcher ignoring catalysts | researcher.py:346 | Catalysts extracted but never weighted into conviction | 4h | HIGH |
| A2 | Fix researcher nested/flat schema mismatch | researcher.py:286 | Reads `technical_reports["reports"]` but writes `technical_reports[symbol]` — all technical signals lost for multi-symbol | 1h | HIGH |
| A3 | Fix researcher _self_audit conviction unbounded | researcher.py:1029 | LLM-returned conviction not clamped to [0,1] | 30min | MEDIUM |
| A4 | Fix researcher _extract_catalysts truncation | researcher.py:836 | JSON truncated mid-string, silently empties catalyst list | 1h | MEDIUM |
| A5 | Add LLM-as-Judge hallucination guard | NEW (learning/) | Verify answers grounded in retrieved chunks before returning to user | 4h | HIGH |
| A6 | Add query rewriting node | NEW (agents/) | 2-4 reformulations before retrieval — +15-20% recall on financial corpus | 3h | HIGH |
| A7 | Add parent-child chunking for RAG | NEW (memory/) | Small chunks for retrieval, full context for generation | 3h | MEDIUM |
| A8 | Add RAGAS nightly evaluation | NEW (observability/) | Faithfulness >0.9, Context Precision >0.85 | 4h | MEDIUM |
| A9 | Add catalyst lift to conviction calculation | researcher.py | Bias conviction by ±0.1 based on catalyst strength | 2h | MEDIUM |
| A10 | Fix market_data cache-on-error | market_data.py:354 | Failed fetches cached for 60s, causing phantom outages | 30min | HIGH |

---

### B) EXECUTION & TRADING (8 items)

| # | Improvement | File | Issue | Effort | Impact |
|---|-------------|------|-------|--------|--------|
| B1 | Add market-hours check to executor | executor.py:59 | Orders placed 24/7 without checking NSE hours | 2h | HIGH |
| B2 | Fix MARKET orders never fill in paper trading | engine.py:204 | Requires `price>0` for MARKET, but executor passes 0.0 | 1h | HIGH |
| B3 | Fix paper trade BUY pnl=0 silencing | engine.py:273 | BUY records show pnl=0, polluting learning loop | 2h | MEDIUM |
| B4 | Add spread model to paper trading | engine.py | Bid-ask spread (0.3-1%) more impactful than slippage for illiquid mid/smallcaps | 3h | MEDIUM |
| B5 | Add paper-live reconciliation job | NEW (paper_trading/) | Compare paper vs live positions nightly | 3h | MEDIUM |
| B6 | Fix executor exchange audit trail | executor.py:360 | Falls to BSE silently, audit shows NSE | 1h | LOW |
| B7 | Add market-hours-aware cache TTL | market_data.py:19 | 60s cache during market hours is stale for quotes | 1h | MEDIUM |
| B8 | Add SEBI Strategy ID tagging | executor.py | Required by Apr 2026 SEBI circular | 4h | HIGH |

---

### C) RISK MANAGEMENT (8 items)

| # | Improvement | File | Issue | Effort | Impact |
|---|-------------|------|-------|--------|--------|
| C1 | Wire peak_portfolio_value into check_drawdown | risk_manager.py:622 | peak never populated, drawdown circuit breaker always inactive | 2h | HIGH |
| C2 | Fix check_guardian hardcoded stop-loss | supervisor_tools.py:1308 | Hardcoded -15% instead of settings.default 8% | 30min | HIGH |
| C3 | Add real pairwise correlation matrix | risk_manager.py:381 | Currently sector-match only, not actual correlation | 4h | HIGH |
| C4 | Add vol-targeting overlay (10% annual) | NEW (risk_manager/) | Scale gross exposure by 0.10/realized_vol_60d | 3h | HIGH |
| C5 | Add regime-conditional covariance | NEW (agents/) | 2-state HMM on Nifty/VIX, per-regime Sigma | 4h | HIGH |
| C6 | Add Kelly criterion with regime gates | risk_manager.py | Fractional Kelly, gate by drawdown/regime/conviction | 3h | MEDIUM |
| C7 | Fix risk_score naive linear formula | risk_manager.py:655 | Ignores joint probability of correlated failures | 2h | MEDIUM |
| C8 | Add correlation-aware drawdown breaker | risk_manager.py | Monitor ρ̄(portfolio, Nifty50), de-gross when ρ̄ > 0.65 | 3h | HIGH |

---

### D) PORTFOLIO MANAGEMENT (7 items)

| # | Improvement | File | Issue | Effort | Impact |
|---|-------------|------|-------|--------|--------|
| D1 | Add Indian tax engine (STCG/LTCG/XIRR) | NEW (portfolio/) | No tax classification despite PRD promise | 6h | HIGH |
| D2 | Fix STCG/LTCG classification (trades table empty) | portfolio_manager.py:182 | Always returns STCG because trades table rarely populated | 2h | HIGH |
| D3 | Add tax-loss harvesting | portfolio_manager.py | PRD §2.3 goal #5 — not implemented | 6h | HIGH |
| D4 | Add LTCG gain harvesting (₹1.25L/yr zero tax) | portfolio_manager.py | Auto-redeem up to ₹1.25L LTCG per FY | 4h | HIGH |
| D5 | Fix sector map duplication (3 sources of truth) | portfolio_manager.py:32 | Duplicated in risk_manager, universe, portfolio_manager | 1h | MEDIUM |
| D6 | Add momentum-aware throttle | portfolio_manager.py:419 | No gating on recent price moves | 2h | MEDIUM |
| D7 | Fix check_news_risk blocks without alternative | risk_manager.py:574 | Blocks trade but suggests no alternative action | 1h | LOW |

---

### E) LEARNING & CALIBRATION (6 items)

| # | Improvement | File | Issue | Effort | Impact |
|---|-------------|------|-------|--------|--------|
| E1 | Add episodic memory (decisions → outcomes → lessons) | NEW (memory/) | No structured past-decision retrieval | 4h | HIGH |
| E2 | Add self-critique loop (weekly hindsight review) | NEW (learning/) | No feedback on whether same decision would be made today | 4h | MEDIUM |
| E3 | Add drift detection (embedding distribution shift) | NEW (learning/) | No monitoring of data/concept drift | 4h | MEDIUM |
| E4 | Add reasonability gate (reject >20% upside in 1wk) | supervisor_tools.py | LLM can recommend unrealistic targets | 2h | HIGH |
| E5 | Fix duplicate embedding implementations | supervisor_tools.py:43 & retriever.py:61 | Two copies of _generate_embedding and cosine_similarity | 2h | MEDIUM |
| E6 | Add _embedding_cache to avoid re-encoding | supervisor_tools.py:43 | Identical texts encoded on every retrieval | 1h | LOW |

---

### F) PERFORMANCE (6 items)

| # | Improvement | File | Issue | Effort | Impact |
|---|-------------|------|-------|--------|--------|
| F1 | Fix screen_candidates sequential scan (60s) | supervisor_tools.py:293 | `for sym in symbols` serial yfinance calls | 2h | HIGH |
| F2 | Fix 21 serial yf.Ticker calls in macro_agent | global_macro.py:357 | 21-60s per morning scan | 1h | HIGH |
| F3 | Fix market_data sequential price fetch | market_data.py:163 | Serial yfinance calls for holdings | 1h | MEDIUM |
| F4 | Fix research fan-out NOT parallel for standard depth | supervisor_tools.py:541 | Sequential awaits, not asyncio.gather | 1h | MEDIUM |
| F5 | Add LLM response caching (diskcache) | llm/client.py | Same prompts re-hit API every run | 4h | MEDIUM |
| F6 | Add sentence-transformers batching | supervisor_tools.py:43 | Single-text encoding, no batch | 2h | LOW |

---

### G) DATA & MACRO (5 items)

| # | Improvement | File | Issue | Effort | Impact |
|---|-------------|------|-------|--------|--------|
| G1 | Fix global_macro hardcoded sector weights | supervisor_tools.py:1472 | Ignores real compute_sector_impact() output | 1h | HIGH |
| G2 | Fix global_macro serial FRED fetches | global_macro.py:81 | Only 2 obs, 24h cache too long for active indicators | 1h | MEDIUM |
| G3 | Add rolling correlation computation | global_macro.py:444 | Hardcoded rules, not real historical correlation | 3h | HIGH |
| G4 | Fix sentiment_agent 5-symbol hard limit | sentiment.py:327 | Only processes 5 holdings, defaults to RELIANCE | 1h | MEDIUM |
| G5 | Add nsepython for FII/DII daily flows | NEW (agents/) | Structural domestic flow data not captured | 2h | MEDIUM |

---

### H) OBSERVABILITY (4 items)

| # | Improvement | File | Issue | Effort | Impact |
|---|-------------|------|-------|--------|--------|
| H1 | Add runs table (per-invocation trace) | NEW (database.py) | No structured per-run log | 2h | HIGH |
| H2 | Add cost_inr tracking per run | observability/ | Approximate cost, no INR tracking | 2h | MEDIUM |
| H3 | Add rolling z-score anomaly detection | NEW (observability/) | No automated drift/performance alerts | 4h | MEDIUM |
| H4 | Add LangSmith free tier integration | NEW (observability/) | 5K traces/mo free, full node audit | 2h | MEDIUM |

---

### I) TESTING (2 items)

| # | Improvement | File | Issue | Effort | Impact |
|---|-------------|------|-------|--------|--------|
| I1 | Add E2E integration test with real LLM | NEW (tests/) | All tests mocked, no real LLM validation | 4h | HIGH |
| I2 | Add walk-forward backtest harness | NEW (backtest/) | Single-pass backtest kills ~75% of edges | 6h | HIGH |

---

## Execution Plan — Priority Waves

### Wave 1: Critical Bugs (do first, ~12h)

| # | Item | Why |
|---|------|-----|
| C2 | Fix hardcoded stop-loss | Wrong risk enforcement — settings say 8%, code says -15% |
| B2 | Fix MARKET orders never fill | Paper engine silently broken for real order flow |
| A2 | Fix researcher nested/flat schema | All technical signals lost for multi-symbol analysis |
| A10 | Fix cache-on-error | 60s phantom outages after transient failures |
| F1 | Fix screen_candidates sequential | 60s morning scan |
| F2 | Fix macro_agent serial fetches | 21-60s per morning scan |
| F4 | Fix research fan-out NOT parallel | Standard depth doesn't use asyncio.gather |
| A3 | Fix conviction unbounded | LLM can return conviction >1.0 |

### Wave 2: Architecture (~20h)

| # | Item | Why |
|---|------|-----|
| C1 | Wire peak_portfolio_value | Drawdown circuit breaker always inactive |
| C3 | Add real pairwise correlation | Sector-match only, not actual correlation |
| G1 | Fix macro sector weights | Ignores real compute_sector_impact output |
| A1 | Fix researcher ignoring catalysts | Pillar 4 of research completely lost |
| B1 | Add market-hours check | Orders placed 24/7 |
| D2 | Fix STCG/LTCG classification | Always returns STCG |
| B3 | Fix paper trade BUY pnl=0 | Learning loop polluted |
| C4 | Add vol-targeting overlay | Next portfolio-construction win |
| E5 | Fix duplicate embeddings | Two model instances, two copies of state |

### Wave 3: New Capabilities (~30h)

| # | Item | Why |
|---|------|-----|
| D1 | Indian tax engine | Day-one feature, STCG/LTCG/XIRR |
| D3 | Tax-loss harvesting | PRD goal, material tax savings |
| D4 | LTCG gain harvesting | ₹1.25L/yr zero tax |
| E1 | Episodic memory | "Last time you bought INFY after 8% drop..." |
| A5 | LLM-as-Judge guard | ~0.1% hallucination rate vs 3-5% |
| E4 | Reasonability gate | Block unrealistic recommendations |
| G3 | Rolling correlation | Not hardcoded rules |
| C8 | Correlation-aware breaker | ρ̄ > 0.65 → de-gross |
| E2 | Self-critique loop | Weekly hindsight review |

### Wave 4: Production Polish (~25h)

| # | Item | Why |
|---|------|-----|
| I2 | Walk-forward backtest | Single-pass kills 75% of edges |
| H1 | Runs table | Per-invocation structured trace |
| B5 | Paper-live reconciliation | Drift detection |
| C5 | Regime-conditional covariance | Per-regime Sigma |
| B8 | SEBI Strategy ID tagging | Required by Apr 2026 |
| A8 | RAGAS nightly eval | Automated quality gate |
| C6 | Kelly criterion with gates | Fractional + regime-aware |
| H3 | Rolling z-score anomalies | Automated drift alerts |
| B4 | Spread model for paper | Mid/smallcap accuracy |

---

## Implementation Rules

1. **Every improvement gets a test** — no change ships without test coverage
2. **Bug fixes first** — Wave 1 before Wave 2
3. **Deterministic code for arithmetic** — LLMs for reasoning, Python for numbers
4. **Single source of truth** — no duplicate sector maps, embedding functions
5. **Backward compatible** — existing tests must still pass
6. **Indian tax compliance** — SEC 111A/112A rules hardcoded, not LLM-guessed

---

## Key Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Multi-agent | Stay single supervisor + DRA specialists | MIT 2026: multi-agent degrades sequential tasks 39-70% |
| Portfolio optimization | Ledoit-Wolf shrinkage (deterministic) | Caner-Capponi 2026: precision-matrix beats Markowitz when screening is stochastic |
| Vol targeting | 10% annualized, scale by realized vol | QuantPedia validated, free to implement |
| Memory | Episodic (SQLite) + Semantic (SQLite FTS5) | Personal scale, no Mem0/Zep needed |
| Tax engine | Deterministic Python, not LLM | SEC rules are complex, must be exact |
| Backtesting | Walk-forward + deflated Sharpe | Kryptera 2026: single-pass backtest kills 75% of edges |
| RAG | Hybrid BM25+vector → rerank → judge | Lushbinary 2026: faithfulness >0.9 |

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/autfin/memory/embeddings.py` | Centralized embedding service |
| `src/autfin/memory/episodic.py` | Episodic memory (decisions→outcomes→lessons) |
| `src/autfin/learning/critique.py` | Self-critique weekly hindsight loop |
| `src/autfin/learning/drift.py` | Data/concept drift detection |
| `src/autfin/risk/vol_targeting.py` | Volatility targeting overlay |
| `src/autfin/risk/regime.py` | Regime detection (HMM 2-state) |
| `src/autfin/risk/kelly.py` | Kelly criterion with regime gates |
| `src/autfin/portfolio/tax_engine.py` | Indian tax (STCG/LTCG/XIRR) |
| `src/autfin/portfolio/harvesting.py` | Tax-loss and LTCG gain harvesting |
| `src/autfin/observability/runs.py` | Per-invocation structured trace |
| `src/autfin/observability/anomaly.py` | Rolling z-score anomaly detection |
| `src/autfin/agents/judge.py` | LLM-as-Judge hallucination guard |
| `src/autfin/agents/rewriter.py` | Query rewriting node |
| `tests/test_risk/test_vol_targeting.py` | Vol targeting tests |
| `tests/test_portfolio/test_tax_engine.py` | Tax engine tests |
| `tests/test_learning/test_episodic.py` | Episodic memory tests |
| `tests/test_learning/test_drift.py` | Drift detection tests |

## Files to Modify

| File | Changes |
|------|---------|
| `src/autfin/tools/supervisor_tools.py` | Parallel screen_candidates, fix sector filter, fix guardian, add reasonability gate, centralize embeddings |
| `src/autfin/agents/researcher.py` | Wire catalysts, fix schema mismatch, clamp conviction |
| `src/autfin/agents/risk_manager.py` | Real correlation, peak portfolio value, vol targeting |
| `src/autfin/agents/portfolio_manager.py` | Tax engine, harvesting, momentum throttle |
| `src/autfin/agents/global_macro.py` | Serial→parallel fetch, rolling correlation |
| `src/autfin/agents/sentiment.py` | Increase 5-symbol limit |
| `src/autfin/agents/executor.py` | Market hours check, SEBI Strategy ID |
| `src/autfin/paper_trading/engine.py` | Fix MARKET fill, spread model, BUY pnl |
| `src/autfin/memory/retrieval.py` | Episodic memory integration |
| `src/autfin/learning/calibrator.py` | Regime-aware calibration |
| `src/autfin/llm/client.py` | Add LLM response cache |
| `src/autfin/database.py` | Add runs table |

---

## Effort Summary

| Wave | Items | Hours | Focus |
|------|-------|-------|-------|
| Wave 1: Critical bugs | 8 | 12 | Stop-loss, fill bugs, serial scans |
| Wave 2: Architecture | 9 | 20 | Peak portfolio, correlation, macro |
| Wave 3: New capabilities | 9 | 30 | Tax engine, episodic memory, judge |
| Wave 4: Production polish | 9 | 25 | Walk-forward, SEBI, regime, Kelly |
| **Total** | **35** | **~87h** | |

---

*Analysis completed June 27, 2026*
*Sources: MIT 2026 multi-agent research, Caner-Capponi precision-matrix paper (2026), MLflow production guide, LangChain state-of-agent-engineering, SEBI Apr 2026 circular, Indian tax rules FY26-27, nsepython/Kite APIs*
