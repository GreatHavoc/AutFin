# AutFin Research Engine — v5.5 Design

> **Date:** July 2026
> **Status:** Complete. Tools architecture implemented.
> **Sources:** Firecrawl research across Hebbia, Daloopa, Morningstar, New Constructs, Quant-Investing, Anand Rathi, Zerodha Varsity, Medium (Multi-Agent Fan-Out), arXiv (Autonomous Systematic Factor Investing)

---

## 1. Core Insight: Research Power = Differentiated Insight × Conviction × Actionability

**Weak research** says "TCS has PE 28, ROE 43%, strong buy." Everyone can compute that. Zero edge.

**Powerful research** says "TCS's stock price implies 8% revenue growth. But BFSI is entering a ₹40,000 Cr upgrade cycle from RBI's digital banking mandate. Combined with 3 new $100M+ cloud deals, actual growth will be 13-15%. The market is underpricing by 40%. Target ₹4,800 (22% upside). Catalyst: Q2 earnings on Oct 10."

Same stock. Completely different analysis. The difference is **layer depth** and **variant perception**.

---

## 2. The Fan-Out Architecture — Speed Through Parallelism

### Why Fan-Out?

A real hedge fund research team runs like this:
- Screening team scans 500+ stocks → 20 candidates (1 hour)
- 6 sector analysts each independently research ~3-4 stocks (2 hours each, parallel)
- PM receives 20 research summaries, selects 5 for deep dive (30 min)
- 2 senior analysts build full stock pitches for top 5 (1 hour each, parallel)

**Total: 3.5 hours for 500 → 5 fully-researched pitches.**

AutFin can do this in 25 minutes because:
- Screening is automated math (60 seconds)
- Research agents run in parallel (20 minutes for ALL 20, not 20 minutes each)
- Stock pitch generation is LLM-driven (5 minutes for all 5)

```
─────────────────────────────────────────────────────────────────
  PHASE 1              PHASE 2                PHASE 3
  SCREEN               FAN-OUT                CONVERGE
  (60 sec)             (20 min)               (5 min)
─────────────────────────────────────────────────────────────────

  Nifty 500         ┌─ fundamentals ─┐
      │              │  (10 min)     │
      ▼              │               │
  Filter 1:         ├─ technical ────┤
  Volume > ₹1Cr     │  (3 min)      │      ┌──────────────┐
  MktCap > ₹500Cr   │               │      │  SUPERVISOR  │
      │              ├─ sentiment ───┤      │  ranks by:   │
      ▼              │  (5 min)      │      │  • conviction│
  420 stocks         │               ├─────▶│  • risk/reward│
      │              ├─ DCF (deep) ──┤      │  • catalyst  │
      ▼              │  (20 min) ← slowest│  │  urgency    │
  Filter 2:          │               │      └──────┬───────┘
  Magic Formula      ├─ peer compare─┤              │
  (EY + ROIC rank)   │  (8 min)      │      ┌───────▼───────┐
      │              │               │      │  TOP 5 PITCHES│
      ▼              └─ moat assess ─┘      │  with full     │
  85 stocks                                  │  6-element    │
      │                                      │  stock pitch  │
      ▼                                      └───────────────┘
  Filter 3:
  F-Score ≥ 6
  Promoter pledge < 10%
  No governance flags
      │
      ▼
  20 candidates → FAN OUT: 20 × 6 agents = 120 analyses concurrently
```

### The Math

| Phase | Sequential Time | Fan-Out Time | Speedup |
|-------|----------------|-------------|---------|
| Screen 500 | 60 sec | 60 sec | 1× |
| Research 20 stocks × 6 agents | 8.5 hours | 20 min | **24×** |
| Build 5 pitches | 40 min | 15 min | 2.7× |
| **Total** | **~10 hours** | **~25 min** | **24×** |

### Amdahl's Law Applied

The total time is bounded by the **slowest agent lane** (DCF valuation at 20 min for deep analysis, or 5 minutes for standard). The key insight from the Multi-Agent Fan-Out research: "the orchestrator can refuse to wait for stragglers." If DCF takes 20 minutes but the other 5 agents finish in 10, the supervisor can proceed with preliminary ranking and refine when DCF arrives.

---

## 3. Phase 1: Screening Pipeline

### Filter 1: Liquidity Gate

```python
def liquidity_filter(stocks: list[dict]) -> list[dict]:
    return [
        s for s in stocks
        if s["avg_daily_volume"] >= 1_00_00_000  # ₹1 Cr minimum
        and s["market_cap"] >= 500_00_00_000      # ₹500 Cr minimum
    ]
```

Eliminates micro-caps and illiquid stocks. Source: yfinance `fast_info.marketCap`, `history(5d).Volume.mean()`.

### Filter 2: Magic Formula (Joel Greenblatt)

```python
def magic_formula_rank(stocks: list[dict]) -> list[dict]:
    """Rank by combined Earnings Yield and ROIC. Take top 85."""
    for s in stocks:
        ebit = s["ebit"]
        ev = s["market_cap"] + s["total_debt"] - s["cash"]
        nfa = s["net_fixed_assets"]
        wc = s["current_assets"] - s["current_liabilities"]
        
        s["earnings_yield"] = ebit / ev if ev > 0 else 0
        s["roic"] = ebit / (nfa + wc) if (nfa + wc) > 0 else 0
    
    # Rank separately, combine
    ey_rank = {s["symbol"]: i for i, s in enumerate(
        sorted(stocks, key=lambda x: x["earnings_yield"], reverse=True))}
    roic_rank = {s["symbol"]: i for i, s in enumerate(
        sorted(stocks, key=lambda x: x["roic"], reverse=True))}
    
    for s in stocks:
        s["magic_rank"] = ey_rank[s["symbol"]] + roic_rank[s["symbol"]]
    
    return sorted(stocks, key=lambda x: x["magic_rank"])[:85]
```

Back-tested: Outperforms market by 4-6% CAGR over 20+ years. Finds cheap, high-quality businesses simultaneously.

### Filter 3: Piotroski F-Score + Indian Red Flags

```python
def piotroski_f_score(current: dict, prior: dict) -> int:
    """9-point financial health checklist. Score ≥ 6 = improving."""
    score = 0
    
    # PROFITABILITY
    score += 1 if current["roa"] > 0 else 0                                    # 1
    score += 1 if current["operating_cf"] / current["total_assets"] > 0 else 0 # 2
    score += 1 if current["roa"] > prior["roa"] else 0                         # 3
    cfo_roa = current["operating_cf"] / current["total_assets"]
    score += 1 if cfo_roa > current["roa"] else 0                              # 4 (accrual quality)
    
    # FINANCIAL HEALTH
    lev = current["long_term_debt"] / current["total_assets"]
    lev_prior = prior["long_term_debt"] / prior["total_assets"]
    score += 1 if lev < lev_prior else 0                                       # 5 (deleveraging)
    cr = current["current_assets"] / current["current_liabilities"]
    cr_prior = prior["current_assets"] / prior["current_liabilities"]
    score += 1 if cr > cr_prior else 0                                         # 6 (liquidity)
    score += 1 if current["shares"] <= prior["shares"] else 0                  # 7 (no dilution)
    
    # EFFICIENCY
    gm = current["gross_profit"] / current["revenue"]
    gm_prior = prior["gross_profit"] / prior["revenue"]
    score += 1 if gm > gm_prior else 0                                         # 8 (margin expansion)
    at = current["revenue"] / current["total_assets"]
    at_prior = prior["revenue"] / prior["total_assets"]
    score += 1 if at > at_prior else 0                                         # 9 (efficiency)
    
    return score  # 0-9

def indian_red_flags(stock: dict) -> list[str]:
    """Indian market-specific governance checks. Any hit = auto-reject."""
    flags = []
    if stock.get("promoter_holding_change_4q", 0) < -0.05:
        flags.append("Promoter holding declined >5% in 4 quarters")
    if stock.get("promoter_pledge_pct", 0) > 20:
        flags.append("Promoter pledge >20%")
    if stock.get("auditor_qualification_2y", False):
        flags.append("Auditor qualification in last 2 years")
    if stock.get("related_party_txn_pct", 0) > 10:
        flags.append("Related-party transactions >10% of revenue")
    if stock.get("negative_ocf_streak", 0) >= 3:
        flags.append("Negative operating cash flow 3+ consecutive quarters")
    if stock.get("sebi_penalty_12m", False):
        flags.append("SEBI investigation/penalty in last 12 months")
    return flags
```

**F-Score back-test evidence:** High F-Score (8-9) stocks outperformed market by 13.4%/yr for 20 years. Combined long high-F-score / short low-F-score delivered 23.0%/yr. Works across all markets (India, Europe, US). **Improves ALL investment strategies** — combined with low P/B, high Earnings Yield, Magic Formula, or Free Cash Flow Yield, the F-Score added 166-284% improvement over 12 years.

---

## 4. Phase 2: Fan-Out Research Agents

### 4.1 Fundamentals Agent (10 min)

**Input:** yfinance 3-year financial statements
**Output:** 16-metric ratio sheet + health score + F-Score + growth trends + red flags

```python
async def fundamentals_agent(state: AgentState, symbol: str) -> dict:
    """Deep fundamental analysis for one stock."""
    import yfinance as yf
    ticker = yf.Ticker(f"{symbol}.NS")
    info = ticker.info
    bs = ticker.balance_sheet.iloc[:, 0]
    inc = ticker.financials.iloc[:, 0]
    cf = ticker.cashflow.iloc[:, 0]
    
    return {
        "symbol": symbol,
        "ratios": {
            "pe": info.get("trailingPE"), "pb": info.get("priceToBook"),
            "roe": info.get("returnOnEquity"), "roic": compute_roic(inc, bs),
            "debt_equity": info.get("debtToEquity"),
            "fcf_yield": compute_fcf_yield(cf, info),
            "earnings_yield": compute_ey(inc, info),
        },
        "health_score": compute_health_score(inc, bs, cf),
        "f_score": piotroski_f_score(current_financials, prior_financials),
        "growth": {
            "revenue_3y": info.get("revenueGrowth"),
            "profit_3y": info.get("earningsGrowth"),
        },
        "red_flags": indian_red_flags(governance_data),
        "recommendation": "BUY" if health_score > 60 and f_score >= 6 else "HOLD",
    }
```

### 4.2 Technical Agent (3 min)

**Input:** yfinance 6-month daily OHLCV
**Output:** Multi-timeframe indicators, patterns, signal + conviction

- SMA 20/50/200, Golden Cross/Death Cross detection
- RSI(14), MACD(12,26,9), Bollinger Bands(20,2)
- ATR(14), OBV, VWAP, Stochastic(14,3)
- Support/Resistance levels, volume confirmation
- Signal: BUY/SELL/HOLD with conviction (0-1.0)

### 4.3 Sentiment Agent (5 min)

**Input:** Marketaux API (50+ articles), NSE announcements
**Output:** Sentiment score, key topics, hard/soft catalysts with dates

- Overall sentiment score (0-100) from weighted article aggregation
- Article count + source credibility weighting
- Key topics extraction: "earnings beat", "regulatory change", "new contract"
- **Catalyst classification with dates:**
  - **HARD:** "Q2 FY27 earnings — Oct 15, 2026", "FOMC meeting — July 30, 2026"
  - **SOFT:** "New product launch expected Q4", "Potential M&A announcement"
- Insider trading activity detection (promoter buy/sell)

### 4.4 Valuation Agent — Reverse DCF (20 min, slowest lane)

**This is the key differentiator.** Traditional DCF guesses growth rates. Reverse DCF tells you what the MARKET is pricing in.

**Input:** Current FCF, shares outstanding, WACC, sector template
**Output:** 3-scenario DCF (Base/Upside/Downside) + Reverse DCF insight

```python
def reverse_dcf(current_price, current_fcf, shares_outstanding, wacc):
    """Back out implied revenue CAGR from current stock price."""
    terminal_growth = 0.03
    low, high = -0.10, 0.30  # Search range for implied growth
    
    for _ in range(100):  # Binary search
        mid = (low + high) / 2
        projected_fcf = [current_fcf * (1 + mid)**i for i in range(1, 11)]
        terminal_value = projected_fcf[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
        pv = sum(fcf / (1 + wacc)**i for i, fcf in enumerate(projected_fcf, 1))
        pv += terminal_value / (1 + wacc)**10
        
        if pv / shares_outstanding > current_price:
            low = mid   # Market expects MORE growth
        else:
            high = mid  # Market expects LESS growth
    
    return (low + high) / 2

# Usage:
implied_growth = reverse_dcf(2500, fcf, shares, wacc)
# → "At ₹2,500, RELIANCE is priced for 6.2% annual FCF growth"
# → "Historical 5yr FCF CAGR: 9.1%"
# → "Gap: +2.9% → 24% undervaluation"
# → "If growth returns to historical avg, fair value = ₹3,100"
```

**Sector-specific WACC templates:**
| Sector | WACC Range | Growth Period | Terminal Growth |
|--------|-----------|---------------|-----------------|
| IT Services | 9-10% | 10 years | 3.5% |
| Banks/NBFCs | 10-11% | 8 years | 3.0% |
| Consumer | 9-10% | 10 years | 3.5% |
| Pharma | 9-10% | 12 years | 3.0% |
| Metals/Mining | 10-12% | 6 years | 2.5% |
| Real Estate | 11-13% | 8 years | 2.5% |

### 4.5 Peer Compare Agent (8 min)

Compares the stock against 4-6 sector peers on 8 metrics:
- PE, EV/EBITDA, ROE, FCF Yield, Revenue Growth, Net Margin, Debt/Equity, Moat Score

Returns percentile ranking: "TCS ranks #2/6 in IT Services on ROE (89th percentile), #4/6 on valuation (33rd percentile)."

### 4.6 Moat Agent (5 min) — Morningstar 5-Source Framework

```python
def assess_moat(symbol, fundamentals, industry) -> dict:
    score = 0
    
    # 1. Intangible Assets (0-20): patents, brands, regulatory licenses
    # 2. Switching Costs (0-20): how expensive/inconvenient to leave
    # 3. Network Effect (0-20): more users = more valuable service
    # 4. Cost Advantage (0-20): structurally lower costs than competitors
    # 5. Efficient Scale (0-20): market size limits competitor entry
    
    # Rating
    if score >= 75: rating = "WIDE"    # 20+ years excess returns
    elif score >= 50: rating = "NARROW" # 10+ years
    else: rating = "NONE"
    
    return {"score": score, "rating": rating, "sources": source_breakdown}
```

Why this matters: A wide-moat company's DCF uses 15+ year growth period and lower WACC. A no-moat company's DCF converges to cost of capital in 5 years. This is Morningstar's approach to adjusting DCF for competitive advantage.

---

## 5. Phase 3: Convergence & Decision

### Composite Ranking

```
Composite = Fundamentals_Score × 0.30
          + (Target_Price / Current_Price - 1) × 0.25   # Upside %
          + Sentiment_Score × 0.15
          + Technical_Signal_Conviction × 0.10
          + Moat_Score/100 × 0.10
          + Peer_Percentile × 0.10
```

### 6-Element Stock Pitch (Buy-Side Standard)

```
# TCS — BUY, Target ₹4,800 (22% upside), Conviction 0.82

1. RECOMMENDATION: BUY | PT ₹4,800 | Stop ₹3,600 | R/R 3.2:1
   3 reasons: BFSI upgrade cycle, 3 new $100M+ cloud deals, margin expansion

2. COMPANY BACKGROUND (summarized):
   Revenue ₹2.4L Cr | Market Cap ₹14L Cr | 6L employees
   Segments: BFSI(31%), Retail(15%), Tech(18%), Life Sciences(12%)

3. INVESTMENT THESIS — VARIANT PERCEPTION:
   Consensus believes TCS is a "steady 8% grower" with limited upside.
   WHAT THE MARKET IS MISSING: RBI's digital banking mandate (Jan 2026)
   requires ALL Indian banks to upgrade core systems. This is a ₹40,000 Cr
   opportunity. TCS has 70% market share in Indian banking IT. Impact: ₹95/share.
   MARGIN STORY: AI-driven delivery automation expanding margins by 120bps/year.
   Street estimates 25.5% margins. Actual: 26.8%. Impact: ₹42/share.

4. CATALYSTS:
   HARD: Q2 FY27 earnings (Oct 10, 2026) — expected 14% YoY revenue, 28% margins
   HARD: 3 BFSI deal announcements (expected Aug-Sep) — $100M+ each TCV
   SOFT: Large deal pipeline commentary in Q2 call → sentiment boost
   SOFT: FY27 guidance upgrade possible if BFSI deals close early

5. VALUATION:
   | Scenario  | Rev CAGR | Fair Value | vs Current |
   |-----------|----------|------------|------------|
   | BASE      | 11%      | ₹4,800     | +22%       |
   | UPSIDE    | 14%      | ₹5,600     | +42%       |
   | DOWNSIDE  | 7%       | ₹3,100     | -22%       |
   
   REVERSE DCF: At ₹3,950, market is pricing in 8.3% revenue CAGR.
   Historical 5yr: 11.4%. Gap: 3.1%. If TCS even returns to 10%,
   fair value = ₹4,450. The market is pricing in a PERMANENT slowdown
   that doesn't account for the BFSI upgrade cycle.

6. RISKS & MITIGANTS:
   - BFSI deals delayed >6 months → -₹95/share. MITIGATE: set stop ₹3,600.
   - US recession reduces tech spending → -₹180/share. MITIGATE: 31% revenue
     from banking which is counter-cyclical (regulatory spending).
   - INR appreciation → -₹45/share. MITIGATE: TCS hedges 12 months forward.
```

### Decision Gate

```
FOR each stock in ranked Top 5:
  IF conviction < 0.5           → DROP. Not enough signal.
  IF risk_reward < 2.0          → DROP. Asymmetric returns only.
  IF sector_exposure >= 30%     → DROP. Concentration limit.
  IF remaining_cash < min_trade  → QUEUE. Wait for top-up.
  ALL CHECKS PASSED              → ADD to trade_queue with Kelly-sized allocation.
```

---

## 6. Guardian Track — Never Sleeps

While discovery runs, holdings are ALWAYS monitored:

```python
async def guardian_check(holding: dict, state: AgentState):
    """Continuous holding health check. Runs parallel to discovery."""
    checks = await asyncio.gather(
        check_stop_loss(holding),         # Price ≤ stop → SELL
        check_profit_target(holding),     # Price ≥ target → BOOK PROFIT
        validate_thesis(holding),         # INTACT/WEAK/BROKEN
        check_catalyst_date(holding),     # Within 7 days of hard catalyst
        check_governance(holding),        # Promoter pledge change, SEBI news
    )
    
    for check in checks:
        if check["action"] == "SELL":
            await execute_sell(holding, check["reason"])
            return  # SELL overrides all other checks
    
    if any(c["severity"] == "HIGH" for c in checks):
        await trigger_detailed_reanalysis(holding)
```

**Guardian runs EVERY 5 minutes during market hours, not just at 9 AM.**

---

## 7. Adrenaline Mode — Price Alert Response

When a stock moves 3%+ intraday, the supervisor fires quick mode:

```
context = "price_alert"
alert_symbol = "RELIANCE"
price_move_pct = -0.035

1. Fan-out 3 agents for ONE stock (concurrent):
   technical_agent + sentiment_agent + validate_thesis
   
2. Wall time: 5 minutes

3. Supervisor decision matrix:
   ┌──────────────────┬──────────────────┬──────────────────┐
   │                   │ Thesis INTACT      │ Thesis BROKEN      │
   ├──────────────────┼──────────────────┼──────────────────┤
   │ Price UP >3%     │ Book profit?       │ Sell into strength │
   ├──────────────────┼──────────────────┼──────────────────┤
   │ Price DOWN >3%   │ Noise. Ignore.     │ SELL IMMEDIATELY   │
   │                   │ (If fundamentals   │                    │
   │                   │  still solid)      │                    │
   └──────────────────┴──────────────────┴──────────────────┘
```

---

## 8. The Data Flywheel — Self-Improving Research

Every trade outcome feeds back into the research calibration:

```
TRADE COMPLETES → log_outcome() → calibrate()
                    │
                    ├→ Conviction calibration by sector
                    │   "IT picks: conv 0.7+ → 71% win rate ✓"
                    │   "Metals picks: conv 0.7+ → 33% win rate ✗"
                    │   → Reduce metals sector weight in screening
                    │
                    ├→ Catalyst accuracy tracking
                    │   "Predicted earnings beat 14 times, correct 9 (64%)"
                    │   → Adjust catalyst impact weight to 0.64 of nominal
                    │
                    ├→ Thesis durability by sector
                    │   "IT theses: avg 8 months until broken"
                    │   "Real estate theses: avg 3 months"
                    │   → Adjust time horizon and re-check frequency
                    │
                    ├→ Pattern library
                    │   "4/5 losing trades: promoter pledge >20% + declining promoter"
                    │   → Auto-elevate to CRITICAL in screening
                    │
                    └→ F-Score threshold adjustment
                        "F-Score ≥ 6: average return +12%. F-Score ≥ 7: +18%."
                        → Raise screening threshold to ≥ 7 during bull markets
```

---

## 9. Sector-Specific Analysis Templates

Different sectors need different analysis. The supervisor selects the right template:

| Sector | Key Metrics | Valuation Method | Red Flags |
|--------|------------|-----------------|-----------|
| **Banks/NBFCs** | NIM, GNPA%, NNPA%, CAR, CASA, credit growth, PCR | P/B, P/ABV, Residual Income | NPA divergence, evergreening, high-cost deposits, promoter-linked lending |
| **IT Services** | Revenue/employee, utilization%, TCV, attrition, digital % | DCF (high FCF), PEG | Client concentration, H1-B dependency, margin compression, captive competition |
| **Pharma** | ANDA pipeline, patent expiry dates, USFDA status, R&D % | DCF + Pipeline NPV, EV/EBITDA | USFDA warning letters, concentration risk, price erosion, regulatory overhang |
| **Consumer** | Volume growth, market share, distribution reach, ad spend%, gross margin | DCF (predictable FCF), PEG | Raw material inflation, promoter pledge, channel stuffing, GST evasion |
| **Metals/Mining** | Capacity utilization, EBITDA/ton, debt/EBITDA, raw material security | EV/EBITDA, P/B (cyclical) | Commodity cycle peak, environmental clearances, leverage, Chinese demand |
| **Real Estate** | Pre-sales, collections%, pending approvals, debt/equity, land bank vintage | NAV, P/B | RERA delays, cash flow mismatch, land bank litigation, interest rate sensitivity |
| **Power/Utilities** | PLF, regulated equity, receivables days, PPA status | DCF (regulated returns), P/BV | State DISCOM health, receivables, fuel risk, regulatory risk |
| **Auto** | Volume growth, market share, capacity utilization, commodity exposure, EV transition | DCF, EV/EBITDA (cyclical) | Cycle peak, inventory buildup, emission norms, EV disruption |
| **Telecom** | ARPU, subscriber growth, spectrum holdings, debt/EBITDA, capex/sales | DCF, EV/EBITDA | Tariff wars, spectrum auction overhang, AGR dues, tech disruption |

The supervisor loads the right template based on the stock's sector classification. This ensures genuinely tailored analysis, not generic PE/ROE screens.

---

## 10. Quantitative Screening Formulas Reference

### Piotroski F-Score (9-point)
See Section 3 for full implementation. Score 0-9. ≥ 6 = quality.

### Magic Formula (Greenblatt)
```
Rank = Rank(EBIT/EV) + Rank(EBIT/Invested Capital)
Take top-ranked decile (10%) or top 85 from 500
```

### Beneish M-Score (Earnings Manipulation Detection)
```
M = -4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
    + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI
M > -2.22 = likely earnings manipulator → REJECT
```

### Altman Z-Score (Bankruptcy Risk)
```
Z = 1.2*(WC/TA) + 1.4*(RE/TA) + 3.3*(EBIT/TA) + 0.6*(MVE/TL) + 1.0*(S/TA)
Z < 1.81 = distress zone → REJECT
```

### Acquirer's Multiple (Tobias Carlisle)
```
Rank = EBIT / Enterprise Value
Take top-ranked decile for deep value plays
```

---

## 11. Implementation Checklist

| # | Component | Complexity | Dependencies |
|---|-----------|-----------|-------------|
| S1 | Nifty 500 symbol list + yfinance bulk fetch | Low | yfinance |
| S2 | Liquidity filter (volume + market cap) | Low | yfinance |
| S3 | Magic Formula ranking engine | Medium | yfinance financials |
| S4 | Piotroski F-Score calculator | Medium | yfinance financials (2yr) |
| S5 | Indian red-flag checker | Medium | NSE announcements, Screener.in |
| F1 | Fundamentals agent (parallel-ready) | Exists | — (enhance existing) |
| F2 | Technical agent (parallel-ready) | Exists | — |
| F3 | Sentiment agent (parallel-ready) | Exists | Marketaux |
| F4 | Valuation agent with Reverse DCF | Medium | Sector templates |
| F5 | Peer compare agent | Low | yfinance peer data |
| F6 | Moat assessment agent | Medium | LLM + Morningstar framework |
| F7 | Fan-out orchestration in supervisor | Medium | asyncio.gather + tool calling |
| C1 | Composite ranking + decision gate | Low | — |
| C2 | 6-element stock pitch generator | Medium | LLM formatting |
| C3 | Guardian continuous monitoring | Medium | APScheduler 5-min interval |
| C4 | Adrenaline quick mode | Low | Price alert trigger |
| D1 | Conviction calibration engine | Medium | Learning journal + vector search |
| D2 | Pattern detection from outcomes | Medium | Vector search |
| D3 | F-Score threshold auto-adjustment | Low | Trade outcome analysis |

---

*This design implements the research from Hebbia (equity research report structure), Daloopa (valuation methods), Morningstar (economic moat framework), New Constructs (reverse DCF / expectations investing), Quant-Investing (Piotroski F-Score and Magic Formula), and the arXiv paper on autonomous systematic factor investing via agentic AI. The fan-out pattern follows AWS and Anthropic's multi-agent research system architecture.*
