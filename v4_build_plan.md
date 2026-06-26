# AutFin v4.1 — Reliability & Test Coverage

> **Date:** June 26, 2026
> **Base:** v4.0 complete (841 tests passing, 20 tools, single supervisor graph)
> **Goal:** Improve reliability (crash recovery, connection pooling), add test coverage for paper trading and supervisor graph, persist observability traces to SQLite.

---

## Architecture: What Changes

```
v3.0 (current)                          v4.0 (target)
─────────────────                       ─────────────────
Macro: 11 indicators                    Macro: 16 indicators + India 10Y
  ^GSPC ^IXIC ^N225 ^FTSE                + ^SSEC ^HSI ^GDAXI ^FCHI ^STOXX50E
  ^VIX DX-Y.NYB ^NSEI ^NSEBANK          + India 10Y G-Sec (FRED)
  CL=F GC=F HG=F NG=F BZ=F

Sentiment: keyword matching              Sentiment: FinBERT (news) + VADER (social)
  37 positive / 33 negative words          ProsusAI/finbert (0.97 accuracy)
  Score: low accuracy                      VADER for Reddit (fast, good for slang)
                                           Score: high accuracy

News: Marketaux only                     News: Marketaux + Moneycontrol + ET + Reddit + NSE
  100 req/day                                Marketaux (API) + Reddit PRAW (free)
                                             + Moneycontrol/ET (scrapers, already built)
                                             + NSE announcements (nselib library)

Social: placeholder                      Social: Reddit r/IndianStockMarket (331K members)
  analyze_social() returns empty           PRAW integration, real-time retail sentiment
```

---

## Dependencies (all free)

```txt
# requirements additions (v4)
transformers>=4.30.0    # FinBERT model (local inference)
torch>=2.0.0            # PyTorch backend for FinBERT
vaderSentiment>=3.3.2   # VADER for social media text
praw>=7.7.0             # Reddit API wrapper (PRAW)
nselib>=0.3.0           # NSE data library (replaces fragile HTTP scraping)
```

**Total install size:** ~1.5GB (mostly PyTorch). Runs on CPU (no GPU required).
**No paid APIs added.** Everything runs locally or uses existing free tiers.

---

## Build Order — 4 Phases, ~5 Days

### Phase 4.1: Quick Wins (Day 1) — No New Dependencies ✅ DONE

**Goal:** Fix gaps in existing infrastructure before adding new features.

| # | Task | File | Detail |
|---|------|------|--------|
| 4.1.1 | Add India 10Y G-Sec yield via FRED | `agents/global_macro.py:68-72` | Add `"india_10y_yield": "INDIRLTLT01STM"` to `FRED_SERIES` dict. Already have FRED infrastructure — just add one series ID. |
| 4.1.2 | Add 5 global indices to yfinance fetcher | `agents/global_macro.py:355-362` | Add `"^SSEC": "shanghai_composite"`, `"^HSI": "hang_seng"`, `"^GDAXI": "dax"`, `"^FCHI": "cac_40"`, `"^STOXX50E": "euro_stoxx_50"` to `index_map`. |
| 4.1.3 | Fix env var inconsistency | `tools/sentiment_tools.py:34` | Change `os.environ.get("MARKETAUX_API_TOKEN")` → `os.environ.get("MARKETAUX_API_KEY")` to match `.env.example`. |
| 4.1.4 | Remove duplicate Chinese chars + dupes | `tools/sentiment_tools.py:75,83` | Remove 4x `"潜力"` from positive_words, 4x `" trouble"` from negative_words. |
| 4.1.5 | Add Reddit env vars to .env.example | `.env.example` | Add `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` (optional, free). |
| 4.1.6 | Update SECTOR_IMPACT_MATRIX for new indices | `agents/global_macro.py:748-869` | Add entries for `shanghai_composite`, `hang_seng`, `dax`, `cac_40`, `euro_stoxx_50` with directional impacts on Indian sectors. |
| 4.1.7 | Wire Bank Nifty into regime detection | `tools/screening.py:352-409` | Currently only uses Nifty 50 SMA. Add Bank Nifty SMA for banking sector regime. |
| 4.1.8 | Update supervisor's `get_market_status()` | `tools/supervisor_tools.py:1458-1524` | Return new indices in response (shanghai, hang_seng, dax, cac_40, euro_stoxx_50, india_10y). |
| 4.1.9 | Tests for new macro indicators | NEW `tests/test_agents/test_global_macro_v4.py` | Test FRED India 10Y fetch, test new index tickers parse correctly, test sector impact matrix entries exist. |

**Expected outcome:** Macro coverage 11→16 indices + India 10Y. No behavior changes, just more data.

---

### Phase 4.2: Sentiment Upgrade (Day 2-3) — Add FinBERT + VADER ✅ DONE

**Goal:** Replace naive keyword sentiment with ML-based analysis.

| # | Task | File | Detail |
|---|------|------|--------|
| 4.2.1 | Create `HybridSentimentAnalyzer` class | NEW `tools/sentiment_analyzer.py` | Core class that routes text to appropriate analyzer: FinBERT for news, VADER for social, keyword fallback. Lazy-loads models on first use. |
| 4.2.2 | Implement FinBERT integration | `tools/sentiment_analyzer.py` | Load `ProsusAI/finbert` via HuggingFace `pipeline("sentiment-analysis")`. Returns `{label, score}` where label ∈ {positive, negative, neutral}. Map to [-1, 1] score. |
| 4.2.3 | Implement VADER integration | `tools/sentiment_analyzer.py` | Use `SentimentIntensityAnalyzer().polarity_scores()`. Returns `{neg, neu, pos, compound}`. Use compound score mapped to [-1, 1]. |
| 4.2.4 | Keep keyword fallback | `tools/sentiment_analyzer.py` | When FinBERT/VADER unavailable (import error), fall back to existing `_calculate_sentiment_score()`. Clean up the Chinese chars and duplicates. |
| 4.2.5 | Update `SentimentTools.get_news_sentiment()` | `tools/sentiment_tools.py:112-195` | Replace `self._calculate_sentiment_score(text)` with `self.analyzer.analyze(text, source_type="news")`. |
| 4.2.6 | Update `SentimentAgent` | `agents/sentiment.py:208-251` | Agent already delegates to `SentimentTools` — no changes needed (uses hybrid analyzer via `SentimentTools`). |
| 4.2.7 | ~~Update sentiment categories~~ | — | Kept existing thresholds (0.6/0.3). FinBERT confidence passed through as-is. |
| 4.2.8 | Add model caching | `tools/sentiment_analyzer.py` | Module-level `_finbert_pipeline`, `_vader_analyzer` globals. First call loads, subsequent calls reuse. |
| 4.2.9 | Tests for sentiment upgrade | NEW `tests/test_tools/test_sentiment_analyzer.py` | 23 tests: keyword fallback (6), VADER (4), FinBERT (4), public API (6), singleton (2). All passing. |

**Architecture of HybridSentimentAnalyzer:**

```python
class HybridSentimentAnalyzer:
    """ML-based sentiment analysis with graceful fallback."""

    def __init__(self):
        self._finbert = None  # Lazy loaded
        self._vader = None    # Lazy loaded

    def _load_finbert(self):
        """Load FinBERT model (called once, cached)."""
        from transformers import pipeline
        self._finbert = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            return_all_scores=False
        )

    def _load_vader(self):
        """Load VADER analyzer (instant, no model)."""
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self._vader = SentimentIntensityAnalyzer()

    def analyze(self, text: str, source_type: str = "news") -> dict:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze
            source_type: "news" (FinBERT), "social" (VADER), or "fallback" (keyword)

        Returns:
            {score: float [-1,1], confidence: float [0,1], method: str}
        """
        if source_type == "social":
            return self._vader_analyze(text)
        elif source_type == "news":
            return self._finbert_analyze(text)
        else:
            return self._keyword_analyze(text)

    def _finbert_analyze(self, text: str) -> dict:
        if not self._finbert:
            self._load_finbert()
        result = self._finbert(text[:512])[0]  # FinBERT max 512 tokens
        label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
        return {
            "score": label_map[result["label"]] * result["score"],
            "confidence": result["score"],
            "method": "finbert"
        }

    def _vader_analyze(self, text: str) -> dict:
        if not self._vader:
            self._load_vader()
        scores = self._vader.polarity_scores(text)
        return {
            "score": scores["compound"],
            "confidence": abs(scores["compound"]),
            "method": "vader"
        }
```

---

### Phase 4.3: New Data Sources (Day 3-4) — Reddit + Scrapers + NSE ✅ DONE

**Goal:** Add retail sentiment from Reddit, wire existing scrapers, fix NSE access.

| # | Task | File | Detail |
|---|------|------|--------|
| 4.3.1 | Create Reddit sentiment tool | NEW `tools/reddit_sentiment.py` | PRAW integration: search r/IndianStockMarket and r/IndiaInvestments for stock mentions. Return sentiment + volume. |
| 4.3.2 | ~~Add Reddit tool to supervisor~~ | — | Deferred to Phase 4.4 (supervisor prompt update). |
| 4.3.3 | Wire Moneycontrol scraper to sentiment | `tools/sentiment_tools.py` | `get_moneycontrol_sentiment()` calls `scrape_moneycontrol_news()` and runs `analyze_batch()`. |
| 4.3.4 | Wire ET scraper to sentiment | `tools/sentiment_tools.py` | `get_economic_times_sentiment()` calls `scrape_economic_times()` and runs `analyze_batch()`. |
| 4.3.5 | ~~Fix NSE announcements~~ | — | Deferred — existing scraper works, nselib not needed yet. |
| 4.3.6 | Add sentiment volume tracking | `tools/sentiment_tools.py` | Each source returns `articles_count` / `posts_count`. |
| 4.3.7 | Multi-source sentiment aggregation | `tools/sentiment_tools.py` | `get_multi_source_sentiment()` — weighted average: Marketaux (0.4) + MC (0.2) + ET (0.2) + Reddit (0.2). |
| 4.3.8 | ~~Update `analyze_sentiment()` supervisor tool~~ | — | Deferred to Phase 4.4. |
| 4.3.9 | Tests for Reddit + scrapers | NEW `tests/test_tools/test_reddit_and_multi_source.py` | 17 tests: Reddit tool (8), multi-source (9). All passing. |

**Reddit Sentiment Architecture:**

```python
# tools/reddit_sentiment.py
import praw
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class RedditSentimentTool:
    """Fetch and analyze Reddit sentiment for Indian stocks."""

    def __init__(self):
        self.reddit = None
        self.analyzer = SentimentIntensityAnalyzer()

    def _get_reddit(self):
        """Lazy init PRAW (free, no API key needed for read-only)."""
        if self.reddit is None:
            self.reddit = praw.Reddit(
                client_id=os.environ.get("REDDIT_CLIENT_ID", ""),
                client_secret=os.environ.get("REDDIT_CLIENT_SECRET", ""),
                user_agent=os.environ.get("REDDIT_USER_AGENT", "AutFin/1.0")
            )
        return self.reddit

    async def analyze(self, symbol: str, subreddits: list = None) -> dict:
        """
        Search Reddit for stock mentions and analyze sentiment.

        Returns:
            {
                "symbol": "RELIANCE",
                "posts_found": 15,
                "avg_sentiment": 0.42,
                "sentiment_category": "bullish",
                "top_posts": [{"title": "...", "score": 0.8, "url": "..."}],
                "volume_signal": "high"  # vs historical average
            }
        """
        if subreddits is None:
            subreddits = ["IndianStockMarket", "IndiaInvestments", "IndianStreetBets"]

        reddit = self._get_reddit()
        posts = []

        for sub_name in subreddits:
            subreddit = reddit.subreddit(sub_name)
            for post in subreddit.search(symbol, limit=20, sort="new", time_filter="week"):
                sentiment = self.analyzer.polarity_scores(post.title + " " + post.selftext[:500])
                posts.append({
                    "title": post.title[:100],
                    "score": sentiment["compound"],
                    "upvotes": post.score,
                    "comments": post.num_comments,
                    "url": f"https://reddit.com{post.permalink}",
                    "subreddit": sub_name
                })

        if not posts:
            return {"symbol": symbol, "posts_found": 0, "avg_sentiment": 0.0}

        avg_sentiment = sum(p["score"] for p in posts) / len(posts)
        return {
            "symbol": symbol,
            "posts_found": len(posts),
            "avg_sentiment": round(avg_sentiment, 3),
            "sentiment_category": self._categorize(avg_sentiment),
            "top_posts": sorted(posts, key=lambda x: abs(x["score"]), reverse=True)[:5],
            "volume_signal": self._volume_signal(len(posts))
        }
```

---

### Phase 4.4: Integration, Dashboard & Tests (Day 5) ✅ DONE

**Goal:** Wire everything together, update dashboard, hit 800+ tests.

| # | Task | File | Detail |
|---|------|------|--------|
| 4.4.1 | Update supervisor system prompt | `prompts/supervisor.py` | Added multi-source sentiment (MC/ET/Reddit), FinBERT, 16 indices, India 10Y, sector impact matrix. |
| 4.4.2 | Update `analyze_macro_context()` | `tools/supervisor_tools.py:1351-1455` | Already has 15 tickers + India 10Y from FRED. Added rates.india_10y in response. |
| 4.4.3 | Update `build_stock_pitch()` | `tools/supervisor_tools.py:611-673` | Added sentiment factor instruction to system prompt. |
| 4.4.4 | Add macro dashboard panel | `dashboard/app.py` | NEW `render_global_markets()` — Nifty/Bank Nifty cards, 12 global indices table, rates (US 10Y, India 10Y, USD/INR), sector weights, macro narrative. |
| 4.4.5 | Add sentiment dashboard panel | `dashboard/app.py` | NEW `render_sentiment()` — multi-select symbols, weighted sentiment cards, source breakdown expanders, Reddit volume signal. |
| 4.4.6 | Update sidebar nav | `dashboard/app.py:158-167` | Added "Global Markets" and "Sentiment" to nav radio. |
| 4.4.7 | Update `render_sidebar()` quick stats | `dashboard/app.py:142-153` | Added Nifty 50 + Bank Nifty from market_snapshot quotes. |
| 4.4.8 | Integration tests | NEW `tests/test_graphs/test_v4_integration.py` | 15 tests: supervisor prompt (4), pitch sentiment (2), macro wiring (3), dashboard panels (4), analyzer API (2). All passing. |
| 4.4.9 | Dashboard tests | NEW `tests/test_dashboard/test_phase4.py` | 13 tests: panel functions (2), sidebar nav (3), main routing (3), supervisor prompt v4 (5). All passing. |
| 4.4.10 | Run full suite, fix regressions | — | 191 tools+dashboard+integration tests pass. Pre-existing `test_auth_failure` + `test_app.py` (st not defined) unchanged. |
| 4.4.11 | Update v4_build_plan.md status | `v4_build_plan.md` | All phases marked ✅ complete. |

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    SUPERVISOR (ReAct Loop)                        │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ analyze_      │  │ analyze_     │  │ build_stock_pitch()  │  │
│  │ macro_context │  │ sentiment()  │  │                      │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────────────┘  │
│         │                 │                                       │
│         ▼                 ▼                                       │
│  ┌──────────────┐  ┌──────────────────────────────────────────┐ │
│  │ Global Macro  │  │ Hybrid Sentiment Analyzer                │ │
│  │               │  │                                          │ │
│  │ yfinance:     │  │ News path:                               │ │
│  │  16 indices   │  │  Marketaux → FinBERT → score             │ │
│  │  5 commodities│  │  Moneycontrol → FinBERT → score          │ │
│  │  INR/USD      │  │  ET → FinBERT → score                    │ │
│  │  VIX, DXY     │  │  → weighted aggregate                    │ │
│  │               │  │                                          │ │
│  │ FRED:         │  │ Social path:                             │ │
│  │  US GDP/CPI   │  │  Reddit PRAW → VADER → score             │ │
│  │  Fed Rate     │  │  → volume + sentiment category           │ │
│  │  India 10Y    │  │                                          │ │
│  │               │  │ Fallback:                                │ │
│  │ Impact:       │  │  Keyword matching (existing)             │ │
│  │  16 indicators│  │                                          │ │
│  │  → sector map │  │ Output:                                  │ │
│  │  → regime     │  │  {score, confidence, sources[], method} │ │
│  └──────────────┘  └──────────────────────────────────────────┘ │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Screen → Research → Pitch → Risk → Trade → Learn          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Sector Impact Matrix — New Entries

```python
# Add to SECTOR_IMPACT_MATRIX in global_macro.py

"shanghai_composite": {
    "bullish": ["metals", "it", "auto"],  # China demand lifts metals, IT outsourcing
    "bearish": ["pharma"],  # China pharma competition
    "weight": 1.0
},
"hang_seng": {
    "bullish": ["it", "banking"],  # HK rally = risk-on
    "bearish": [],
    "weight": 0.8
},
"dax": {
    "bullish": ["auto", "metals", "it"],  # Germany = auto/metal demand
    "bearish": [],
    "weight": 1.0
},
"cac_40": {
    "bullish": ["fmcg", "pharma"],  # Luxury + pharma exposure
    "bearish": [],
    "weight": 0.7
},
"euro_stoxx_50": {
    "bullish": ["it", "banking"],  # Broad Europe risk-on
    "bearish": [],
    "weight": 0.9
},
"india_10y_yield": {
    "bullish": [],  # Lower yield = bullish
    "bearish": ["banking", "realty", "infra"],  # Higher yield = pressure
    "weight": 1.5  # Very important for rate-sensitive sectors
}
```

---

## Test Plan

| Phase | New Tests | Cumulative | Focus |
|-------|----------|------------|-------|
| 4.1 | +13 ✅ | 759 | ✅ FRED India 10Y, 5 new indices, env var fix, Bank Nifty regime, sector matrix |
| 4.2 | +23 ✅ | 782 | ✅ HybridSentimentAnalyzer: keyword (6), VADER (4), FinBERT (4), public API (6), singleton (2) |
| 4.3 | +17 ✅ | 813 | ✅ Reddit PRAW tool, MC/ET scraper wiring, multi-source aggregation, weighted scoring |
| 4.4 | +28 ✅ | 841 | ✅ Supervisor prompt, pitch sentiment, 2 dashboard panels, sidebar, 15 integration + 13 dashboard tests |
| **v4.1** | **+73** | **914** | ✅ P2: checkpointer, lazy imports, lookback_days fix. P3: 43 paper trading + 21 supervisor tests. P5: connection pool, trace persistence. |

---

## Success Criteria

| Criterion | v3.0 (current) | v4.0 (target) |
|-----------|----------------|----------------|
| Macro indicators | 11 | 16 + India 10Y |
| Sentiment method | Keyword (37/33 words) | FinBERT (ML) + VADER (social) |
| Sentiment accuracy | Low (~60%) | High (~95% news, ~80% social) |
| News sources | 1 (Marketaux) | 4 (Marketaux + MC + ET + Reddit) |
| Social sentiment | Placeholder | Reddit (331K members) |
| Global coverage | US + Japan + UK | + China + HK + Germany + France + Europe |
| Tests | 759 | 914+ |
| Cost | $0/month | $0/month |

---

## Non-Goals (v4.0 explicitly does NOT do)

- Twitter/X sentiment — no free tier for reads ($49+/month minimum)
- StockTwits — negligible Indian stock coverage
- GPU inference — FinBERT runs on CPU (~100ms/inference, sufficient for batch analysis)
- Real-time news streaming — batch-based per scheduler cycle
- Paid NLP APIs — everything runs locally via HuggingFace/VADER
- Options/Futures sentiment — equity-focused only

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| FinBERT slow on CPU | Lazy load, cache model, process max 5 articles per symbol (cap at 500ms) |
| Reddit rate limiting | PRAW handles this automatically; cache results for 30 min |
| NSE library breaks | Fallback to existing direct HTTP scraper |
| Marketaux quota hit (100/day) | Gracefully degrade to MC/ET scrapers |
| FRED India 10Y monthly only | Use US 10Y as daily proxy, FRED as monthly reference |
| yfinance indices fail | Existing error handling (try/except with fallback) |
