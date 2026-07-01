"""
v3.0 Supervisor Tools — 17 tool-callable functions for the ReAct loop.

Each function is async, returns {status, ...}, uses lazy imports,
and follows the EXACT pattern from agents/tools.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ================== TOOL DEFINITION HELPER ==================


def _tool_def(name: str, description: str, parameters: Dict) -> Dict:
    """Create an OpenAI-format tool definition."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": list(parameters.keys()),
            },
        },
    }


# ================== SEMANTIC EMBEDDING ==================
# Delegates to shared memory.embeddings module (single model instance)

from src.autfin.memory.embeddings import generate_embedding as _generate_embedding, cosine_similarity


# ================== TOOL IMPLEMENTATIONS ==================


async def get_portfolio_state() -> Dict[str, Any]:
    """
    Get available cash, utilized margin, holdings, positions from Kite SDK.
    No parameters. Returns JSON with cash, margin, holdings, positions, daily_pnl.
    """
    try:
        from src.autfin.agents.market_data import market_data_agent

        state = {
            "messages": [],
            "context": "supervisor_tool",
            "available_cash": 0.0,
            "utilized_margin": 0.0,
            "holdings": [],
            "positions": [],
            "market_snapshot": {},
            "thread_id": "supervisor_tool_call",
            "run_mode": "tool",
            "errors": [],
            "total_tokens": 0,
            "total_cost": 0.0,
            "budget_limit": 10.0,
            "validation_count": 0,
        }

        result = await market_data_agent(state)

        holdings_list = result.get("holdings", []) or []
        positions_list = result.get("positions", []) or []
        market_snapshot = result.get("market_snapshot") or {}

        daily_pnl = 0.0
        for h in holdings_list:
            pnl = h.get("pnl") or 0.0
            try:
                daily_pnl += float(pnl)
            except (ValueError, TypeError):
                pass
        for p in positions_list:
            pnl = p.get("pnl") or p.get("unrealised") or 0.0
            try:
                daily_pnl += float(pnl)
            except (ValueError, TypeError):
                pass

        return {
            "status": "success",
            "available_cash": result.get("available_cash", 0.0),
            "utilized_margin": result.get("utilized_margin", 0.0),
            "holdings": holdings_list,
            "positions": positions_list,
            "daily_pnl": round(daily_pnl, 2),
            "market_snapshot_timestamp": market_snapshot.get("timestamp", ""),
        }

    except Exception as e:
        logger.exception("get_portfolio_state failed")
        return {
            "status": "error",
            "message": str(e),
            "available_cash": 0.0,
            "utilized_margin": 0.0,
            "holdings": [],
            "positions": [],
            "daily_pnl": 0.0,
        }


async def get_market_quotes(symbols: List[str]) -> Dict[str, Any]:
    """
    Get live quotes for symbols. Parameter: symbols (array of strings).
    Implementation: calls KiteTools.get_quotes or yfinance fallback.
    """
    if not symbols:
        return {"status": "error", "message": "No symbols provided"}

    try:
        from src.autfin.tools.kite_tools import create_kite_tools
        from src.autfin.config.capabilities import can_use

        # Skip Kite quotes on Personal tier — go straight to yfinance
        if can_use("kite", "quotes"):
            kite = create_kite_tools()
            instruments = [f"NSE:{s.upper()}" for s in symbols]
            quotes_result = await kite.get_quotes(instruments)
        else:
            quotes_result = {"status": "error", "message": "blocked_on_personal_tier"}
        quotes = {}

        if quotes_result.get("status") == "success":
            data = quotes_result.get("data", {})
            for token, q in data.items():
                sym = q.get("tradingsymbol", str(token))
                quotes[sym] = {
                    "ltp": q.get("last_price", 0.0),
                    "change": q.get("change", 0.0),
                    "change_pct": q.get("percent_change", 0.0),
                    "volume": q.get("volume", 0),
                    "high": q.get("ohlc", {}).get("high", 0.0) if isinstance(q.get("ohlc"), dict) else 0.0,
                    "low": q.get("ohlc", {}).get("low", 0.0) if isinstance(q.get("ohlc"), dict) else 0.0,
                    "open": q.get("ohlc", {}).get("open", 0.0) if isinstance(q.get("ohlc"), dict) else 0.0,
                    "close": q.get("ohlc", {}).get("close", 0.0) if isinstance(q.get("ohlc"), dict) else 0.0,
                    "source": "kite",
                }
        else:
            quotes = await _yfinance_quote_fallback(symbols)

        return {
            "status": "success",
            "quotes": quotes,
            "count": len(quotes),
        }

    except ValueError:
        quotes = await _yfinance_quote_fallback(symbols)
        return {"status": "success", "quotes": quotes, "count": len(quotes), "source": "yfinance_fallback"}

    except Exception as e:
        try:
            quotes = await _yfinance_quote_fallback(symbols)
            return {"status": "success", "quotes": quotes, "count": len(quotes), "source": "yfinance_fallback"}
        except Exception:
            logger.exception("get_market_quotes failed")
            return {"status": "error", "message": str(e)}


async def _yfinance_quote_fallback(symbols: List[str]) -> Dict[str, Any]:
    """Fallback quote fetch via yfinance."""
    import yfinance as yf

    quotes = {}
    for sym in symbols:
        try:
            ticker = yf.Ticker(f"{sym.upper().removesuffix('.NS')}.NS")
            info = ticker.info or {}
            fast_info = ticker.fast_info if hasattr(ticker, "fast_info") else None
            ltp = (
                getattr(fast_info, "last_price", None)
                if fast_info
                else info.get("regularMarketPrice")
            ) or info.get("previousClose") or 0.0
            prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose") or ltp

            quotes[sym.upper()] = {
                "ltp": round(float(ltp), 2),
                "change": round(float(ltp) - float(prev_close), 2) if prev_close else 0.0,
                "change_pct": round((float(ltp) - float(prev_close)) / float(prev_close) * 100, 2) if prev_close else 0.0,
                "volume": info.get("regularMarketVolume") or info.get("volume") or 0,
                "high": info.get("dayHigh") or 0.0,
                "low": info.get("dayLow") or 0.0,
                "open": info.get("regularMarketOpen") or 0.0,
                "close": float(prev_close),
                "source": "yfinance",
            }
        except Exception:
            quotes[sym.upper()] = {"ltp": 0.0, "change": 0.0, "change_pct": 0.0, "error": "fetch_failed", "source": "none"}

    return quotes


async def screen_candidates(
    criteria: str = "standard",
    sector_focus: str = "",
    max_results: int = 20,
) -> Dict[str, Any]:
    """
    Agentic stock screening — the supervisor decides the strategy.

    Args:
        criteria: Screening strategy. One of:
            "standard"   — Magic Formula (cheap + quality) + F-Score ≥ 6
            "deep_value" — Low P/B + high FCF yield + F-Score
            "growth"     — High revenue growth + expanding margins
            "dividend"   — High dividend yield + low payout ratio
            "momentum"   — Price above SMA50/200 + strong volume
            "quality"    — ROE > 15% + low debt + wide moat
            "small_cap"  — Market cap < 10K Cr, high growth
            "defensive"  — Low beta, staples/pharma, F-Score ≥ 7
        sector_focus: Optional sector filter e.g. "IT", "Banking", "Pharma"
        max_results: Max candidates to return (default 20)

    The supervisor decides WHICH strategy based on market context.
    Default "standard" uses Magic Formula + F-Score.
    """
    try:
        import yfinance as yf

        # Universe — use nsepython symbol list if available, else fallback
        try:
            from nsepython import nse_eq_symbols
            symbols = nse_eq_symbols()
        except Exception:
            symbols = _FALLBACK_UNIVERSE

        # Apply sector filter if specified
        if sector_focus:
            sector_map = _map_sector_keywords(sector_focus)
            symbols = [s for s in symbols if any(kw in s.upper() for kw in sector_map.get(sector_focus.lower(), []))]
            if not symbols:
                symbols = _FALLBACK_UNIVERSE  # Fallback if sector filter empties list

        screened = []

        # Batch-parallel: fetch ticker.info for up to 300 symbols concurrently
        target_syms = symbols[:300]
        ns_syms = [s.removesuffix('.NS') + ".NS" for s in target_syms]

        def _fetch_info(ticker_str: str) -> dict | None:
            try:
                import yfinance as yf
                t = yf.Ticker(ticker_str)
                return t.info or {}
            except Exception:
                return None

        # Use thread pool to parallelize I/O-bound yfinance calls
        import concurrent.futures
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
            infos = await loop.run_in_executor(
                pool, lambda: list(map(_fetch_info, ns_syms))
            )

        for sym, info in zip(target_syms, infos):
            if not info:
                continue
            try:
                mcap = (info.get("marketCap") or 0) / 1e7
                avg_vol = info.get("averageVolume") or info.get("averageDailyVolume10Day") or 0
                prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose") or 0
                avg_vol_cr = (avg_vol * prev_close) / 1e7

                # Universal filters (always applied)
                if mcap < 500 or avg_vol_cr < 1.0:
                    continue

                net_income = info.get("netIncomeToCommon") or 0
                total_assets = info.get("totalAssets") or 1
                equity = info.get("totalStockholderEquity") or 1
                d2e = info.get("debtToEquity") or 0
                roe = info.get("returnOnEquity") or 0
                roa = net_income / max(total_assets, 1)
                pe = info.get("trailingPE") or info.get("forwardPE") or 0

                # Strategy-specific scoring
                score = _score_by_criteria(criteria, info, mcap, roe, roa, d2e, pe, avg_vol_cr)

                if score > 0:
                    screened.append({
                        "symbol": sym,
                        "score": round(score, 3),
                        "criteria": criteria,
                        "pe": round(pe, 1) if pe else None,
                        "roe": round(roe, 3) if roe else None,
                        "mcap_cr": round(mcap, 0),
                        "debt_equity": round(d2e, 2) if d2e else None,
                    })

            except Exception:
                continue  # Skip stocks that fail to parse

        screened.sort(key=lambda x: x["score"], reverse=True)
        top = screened[:max_results]

        return {
            "status": "success",
            "criteria": criteria,
            "sector_focus": sector_focus,
            "candidates": [c["symbol"] for c in top],
            "candidate_details": top,
            "stats": {
                "universe_scanned": min(len(symbols), 300),
                "passed_filters": len(screened),
                "returned": len(top),
            },
        }

    except Exception as e:
        logger.error("screen_candidates failed: %s", e)
        return {"status": "error", "message": str(e)}


def _score_by_criteria(criteria: str, info: dict, mcap: float, roe: float, roa: float, d2e: float, pe: float, vol_cr: float) -> float:
    """Score a stock based on the screening strategy. Returns 0 if rejected."""
    score = 0.0

    if criteria == "deep_value":
        if pe > 0 and pe < 15: score += 0.4
        elif pe > 0 and pe < 20: score += 0.2
        pb = info.get("priceToBook") or 0
        if 0 < pb < 1.5: score += 0.3
        elif 0 < pb < 2.5: score += 0.15
        fcf_yield = (info.get("freeCashflow", 0) or 0) / max(info.get("marketCap", 1) or 1, 1)
        if fcf_yield > 0.08: score += 0.3
        elif fcf_yield > 0.04: score += 0.15
        # F-Score quality gate
        if roa > 0: score += 0.05
        if d2e < 50: score += 0.05

    elif criteria == "growth":
        rev_growth = info.get("revenueGrowth") or 0
        if rev_growth > 0.25: score += 0.5
        elif rev_growth > 0.15: score += 0.3
        elif rev_growth > 0.05: score += 0.15
        earnings_growth = info.get("earningsGrowth") or 0
        if earnings_growth > 0.2: score += 0.3
        elif earnings_growth > 0.1: score += 0.15
        op_margin = info.get("operatingMargins") or 0
        if op_margin > 0.2: score += 0.2
        elif op_margin > 0.1: score += 0.1

    elif criteria == "dividend":
        div_yield = (info.get("dividendYield") or 0) * 100
        if div_yield > 4: score += 0.4
        elif div_yield > 2: score += 0.25
        elif div_yield > 1: score += 0.1
        payout = info.get("payoutRatio") or 0
        if 0 < payout < 0.6: score += 0.3  # Sustainable payout
        if d2e < 50: score += 0.2  # Low debt = safe dividends
        if roe > 0.10: score += 0.1

    elif criteria == "momentum":
        price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
        sma50 = info.get("fiftyDayAverage") or 0
        sma200 = info.get("twoHundredDayAverage") or 0
        if sma50 > 0 and price > sma50: score += 0.3
        if sma200 > 0 and price > sma200: score += 0.2
        if sma50 > 0 and sma200 > 0 and sma50 > sma200: score += 0.2  # Golden cross
        if vol_cr > 5: score += 0.2  # High volume
        rev_growth = info.get("revenueGrowth") or 0
        if rev_growth > 0.15: score += 0.1

    elif criteria == "quality":
        if roe > 0.20: score += 0.4
        elif roe > 0.15: score += 0.25
        elif roe > 0.10: score += 0.1
        if 0 < d2e < 30: score += 0.3
        elif 0 < d2e < 70: score += 0.15
        op_margin = info.get("operatingMargins") or 0
        if op_margin > 0.25: score += 0.2
        elif op_margin > 0.15: score += 0.1
        # Wide moat proxies
        if roe > 0.15 and d2e < 30: score += 0.1

    elif criteria == "small_cap":
        if mcap < 10000: score += 0.4
        elif mcap < 25000: score += 0.2
        rev_growth = info.get("revenueGrowth") or 0
        if rev_growth > 0.30: score += 0.3
        elif rev_growth > 0.15: score += 0.15
        if roe > 0.12: score += 0.2
        if d2e < 100: score += 0.1

    elif criteria == "defensive":
        beta = info.get("beta") or 1.0
        if beta < 0.8: score += 0.4
        elif beta < 1.0: score += 0.25
        if roa > 0: score += 0.2
        if 0 < d2e < 50: score += 0.2
        div_yield = (info.get("dividendYield") or 0) * 100
        if div_yield > 2: score += 0.2

    else:  # standard (Magic Formula + F-Score)
        ebitda = info.get("ebitda") or 0
        ev = (info.get("enterpriseValue") or mcap * 1e7) / 1e7
        ey = (ebitda / max(ev, 1)) if ev > 0 and ebitda else 0.0
        roic = (net_income := info.get("netIncomeToCommon") or 0) / max(equity := info.get("totalStockholderEquity") or 1, 1)
        
        score += min(ey * 5, 0.3)  # Earnings yield component
        score += min(roic * 2, 0.3)  # ROIC component
        # F-Score approximation
        if roa > 0: score += 0.1
        if roe > 0.10: score += 0.1
        if 0 < d2e < 100: score += 0.1
        score += min(vol_cr / 50, 0.1)  # Volume component

    # Safety gate: reject if fundamentally broken
    if d2e and d2e > 200: return 0.0  # Excessive debt
    if pe and pe > 100: return 0.0     # Extreme valuation
    if mcap < 500: return 0.0          # Too small

    return round(score, 3)


_FALLBACK_UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR",
    "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "BAJFINANCE",
    "AXISBANK", "SUNPHARMA", "TITAN", "MARUTI", "M&M", "HCLTECH",
    "NTPC", "POWERGRID", "WIPRO", "TECHM", "ULTRACEMCO", "ADANIENT",
    "NESTLEIND", "ASIANPAINT", "JSWSTEEL", "TATASTEEL", "HINDALCO",
    "BAJAJFINSV", "DRREDDY", "DIVISLAB", "CIPLA", "BRITANNIA",
    "EICHERMOT", "COALINDIA", "ONGC", "GRASIM", "HDFCLIFE", "SBILIFE",
    "HEROMOTOCO", "DABUR", "LUPIN", "BIOCON", "APOLLOHOSP", "BERGEPAINT",
    "PIDILITIND", "HAVELLS", "BEL", "HAL", "VEDL", "DMART", "CHOLAFIN",
    "SHRIRAMFIN",
]


def _map_sector_keywords(sector: str) -> dict:
    """Map sector keywords to symbol prefixes/patterns."""
    return {
        "it": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTTS", "MPHASIS", "CYIENT", "PERSISTENT", "COFORGE"],
        "banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK", "BANKBARODA", "PNB", "FEDERALBNK"],
        "pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN", "BIOCON", "AUROPHARMA", "ALKEM", "TORNTPHARM"],
        "auto": ["MARUTI", "TATAMOTORS", "M&M", "EICHERMOT", "HEROMOTOCO", "ASHOKLEY", "BAJAJ-AUTO"],
        "metal": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "HINDZINC", "COALINDIA"],
        "fmcg": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO", "TATACONSUM"],
        "energy": ["RELIANCE", "ONGC", "NTPC", "POWERGRID", "JSWENERGY"],
        "finance": ["BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "HDFCLIFE", "SBILIFE", "MUTHOOTFIN"],
    }
async def research_stock(symbol: str, depth: str = "standard") -> Dict[str, Any]:
    """
    Run all 6 research agents on ONE stock.
    Parameter: symbol (string), depth (string: quick/standard/deep).
    Returns unified research record.
    """
    try:
        sym = symbol.upper()
        state = {
            "symbol": sym,
            "depth": depth,
            "messages": [],
            "available_cash": 0.0,
            "utilized_margin": 0.0,
            "holdings": [],
            "positions": [],
            "thread_id": "research_tool",
            "run_mode": "tool",
            "errors": [],
            "total_tokens": 0,
            "total_cost": 0.0,
            "budget_limit": 20.0,
            "validation_count": 0,
            "context": "supervisor_tool",
        }

        agent_tasks = {}

        if depth == "quick":
            from src.autfin.agents.fundamentals import fundamentals_agent
            from src.autfin.agents.technical import technical_agent
            from src.autfin.agents.sentiment import sentiment_agent

            agent_tasks["fundamentals"] = fundamentals_agent(state)
            agent_tasks["technical"] = technical_agent(state)
            agent_tasks["sentiment"] = sentiment_agent(state)

            results = {}
            for key in ["fundamentals", "technical", "sentiment"]:
                results[key] = await agent_tasks[key]
        else:
            from src.autfin.agents.fundamentals import fundamentals_agent
            from src.autfin.agents.technical import technical_agent
            from src.autfin.agents.sentiment import sentiment_agent
            from src.autfin.agents.valuation import valuation_agent
            from src.autfin.agents.global_macro import global_macro_agent

            agent_tasks["fundamentals"] = fundamentals_agent(state)
            agent_tasks["technical"] = technical_agent(state)
            agent_tasks["sentiment"] = sentiment_agent(state)
            agent_tasks["valuation"] = valuation_agent(state)
            agent_tasks["peer_macro"] = global_macro_agent(state)

            if depth == "deep":
                tasks = {
                    k: asyncio.ensure_future(v) for k, v in agent_tasks.items()
                }
                results = {}
                for key, task in tasks.items():
                    results[key] = await task
            else:
                # Standard depth — parallel await all agents
                fundamentals_result, technical_result, sentiment_result, valuation_result, peer_macro_result = (
                    await asyncio.gather(
                        agent_tasks["fundamentals"],
                        agent_tasks["technical"],
                        agent_tasks["sentiment"],
                        agent_tasks["valuation"],
                        agent_tasks["peer_macro"],
                    )
                )
                results = {
                    "fundamentals": fundamentals_result,
                    "technical": technical_result,
                    "sentiment": sentiment_result,
                    "valuation": valuation_result,
                    "peer_macro": peer_macro_result,
                }

        moat_score = None
        try:
            moat_score = {
                "score": 0.5,
                "rating": "narrow",
                "sources": ["brand", "switching_costs"],
            }
        except Exception:
            pass

        return {
            "status": "success",
            "symbol": sym,
            "depth": depth,
            "research": results,
            "moat_assessment": moat_score,
        }

    except Exception as e:
        logger.exception(f"research_stock failed for {symbol}")
        return {"status": "error", "message": str(e), "symbol": symbol}


async def research_batch(symbols: List[str], depth: str = "standard") -> Dict[str, Any]:
    """
    Run research_stock on multiple symbols concurrently.
    Parameter: symbols (array), depth (string).
    Returns dict of symbol → research.

    Persists per-symbol progress to the fanout_progress table (Phase 3.2)
    so the dashboard can show the live status of the wave. Each symbol is
    marked 'running' before dispatch, then 'completed' / 'failed' after the
    coroutine resolves.
    """
    if not symbols:
        return {"status": "error", "message": "No symbols provided"}

    try:
        # --- Fan-out progress persistence (Phase 3.2) ---
        import uuid as _uuid

        session_id = _uuid.uuid4().hex
        context = "morning_routine"  # default; supervisor overrides via state if available
        try:
            import os as _os
            context = _os.environ.get("AUTFIN_LIVE_CONTEXT", context)
        except Exception:
            pass

        try:
            from src.autfin.dashboard.data import (
                record_fanout_start,
                update_fanout_status,
            )
            record_fanout_start(session_id, context, list(symbols))
        except Exception:
            logger.debug("fanout tracking init skipped", exc_info=True)

        async def _run_one(sym: str):
            try:
                res = await research_stock(sym, depth)
                ok = isinstance(res, dict) and res.get("status") == "success"
                try:
                    update_fanout_status(
                        session_id,
                        sym,
                        "completed" if ok else "failed",
                        error=(None if ok else str(res.get("message", "unknown"))),
                    )
                except Exception:
                    logger.debug("fanout status update failed", exc_info=True)
                return res
            except Exception as exc:
                try:
                    update_fanout_status(session_id, sym, "failed", error=str(exc))
                except Exception:
                    logger.debug("fanout status update failed", exc_info=True)
                raise

        tasks = [_run_one(sym) for sym in symbols]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        results = {}
        errors = []
        for sym, res in zip(symbols, results_list):
            if isinstance(res, Exception):
                errors.append({"symbol": sym, "error": str(res)})
                results[sym] = {"status": "error", "message": str(res)}
            else:
                results[sym] = res

        return {
            "status": "success",
            "session_id": session_id,
            "results": results,
            "errors": errors,
            "count": len(results),
            "depth": depth,
        }

    except Exception as e:
        logger.exception("research_batch failed")
        return {"status": "error", "message": str(e)}


async def build_stock_pitch(symbol: str, research_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate 6-element buy-side stock pitch for a symbol.
    Parameter: symbol (string), research_data (dict).
    Implementation: calls researcher_agent with instruction to produce 6-element pitch format. LLM-driven.

    On success the pitch is persisted to the stock_pitches table so the dashboard
    can render it (Phase 3.1). The context is read from AgentState if available,
    otherwise defaults to "manual".
    """
    try:
        from src.autfin.llm.client import get_llm_client

        client = get_llm_client()
        if client is None:
            return {"status": "error", "message": "LLM client unavailable"}

        research_summary = json.dumps(research_data, default=str, indent=2)[:8000]

        system_prompt = (
            "You are a buy-side analyst. Generate a professional 6-element stock pitch. "
            "Output ONLY valid JSON with keys: investment_thesis, catalysts, risks, "
            "valuation_summary, competitive_moat, price_target. "
            "investment_thesis is a 2-3 sentence summary. "
            "catalysts is an array of 2-4 catalysts. "
            "risks is an array of 2-4 key risks. "
            "valuation_summary is a 2-3 sentence analysis. "
            "competitive_moat is a short paragraph. "
            "price_target is a dict with base, upside, downside and rationale. "
            "If sentiment data is included, factor it into your thesis and risk assessment."
        )

        response = await client.generate(
            prompt=f"Build a stock pitch for {symbol} based on:\n{research_summary}",
            system_prompt=system_prompt,
        )

        if response:
            try:
                pitch = json.loads(response) if isinstance(response, str) else response
            except json.JSONDecodeError:
                pitch = {"raw_response": str(response)[:2000]}

            # Persist pitch for the dashboard (Phase 3.1).
            # Read context from current research_data if available, else default.
            persist_ctx = research_data.get("context", "manual") if isinstance(research_data, dict) else "manual"
            try:
                from src.autfin.dashboard.data import save_stock_pitch
                save_stock_pitch(symbol=symbol, context=str(persist_ctx), pitch=pitch)
            except Exception:
                logger.debug("save_stock_pitch persistence skipped", exc_info=True)

            return {
                "status": "success",
                "symbol": symbol.upper(),
                "pitch": pitch,
            }

        return {"status": "error", "message": "LLM returned no response", "symbol": symbol}

    except Exception as e:
        logger.exception(f"build_stock_pitch failed for {symbol}")
        return {"status": "error", "message": str(e), "symbol": symbol}


async def validate_thesis(symbol: str, holding_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if original investment thesis for a holding is still valid.
    Parameter: symbol (string), holding_data (dict).
    Returns INTACT/WEAK/BROKEN + reasoning.
    """
    try:
        from src.autfin.llm.client import get_llm_client

        client = get_llm_client()
        if client is None:
            return {
                "status": "success",
                "symbol": symbol.upper(),
                "verdict": "WEAK",
                "reasoning": "LLM unavailable - thesis requires manual review",
                "confidence": 0.4,
            }

        holding_str = json.dumps(holding_data, default=str, indent=2)[:6000]

        import yfinance as yf

        current_info = {}
        try:
            t = yf.Ticker(f"{symbol.upper().removesuffix('.NS')}.NS")
            info = t.info or {}
            current_info = {
                "ltp": info.get("currentPrice") or info.get("regularMarketPreviousClose", 0),
                "change_pct": info.get("regularMarketChangePercent", 0),
                "pe": info.get("trailingPE", "N/A"),
                "market_cap": info.get("marketCap", "N/A"),
                "revenue_growth": info.get("revenueGrowth", "N/A"),
                "profit_margins": info.get("profitMargins", "N/A"),
            }
        except Exception:
            current_info = {"error": "yfinance fetch failed"}

        prompt = (
            "Compare the original investment thesis with the current data. "
            "Determine if the thesis is INTACT, WEAK, or BROKEN. "
            "Output ONLY valid JSON with keys: verdict (one of INTACT/WEAK/BROKEN), "
            "reasoning (2-3 sentences), confidence (0.0-1.0)."
        )

        response = await client.generate(
            prompt=f"Holding data:\n{holding_str}\n\nCurrent data:\n{json.dumps(current_info)}",
            system_prompt=prompt,
            max_tokens=500,
        )

        if response:
            try:
                result = json.loads(response) if isinstance(response, str) else response
            except json.JSONDecodeError:
                result = {
                    "verdict": "WEAK",
                    "reasoning": "Unable to parse LLM response",
                    "confidence": 0.5,
                }
        else:
            result = {"verdict": "WEAK", "reasoning": "LLM returned no response", "confidence": 0.5}

        return {
            "status": "success",
            "symbol": symbol.upper(),
            "verdict": result.get("verdict", "WEAK"),
            "reasoning": result.get("reasoning", ""),
            "confidence": result.get("confidence", 0.5),
        }

    except Exception as e:
        logger.exception(f"validate_thesis failed for {symbol}")
        return {
            "status": "error",
            "message": str(e),
            "symbol": symbol.upper(),
            "verdict": "WEAK",
            "reasoning": "Error during validation",
            "confidence": 0.0,
        }


async def run_risk_checks(trade: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run 14 risk checks on a proposed trade.
    Parameter: trade (dict with symbol, action, quantity, price).
    Returns approved/vetoed + score + suggested size.
    """
    try:
        from src.autfin.agents.risk_manager import RiskManagerAgent

        symbol = trade.get("symbol", "")
        action = trade.get("action", "BUY")
        quantity = int(trade.get("quantity", 0))
        price = float(trade.get("price", 0))

        # Normalize holdings: ensure list of dicts (not list of strings)
        raw_holdings = trade.get("holdings", [])
        holdings = []
        for h in raw_holdings:
            if isinstance(h, str):
                holdings.append({"tradingsymbol": h, "qty": 0, "ltp": 0, "avg_price": 0})
            elif isinstance(h, dict):
                holdings.append(h)

        portfolio = {
            "holdings": holdings,
            "available_cash": trade.get("available_cash", 1000000.0),
            "positions": trade.get("positions", []),
        }

        agent = RiskManagerAgent()
        trade = {
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
        }
        results = agent.check_all_risks(
            trade=trade,
            portfolio=portfolio,
        )

        # check_all_risks returns: {risk_score: 0-100, risk_level, checks, flagged_issues, ...}
        risk_score_raw = results.get("risk_score", 50)

        # Determine approval: use apply_veto if available, else fall back to "approved" field
        approved = results.get("approved", True)
        veto_reason = None
        try:
            veto_result = agent.apply_veto(risk_score_raw)
            if isinstance(veto_result, tuple) and len(veto_result) == 2:
                should_veto, veto_reason = veto_result
                approved = not should_veto
        except Exception:
            pass

        # Normalize risk_score to 0.0-1.0 for output
        if isinstance(risk_score_raw, (int, float)) and risk_score_raw > 1:
            risk_score_norm = risk_score_raw / 100.0
        else:
            risk_score_norm = float(risk_score_raw)

        flagged = results.get("flagged_issues", results.get("issues", []))

        # Find max position size from failed checks
        suggested_qty = quantity
        for check in results.get("checks", []):
            if not check.get("passes", True) and check.get("name") == "position_size":
                # Extract suggested qty from position_size check reason
                suggested_qty = max(0, quantity - 1)
                break

        return {
            "status": "success",
            "symbol": symbol.upper(),
            "action": action,
            "approved": approved,
            "risk_score": round(risk_score_norm, 3),
            "risk_level": results.get("risk_level", "UNKNOWN"),
            "checks": results.get("checks", []),
            "issues": flagged,
            "suggested_size": suggested_qty,
            "veto_reason": veto_reason if not approved else None,
        }

    except Exception as e:
        logger.exception(f"run_risk_checks failed for {trade.get('symbol', '')}")
        return {
            "status": "error",
            "message": str(e),
            "symbol": trade.get("symbol", ""),
            "approved": False,
            "risk_score": 0.0,
        }


async def allocate_capital(approved_trades: List[Dict[str, Any]], available_cash: float) -> Dict[str, Any]:
    """
    Allocate capital across approved trades.
    Parameter: approved_trades (array), available_cash (number).
    Returns trade_queue.
    """
    if not approved_trades:
        return {"status": "error", "message": "No approved trades provided", "trade_queue": []}

    try:
        from src.autfin.agents.portfolio_manager import portfolio_manager_agent

        state = {
            "messages": [],
            "context": "capital_allocation",
            "available_cash": available_cash,
            "utilized_margin": 0.0,
            "holdings": [],
            "positions": [],
            "trade_queue": approved_trades,
            "thread_id": "allocate_tool",
            "run_mode": "tool",
            "errors": [],
            "total_tokens": 0,
            "total_cost": 0.0,
            "budget_limit": 10.0,
            "validation_count": 0,
        }

        result = await portfolio_manager_agent(state)

        trade_queue = result.get("trade_queue", []) or []
        total_allocated = sum(
            float(t.get("position_size", t.get("amount", 0)) or 0) for t in trade_queue
        )

        return {
            "status": "success",
            "trade_queue": trade_queue,
            "total_allocated": round(total_allocated, 2),
            "available_cash": round(available_cash, 2),
            "remaining_cash": round(available_cash - total_allocated, 2),
            "allocation_count": len(trade_queue),
        }

    except Exception as e:
        logger.exception("allocate_capital failed")
        return {"status": "error", "message": str(e), "trade_queue": []}


async def execute_order(order: Dict[str, Any]) -> Dict[str, Any]:
    """
    Place a trade order via Kite SDK or paper trade.
    Parameter: order (dict).
    Returns {status, order_id, fill_price, mode}.
    """
    try:
        from src.autfin.agents.executor import executor_agent
        from src.autfin.config.settings import AutFinSettings

        raw_conviction = order.get("conviction", 0.5)
        try:
            from src.autfin.learning.calibrator import calibrate_conviction
            adjusted = await calibrate_conviction(raw_conviction, order.get("sector"))
            order["conviction_raw"] = raw_conviction
            order["conviction_calibrated"] = adjusted
            order["conviction"] = adjusted
        except Exception:
            pass

        settings = AutFinSettings()
        run_mode = settings.execution_mode or "recommendation"

        state = {
            "messages": [],
            "context": "order_execution",
            "available_cash": order.get("available_cash", 0.0),
            "utilized_margin": order.get("utilized_margin", 0.0),
            "holdings": order.get("holdings", []),
            "positions": [],
            "trade_queue": [order],
            "thread_id": "exec_tool",
            "run_mode": run_mode,
            "errors": [],
            "total_tokens": 0,
            "total_cost": 0.0,
            "budget_limit": 10.0,
            "validation_count": 0,
        }

        result = await executor_agent(state)

        exec_reports = result.get("execution_reports", []) or []
        report = exec_reports[0] if exec_reports else {}

        return {
            "status": "success",
            "order_id": report.get("order_id", ""),
            "fill_price": report.get("fill_price", order.get("price", 0.0)),
            "mode": run_mode,
            "symbol": order.get("symbol", ""),
            "action": order.get("action", ""),
            "quantity": order.get("quantity", order.get("qty", 0)),
            "execution_report": report,
        }

    except Exception as e:
        logger.exception(f"execute_order failed for {order.get('symbol', '')}")

        import uuid
        fallback_id = f"paper-{uuid.uuid4().hex[:8]}"

        return {
            "status": "failed",
            "order_id": fallback_id,
            "fill_price": order.get("price", 0.0),
            "mode": "paper_fallback",
            "symbol": order.get("symbol", ""),
            "action": order.get("action", ""),
            "quantity": order.get("quantity", order.get("qty", 0)),
            "error": str(e)[:200],
            "note": f"Live execution failed — order NOT placed: {str(e)[:200]}",
        }


async def retrieve_learnings(query: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Search learning journal for similar past setups.
    Parameter: query (string — natural language description of current setup),
    filters (dict — optional sector/pattern filters).
    Returns {similar_setups: [...], calibration: {tier_win_rates}, patterns: [...]}.
    """
    try:
        from src.autfin.memory.database import get_connection, DB_PATH

        conn = get_connection()
        similar_setups = []
        calibration = {}
        patterns = []

        try:
            cursor = conn.cursor()

            where_clauses = ["1=1"]
            params = []
            if filters:
                if filters.get("symbol"):
                    where_clauses.append("symbol = ?")
                    params.append(filters["symbol"].upper())
                if filters.get("outcome"):
                    where_clauses.append("outcome = ?")
                    params.append(1 if filters["outcome"] == "win" else 0)

            keyword_terms = query.lower().split()[:5]
            for term in keyword_terms:
                if len(term) > 2:
                    where_clauses.append("(thesis_summary LIKE ? OR symbol LIKE ?)")
                    params.extend([f"%{term}%", f"%{term}%"])

            where_sql = " AND ".join(where_clauses)

            cursor.execute(
                f"SELECT symbol, created_at, pnl_pct, outcome, thesis_summary FROM learning_journal WHERE {where_sql} ORDER BY created_at DESC LIMIT 10",
                params,
            )
            rows = cursor.fetchall()

            for row in rows:
                similar_setups.append({
                    "symbol": row[0],
                    "date": row[1],
                    "pnl": row[2],
                    "outcome": row[3],
                    "lesson": row[4] or "",
                })

            cursor.execute(
                "SELECT outcome, COUNT(*) FROM learning_journal GROUP BY outcome"
            )
            win_counts = dict(cursor.fetchall())
            total = sum(win_counts.values())
            if total > 0:
                wins = win_counts.get(1, 0)
                losses = win_counts.get(0, 0)
                calibration["win_rate"] = round(wins / total * 100, 1)
                calibration["total_trades"] = total
                calibration["wins"] = wins
                calibration["losses"] = losses
            else:
                calibration = {"win_rate": 0.0, "total_trades": 0, "wins": 0, "losses": 0}

            cursor.execute(
                "SELECT DISTINCT symbol FROM learning_journal ORDER BY created_at DESC LIMIT 20"
            )
            pattern_rows = cursor.fetchall()
            patterns = [p[0] for p in pattern_rows]

        finally:
            try:
                conn.close()
            except Exception:
                pass

        query_lower = query.lower()
        filtered_setups = []
        for s in similar_setups:
            relevance = 0
            if s.get("symbol", "").lower() in query_lower:
                relevance += 3
            lesson = (s.get("lesson") or "").lower()
            for term in query_lower.split():
                if term in lesson:
                    relevance += 1
            s["relevance"] = relevance
            filtered_setups.append(s)

        filtered_setups.sort(key=lambda x: x["relevance"], reverse=True)

        return {
            "status": "success",
            "similar_setups": filtered_setups[:10],
            "calibration": calibration,
            "patterns": patterns,
            "query": query,
        }

    except Exception as e:
        logger.exception("retrieve_learnings failed")
        return {
            "status": "error",
            "message": str(e),
            "similar_setups": [],
            "calibration": {},
            "patterns": [],
        }


async def validate_reasonability(details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gate unrealistic LLM-generated recommendations before they reach the user.

    Checks:
    - Target upside > 20% in 1 week is suspicious
    - Conviction > 0.95 without justification is suspicious
    - Stop-loss > 25% is too wide
    - Risk/reward < 1:2 is unfavorable

    Returns {valid: bool, warnings: list[str], blocked: bool}.
    """
    warnings = []
    blocked = False

    target_upside_pct = details.get("target_upside_pct", 0)
    if target_upside_pct > 20:
        warnings.append(f"Target upside {target_upside_pct:.1f}% in short timeframe is unrealistic")
        blocked = True

    conviction = details.get("conviction", details.get("conviction_score", 0))
    if conviction > 0.95:
        warnings.append(f"Conviction {conviction:.2f} > 0.95 — unusually high, verify evidence")
        blocked = True

    stop_loss_pct = abs(details.get("stop_loss_pct", 0))
    if stop_loss_pct > 25:
        warnings.append(f"Stop-loss {stop_loss_pct:.1f}% is wider than 25% — unusual for Indian equities")
        blocked = True

    risk_reward = details.get("risk_reward_ratio", 0)
    if risk_reward > 0 and risk_reward < 2.0:
        warnings.append(f"Risk/Reward 1:{risk_reward:.1f} below 1:2 minimum")

    return {
        "valid": not blocked,
        "blocked": blocked,
        "warnings": warnings,
    }


async def log_decision(decision_type: str, details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Write a decision to the learning journal.
    Parameter: decision_type (string: trade/buy/sell/thesis_update), details (dict).
    Writes to SQLite learning_journal + ChromaDB vector store.
    Returns {logged: true}.
    """
    try:
        from src.autfin.memory.database import get_connection, init_database

        init_database()
        conn = get_connection()
        import uuid

        decision_id = details.get("decision_id") or str(uuid.uuid4())
        symbol = details.get("symbol", "")
        timestamp = details.get("timestamp") or datetime.now(timezone.utc).isoformat()
        action = details.get("action", "BUY")
        conviction = details.get("conviction", details.get("conviction_score", 0.5))
        pnl_pct = details.get("pnl_pct", details.get("pnl"))
        outcome = details.get("outcome", "completed" if details.get("was_correct") else ("loss" if "was_correct" in details else "open"))
        thesis = details.get("thesis_summary", json.dumps(details) if isinstance(details, dict) else str(details))

        try:
            conn.execute(
                "INSERT INTO learning_journal (symbol, action, conviction, pnl_pct, outcome, thesis_summary, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (symbol, action, conviction, pnl_pct, outcome, thesis, timestamp),
            )
            conn.commit()
        finally:
            try:
                conn.close()
            except Exception:
                pass

        try:
            from src.autfin.memory.vector_store import VectorStore

            vs = VectorStore(mode="persistent")
            lesson_text = details.get("lesson", json.dumps(details, default=str))
            embedding = _generate_embedding(lesson_text)
            vs.add(
                collection_name="learning_journal",
                embeddings=[embedding],
                metadatas=[{
                    "decision_id": decision_id,
                    "decision_type": decision_type,
                    "symbol": symbol,
                    "timestamp": timestamp,
                }],
                ids=[decision_id],
            )
        except Exception as vs_err:
            logger.warning(f"Vector store write skipped: {vs_err}")

        return {
            "status": "success",
            "logged": True,
            "decision_id": decision_id,
            "decision_type": decision_type,
        }

    except Exception as e:
        logger.exception("log_decision failed")
        return {"status": "error", "message": str(e), "logged": False}


async def record_learning_tool(
    symbol: str,
    action: str,
    conviction: float,
    pnl_pct: float,
    thesis_summary: str,
    outcome: str = "completed",
    sector: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Record a trade outcome in the learning journal for future retrieval and calibration.
    Parameters: symbol, action (BUY/SELL), conviction (0-1), pnl_pct (decimal, e.g. 0.05 = 5%),
    thesis_summary, outcome (open/completed), sector.
    """
    try:
        from src.autfin.learning.calibrator import record_learning

        ok = await record_learning(
            symbol=symbol,
            action=action,
            conviction=conviction,
            pnl_pct=pnl_pct,
            thesis_summary=thesis_summary,
            outcome=outcome,
            sector=sector,
        )
        return {"status": "success" if ok else "error", "recorded": ok, "symbol": symbol}
    except Exception as e:
        logger.exception("record_learning_tool failed")
        return {"status": "error", "message": str(e), "recorded": False}


async def calibrate_conviction_tool(
    raw_conviction: float,
    sector: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get a conviction adjustment based on historical calibration data.
    Parameters: raw_conviction (0-1), optional sector.
    Returns adjusted conviction and calibration details.
    """
    try:
        from src.autfin.learning.calibrator import CalibrationTracker

        tracker = CalibrationTracker()
        result = await tracker.suggested_conviction_adjustment(raw_conviction, sector)
        return {"status": "success", **result}
    except Exception as e:
        logger.exception("calibrate_conviction_tool failed")
        return {
            "status": "error",
            "message": str(e),
            "adjusted_conviction": raw_conviction,
        }


async def detect_patterns_tool(min_samples: int = 10) -> Dict[str, Any]:
    """
    Detect recurring patterns in trade outcomes (sector win rates, day-of-week patterns).
    Parameter: min_samples (minimum trades per pattern, default 10).
    """
    try:
        from src.autfin.learning.calibrator import detect_patterns

        patterns = await detect_patterns(min_samples=min_samples)
        return {"status": "success", "patterns": patterns, "count": len(patterns)}
    except Exception as e:
        logger.exception("detect_patterns_tool failed")
        return {"status": "error", "message": str(e), "patterns": []}


async def request_human_approval(question: str, trade_details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pause for human review of a trade.
    Parameter: question (string), trade_details (dict with symbol, action, quantity, price, reason, conviction).
    Sets pending_human_approval in state. Returns {requires_approval: true}.
    """
    import uuid

    approval_id = str(uuid.uuid4())

    pending = {
        "approval_id": approval_id,
        "question": question,
        "symbol": trade_details.get("symbol", ""),
        "action": trade_details.get("action", "BUY"),
        "quantity": trade_details.get("quantity", 0),
        "price": trade_details.get("price", 0.0),
        "reason": trade_details.get("reason", ""),
        "conviction": trade_details.get("conviction", 0.5),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(f"Human approval requested: {pending['symbol']} {pending['action']} @ {pending['price']}")

    return {
        "status": "success",
        "requires_approval": True,
        "pending_approval": pending,
    }


async def check_guardian() -> Dict[str, Any]:
    """
    Check all holdings for stop-loss, profit target, thesis breaks, governance red flags.
    No parameters.
    Returns {sell_signals: [...], alerts: [...]}.
    """
    try:
        from src.autfin.agents.market_data import market_data_agent

        state = {
            "messages": [],
            "context": "guardian_check",
            "available_cash": 0.0,
            "utilized_margin": 0.0,
            "holdings": [],
            "positions": [],
            "thread_id": "guardian_tool",
            "run_mode": "tool",
            "errors": [],
            "total_tokens": 0,
            "total_cost": 0.0,
            "budget_limit": 10.0,
            "validation_count": 0,
        }

        result = await market_data_agent(state)
        holdings = result.get("holdings", []) or []
        market_snapshot = result.get("market_snapshot") or {}
        quotes = market_snapshot.get("quotes", {})

        sell_signals = []
        alerts = []

        from src.autfin.config.settings import get_settings
        _settings = get_settings()
        STOP_LOSS_PCT = -abs(_settings.default_stop_loss_pct) * 100  # 8% → -8.0
        PROFIT_TARGET_PCT = _settings.profit_target_pct * 100  # 15% → 15.0
        DEGRADATION_THRESHOLD_PCT = -25.0

        for holding in holdings:
            symbol = (
                holding.get("tradingsymbol")
                or holding.get("symbol")
                or holding.get("trading_symbol")
                or ""
            )
            if not symbol:
                continue

            avg_price = float(holding.get("avg_price", 0) or 0)
            qty = int(holding.get("qty", holding.get("quantity", 0) or 0))

            quote_obj = quotes.get(symbol) or {}
            ltp = float(quote_obj.get("ltp", 0) or 0)

            if avg_price <= 0 or ltp <= 0:
                continue

            pnl_pct = ((ltp - avg_price) / avg_price) * 100

            signal = {
                "symbol": symbol,
                "avg_price": avg_price,
                "ltp": ltp,
                "pnl_pct": round(pnl_pct, 2),
                "triggers": [],
            }

            if pnl_pct <= STOP_LOSS_PCT:
                signal["triggers"].append(f"STOP_LOSS ({pnl_pct:.1f}%)")
                sell_signals.append(signal)
                alerts.append(f"STOP LOSS: {symbol} at {pnl_pct:.1f}% (LTP: {ltp})")

            elif pnl_pct >= PROFIT_TARGET_PCT:
                signal["triggers"].append(f"PROFIT_TARGET ({pnl_pct:.1f}%)")
                alerts.append(f"PROFIT TARGET: {symbol} at {pnl_pct:.1f}% (LTP: {ltp})")

            if "fundamental_reports" in holding:
                health = holding.get("fundamental_reports", {}).get("health_score", 1.0)
                if float(health) < 0.3:
                    signal["triggers"].append("FUNDAMENTAL_DEGRADATION")
                    sell_signals.append(signal)

            if holding.get("thesis_status") == "BROKEN":
                signal["triggers"].append("THESIS_BROKEN")
                sell_signals.append(signal)

            if signal["triggers"]:
                pass

        if sell_signals:
            try:
                from src.autfin.learning.calibrator import record_learning
                for sig in sell_signals:
                    trigger = sig["triggers"][0] if sig["triggers"] else "UNKNOWN"
                    await record_learning(
                        symbol=sig["symbol"],
                        action="SELL",
                        conviction=0.5,
                        pnl_pct=sig["pnl_pct"] / 100.0,
                        thesis_summary=f"Guardian trigger: {trigger}",
                        outcome="completed",
                    )
            except Exception as learn_err:
                logger.debug("Guardian learning backfill skipped: %s", learn_err)

        return {
            "status": "success",
            "sell_signals": sell_signals,
            "alerts": alerts,
            "holdings_checked": len(holdings),
            "sell_count": len(sell_signals),
            "alert_count": len(alerts),
        }

    except Exception as e:
        logger.exception("check_guardian failed")
        return {
            "status": "error",
            "message": str(e),
            "sell_signals": [],
            "alerts": [],
        }


async def analyze_macro_context() -> Dict[str, Any]:
    """
    Fetch global indices, rates, commodities, compute India impact.
    No parameters.
    Returns {indices, rates, india_impact, narrative, sector_weights}.
    """
    try:
        from src.autfin.agents.global_macro import global_macro_agent

        state = {
            "messages": [],
            "context": "macro_analysis",
            "available_cash": 0.0,
            "utilized_margin": 0.0,
            "holdings": [],
            "positions": [],
            "thread_id": "macro_tool",
            "run_mode": "tool",
            "errors": [],
            "total_tokens": 0,
            "total_cost": 0.0,
            "budget_limit": 20.0,
            "validation_count": 0,
        }

        result = await global_macro_agent(state)
        macro_signals = result.get("macro_signals", {}) or {}

        import yfinance as yf

        global_tickers = {
            "^GSPC": "S&P 500",
            "^IXIC": "NASDAQ",
            "^VIX": "VIX",
            "^NSEI": "Nifty 50",
            "^NSEBANK": "Bank Nifty",
            "DX-Y.NYB": "USD Index",
            "CL=F": "Crude Oil",
            "GC=F": "Gold",
            "^TNX": "10Y Treasury",
            "USDINR=X": "USD/INR",
            "000001.SS": "Shanghai Composite",
            "^HSI": "Hang Seng",
            "^GDAXI": "DAX",
            "^FCHI": "CAC 40",
            "^STOXX50E": "Euro Stoxx 50",
        }

        indices = {}
        for ticker_id, label in global_tickers.items():
            try:
                t = yf.Ticker(ticker_id)
                info = t.info or {}
                fast = t.fast_info if hasattr(t, "fast_info") else None
                ltp = (
                    getattr(fast, "last_price", None)
                    if fast
                    else info.get("regularMarketPrice")
                ) or info.get("previousClose", 0)
                change_pct = info.get("regularMarketChangePercent", 0)
                indices[label] = {
                    "ticker": ticker_id,
                    "ltp": round(float(ltp), 2) if ltp else 0.0,
                    "change_pct": round(float(change_pct), 2) if change_pct else 0.0,
                }
            except Exception:
                indices[label] = {"ticker": ticker_id, "ltp": 0.0, "change_pct": 0.0, "error": "fetch_failed"}

        india_impact = "neutral"
        if indices.get("Nifty 50", {}).get("change_pct", 0) > 0.5:
            india_impact = "bullish"
        elif indices.get("Nifty 50", {}).get("change_pct", 0) < -0.5:
            india_impact = "bearish"

        # Use real compute_sector_impact() instead of hardcoded weights
        try:
            from src.autfin.agents.global_macro import compute_sector_impact
            sector_impact_raw = compute_sector_impact(macro_signals)
            sector_weights = {}
            for sector, info in sector_impact_raw.items():
                sector_weights[sector] = {
                    "weight": 0.10,  # Base weight — adjusted by impact direction
                    "bias": info.get("impact_direction", "neutral"),
                    "strength": info.get("strength", "low"),
                    "drivers": info.get("drivers", []),
                    "reasoning": info.get("reasoning", ""),
                }
        except Exception:
            # Fallback to minimal defaults
            sector_weights = {
                "IT": {"weight": 0.15, "bias": "neutral"},
                "Bank": {"weight": 0.20, "bias": "neutral"},
                "Pharma": {"weight": 0.10, "bias": "neutral"},
                "Auto": {"weight": 0.08, "bias": "neutral"},
                "FMCG": {"weight": 0.12, "bias": "neutral"},
                "Energy": {"weight": 0.15, "bias": "neutral"},
                "Metals": {"weight": 0.07, "bias": "neutral"},
            }

        narrative = f"Global indices mixed. India impact: {india_impact}."

        return {
            "status": "success",
            "indices": indices,
            "rates": {
                "us_10y": indices.get("10Y Treasury", {}).get("ltp", 0.0),
                "india_10y": macro_signals.get("india_10y_yield", {}).get("value"),
                "usd_inr": indices.get("USD/INR", {}).get("ltp", 0.0),
            },
            "india_impact": india_impact,
            "narrative": narrative,
            "sector_weights": sector_weights,
            "macro_signals": macro_signals,
        }

    except Exception as e:
        logger.exception("analyze_macro_context failed")
        return {
            "status": "error",
            "message": str(e),
            "indices": {},
            "rates": {},
            "india_impact": "unknown",
            "narrative": "Macro analysis unavailable",
            "sector_weights": {},
        }


async def get_market_status() -> Dict[str, Any]:
    """
    Check if market is open, get Nifty level, Bank Nifty, global indices, sector performance.
    No parameters.
    Returns {is_open, nifty_level, nifty_change_pct, bank_nifty, global_indices, sectors: {}}.
    """
    try:
        import yfinance as yf

        nifty = yf.Ticker("^NSEI")
        info = nifty.info or {}
        fast = nifty.fast_info if hasattr(nifty, "fast_info") else None

        nifty_ltp = (
            getattr(fast, "last_price", None)
            if fast
            else info.get("regularMarketPrice")
        ) or info.get("previousClose") or 0.0
        nifty_change_pct = info.get("regularMarketChangePercent") or 0.0

        # Bank Nifty
        bank_nifty_ltp = 0.0
        bank_nifty_change_pct = 0.0
        try:
            bn = yf.Ticker("^NSEBANK")
            bn_info = bn.info or {}
            bn_fast = bn.fast_info if hasattr(bn, "fast_info") else None
            bank_nifty_ltp = (
                getattr(bn_fast, "last_price", None)
                if bn_fast
                else bn_info.get("regularMarketPrice")
            ) or bn_info.get("previousClose") or 0.0
            bank_nifty_change_pct = bn_info.get("regularMarketChangePercent") or 0.0
        except Exception:
            pass

        # Global indices
        global_index_map = {
            "^GSPC": "S&P 500",
            "^IXIC": "NASDAQ",
            "^N225": "Nikkei 225",
            "^FTSE": "FTSE 100",
            "000001.SS": "Shanghai Composite",
            "^HSI": "Hang Seng",
            "^GDAXI": "DAX",
            "^FCHI": "CAC 40",
            "^STOXX50E": "Euro Stoxx 50",
            "^VIX": "VIX",
            "DX-Y.NYB": "USD Index",
            "USDINR=X": "USD/INR",
        }
        global_indices = {}
        for ticker_id, label in global_index_map.items():
            try:
                t = yf.Ticker(ticker_id)
                t_info = t.info or {}
                t_fast = t.fast_info if hasattr(t, "fast_info") else None
                ltp = (
                    getattr(t_fast, "last_price", None)
                    if t_fast
                    else t_info.get("regularMarketPrice")
                ) or t_info.get("previousClose", 0)
                change_pct = t_info.get("regularMarketChangePercent", 0)
                global_indices[label] = {
                    "ltp": round(float(ltp), 2) if ltp else 0.0,
                    "change_pct": round(float(change_pct), 2) if change_pct else 0.0,
                }
            except Exception:
                global_indices[label] = {"ltp": 0.0, "change_pct": 0.0}

        now = datetime.now()
        weekday = now.weekday()
        is_weekday = weekday < 5
        market_open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
        is_open = is_weekday and market_open_time <= now <= market_close_time

        sector_indices = {}  # Nifty sectoral indices not available on yfinance

        sectors = {}
        for ticker_id, label in sector_indices.items():
            try:
                t = yf.Ticker(f"^{ticker_id}")
                sec_info = t.info or {}
                sec_fast = t.fast_info if hasattr(t, "fast_info") else None
                sec_ltp = (
                    getattr(sec_fast, "last_price", None)
                    if sec_fast
                    else sec_info.get("regularMarketPrice")
                ) or sec_info.get("previousClose") or 0.0
                sec_change = sec_info.get("regularMarketChangePercent") or 0.0
                sectors[label] = {
                    "ltp": round(float(sec_ltp), 2),
                    "change_pct": round(float(sec_change), 2),
                }
            except Exception:
                sectors[label] = {"ltp": 0.0, "change_pct": 0.0, "error": "fetch_failed"}

        return {
            "status": "success",
            "is_open": is_open,
            "timestamp": now.isoformat(),
            "nifty_level": round(float(nifty_ltp), 2),
            "nifty_change_pct": round(float(nifty_change_pct), 2),
            "bank_nifty_level": round(float(bank_nifty_ltp), 2),
            "bank_nifty_change_pct": round(float(bank_nifty_change_pct), 2),
            "global_indices": global_indices,
            "sectors": sectors,
        }

    except Exception as e:
        logger.exception("get_market_status failed")
        return {
            "status": "error",
            "message": str(e),
            "is_open": False,
            "nifty_level": 0.0,
            "nifty_change_pct": 0.0,
            "bank_nifty_level": 0.0,
            "bank_nifty_change_pct": 0.0,
            "global_indices": {},
            "sectors": {},
        }


async def finalize_session() -> Dict[str, Any]:
    """
    Generate daily summary, log all decisions, send alerts.
    No parameters.
    Returns {summary: string, alerts_sent: number}.
    """
    try:
        summary_parts = []
        alerts_sent = 0

        try:
            portfolio = await get_portfolio_state()
            if portfolio["status"] == "success":
                summary_parts.append(
                    f"Portfolio: Cash ₹{portfolio['available_cash']:,.0f}, "
                    f"Margin Used ₹{portfolio['utilized_margin']:,.0f}, "
                    f"Daily P&L ₹{portfolio['daily_pnl']:,.0f}"
                )
        except Exception as pe:
            summary_parts.append("Portfolio: unavailable")
            logger.warning(f"Portfolio fetch in finalize: {pe}")

        try:
            macro = await analyze_macro_context()
            if macro["status"] == "success":
                summary_parts.append(
                    f"Macro: India impact {macro['india_impact']}, "
                    f"Nifty {'up' if macro.get('indices', {}).get('Nifty 50', {}).get('change_pct', 0) >= 0 else 'down'}"
                )
        except Exception as me:
            logger.warning(f"Macro fetch in finalize: {me}")

        try:
            guardian = await check_guardian()
            if guardian["status"] == "success" and guardian.get("alerts"):
                summary_parts.append(f"Alerts: {len(guardian['alerts'])} active")
                alerts_sent = len(guardian["alerts"])
        except Exception as ge:
            logger.warning(f"Guardian check in finalize: {ge}")

        try:
            market = await get_market_status()
            if market["status"] == "success":
                status_str = "OPEN" if market["is_open"] else "CLOSED"
                summary_parts.append(
                    f"Market: {status_str}, Nifty {market['nifty_level']:,.0f} ({market['nifty_change_pct']:+.2f}%)"
                )
        except Exception as me2:
            logger.warning(f"Market status in finalize: {me2}")

        summary = " | ".join(summary_parts) if summary_parts else "Session summary unavailable"

        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            from src.autfin.memory.database import get_connection, init_database

            init_database()
            conn = get_connection()
            try:
                portfolio = await get_portfolio_state()
                holdings = portfolio.get("holdings", [])
                holdings_value = sum(
                    h.get("ltp", 0) * h.get("quantity", 0) for h in holdings
                )
                total_pnl = sum(h.get("pnl", 0) for h in holdings)
                total_pnl_pct = (
                    (total_pnl / holdings_value * 100) if holdings_value > 0 else 0.0
                )
                conn.execute(
                    "INSERT INTO portfolio_snapshots (date, available_cash, utilized_margin, "
                    "total_holdings_value, total_pnl, total_pnl_pct, holdings_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        timestamp,
                        portfolio.get("available_cash", 0.0),
                        portfolio.get("utilized_margin", 0.0),
                        holdings_value,
                        total_pnl,
                        total_pnl_pct,
                        json.dumps(holdings, default=str),
                    ),
                )
                conn.commit()
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

            await log_decision(
                decision_type="session_summary",
                details={"summary": summary, "alerts_sent": alerts_sent, "timestamp": timestamp},
            )
        except Exception as db_err:
            logger.warning(f"Database write in finalize: {db_err}")

        return {
            "status": "success",
            "summary": summary,
            "alerts_sent": alerts_sent,
            "timestamp": timestamp,
        }

    except Exception as e:
        logger.exception("finalize_session failed")
        return {
            "status": "error",
            "message": str(e),
            "summary": "Session finalization failed",
            "alerts_sent": 0,
        }


# ================== TOOL DEFINITIONS ==================

SUPERVISOR_TOOLS: List[Dict[str, Any]] = [
    _tool_def(
        "get_portfolio_state",
        "Get available cash, utilized margin, holdings, positions from Kite SDK. No parameters. Returns JSON with cash, margin, holdings, positions, daily_pnl.",
        {},
    ),
    _tool_def(
        "get_market_quotes",
        "Get live quotes for symbols via Kite SDK or yfinance fallback.",
        {
            "symbols": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of stock symbols (e.g. ['RELIANCE', 'TCS'])",
            },
        },
    ),
    _tool_def(
        "screen_candidates",
        "Run 3-filter screening: liquidity → magic formula → F-Score + red flags. Returns 15-25 top scored symbols.",
        {},
    ),
    _tool_def(
        "research_stock",
        "Run all 6 research agents on ONE stock (fundamentals, technical, sentiment, valuation, peer compare, moat).",
        {
            "symbol": {
                "type": "string",
                "description": "Stock symbol (e.g. RELIANCE)",
            },
            "depth": {
                "type": "string",
                "description": "Research depth: quick, standard, or deep",
            },
        },
    ),
    _tool_def(
        "research_batch",
        "Run research_stock on multiple symbols concurrently.",
        {
            "symbols": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of stock symbols to research",
            },
            "depth": {
                "type": "string",
                "description": "Research depth: quick, standard, or deep",
            },
        },
    ),
    _tool_def(
        "build_stock_pitch",
        "Generate 6-element buy-side stock pitch for a symbol using LLM synthesis.",
        {
            "symbol": {
                "type": "string",
                "description": "Stock symbol",
            },
            "research_data": {
                "type": "object",
                "description": "Research data dict from research_stock",
            },
        },
    ),
    _tool_def(
        "validate_thesis",
        "Check if original investment thesis for a holding is still valid. Returns INTACT/WEAK/BROKEN + reasoning.",
        {
            "symbol": {
                "type": "string",
                "description": "Stock symbol",
            },
            "holding_data": {
                "type": "object",
                "description": "Holding data including original thesis, entry price, conviction",
            },
        },
    ),
    _tool_def(
        "run_risk_checks",
        "Run 14 risk checks on a proposed trade. Returns approved/vetoed + score + suggested size.",
        {
            "trade": {
                "type": "object",
                "description": "Trade dict with symbol, action, quantity, price, and optional holdings/positions context",
            },
        },
    ),
    _tool_def(
        "allocate_capital",
        "Allocate capital across approved trades using Kelly + conviction-weighted sizing.",
        {
            "approved_trades": {
                "type": "array",
                "items": {"type": "object"},
                "description": "List of approved trade dicts",
            },
            "available_cash": {
                "type": "number",
                "description": "Available cash to allocate in INR",
            },
        },
    ),
    _tool_def(
        "execute_order",
        "Place a trade order via Kite SDK or paper trade based on run mode.",
        {
            "order": {
                "type": "object",
                "description": "Order dict with symbol, action, quantity/qty, price, order_type",
            },
        },
    ),
    _tool_def(
        "retrieve_learnings",
        "Search learning journal for similar past setups using keyword + vector search.",
        {
            "query": {
                "type": "string",
                "description": "Natural language description of current setup",
            },
            "filters": {
                "type": "object",
                "description": "Optional filters: sector, pattern, outcome (win/loss)",
            },
        },
    ),
    _tool_def(
        "log_decision",
        "Write a decision to the learning journal (SQLite + ChromaDB).",
        {
            "decision_type": {
                "type": "string",
                "description": "Decision type: trade, buy, sell, thesis_update",
            },
            "details": {
                "type": "object",
                "description": "Decision details dict",
            },
        },
    ),
    _tool_def(
        "request_human_approval",
        "Pause for human review of a trade before execution.",
        {
            "question": {
                "type": "string",
                "description": "Question for the human reviewer",
            },
            "trade_details": {
                "type": "object",
                "description": "Trade details: symbol, action, quantity, price, reason, conviction",
            },
        },
    ),
    _tool_def(
        "check_guardian",
        "Check all holdings for stop-loss, profit target, thesis breaks, governance red flags.",
        {},
    ),
    _tool_def(
        "analyze_macro_context",
        "Fetch global indices, rates, commodities, compute India impact.",
        {},
    ),
    _tool_def(
        "get_market_status",
        "Check if market is open, get Nifty level, sector performance.",
        {},
    ),
    _tool_def(
        "finalize_session",
        "Generate daily summary, log all decisions, send alerts.",
        {},
    ),
    _tool_def(
        "record_learning",
        "Record a trade outcome in the learning journal for future retrieval and calibration.",
        {
            "symbol": {"type": "string", "description": "Stock symbol (e.g. RELIANCE)"},
            "action": {"type": "string", "description": "Trade action: BUY or SELL"},
            "conviction": {"type": "number", "description": "Conviction score (0-1)"},
            "pnl_pct": {"type": "number", "description": "P&L as decimal (e.g. 0.05 = 5% profit)"},
            "thesis_summary": {"type": "string", "description": "Brief thesis or trade rationale"},
            "outcome": {"type": "string", "description": "Trade outcome: open or completed"},
            "sector": {"type": "string", "description": "Optional sector name"},
        },
    ),
    _tool_def(
        "calibrate_conviction",
        "Get a conviction adjustment based on historical calibration data.",
        {
            "raw_conviction": {"type": "number", "description": "Raw conviction score (0-1)"},
            "sector": {"type": "string", "description": "Optional sector for sector-specific calibration"},
        },
    ),
    _tool_def(
        "detect_patterns",
        "Detect recurring patterns in trade outcomes (sector win rates, day-of-week patterns).",
        {
            "min_samples": {"type": "integer", "description": "Minimum trades per pattern (default 10)"},
        },
    ),
]


# ================== TOOL CALLABLES ==================

SUPERVISOR_CALLABLES: Dict[str, Any] = {
    "get_portfolio_state": get_portfolio_state,
    "get_market_quotes": get_market_quotes,
    "screen_candidates": screen_candidates,
    "research_stock": research_stock,
    "research_batch": research_batch,
    "build_stock_pitch": build_stock_pitch,
    "validate_thesis": validate_thesis,
    "run_risk_checks": run_risk_checks,
    "allocate_capital": allocate_capital,
    "execute_order": execute_order,
    "retrieve_learnings": retrieve_learnings,
    "log_decision": log_decision,
    "request_human_approval": request_human_approval,
    "check_guardian": check_guardian,
    "analyze_macro_context": analyze_macro_context,
    "get_market_status": get_market_status,
    "finalize_session": finalize_session,
    "record_learning": record_learning_tool,
    "calibrate_conviction": calibrate_conviction_tool,
    "detect_patterns": detect_patterns_tool,
}
