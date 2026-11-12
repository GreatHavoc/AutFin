"""v3.0 Stock Screening Pipeline — Liquidity, Magic Formula, Piotroski F-Score & Indian Red Flags."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

NIFTY_500_SYMBOLS: list[str] = [
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "INFY",
    "ICICIBANK",
    "HINDUNILVR",
    "ITC",
    "KOTAKBANK",
    "SBIN",
    "BHARTIARTL",
    "LT",
    "AXISBANK",
    "BAJFINANCE",
    "ASIANPAINT",
    "MARUTI",
    "TITAN",
    "SUNPHARMA",
    "HCLTECH",
    "WIPRO",
    "NTPC",
    "POWERGRID",
    "ONGC",
    "ADANIENT",
    "ADANIPORTS",
    "JSWSTEEL",
    "TATASTEEL",
    # TATAMOTORS removed — not available on yfinance (NSE symbol exists, Yahoo data missing)
    "TECHM",
    "ULTRACEMCO",
    "BAJAJFINSV",
    "NESTLEIND",
    "DRREDDY",
    "CIPLA",
    "DIVISLAB",
    "APOLLOHOSP",
    "BRITANNIA",
    "EICHERMOT",
    "INDUSINDBK",
    "PNB",
    "BANKBARODA",
    "COALINDIA",
    "GRASIM",
    "HDFCLIFE",
    "SBILIFE",
    "HEROMOTOCO",
    "BAJAJ-AUTO",
    "HINDZINC",
    "VEDL",
    "DLF",
    "GODREJCP",
    "HAVELLS",
    "ABB",
    "SIEMENS",
    "PIDILITIND",
    "SHRIRAMFIN",  # formerly SRTRANSFIN (renamed after merger)
    "CHOLAFIN",
    "BAJAJHLDNG",
    "MUTHOOTFIN",
    "BHARATFORG",
    "ASHOKLEY",
    "BEL",
    "HAL",
]


def liquidity_filter(stocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Eliminate stocks with avg daily volume < ₹1 Cr or market cap < ₹500 Cr."""
    return [
        s
        for s in stocks
        if (s.get("avg_volume") or 0) >= 1_00_00_000
        and (s.get("market_cap") or 0) >= 500_00_00_000
    ]


def magic_formula_rank(stocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank by combined Earnings Yield rank + ROIC rank (Joel Greenblatt).

    Returns the top 85 stocks.
    """
    for s in stocks:
        ebit = s.get("ebit") or 0
        market_cap = s.get("market_cap") or 0
        total_debt = s.get("total_debt") or 0
        cash = s.get("cash") or 0
        ev = market_cap + total_debt - cash
        nfa = s.get("net_fixed_assets") or 0
        current_assets = s.get("current_assets") or 0
        current_liabilities = s.get("current_liabilities") or 0
        wc = current_assets - current_liabilities

        s["earnings_yield"] = ebit / ev if ev > 0 else 0.0
        s["roic"] = ebit / (nfa + wc) if (nfa + wc) > 0 else 0.0

    ey_rank = {
        row["symbol"]: i
        for i, row in enumerate(
            sorted(stocks, key=lambda x: x.get("earnings_yield", 0), reverse=True)
        )
    }
    roic_rank = {
        row["symbol"]: i
        for i, row in enumerate(
            sorted(stocks, key=lambda x: x.get("roic", 0), reverse=True)
        )
    }

    for s in stocks:
        s["magic_rank"] = ey_rank.get(s["symbol"], 999) + roic_rank.get(s["symbol"], 999)

    return sorted(stocks, key=lambda x: x.get("magic_rank", 999))[:85]


def piotroski_f_score(current: dict[str, Any], prior: dict[str, Any]) -> int:
    """9-point financial health score. Returns 0-9; >= 6 indicates fundamentally improving."""
    score = 0

    # --- Profitability (4 points) ---
    # 1. Positive ROA
    score += 1 if (current.get("roa") or 0) > 0 else 0

    # 2. Positive operating cash flow / total assets
    cur_ocf = current.get("operating_cf") or 0
    cur_ta = max(current.get("total_assets") or 1, 1)
    score += 1 if cur_ocf / cur_ta > 0 else 0

    # 3. ROA improved vs prior year
    score += 1 if (current.get("roa") or 0) > (prior.get("roa") or 0) else 0

    # 4. Operating CF > ROA (accrual quality check)
    cfo_roa = cur_ocf / cur_ta
    score += 1 if cfo_roa > (current.get("roa") or 0) else 0

    # --- Financial health (3 points) ---
    # 5. Lower leverage vs prior year
    cur_ltdebt = current.get("long_term_debt") or 0
    prev_ltdebt = prior.get("long_term_debt") or 0
    cur_ta_fin = max(current.get("total_assets") or 1, 1)
    prev_ta_fin = max(prior.get("total_assets") or 1, 1)
    lev = cur_ltdebt / cur_ta_fin
    lev_prior = prev_ltdebt / prev_ta_fin
    score += 1 if lev < lev_prior else 0

    # 6. Improved current ratio
    cur_ca = current.get("current_assets") or 0
    cur_cl = max(current.get("current_liabilities") or 1, 1)
    prev_ca = prior.get("current_assets") or 0
    prev_cl = max(prior.get("current_liabilities") or 1, 1)
    cr = cur_ca / cur_cl
    cr_prior = prev_ca / prev_cl
    score += 1 if cr > cr_prior else 0

    # 7. Shares outstanding not increased (no dilution)
    cur_shares = current.get("shares_outstanding") or 1e12
    prev_shares = prior.get("shares_outstanding") or 1e12
    score += 1 if cur_shares <= prev_shares else 0

    # --- Efficiency (2 points) ---
    # 8. Improved gross margin
    cur_gp = current.get("gross_profit") or 0
    cur_rev_eff = max(current.get("revenue") or 1, 1)
    prev_gp = prior.get("gross_profit") or 0
    prev_rev_eff = max(prior.get("revenue") or 1, 1)
    gm = cur_gp / cur_rev_eff
    gm_prior = prev_gp / prev_rev_eff
    score += 1 if gm > gm_prior else 0

    # 9. Improved asset turnover
    cur_rev_at = current.get("revenue") or 0
    cur_ta_at = max(current.get("total_assets") or 1, 1)
    prev_rev_at = prior.get("revenue") or 0
    prev_ta_at = max(prior.get("total_assets") or 1, 1)
    at = cur_rev_at / cur_ta_at
    at_prior = prev_rev_at / prev_ta_at
    score += 1 if at > at_prior else 0

    return score


def indian_red_flags(stock: dict[str, Any]) -> list[str]:
    """Indian market-specific governance checks. Any hit = auto-reject from candidates."""
    flags: list[str] = []

    promoter_change = stock.get("promoter_holding_change_4q") or 0
    if promoter_change < -0.05:
        flags.append(
            f"Promoter holding declined {abs(promoter_change) * 100:.0f}% in 4 quarters"
        )

    promoter_pledge = stock.get("promoter_pledge_pct") or 0
    if promoter_pledge > 20:
        flags.append(f"Promoter pledge {promoter_pledge:.0f}% (>20%)")

    if stock.get("auditor_qualification_2y"):
        flags.append("Auditor qualification in last 2 years")

    related_party = stock.get("related_party_txn_pct") or 0
    if related_party > 10:
        flags.append(
            f"Related-party transactions {related_party:.0f}% of revenue (>10%)"
        )

    ocf_streak = stock.get("negative_ocf_streak") or 0
    if ocf_streak >= 3:
        flags.append("Negative operating cash flow 3+ consecutive quarters")

    if stock.get("sebi_penalty_12m"):
        flags.append("SEBI investigation/penalty in last 12 months")

    debt_equity = stock.get("debt_equity") or 0
    if promoter_pledge > 50 and debt_equity > 2:
        flags.append(
            "CRITICAL: High promoter pledge (>50%) + high debt (D/E>2) — VETO"
        )

    return flags


async def fetch_yfinance_bulk(symbols: list[str]) -> list[dict[str, Any]]:
    """Fetch key financials for multiple symbols from yfinance in batch.

    Returns list of dicts with: symbol, market_cap, avg_volume, revenue,
    ebit, roa, total_assets, current_assets, current_liabilities,
    long_term_debt, total_debt, cash, net_fixed_assets, gross_profit,
    operating_cf, shares_outstanding, pe, pb, roe, debt_equity,
    fcf_yield, promoter_holding, promoter_pledge_pct, sector,
    plus prior-year values for F-Score comparison.
    """
    results: list[dict[str, Any]] = []

    for symbol in symbols:
        try:
            import yfinance as yf

            from autfin.agents.yfinance_utils import normalize_symbol

            ticker = yf.Ticker(normalize_symbol(symbol))
            info = ticker.info or {}

            hist = ticker.history(period="5d")
            avg_volume = (
                float(hist["Volume"].mean())
                if hist is not None
                and not hist.empty
                and "Volume" in hist.columns
                else 0.0
            )

            bs = ticker.balance_sheet
            inc = ticker.financials
            cf = ticker.cashflow

            bs_latest: dict[str, Any] = (
                bs.iloc[:, 0].to_dict() if bs is not None and not bs.empty else {}
            )
            inc_latest: dict[str, Any] = (
                inc.iloc[:, 0].to_dict() if inc is not None and not inc.empty else {}
            )
            cf_latest: dict[str, Any] = (
                cf.iloc[:, 0].to_dict() if cf is not None and not cf.empty else {}
            )
            bs_prior: dict[str, Any] = (
                bs.iloc[:, 1].to_dict()
                if bs is not None and bs.shape[1] > 1
                else {}
            )
            inc_prior: dict[str, Any] = (
                inc.iloc[:, 1].to_dict()
                if inc is not None and inc.shape[1] > 1
                else {}
            )

            net_income = inc_latest.get("Net Income") or 0
            total_assets = bs_latest.get("Total Assets") or 0
            market_cap = info.get("marketCap") or 0

            results.append(
                {
                    "symbol": symbol,
                    "market_cap": market_cap,
                    "avg_volume": avg_volume,
                    "sector": info.get("sector") or "Unknown",
                    # Income statement
                    "revenue": inc_latest.get("Total Revenue") or 0,
                    "ebit": inc_latest.get("EBIT")
                    or inc_latest.get("Operating Income")
                    or 0,
                    "gross_profit": inc_latest.get("Gross Profit") or 0,
                    # Balance sheet
                    "roa": net_income / max(total_assets or 1, 1),
                    "total_assets": total_assets,
                    "current_assets": bs_latest.get("Current Assets") or 0,
                    "current_liabilities": bs_latest.get("Current Liabilities") or 0,
                    "long_term_debt": bs_latest.get("Long Term Debt") or 0,
                    "total_debt": bs_latest.get("Total Debt") or 0,
                    "cash": bs_latest.get("Cash And Cash Equivalents") or 0,
                    "net_fixed_assets": bs_latest.get("Net PPE") or 0,
                    # Cash flow
                    "operating_cf": cf_latest.get("Operating Cash Flow") or 0,
                    # Shares
                    "shares_outstanding": info.get("sharesOutstanding") or 0,
                    # Prior-period values for F-Score
                    "revenue_prior": inc_prior.get("Total Revenue") or 0,
                    "roa_prior": (inc_prior.get("Net Income") or 0)
                    / max(bs_prior.get("Total Assets") or 1, 1),
                    "total_assets_prior": bs_prior.get("Total Assets") or 0,
                    "current_assets_prior": bs_prior.get("Current Assets") or 0,
                    "current_liabilities_prior": bs_prior.get("Current Liabilities")
                    or 0,
                    "long_term_debt_prior": bs_prior.get("Long Term Debt") or 0,
                    "gross_profit_prior": inc_prior.get("Gross Profit") or 0,
                    "shares_outstanding_prior": info.get("sharesOutstanding") or 0,
                    # Valuation / quality
                    "pe": info.get("trailingPE") or info.get("forwardPE"),
                    "pb": info.get("priceToBook"),
                    "roe": info.get("returnOnEquity"),
                    "debt_equity": info.get("debtToEquity")
                    or (
                        (info.get("totalDebt") or 0)
                        / max(info.get("totalEquity") or 1, 1)
                    ),
                    "fcf_yield": (cf_latest.get("Free Cash Flow") or 0)
                    / max(market_cap or 1, 1),
                    "promoter_holding": info.get("heldPercentInsiders"),
                    # Fields not available from yfinance — require Indian data source
                    "promoter_pledge_pct": 0,
                    "promoter_holding_change_4q": 0,
                    "auditor_qualification_2y": False,
                    "related_party_txn_pct": 0,
                    "negative_ocf_streak": 0,
                    "sebi_penalty_12m": False,
                }
            )
        except Exception:
            logger.warning("Failed to fetch yfinance data for %s", symbol, exc_info=True)

    return results


# ---------------------------------------------------------------------------
# Market Regime Detection (Phase 2.6)
# ---------------------------------------------------------------------------

async def detect_market_regime() -> dict[str, Any]:
    """Detect current market regime (bull/bear/neutral) from Nifty 50 and Bank Nifty.

    Uses 50-day and 200-day SMA crossover + recent momentum on Nifty 50.
    Bank Nifty SMA is tracked separately for banking sector regime.

    Returns: {regime: str, nifty_price: float, sma_50: float, sma_200: float,
              confidence: float, bank_nifty_regime: str}
    """
    try:
        import yfinance as yf

        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(period="1y")

        if hist.empty or len(hist) < 200:
            return {"regime": "neutral", "nifty_price": 0, "sma_50": 0, "sma_200": 0,
                    "confidence": 0, "bank_nifty_regime": "neutral"}

        close = hist["Close"]
        current = float(close.iloc[-1])
        sma_50 = float(close.rolling(50).mean().iloc[-1])
        sma_200 = float(close.rolling(200).mean().iloc[-1])

        above_200 = current > sma_200
        golden_cross = sma_50 > sma_200

        if above_200 and golden_cross:
            regime = "bull"
            confidence = min(1.0, (current - sma_200) / sma_200 * 10 + 0.5)
        elif not above_200 and not golden_cross:
            regime = "bear"
            confidence = min(1.0, (sma_200 - current) / sma_200 * 10 + 0.5)
        else:
            regime = "neutral"
            confidence = 0.5

        # Bank Nifty regime for banking sector context
        bank_nifty_regime = "neutral"
        try:
            bank_nifty = yf.Ticker("^NSEBANK")
            bn_hist = bank_nifty.history(period="1y")
            if not bn_hist.empty and len(bn_hist) >= 200:
                bn_close = bn_hist["Close"]
                bn_current = float(bn_close.iloc[-1])
                bn_sma_50 = float(bn_close.rolling(50).mean().iloc[-1])
                bn_sma_200 = float(bn_close.rolling(200).mean().iloc[-1])
                if bn_current > bn_sma_200 and bn_sma_50 > bn_sma_200:
                    bank_nifty_regime = "bull"
                elif bn_current < bn_sma_200 and bn_sma_50 < bn_sma_200:
                    bank_nifty_regime = "bear"
        except Exception:
            pass

        return {
            "regime": regime,
            "nifty_price": round(current, 2),
            "sma_50": round(sma_50, 2),
            "sma_200": round(sma_200, 2),
            "confidence": round(confidence, 3),
            "bank_nifty_regime": bank_nifty_regime,
        }
    except Exception as e:
        logger.debug("Market regime detection failed: %s", e)
        return {"regime": "neutral", "nifty_price": 0, "sma_50": 0, "sma_200": 0,
                "confidence": 0, "bank_nifty_regime": "neutral"}


def get_fscore_threshold(regime: str) -> int:
    """Return F-Score threshold based on market regime.

    - Bull market:  5 (more aggressive, accept weaker fundamentals)
    - Bear market:  7 (more defensive, only strong fundamentals)
    - Neutral:      6 (standard)
    """
    thresholds = {"bull": 5, "bear": 7, "neutral": 6}
    return thresholds.get(regime, 6)


async def run_screening_pipeline() -> dict[str, Any]:
    """Run the full 3-filter stock screening pipeline.

    Filters:
      1. Liquidity — eliminate illiquid / micro-cap stocks.
      2. Magic Formula — top 85 by combined Earnings Yield + ROIC rank.
      3. Piotroski F-Score >= threshold (auto-adjusted by market regime) and zero Indian red flags.

    Returns {"candidates": [...], "candidate_details": [...], "stats": {...}}.
    """
    symbols = NIFTY_500_SYMBOLS

    regime_data = await detect_market_regime()
    regime = regime_data.get("regime", "neutral")
    fscore_threshold = get_fscore_threshold(regime)

    stocks = await fetch_yfinance_bulk(symbols)

    stocks = liquidity_filter(stocks)

    stocks = magic_formula_rank(stocks)

    candidates: list[dict[str, Any]] = []
    for s in stocks:
        current = {
            "roa": s.get("roa") or 0,
            "total_assets": s.get("total_assets") or 0,
            "current_assets": s.get("current_assets") or 0,
            "current_liabilities": s.get("current_liabilities") or 0,
            "long_term_debt": s.get("long_term_debt") or 0,
            "gross_profit": s.get("gross_profit") or 0,
            "revenue": s.get("revenue") or 0,
            "shares_outstanding": s.get("shares_outstanding") or 0,
            "operating_cf": s.get("operating_cf") or 0,
        }
        prior = {
            "roa": s.get("roa_prior") or 0,
            "total_assets": s.get("total_assets_prior") or 0,
            "current_assets": s.get("current_assets_prior") or 0,
            "current_liabilities": s.get("current_liabilities_prior") or 0,
            "long_term_debt": s.get("long_term_debt_prior") or 0,
            "gross_profit": s.get("gross_profit_prior") or 0,
            "revenue": s.get("revenue_prior") or 0,
            "shares_outstanding": s.get("shares_outstanding_prior") or 0,
            "operating_cf": 0,
        }

        f_score = piotroski_f_score(current, prior)
        red_flags = indian_red_flags(s)

        if f_score >= fscore_threshold and not red_flags:
            s["f_score"] = f_score
            s["composite_score"] = (
                (s.get("earnings_yield") or 0) * 0.5
                + (s.get("roic") or 0) * 0.3
                + (f_score / 9) * 0.2
            )
            candidates.append(s)

    candidates = sorted(
        candidates, key=lambda x: x.get("composite_score", 0), reverse=True
    )[:20]

    return {
        "candidates": [c["symbol"] for c in candidates],
        "candidate_details": candidates,
        "stats": {
            "universe_size": len(symbols),
            "after_liquidity": len(stocks),
            "after_magic_formula": min(85, len(stocks)),
            "final_candidates": len(candidates),
            "market_regime": regime,
            "fscore_threshold": fscore_threshold,
            "regime_data": regime_data,
        },
    }


# ---------------------------------------------------------------------------
# Sector-Rotation Overlay (Phase 1.10)
# ---------------------------------------------------------------------------

# Sector classification for Nifty 500 symbols
_SECTOR_CLASSIFICATION: dict[str, str] = {
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT", "TECHM": "IT",
    "LTIM": "IT", "LTTS": "IT", "MPHASIS": "IT", "PERSISTENT": "IT",
    "COFORGE": "IT", "CYIENT": "IT", "KPITTECH": "IT",
    "HDFCBANK": "BANKING", "ICICIBANK": "BANKING", "KOTAKBANK": "BANKING",
    "AXISBANK": "BANKING", "SBIN": "BANKING", "INDUSINDBK": "BANKING",
    "BANDHANBNK": "BANKING", "FEDERALBNK": "BANKING", "PNB": "BANKING",
    "BANKBARODA": "BANKING", "AUBANK": "BANKING",
    "SUNPHARMA": "PHARMA", "DRREDDY": "PHARMA", "CIPLA": "PHARMA",
    "DIVISLAB": "PHARMA", "LUPIN": "PHARMA", "BIOCON": "PHARMA",
    "MARUTI": "AUTO", "TATAMOTORS": "AUTO", "M&M": "AUTO",
    "EICHERMOT": "AUTO", "HEROMOTOCO": "AUTO", "ASHOKLEY": "AUTO",
    "BAJAJ-AUTO": "AUTO", "TVSMOTOR": "AUTO",
    "TATASTEEL": "METALS", "JSWSTEEL": "METALS", "HINDALCO": "METALS",
    "VEDL": "METALS", "HINDZINC": "METALS", "COALINDIA": "METALS",
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "DABUR": "FMCG", "MARICO": "FMCG",
    "TATACONSUM": "FMCG", "GODREJCP": "FMCG",
    "RELIANCE": "ENERGY", "ONGC": "ENERGY", "NTPC": "ENERGY",
    "POWERGRID": "ENERGY", "ADANIENT": "ENERGY", "ADANIPORTS": "ENERGY",
    "TATAPOWER": "ENERGY", "ADANIPOWER": "ENERGY",
    "BAJFINANCE": "FINANCE", "BAJAJFINSV": "FINANCE", "CHOLAFIN": "FINANCE",
    "HDFCLIFE": "FINANCE", "SBILIFE": "FINANCE", "MUTHOOTFIN": "FINANCE",
    "LT": "INFRA", "DLF": "INFRA", "ULTRACEMCO": "INFRA",
}


def _get_symbol_sector(symbol: str) -> str:
    """Get sector classification for a symbol."""
    return _SECTOR_CLASSIFICATION.get(symbol.upper(), "OTHER")


def apply_sector_rotation(
    candidates: list[dict[str, Any]],
    sector_biases: dict[str, float],
) -> list[dict[str, Any]]:
    """
    Adjust candidate composite scores based on macro sector biases.

    Args:
        candidates: List of candidate dicts with 'symbol' and 'composite_score'
        sector_biases: Dict of sector → bias (-1.0 bearish to +1.0 bullish)
                       from analyze_macro_context sector_weights

    Returns:
        Re-sorted candidates with adjusted scores
    """
    for c in candidates:
        sector = _get_symbol_sector(c.get("symbol", ""))
        bias = sector_biases.get(sector, 0.0)
        # Bias adjustment: ±25% of composite score
        adjustment = 1.0 + (bias * 0.25)
        c["sector"] = sector
        c["sector_bias"] = round(bias, 3)
        c["adjusted_score"] = round(c.get("composite_score", 0) * adjustment, 4)

    return sorted(candidates, key=lambda x: x.get("adjusted_score", 0), reverse=True)


async def run_screening_pipeline_with_sector_rotation(
    sector_biases: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Run the full screening pipeline with macro-overlayed sector rotation.

    Args:
        sector_biases: Sector bias dict from analyze_macro_context sector_weights.
                       If None, runs standard pipeline without rotation.

    Returns:
        Same format as run_screening_pipeline + sector_rotation_applied flag
    """
    base = await run_screening_pipeline()

    if sector_biases is None:
        base["sector_rotation_applied"] = False
        return base

    candidate_details = base.get("candidate_details", [])
    if not candidate_details:
        base["sector_rotation_applied"] = False
        return base

    rotated = apply_sector_rotation(candidate_details, sector_biases)
    top = rotated[:20]

    base["candidates"] = [c["symbol"] for c in top]
    base["candidate_details"] = top
    base["sector_rotation_applied"] = True
    base["sector_biases"] = sector_biases
    return base
