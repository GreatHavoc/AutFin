"""
Nifty 500 Universe — symbol list, sector mapping, and bulk yfinance data fetcher.

Phase 1.1: Dedicated module for the investable universe. Separates the symbol
list and batch data fetching from the screening pipeline so both can be used
independently (e.g., fan-out research, guardian checks, macro overlays).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Nifty 500 symbols (NSE only, yfinance-compatible)
# ---------------------------------------------------------------------------

NIFTY_500: list[str] = sorted(set([
    # ── Large Cap (Nifty 50) ──
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY",
    "HINDUNILVR", "ITC", "KOTAKBANK", "SBIN", "BHARTIARTL",
    "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI",
    "TITAN", "SUNPHARMA", "HCLTECH", "WIPRO", "NTPC",
    "POWERGRID", "ONGC", "ADANIENT", "ADANIPORTS", "JSWSTEEL",
    "TATASTEEL", "TECHM", "ULTRACEMCO", "BAJAJFINSV", "NESTLEIND",
    "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP", "BRITANNIA",
    "EICHERMOT", "INDUSINDBK", "PNB", "BANKBARODA", "COALINDIA",
    "GRASIM", "HDFCLIFE", "SBILIFE", "HEROMOTOCO", "BAJAJ-AUTO",
    "HINDZINC", "VEDL", "DLF", "GODREJCP", "HAVELLS",
    # ── Mid/Small Cap (Nifty Next 50 / Midcap 100) ──
    "ABB", "SIEMENS", "PIDILITIND", "SHRIRAMFIN", "CHOLAFIN",
    "BAJAJHLDNG", "MUTHOOTFIN", "BHARATFORG", "ASHOKLEY", "BEL",
    "HAL", "DMART", "VBL", "MARICO", "TATACONSUM",
    "UBL", "COLPAL", "DABUR", "LUPIN",
    "BIOCON", "AUROPHARMA", "ALKEM", "TORNTPHARM", "IPCALAB",
    "ACC", "AMBUJACEM", "DALBHARAT", "HEIDELBERG", "SHREECEM",
    "TATAPOWER", "ADANIPOWER", "NHPC", "SJVN", "IRFC",
    "RECLTD", "PFC", "LICI", "SBICARD",
    "MANAPPURAM", "CANFINHOME", "LICHSGFIN",
    "FEDERALBNK", "BANDHANBNK", "IDFCFIRSTB", "AUBANK", "BANKINDIA",
    "CANBK", "UNIONBANK", "INDIANB", "CENTRALBK", "MAHABANK",
    "TATAMOTORS", "M&M", "TVSMOTOR", "EXIDEIND", "AMARARAJA",
    "BOSCHLALL", "SONACOMS", "SCHAEFFLER", "SKFINDIA", "TIMKEN",
    "DIXON", "CLEAN", "ZOMATO", "PAYTM", "POLICYBZR",
    "NYKAA", "DELHIVERY", "MAPMYINDIA", "KPITTECH", "PERSISTENT",
    "COFORGE", "MPHASIS", "LTIM", "LTTS", "CYIENT",
    "CAMS", "KFINTECH", "BSE", "CDSL", "MCX",
]))

# Sector mapping for rotation overlay
SECTOR_MAP: dict[str, list[str]] = {
    "IT": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM", "LTTS",
            "MPHASIS", "PERSISTENT", "COFORGE", "CYIENT", "KPITTECH",
            "BSE", "CAMS", "CDSL", "KFINTECH", "MCX"],
    "BANKING": ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN",
                "INDUSINDBK", "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB",
                "PNB", "BANKBARODA", "CANBK", "UNIONBANK", "AUBANK",
                "BANKINDIA", "CENTRALBK", "INDIANB", "MAHABANK", "SBICARD"],
    "PHARMA": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN",
               "BIOCON", "AUROPHARMA", "ALKEM", "TORNTPHARM", "IPCALAB",
               "APOLLOHOSP"],
    "AUTO": ["MARUTI", "TATAMOTORS", "M&M", "EICHERMOT", "HEROMOTOCO",
             "ASHOKLEY", "BAJAJ-AUTO", "TVSMOTOR", "AMARARAJA",
             "EXIDEIND", "BOSCHLTD", "SONACOMS", "BHARATFORG"],
    "METALS": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "HINDZINC",
               "COALINDIA", "SAIL", "JINDALSTEL", "HEIDELBERG"],
    "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR",
             "MARICO", "TATACONSUM", "GODREJCP", "VBL", "UBL", "COLPAL",
             "ASIANPAINT", "TITAN", "DMART"],
    "ENERGY": ["RELIANCE", "ONGC", "NTPC", "POWERGRID", "ADANIENT",
               "ADANIPORTS", "TATAPOWER", "ADANIPOWER", "NHPC", "SJVN",
               "BEL", "HAL", "IRFC", "LICI"],
    "FINANCE": ["BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "HDFCLIFE",
                "SBILIFE", "MUTHOOTFIN", "MANAPPURAM", "BAJAJHLDNG",
                "LICHSGFIN", "CANFINHOME", "RECLTD", "PFC",
                "SHRIRAMFIN", "POLICYBZR"],
    "INFRA": ["LT", "ADANIENT", "ADANIPORTS", "DLF", "GODREJCP",
              "ACC", "AMBUJACEM", "ULTRACEMCO", "SHREECEM", "DALBHARAT",
              "ABB", "SIEMENS", "HAVELLS", "GRASIM", "PIDILITIND",
              "SCHAEFFLER", "SKFINDIA", "TIMKEN"],
    "TECH_PLATFORM": ["ZOMATO", "PAYTM", "NYKAA", "DELHIVERY",
                      "MAPMYINDIA", "CLEAN", "DIXON"],
}


def get_sector(symbol: str) -> str:
    """Return the sector for a given symbol, or 'UNKNOWN'."""
    sym = symbol.upper()
    for sector, symbols in SECTOR_MAP.items():
        if sym in symbols:
            return sector
    return "UNKNOWN"


def get_sector_symbols(sector: str) -> list[str]:
    """Return all symbols in a given sector."""
    return SECTOR_MAP.get(sector.upper(), [])


def get_sector_weight_map(sector_weights: dict[str, float]) -> dict[str, float]:
    """
    Map Nifty 500 symbols to sector-adjusted weights.

    Args:
        sector_weights: Dict of sector → bias score (e.g., {"IT": 0.8, "BANKING": 0.3})

    Returns:
        Dict of symbol → adjusted weight (higher = more attractive)
    """
    weights: dict[str, float] = {}
    for symbol in NIFTY_500:
        sector = get_sector(symbol)
        base_weight = 1.0
        bias = sector_weights.get(sector, 0.0)
        # bias range: -1.0 (bearish) to +1.0 (bullish)
        adjusted = base_weight * (1.0 + bias * 0.3)  # ±30% adjustment
        weights[symbol] = max(adjusted, 0.1)  # Floor at 0.1
    return weights


# ---------------------------------------------------------------------------
# Bulk data fetcher
# ---------------------------------------------------------------------------

async def fetch_universe_data(
    symbols: list[str] | None = None,
    max_concurrent: int = 10,
) -> list[dict[str, Any]]:
    """
    Fetch key financials for a batch of symbols from yfinance.

    Uses asyncio.Semaphore to limit concurrent requests and avoid rate limits.

    Args:
        symbols: Symbols to fetch (defaults to NIFTY_500)
        max_concurrent: Max concurrent yfinance requests

    Returns:
        List of dicts with symbol, market_cap, avg_volume, sector, financials
    """
    if symbols is None:
        symbols = NIFTY_500

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _fetch_one(sym: str) -> dict[str, Any] | None:
        try:
            import yfinance as yf
            from src.autfin.agents.yfinance_utils import normalize_symbol

            async with semaphore:
                ticker = yf.Ticker(normalize_symbol(sym))
                info = ticker.info or {}

                hist = ticker.history(period="5d")
                avg_volume = (
                    float(hist["Volume"].mean())
                    if hist is not None and not hist.empty and "Volume" in hist.columns
                    else 0.0
                )

                bs = ticker.balance_sheet
                inc = ticker.financials
                cf = ticker.cashflow

                bs_latest = bs.iloc[:, 0].to_dict() if bs is not None and not bs.empty else {}
                inc_latest = inc.iloc[:, 0].to_dict() if inc is not None and not inc.empty else {}
                cf_latest = cf.iloc[:, 0].to_dict() if cf is not None and not cf.empty else {}
                bs_prior = bs.iloc[:, 1].to_dict() if bs is not None and bs.shape[1] > 1 else {}
                inc_prior = inc.iloc[:, 1].to_dict() if inc is not None and inc.shape[1] > 1 else {}

                net_income = inc_latest.get("Net Income") or 0
                total_assets = bs_latest.get("Total Assets") or 0
                market_cap = info.get("marketCap") or 0

                return {
                    "symbol": sym,
                    "market_cap": market_cap,
                    "avg_volume": avg_volume,
                    "sector": info.get("sector") or get_sector(sym),
                    # Income statement
                    "revenue": inc_latest.get("Total Revenue") or 0,
                    "ebit": inc_latest.get("EBIT") or inc_latest.get("Operating Income") or 0,
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
                    "roa_prior": (inc_prior.get("Net Income") or 0) / max(bs_prior.get("Total Assets") or 1, 1),
                    "total_assets_prior": bs_prior.get("Total Assets") or 0,
                    "current_assets_prior": bs_prior.get("Current Assets") or 0,
                    "current_liabilities_prior": bs_prior.get("Current Liabilities") or 0,
                    "long_term_debt_prior": bs_prior.get("Long Term Debt") or 0,
                    "gross_profit_prior": inc_prior.get("Gross Profit") or 0,
                    "shares_outstanding_prior": info.get("sharesOutstanding") or 0,
                    # Valuation / quality
                    "pe": info.get("trailingPE") or info.get("forwardPE"),
                    "pb": info.get("priceToBook"),
                    "roe": info.get("returnOnEquity"),
                    "debt_equity": info.get("debtToEquity") or (
                        (info.get("totalDebt") or 0) / max(info.get("totalEquity") or 1, 1)
                    ),
                    "fcf_yield": (cf_latest.get("Free Cash Flow") or 0) / max(market_cap or 1, 1),
                    "promoter_holding": info.get("heldPercentInsiders"),
                    # Fields not available from yfinance
                    "promoter_pledge_pct": 0,
                    "promoter_holding_change_4q": 0,
                    "auditor_qualification_2y": False,
                    "related_party_txn_pct": 0,
                    "negative_ocf_streak": 0,
                    "sebi_penalty_12m": False,
                }
        except Exception:
            logger.debug("Failed to fetch data for %s", sym, exc_info=True)
            return None

    tasks = [_fetch_one(sym) for sym in symbols[:200]]  # Cap at 200 for rate limits
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return [r for r in results if isinstance(r, dict)]
