"""Fundamental analysis tools for AutFin — ratio computation, DCF modeling, health scoring."""

from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class FundamentalTools:
    """Tools for fundamental analysis of Indian equities."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}

    async def get_financials(self, symbol: str) -> Dict[str, Any]:
        """Fetch financial statements from yfinance and compute key ratios.

        Returns dict with: pe, pb, roe, roic, debt_equity, current_ratio,
        fcf_yield, earnings_yield, dividend_yield, market_cap, enterprise_value,
        revenue_growth_3y, profit_growth_3y, promoter_holding, pledged_pct.
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(f"{symbol.upper().removesuffix('.NS')}.NS")
            info = ticker.info or {}

            return {
                "symbol": symbol,
                "pe": info.get("trailingPE") or info.get("forwardPE"),
                "pb": info.get("priceToBook"),
                "roe": info.get("returnOnEquity"),
                "roic": info.get("returnOnCapitalEmployed"),
                "debt_equity": info.get("debtToEquity") or (info.get("totalDebt", 0) / info.get("totalEquity", 1) if info.get("totalEquity") else None),
                "current_ratio": info.get("currentRatio"),
                "fcf_yield": (info.get("freeCashflow", 0) / info.get("marketCap", 1)) if info.get("marketCap") else None,
                "earnings_yield": (1 / info["trailingPE"]) if info.get("trailingPE") and info["trailingPE"] > 0 else None,
                "dividend_yield": info.get("dividendYield"),
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                "revenue_growth_3y": info.get("revenueGrowth"),
                "profit_growth_3y": info.get("earningsGrowth"),
                "promoter_holding": info.get("heldPercentInsiders"),
                "pledged_pct": None,  # Not available via yfinance
            }
        except Exception as e:
            logger.warning("Failed to fetch financials for %s: %s", symbol, e)
            return {"symbol": symbol, "error": str(e)}

    async def get_ratios(self, symbol: str) -> Dict[str, float]:
        """Get clean financial ratios dict. Returns only numeric values."""
        data = await self.get_financials(symbol)
        if "error" in data:
            return {}
        return {k: v for k, v in data.items() if isinstance(v, (int, float)) and v is not None}

    async def compute_health_score(self, symbol: str) -> Dict[str, Any]:
        """Compute a composite financial health score (0-100).

        Components (each 0-20 points):
        - Profitability (ROE > 15% = 20pts, 10-15% = 15pts, 0-10% = 10pts, negative = 0pts)
        - Debt management (D/E < 0.3 = 20pts, 0.3-0.7 = 15pts, 0.7-1.5 = 10pts, >1.5 = 5pts)
        - Growth (Revenue growth > 20% = 20pts, 10-20% = 15pts, 0-10% = 10pts, negative = 5pts)
        - Cash generation (FCF yield > 5% = 20pts, 2-5% = 15pts, 0-2% = 10pts, negative = 5pts)
        - Valuation (PE < 12 = 20pts, 12-18 = 15pts, 18-25 = 10pts, >25 or negative = 5pts)
        """
        data = await self.get_financials(symbol)
        if "error" in data:
            return {"symbol": symbol, "health_score": 0, "error": data["error"]}

        score = 0
        details = {}

        # Profitability (ROE)
        roe = data.get("roe") or 0
        if roe >= 0.15:
            pts, detail = 20, "excellent"
        elif roe >= 0.10:
            pts, detail = 15, "good"
        elif roe > 0:
            pts, detail = 10, "weak"
        else:
            pts, detail = 0, "negative"
        score += pts
        details["profitability"] = {"roe": roe, "score": pts, "rating": detail}

        # Debt
        de = data.get("debt_equity") or 99
        if de < 0.3:
            pts, detail = 20, "conservative"
        elif de < 0.7:
            pts, detail = 15, "moderate"
        elif de < 1.5:
            pts, detail = 10, "elevated"
        else:
            pts, detail = 5, "high"
        score += pts
        details["debt"] = {"debt_equity": de, "score": pts, "rating": detail}

        # Growth
        growth = data.get("revenue_growth_3y") or 0
        if growth >= 0.20:
            pts, detail = 20, "strong"
        elif growth >= 0.10:
            pts, detail = 15, "moderate"
        elif growth > 0:
            pts, detail = 10, "slow"
        else:
            pts, detail = 5, "declining"
        score += pts
        details["growth"] = {"revenue_growth_3y": growth, "score": pts, "rating": detail}

        # Cash
        fcf = data.get("fcf_yield") or 0
        if fcf >= 0.05:
            pts, detail = 20, "strong"
        elif fcf >= 0.02:
            pts, detail = 15, "adequate"
        elif fcf > 0:
            pts, detail = 10, "weak"
        else:
            pts, detail = 5, "negative"
        score += pts
        details["cash"] = {"fcf_yield": fcf, "score": pts, "rating": detail}

        # Valuation
        pe = data.get("pe") or 999
        if 0 < pe < 12:
            pts, detail = 20, "undervalued"
        elif pe < 18:
            pts, detail = 15, "fair"
        elif pe < 25:
            pts, detail = 10, "expensive"
        else:
            pts, detail = 5, "very_expensive"
        score += pts
        details["valuation"] = {"pe": pe, "score": pts, "rating": detail}

        rating = "STRONG" if score >= 80 else "GOOD" if score >= 60 else "FAIR" if score >= 40 else "WEAK"
        return {"symbol": symbol, "health_score": score, "rating": rating, "details": details}

    async def compare_peers(self, symbol: str, peers: List[str]) -> Dict[str, Any]:
        """Compare key ratios across a peer group. Returns dict of symbol -> key ratios."""
        results = {}
        for sym in [symbol] + peers:
            ratios = await self.get_ratios(sym)
            if ratios:
                results[sym] = {
                    "pe": ratios.get("pe"),
                    "roe": ratios.get("roe"),
                    "debt_equity": ratios.get("debt_equity"),
                    "revenue_growth_3y": ratios.get("revenue_growth_3y"),
                    "fcf_yield": ratios.get("fcf_yield"),
                }
        return {"symbol": symbol, "peers": results}

    async def get_balance_sheet(self, symbol: str) -> Dict[str, Any]:
        """Fetch balance sheet data from yfinance."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(f"{symbol.upper().removesuffix('.NS')}.NS")
            bs = ticker.balance_sheet
            if bs is not None and not bs.empty:
                latest = bs.iloc[:, 0]
                return {
                    "symbol": symbol,
                    "total_assets": float(latest.get("Total Assets", 0)),
                    "total_debt": float(latest.get("Total Debt", 0)),
                    "total_equity": float(latest.get("Total Equity Gross Minority Interest", 0)),
                    "cash": float(latest.get("Cash And Cash Equivalents", 0)),
                    "current_assets": float(latest.get("Current Assets", 0)),
                    "current_liabilities": float(latest.get("Current Liabilities", 0)),
                }
            return {"symbol": symbol, "error": "no_balance_sheet_data"}
        except Exception as e:
            return {"symbol": symbol, "error": str(e)}

    async def get_income_statement(self, symbol: str) -> Dict[str, Any]:
        """Fetch income statement data from yfinance."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(f"{symbol.upper().removesuffix('.NS')}.NS")
            inc = ticker.financials
            if inc is not None and not inc.empty:
                latest = inc.iloc[:, 0]
                return {
                    "symbol": symbol,
                    "revenue": float(latest.get("Total Revenue", 0)),
                    "ebitda": float(latest.get("EBITDA", 0)),
                    "net_income": float(latest.get("Net Income", 0)),
                    "eps": float(latest.get("Basic EPS", 0)),
                    "gross_margin": float(latest.get("Gross Profit", 0)) / float(latest.get("Total Revenue", 1)) if latest.get("Total Revenue") else None,
                }
            return {"symbol": symbol, "error": "no_income_data"}
        except Exception as e:
            return {"symbol": symbol, "error": str(e)}

    async def get_cashflow(self, symbol: str) -> Dict[str, Any]:
        """Fetch cash flow statement data from yfinance."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(f"{symbol.upper().removesuffix('.NS')}.NS")
            cf = ticker.cashflow
            if cf is not None and not cf.empty:
                latest = cf.iloc[:, 0]
                return {
                    "symbol": symbol,
                    "operating_cf": float(latest.get("Operating Cash Flow", 0)),
                    "capex": float(latest.get("Capital Expenditure", 0)),
                    "free_cash_flow": float(latest.get("Free Cash Flow", 0)),
                    "dividends_paid": float(latest.get("Dividends Paid", 0)),
                }
            return {"symbol": symbol, "error": "no_cashflow_data"}
        except Exception as e:
            return {"symbol": symbol, "error": str(e)}
