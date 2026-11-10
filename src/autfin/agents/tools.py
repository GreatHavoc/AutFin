"""
LangChain-compatible tool definitions for AutFin agents.

Each tool is defined as a standalone async function with:
- A docstring that serves as the tool description
- Type hints for parameter validation
- OpenAI-format tool definition

Tools are organized by agent domain. Each agent picks the tools it needs.
"""

import warnings
from typing import Any, Callable, Dict, List

warnings.filterwarnings("ignore", message=".*possibly delisted.*")
warnings.filterwarnings("ignore", message=".*no price data found.*")
import pandas as pd

# ================== TOOL REGISTRY ==================

AgentTools = Dict[str, Callable]
ToolDefs = List[Dict[str, Any]]


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


# ================== YFINANCE TOOLS (Fundamentals + Market Data) ==================


async def fetch_financial_info(symbol: str) -> str:
    """Fetch all financial info for a stock from yfinance (ratios, growth, balance sheet)."""
    try:
        import yfinance as yf

        from src.autfin.agents.yfinance_utils import normalize_symbol

        ticker = yf.Ticker(normalize_symbol(symbol.upper()))
        info = ticker.info or {}

        keys = {
            "trailingPE", "forwardPE", "priceToBook", "priceToSalesTrailing12Months",
            "returnOnEquity", "returnOnAssets", "profitMargins", "operatingMargins",
            "debtToEquity", "interestCoverage", "enterpriseValue", "ebitda",
            "totalRevenue", "netIncomeToCommon", "totalAssets", "totalStockholderEquity",
            "totalDebt", "totalCash", "operatingIncome", "freeCashflow", "marketCap",
            "revenueGrowth", "earningsGrowth", "currentRatio", "bookValue",
            "forwardEps", "trailingEps", "pegRatio", "dividendYield",
            "sector", "industry", "shortName", "longName",
        }
        result = {k: info.get(k) for k in keys if k in info}
        result["symbol"] = symbol
        return str(result)
    except Exception as e:
        return f"Error fetching financials for {symbol}: {str(e)}"


async def fetch_balance_sheet(symbol: str) -> str:
    """Fetch the balance sheet for a stock from yfinance."""
    try:
        import yfinance as yf

        from src.autfin.agents.yfinance_utils import normalize_symbol

        ticker = yf.Ticker(normalize_symbol(symbol.upper()))
        bs = ticker.balance_sheet
        if bs is None or bs.empty:
            return str({"error": f"No balance sheet data for {symbol}"})
        result = bs.to_dict()
        return str({symbol: result})
    except Exception as e:
        return f"Error fetching balance sheet for {symbol}: {str(e)}"


async def fetch_income_statement(symbol: str) -> str:
    """Fetch the income statement for a stock from yfinance."""
    try:
        import yfinance as yf

        from src.autfin.agents.yfinance_utils import normalize_symbol

        ticker = yf.Ticker(normalize_symbol(symbol.upper()))
        inc = ticker.financials
        if inc is None or inc.empty:
            return str({"error": f"No income statement data for {symbol}"})
        result = inc.to_dict()
        return str({symbol: result})
    except Exception as e:
        return f"Error fetching income statement for {symbol}: {str(e)}"


async def fetch_cashflow(symbol: str) -> str:
    """Fetch the cash flow statement for a stock from yfinance."""
    try:
        import yfinance as yf

        from src.autfin.agents.yfinance_utils import normalize_symbol

        ticker = yf.Ticker(normalize_symbol(symbol.upper()))
        cf = ticker.cashflow
        if cf is None or cf.empty:
            return str({"error": f"No cash flow data for {symbol}"})
        result = cf.to_dict()
        return str({symbol: result})
    except Exception as e:
        return f"Error fetching cash flow for {symbol}: {str(e)}"


async def fetch_historical_ohlcv(symbol: str, period: str = "6mo", interval: str = "1d") -> str:
    """Fetch OHLCV historical price data from yfinance.

    Args:
        symbol: Stock symbol (e.g. 'RELIANCE')
        period: Time period (1mo, 3mo, 6mo, 1y, 2y, 5y, max)
        interval: Candle interval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)
    """
    try:
        import yfinance as yf

        from src.autfin.agents.yfinance_utils import normalize_symbol

        ticker = yf.Ticker(normalize_symbol(symbol.upper()))
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return str({"error": f"No price data for {symbol}"})
        result = df.tail(200).to_dict()
        return str({symbol: result})
    except Exception as e:
        return f"Error fetching historical data for {symbol}: {str(e)}"


async def fetch_company_info(symbol: str) -> str:
    """Fetch company profile info (sector, industry, market cap, employees)."""
    try:
        import yfinance as yf

        from src.autfin.agents.yfinance_utils import normalize_symbol

        ticker = yf.Ticker(normalize_symbol(symbol.upper()))
        info = ticker.info or {}
        profile = {
            "symbol": symbol,
            "name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "employees": info.get("fullTimeEmployees"),
            "country": info.get("country"),
            "website": info.get("website"),
            "summary": info.get("longBusinessSummary", "")[:500],
        }
        return str(profile)
    except Exception as e:
        return f"Error fetching company info for {symbol}: {str(e)}"


# ================== TECHNICAL ANALYSIS TOOLS ==================


async def compute_sma(prices_json: str, periods: str = "20,50,200") -> str:
    """Compute Simple Moving Average for multiple periods.

    Args:
        prices_json: JSON string of price data (array of close prices)
        periods: Comma-separated list of periods (e.g. '20,50,200')
    """
    try:
        prices = __import__("json").loads(prices_json)
        close = pd.Series([p.get("Close", p) for p in (prices if isinstance(prices, list) else [prices])])
        if isinstance(prices, dict):
            close = pd.Series(pd.to_numeric(pd.Series(prices.get("Close", prices))))
        periods_list = [int(p.strip()) for p in periods.split(",")]
        result = {}
        for p_ in periods_list:
            sma = close.rolling(window=p_).mean()
            result[f"SMA_{p_}"] = float(sma.iloc[-1]) if not sma.empty else 0.0
        return str(result)
    except Exception as e:
        return f"Error computing SMA: {str(e)}"


async def compute_rsi(prices_json: str, period: int = 14) -> str:
    """Compute RSI (Relative Strength Index).

    Args:
        prices_json: JSON string of close price array
        period: RSI period (default 14)
    """
    try:
        closes = pd.Series(pd.to_numeric(pd.Series(__import__("json").loads(prices_json))))
        delta = closes.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss.replace(0, 1)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return str({"RSI": float(rsi.iloc[-1]) if not rsi.empty else 50.0})
    except Exception as e:
        return f"Error computing RSI: {str(e)}"


async def compute_macd(prices_json: str, fast: int = 12, slow: int = 26, signal: int = 9) -> str:
    """Compute MACD (Moving Average Convergence Divergence).

    Args:
        prices_json: JSON string of close price array
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)
    """
    try:
        closes = pd.Series(pd.to_numeric(pd.Series(__import__("json").loads(prices_json))))
        ema_fast = closes.ewm(span=fast, adjust=False).mean()
        ema_slow = closes.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return str({
            "MACD": float(macd_line.iloc[-1]),
            "Signal": float(signal_line.iloc[-1]),
            "Histogram": float(histogram.iloc[-1]),
        })
    except Exception as e:
        return f"Error computing MACD: {str(e)}"


async def compute_bollinger_bands(prices_json: str, period: int = 20, std_dev: float = 2.0) -> str:
    """Compute Bollinger Bands.

    Args:
        prices_json: JSON string of close price array
        period: SMA period (default 20)
        std_dev: Number of standard deviations (default 2)
    """
    try:
        closes = pd.Series(pd.to_numeric(pd.Series(__import__("json").loads(prices_json))))
        sma = closes.rolling(window=period).mean()
        std = closes.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return str({
            "Upper": float(upper.iloc[-1]),
            "Middle": float(sma.iloc[-1]),
            "Lower": float(lower.iloc[-1]),
        })
    except Exception as e:
        return f"Error computing Bollinger Bands: {str(e)}"


async def compute_atr(high_json: str, low_json: str, close_json: str, period: int = 14) -> str:
    """Compute Average True Range (volatility indicator).

    Args:
        high_json: JSON string of high price array
        low_json: JSON string of low price array
        close_json: JSON string of close price array
        period: ATR period (default 14)
    """
    try:
        high = pd.Series(pd.to_numeric(pd.Series(__import__("json").loads(high_json))))
        low = pd.Series(pd.to_numeric(pd.Series(__import__("json").loads(low_json))))
        close = pd.Series(pd.to_numeric(pd.Series(__import__("json").loads(close_json))))
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return str({"ATR": float(atr.iloc[-1])})
    except Exception as e:
        return f"Error computing ATR: {str(e)}"


async def detect_candlestick_patterns(high_json: str, low_json: str,
                                       open_json: str, close_json: str) -> str:
    """Detect common candlestick patterns (engulfing, doji, hammer, etc.)."""
    try:
        opens = pd.Series(pd.to_numeric(pd.Series(__import__("json").loads(open_json))))
        highs = pd.Series(pd.to_numeric(pd.Series(__import__("json").loads(high_json))))
        lows = pd.Series(pd.to_numeric(pd.Series(__import__("json").loads(low_json))))
        closes = pd.Series(pd.to_numeric(pd.Series(__import__("json").loads(close_json))))

        patterns = {}
        body = (closes - opens).abs()
        shadow_up = highs - closes.clip(lower=opens)
        shadow_low = opens.clip(lower=closes) - lows
        avg_body = body.iloc[-5:].mean() if len(body) >= 5 else body.mean()
        avg_body = max(avg_body, 0.01)

        latest = len(closes) - 1
        if latest >= 1:
            if (closes.iloc[latest] > opens.iloc[latest] and
                closes.iloc[latest - 1] < opens.iloc[latest - 1] and
                closes.iloc[latest] > opens.iloc[latest - 1] and
                opens.iloc[latest] < closes.iloc[latest - 1]):
                patterns["bullish_engulfing"] = True
            if (closes.iloc[latest] < opens.iloc[latest] and
                closes.iloc[latest - 1] > opens.iloc[latest - 1] and
                closes.iloc[latest] < opens.iloc[latest - 1] and
                opens.iloc[latest] > closes.iloc[latest - 1]):
                patterns["bearish_engulfing"] = True

        if body.iloc[latest] < avg_body * 0.3:
            if shadow_up.iloc[latest] < body.iloc[latest] * 0.1 and shadow_low.iloc[latest] > body.iloc[latest] * 2:
                patterns["hammer"] = True
            elif shadow_low.iloc[latest] < body.iloc[latest] * 0.1 and shadow_up.iloc[latest] > body.iloc[latest] * 2:
                patterns["shooting_star"] = True
            else:
                patterns["doji"] = True

        return str({"patterns": patterns, "latest_price": float(closes.iloc[-1])})
    except Exception as e:
        return f"Error detecting patterns: {str(e)}"


async def compute_volume_analysis(volume_json: str, price_json: str, period: int = 20) -> str:
    """Compute volume-based analysis (volume trend, volume-price relationship).

    Args:
        volume_json: JSON string of volume array
        price_json: JSON string of close price array
        period: Analysis period (default 20)
    """
    try:
        volume = pd.Series(pd.to_numeric(pd.Series(__import__("json").loads(volume_json))))
        price = pd.Series(pd.to_numeric(pd.Series(__import__("json").loads(price_json))))
        vol_ma = volume.rolling(window=period).mean()
        vol_trend = "increasing" if float(vol_ma.iloc[-1]) > float(vol_ma.iloc[-period]) else "decreasing"

        price_change = price.diff()
        high_vol_days = len(volume[volume > vol_ma * 1.5])
        vol_price_corr = float(volume.iloc[-period:].corr(price.iloc[-period:])) if len(price) >= period else 0.0

        return str({
            "latest_volume": float(volume.iloc[-1]),
            "avg_volume": float(vol_ma.iloc[-1]),
            "volume_trend": vol_trend,
            "high_volume_days": high_vol_days,
            "volume_price_correlation": vol_price_corr,
        })
    except Exception as e:
        return f"Error computing volume analysis: {str(e)}"


# ================== MACRO / FRED TOOLS ==================


async def fetch_macro_indicator(indicator: str) -> str:
    """Fetch a global macro indicator from yfinance (^GSPC, ^IXIC, ^VIX, DX-Y.NYB, CL=F, GC=F, ^TNX, USDINR=X).

    Args:
        indicator: Yahoo ticker symbol (e.g. '^VIX', 'CL=F', 'USDINR=X')
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(indicator)
        info = ticker.info or {}
        fast = ticker.fast_info if hasattr(ticker, "fast_info") else None

        result = {
            "symbol": indicator,
            "last_price": getattr(fast, "last_price", None) if fast else info.get("regularMarketPrice"),
            "previous_close": info.get("regularMarketPreviousClose") or info.get("previousClose"),
            "open": info.get("regularMarketOpen"),
            "day_high": info.get("dayHigh"),
            "day_low": info.get("dayLow"),
            "change_pct": info.get("regularMarketChangePercent"),
            "name": info.get("shortName") or info.get("longName"),
        }
        return str(result)
    except Exception as e:
        return f"Error fetching {indicator}: {str(e)}"


async def fetch_fred_indicator(series_id: str) -> str:
    """Fetch a US economic indicator from FRED (Federal Reserve Economic Data).

    Args:
        series_id: FRED series ID (GDP, CPIAUCSL, FEDFUNDS, UNRATE, DGS10)
    """
    try:
        import os

        import httpx

        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            return str({"error": "FRED_API_KEY not set in environment"})

        response = httpx.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 5,
            },
            timeout=15,
        )
        if response.status_code != 200:
            return str({"error": f"FRED API error: {response.status_code}"})

        data = response.json()
        observations = data.get("observations", [])
        values = [float(o["value"]) for o in observations if o.get("value").replace(".", "").replace("-", "").isdigit()]
        latest = values[0] if values else None

        labels = {
            "GDP": "Gross Domestic Product (billions USD, quarterly)",
            "CPIAUCSL": "Consumer Price Index (urban, monthly)",
            "FEDFUNDS": "Federal Funds Effective Rate (%)",
            "UNRATE": "Unemployment Rate (%)",
            "DGS10": "10-Year Treasury Yield (%)",
        }
        return str({
            "series_id": series_id,
            "label": labels.get(series_id, series_id),
            "latest_value": latest,
            "recent_values": values[:5],
            "unit": "percent" if series_id in ("FEDFUNDS", "UNRATE", "DGS10") else "index",
        })
    except Exception as e:
        return f"Error fetching FRED {series_id}: {str(e)}"


# ================== NEWS / SENTIMENT TOOLS ==================


async def fetch_marketaux_news(symbol: str, limit: int = 5) -> str:
    """Fetch recent news articles from Marketaux for sentiment analysis.

    Args:
        symbol: Stock symbol
        limit: Number of articles (default 5)
    """
    try:
        import os
        from datetime import datetime, timedelta

        import httpx

        api_key = os.environ.get("MARKETAUX_API_KEY")
        if not api_key:
            return str({"error": "MARKETAUX_API_KEY not set"})

        response = httpx.get(
            "https://api.marketaux.com/v1/news/all",
            params={
                "symbols": symbol.upper(),
                "filter_entities": "true",
                "language": "en",
                "api_token": api_key,
                "limit": limit,
                "published_after": (datetime.now() - timedelta(days=7)).isoformat(),
            },
            timeout=15,
        )
        if response.status_code != 200:
            return str({"error": f"Marketaux API error: {response.status_code}"})

        data = response.json()
        articles = data.get("data", [])[:limit]
        result = []
        for a in articles:
            result.append({
                "title": a.get("title"),
                "snippet": (a.get("snippet") or a.get("description", ""))[:300],
                "sentiment": a.get("sentiment"),
                "source": a.get("source"),
                "published": a.get("published_at"),
            })
        return str({"symbol": symbol, "articles": result, "count": len(result)})
    except Exception as e:
        return f"Error fetching Marketaux news: {str(e)}"


async def fetch_nse_announcements(symbol: str) -> str:
    """Fetch corporate announcements from NSE for a stock. Primary source for Indian stocks.

    Args:
        symbol: NSE trading symbol (e.g. 'RELIANCE', 'TCS')
    """
    try:
        import httpx

        response = httpx.get(
            "https://www.nseindia.com/api/corporate-announcements",
            params={"index": "equities", "symbol": symbol.upper()},
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.5",
            },
            timeout=15,
        )
        if response.status_code != 200:
            return str({"error": f"NSE API error: {response.status_code}"})

        data = response.json()
        announcements = data[:10] if isinstance(data, list) else []
        result = []
        for ann in announcements:
            desc = ann.get("desc", "")[:200]
            result.append({
                "description": desc,
                "date": ann.get("broadcast_dt") or ann.get("dt"),
                "category": ann.get("subj"),
            })

        # Keyword-based sentiment scoring
        positive_words = {"profit", "growth", "dividend", "bonus", "buyback", "expansion",
                          "acquisition", "revenue", "upgrade", "positive", "record"}
        negative_words = {"loss", "decline", "investigation", "penalty", "fraud",
                          "resign", "default", "negative", "downgrade", "restructuring"}
        pos, neg = 0, 0
        for ann in announcements:
            text = (ann.get("desc", "") + " " + ann.get("subj", "")).lower()
            for w in positive_words:
                if w in text:
                    pos += 1
            for w in negative_words:
                if w in text:
                    neg += 1

        total = pos + neg
        sentiment_score = round((pos - neg) / max(total, 1), 2)
        return str({
            "symbol": symbol,
            "announcements_count": len(announcements),
            "announcements": result,
            "sentiment_score": sentiment_score,
            "positive_hits": pos,
            "negative_hits": neg,
        })
    except Exception as e:
        return f"Error fetching NSE announcements: {str(e)}"


# ================== DCF / VALUATION TOOLS ==================


async def compute_dcf_valuation(
    free_cash_flow: float,
    revenue_growth: float,
    shares_outstanding: int,
    wacc: float = 0.10,
    terminal_growth: float = 0.03,
    projection_years: int = 5,
) -> str:
    """Compute DCF (Discounted Cash Flow) intrinsic valuation.

    Args:
        free_cash_flow: Latest annual FCF in crores
        revenue_growth: Expected annual growth rate (e.g. 0.15 for 15%)
        shares_outstanding: Number of shares in crores
        wacc: Weighted Average Cost of Capital (default 0.10)
        terminal_growth: Long-term growth for terminal value (default 0.03)
        projection_years: Number of years to project (default 5)
    """
    try:
        scenarios = {
            "upside": revenue_growth * 1.5,
            "base": revenue_growth,
            "downside": max(revenue_growth * 0.5, 0.01),
        }

        results = {}
        for scenario, growth in scenarios.items():
            fcf_projections = []
            for yr in range(1, projection_years + 1):
                fcf_yr = free_cash_flow * (1.0 + growth) ** yr
                fcf_projections.append(fcf_yr)

            terminal_value = (fcf_projections[-1] * (1.0 + terminal_growth)) / (wacc - terminal_growth)

            pv_fcfs = sum(fcf / (1.0 + wacc) ** yr for yr, fcf in enumerate(fcf_projections, 1))
            pv_terminal = terminal_value / (1.0 + wacc) ** projection_years
            enterprise_value = pv_fcfs + pv_terminal

            equity_value = enterprise_value
            target_price = equity_value / max(shares_outstanding, 1)

            margin_of_safety = round((target_price / max(free_cash_flow * 10 / max(shares_outstanding, 1), 0.01) - 1) * 100, 1)
            results[scenario] = {
                "target_price": round(target_price, 2),
                "margin_of_safety_pct": margin_of_safety,
            }

        upside_pct = round((results["upside"]["target_price"] / max(results["base"]["target_price"], 0.01) - 1) * 100, 1)
        downside_pct = round((1 - results["downside"]["target_price"] / max(results["base"]["target_price"], 0.01)) * 100, 1)
        risk_reward = abs(upside_pct / max(abs(downside_pct), 1))

        return str({
            "scenarios": results,
            "risk_reward_ratio": risk_reward,
            "upside_potential_pct": upside_pct,
            "downside_risk_pct": downside_pct,
            "wacc": wacc,
            "terminal_growth": terminal_growth,
        })
    except Exception as e:
        return f"Error computing DCF: {str(e)}"


# ================== COMPARATIVE TOOLS ==================


async def compare_peers(symbol: str, metric: str = "valuation") -> str:
    """Compare a stock against sector peers on valuation or profitability metrics.

    Args:
        symbol: Stock symbol
        metric: Type of comparison ('valuation', 'profitability', or 'growth')
    """
    try:
        sector_peers = {
            "IT": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
            "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK"],
            "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP"],
            "Auto": ["TATAMOTORS", "MARUTI", "M&M", "BAJFINANCE", "EICHERMOT"],
            "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR"],
            "Energy": ["RELIANCE", "ONGC", "NTPC", "POWERGRID", "ADANIENT"],
            "Metals": ["TATASTEEL", "HINDALCO", "JSWSTEEL", "ADANIENT", "VEDL"],
        }

        try:
            import yfinance as yf

            from src.autfin.agents.yfinance_utils import normalize_symbol
            ticker = yf.Ticker(normalize_symbol(symbol.upper()))
            info = ticker.info or {}
            sector = info.get("sector", "N/A")
        except Exception:
            sector = "N/A"

        matching_sector = "N/A"
        for sector_name, peers in sector_peers.items():
            if symbol.upper() in peers:
                matching_sector = sector_name
                break

        peer_list = sector_peers.get(matching_sector, [symbol.upper()])[:5]
        result = {"symbol": symbol, "sector": matching_sector, "peers": peer_list, "metric": metric}
        return str(result)
    except Exception as e:
        return f"Error comparing peers for {symbol}: {str(e)}"


# ================== TOOL REGISTRIES BY AGENT ==================


FUNDAMENTALS_TOOLS: ToolDefs = [
    _tool_def("fetch_financial_info", "Fetch all financial ratios and metrics for a stock from yfinance",
              {"symbol": {"type": "string", "description": "Stock symbol (e.g. RELIANCE)"}}),
    _tool_def("fetch_balance_sheet", "Fetch the balance sheet for a stock",
              {"symbol": {"type": "string", "description": "Stock symbol"}}),
    _tool_def("fetch_income_statement", "Fetch the income statement for a stock",
              {"symbol": {"type": "string", "description": "Stock symbol"}}),
    _tool_def("fetch_cashflow", "Fetch the cash flow statement for a stock",
              {"symbol": {"type": "string", "description": "Stock symbol"}}),
    _tool_def("fetch_company_info", "Fetch company profile (sector, industry, employees, summary)",
              {"symbol": {"type": "string", "description": "Stock symbol"}}),
    _tool_def("compare_peers", "Compare stock against sector peers",
              {"symbol": {"type": "string", "description": "Stock symbol"},
               "metric": {"type": "string", "description": "Comparison metric: valuation, profitability, growth"}}),
    _tool_def("compute_dcf_valuation", "Compute DCF intrinsic valuation with 3 scenarios",
              {"free_cash_flow": {"type": "number", "description": "Latest annual FCF in crores"},
               "revenue_growth": {"type": "number", "description": "Expected growth rate (e.g. 0.15)"},
               "shares_outstanding": {"type": "integer", "description": "Shares in crores"},
               "wacc": {"type": "number", "description": "WACC (default 0.10)"},
               "terminal_growth": {"type": "number", "description": "Terminal growth (default 0.03)"}}),
]

FUNDAMENTALS_CALLABLES: AgentTools = {
    "fetch_financial_info": fetch_financial_info,
    "fetch_balance_sheet": fetch_balance_sheet,
    "fetch_income_statement": fetch_income_statement,
    "fetch_cashflow": fetch_cashflow,
    "fetch_company_info": fetch_company_info,
    "compare_peers": compare_peers,
    "compute_dcf_valuation": compute_dcf_valuation,
}

TECHNICAL_TOOLS: ToolDefs = [
    _tool_def("fetch_historical_ohlcv", "Fetch OHLCV price history from yfinance",
              {"symbol": {"type": "string", "description": "Stock symbol"},
               "period": {"type": "string", "description": "1mo, 3mo, 6mo, 1y, 2y, 5y, max"},
               "interval": {"type": "string", "description": "1m, 5m, 15m, 30m, 1h, 1d"}}),
    _tool_def("compute_sma", "Compute Simple Moving Average for periods (e.g. 20,50,200)",
              {"prices_json": {"type": "string", "description": "JSON of close price data"},
               "periods": {"type": "string", "description": "Comma-separated periods (e.g. '20,50,200')"}}),
    _tool_def("compute_rsi", "Compute RSI (Relative Strength Index)",
              {"prices_json": {"type": "string", "description": "JSON of close price array"},
               "period": {"type": "integer", "description": "RSI period (default 14)"}}),
    _tool_def("compute_macd", "Compute MACD with signal line and histogram",
              {"prices_json": {"type": "string", "description": "JSON of close price array"}}),
    _tool_def("compute_bollinger_bands", "Compute Bollinger Bands (upper, middle, lower)",
              {"prices_json": {"type": "string", "description": "JSON of close price array"},
               "period": {"type": "integer", "description": "SMA period (default 20)"}}),
    _tool_def("compute_atr", "Compute Average True Range (volatility)",
              {"high_json": {"type": "string", "description": "JSON of high price array"},
               "low_json": {"type": "string", "description": "JSON of low price array"},
               "close_json": {"type": "string", "description": "JSON of close price array"}}),
    _tool_def("detect_candlestick_patterns", "Detect candlestick patterns (engulfing, doji, hammer, etc.)",
              {"high_json": {"type": "string"}, "low_json": {"type": "string"},
               "open_json": {"type": "string"}, "close_json": {"type": "string"}}),
    _tool_def("compute_volume_analysis", "Analyze volume trends and volume-price relationship",
              {"volume_json": {"type": "string", "description": "JSON of volume array"},
               "price_json": {"type": "string", "description": "JSON of close price array"}}),
]

TECHNICAL_CALLABLES: AgentTools = {
    "fetch_historical_ohlcv": fetch_historical_ohlcv,
    "compute_sma": compute_sma,
    "compute_rsi": compute_rsi,
    "compute_macd": compute_macd,
    "compute_bollinger_bands": compute_bollinger_bands,
    "compute_atr": compute_atr,
    "detect_candlestick_patterns": detect_candlestick_patterns,
    "compute_volume_analysis": compute_volume_analysis,
}

SENTIMENT_TOOLS: ToolDefs = [
    _tool_def("fetch_marketaux_news", "Fetch recent news from Marketaux with sentiment scores",
              {"symbol": {"type": "string", "description": "Stock symbol"},
               "limit": {"type": "integer", "description": "Number of articles (default 5)"}}),
    _tool_def("fetch_nse_announcements", "Fetch corporate announcements from NSE (primary Indian source)",
              {"symbol": {"type": "string", "description": "Stock symbol"}}),
    _tool_def("fetch_company_info", "Fetch company profile for context on sentiment relevance",
              {"symbol": {"type": "string", "description": "Stock symbol"}}),
]

SENTIMENT_CALLABLES: AgentTools = {
    "fetch_marketaux_news": fetch_marketaux_news,
    "fetch_nse_announcements": fetch_nse_announcements,
    "fetch_company_info": fetch_company_info,
}

MACRO_TOOLS: ToolDefs = [
    _tool_def("fetch_macro_indicator", "Fetch a global macro indicator (^GSPC, ^VIX, CL=F, GC=F, ^TNX, USDINR=X, etc.)",
              {"indicator": {"type": "string", "description": "Yahoo ticker (e.g. ^VIX, CL=F, USDINR=X)"}}),
    _tool_def("fetch_fred_indicator", "Fetch US economic data from FRED (GDP, CPI, Fed Funds, Unemployment)",
              {"series_id": {"type": "string", "description": "FRED series ID: GDP, CPIAUCSL, FEDFUNDS, UNRATE, DGS10"}}),
]

MACRO_CALLABLES: AgentTools = {
    "fetch_macro_indicator": fetch_macro_indicator,
    "fetch_fred_indicator": fetch_fred_indicator,
}

# Master registries
ALL_TOOLS: Dict[str, ToolDefs] = {
    "fundamentals": FUNDAMENTALS_TOOLS,
    "technical": TECHNICAL_TOOLS,
    "sentiment": SENTIMENT_TOOLS,
    "macro": MACRO_TOOLS,
}

ALL_CALLABLES: Dict[str, AgentTools] = {
    "fundamentals": FUNDAMENTALS_CALLABLES,
    "technical": TECHNICAL_CALLABLES,
    "sentiment": SENTIMENT_CALLABLES,
    "macro": MACRO_CALLABLES,
}
