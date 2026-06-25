"""Global Macro Agent — agentic with tool-calling loop for economic indicators."""

import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langgraph.config import RunnableConfig

from src.autfin.state.agent_state import AgentState
from src.autfin.llm.client import get_llm_client
from src.autfin.agents.tools import MACRO_TOOLS, MACRO_CALLABLES

import logging
logger = logging.getLogger(__name__)


# ================== MACRO DATA CACHING ==================

class MacroDataCache:
    """Caches macroeconomic data to avoid redundant API calls."""
    
    def __init__(self, ttl_seconds: int = 300):  # 5 minutes TTL
        """
        Initialize cache with TTL.
        
        Args:
            ttl_seconds: Time-to-live for cache entries (default 300 seconds)
        """
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, datetime] = {}
        self._ttl = timedelta(seconds=ttl_seconds)
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self._timestamps:
            return True
        return datetime.now() - self._timestamps[key] > self._ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if valid."""
        if key not in self._cache:
            return None
        if self._is_expired(key):
            del self._cache[key]
            del self._timestamps[key]
            return None
        return self._cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache with timestamp."""
        self._cache[key] = value
        self._timestamps[key] = datetime.now()
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._timestamps.clear()


# Global cache instance
_macro_cache = MacroDataCache(ttl_seconds=300)


# ================== FRED API FETCHER ==================

# FRED series IDs for US economic indicators
FRED_SERIES = {
    "us_gdp": "GDP",
    "us_cpi": "CPIAUCSL",
    "fed_funds_rate": "FEDFUNDS",
}

# FRED data is quarterly (GDP) or monthly (CPI, Fed Funds) — cache for 24h
_fred_cache: Dict[str, Any] = {}
_fred_timestamps: Dict[str, datetime] = {}
_FRED_TTL = timedelta(hours=24)


def _fetch_fred_indicator(series_id: str) -> Optional[float]:
    """
    Fetch a single indicator from the FRED API.
    
    Args:
        series_id: FRED series ID (e.g., 'GDP', 'CPIAUCSL', 'FEDFUNDS')
        
    Returns:
        Latest value as float, or None on failure
    """
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        return None
    
    # Check cache
    if series_id in _fred_cache:
        ts = _fred_timestamps.get(series_id)
        if ts and datetime.now() - ts < _FRED_TTL:
            return _fred_cache[series_id]
    
    try:
        import httpx
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 2,
        }
        resp = httpx.get(url, params=params, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        observations = data.get("observations", [])
        if observations:
            # Get the most recent non-null value
            for obs in observations:
                val = obs.get("value")
                if val and val != ".":
                    value = float(val)
                    _fred_cache[series_id] = value
                    _fred_timestamps[series_id] = datetime.now()
                    return value
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(f"FRED API fetch failed for {series_id}: {e}")
    
    return None


def fetch_fred_indicators() -> Dict[str, Dict[str, Any]]:
    """
    Fetch all FRED economic indicators (GDP, CPI, Fed Funds Rate).
    
    Returns:
        Dictionary of indicator data with value, unit, source, last_updated
    """
    results = {}
    
    labels = {
        "us_gdp": ("percent", "BEA (quarterly)"),
        "us_cpi": ("index", "BLS (monthly)"),
        "fed_funds_rate": ("percent", "Federal Reserve (monthly)"),
    }
    
    for series_id, (unit, source) in labels.items():
        value = _fetch_fred_indicator(series_id)
        results[series_id] = {
            "value": value,
            "unit": unit,
            "source": f"FRED API — {source}" if value else f"{source} — FRED unavailable",
            "last_updated": datetime.now().isoformat() if value else "unavailable",
        }
    
    return results


# ================== MACRO NARRATIVE GENERATOR ==================

async def generate_macro_narrative(
    indicators: Dict[str, Any],
    impact_analysis: Dict[str, Any],
) -> str:
    """
    Generate a 2-3 sentence macro narrative using LLM.
    
    Provides a concise summary of macro conditions and their
    expected impact on Indian equity markets for downstream agents.
    
    Args:
        indicators: Raw macro indicator data
        impact_analysis: Impact analysis results
        
    Returns:
        2-3 sentence narrative string
    """
    from src.autfin.llm.client import get_llm_client
    
    llm = get_llm_client()
    if not llm.is_available:
        return _rule_based_narrative(indicators, impact_analysis)
    
    # Build context for LLM
    data = indicators.get("data", {})
    key_indicators = {}
    for key in ["nifty_50", "us_10y_yield", "fed_funds_rate", "us_cpi", "us_gdp",
                 "crude_oil_wti", "gold_futures", "inr_usd", "dxy", "vix"]:
        item = data.get(key, {})
        if item and item.get("value") is not None:
            key_indicators[key] = {
                "value": item["value"],
                "change_pct": item.get("change_pct", 0),
            }
    
    impact_score = impact_analysis.get("impact_score", "neutral")
    recommendations = impact_analysis.get("recommendations", [])[:3]
    
    prompt = (
        f"Key macro indicators:\n{key_indicators}\n\n"
        f"Impact score: {impact_score}\n"
        f"Top recommendations: {recommendations}\n\n"
        "Write a 2-3 sentence summary of current macro conditions and their "
        "expected impact on Indian equity markets. Be specific about key drivers."
    )
    
    result = await llm.generate(
        prompt=prompt,
        system_prompt=(
            "You are a macro analyst. Write concise 2-3 sentence market summaries. "
            "Focus on: what changed, why it matters for India, and key risks. "
            "No disclaimers, no filler."
        ),
        model=llm.fast_model,
        max_tokens=200,
    )
    
    return result if result else _rule_based_narrative(indicators, impact_analysis)


def _rule_based_narrative(
    indicators: Dict[str, Any],
    impact_analysis: Dict[str, Any],
) -> str:
    """Fallback narrative when LLM is unavailable."""
    data = indicators.get("data", {})
    impact_score = impact_analysis.get("impact_score", "neutral")
    
    parts = []
    
    nifty = data.get("nifty_50", {})
    if nifty.get("value"):
        chg = nifty.get("change_pct", 0)
        direction = "up" if chg > 0 else "down" if chg < 0 else "flat"
        parts.append(f"Nifty 50 is {direction} {abs(chg):.1f}%")
    
    crude = data.get("crude_oil_wti", {})
    if crude.get("value"):
        parts.append(f"Crude at ${crude['value']:.1f}")
    
    inr = data.get("inr_usd", {})
    if inr.get("value"):
        parts.append(f"INR/USD at {inr['value']:.2f}")
    
    summary = f"Macro conditions are {impact_score}. " + ", ".join(parts) + "."
    
    recs = impact_analysis.get("recommendations", [])
    if recs:
        summary += f" Key action: {recs[0]}."
    
    return summary


# ================== MACRO INDICATOR DEFINITIONS ==================

# US Economic Indicators
US_INDICATORS = {
    "us_gdp": "US GDP (Quarterly)",
    "us_cpi": "US CPI - Consumer Price Index (Monthly)",
    "fed_funds_rate": "Fed Funds Rate (Monthly)",
    "non_farm_payrolls": "US Non-Farm Payrolls (Monthly)",
    "unemployment_rate": "US Unemployment Rate (Monthly)",
}

# Commodities
COMMODITY_INDICATORS = {
    "crude_oil_wti": "Crude Oil WTI",
    "gold_futures": "Gold Futures",
    "copper_futures": "Copper Futures",
    "natural_gas": "Natural Gas",
}

# Global Markets
GLOBAL_MARKET_INDICATORS = {
    "sp_500": "S&P 500 Index",
    "nasdaq_composite": "NASDAQ Composite",
    "nikkei_225": "Nikkei 225",
    "ftse_100": "FTSE 100",
}

# India Specific
INDIA_INDICATORS = {
    "nifty_50": "Nifty 50 Index",
    "bank_nifty": "Bank Nifty Index",
    "inr_usd": "INR/USD Exchange Rate",
    "g_sec_yield_10y": "10Y G-Sec Yield",
    "crisil_rating": "CRISIL Rating",
}

# Other
OTHER_INDICATORS = {
    "vix": "VIX (Volatility Index)",
    "dxy": "US Dollar Index (DXY)",
    "bitcoin_price": "Bitcoin Price",
    "global_oil_prices": "Global Oil Prices",
}

# Combined indicator map
ALL_INDICATORS = {
    **US_INDICATORS,
    **COMMODITY_INDICATORS,
    **GLOBAL_MARKET_INDICATORS,
    **INDIA_INDICATORS,
    **OTHER_INDICATORS,
}


# ================== MACRO DATA FETCHER ==================

async def fetch_macro_indicators() -> Dict[str, Any]:
    """
    Fetch all macroeconomic indicators using yfinance for market data.

    Returns:
        Dictionary with indicator values, changes, and metadata
    """
    # Check cache first
    cache_key = "macro_indicators"
    cached_data = _macro_cache.get(cache_key)
    
    if cached_data is not None:
        return cached_data
    
    # Initialize result structure
    indicators = {
        "timestamp": datetime.now().isoformat(),
        "data": {},
        "errors": [],
    }
    
    try:
        import yfinance as yf

        # Helper to fetch ticker with error handling
        def _fetch(ticker_symbol: str, label: str) -> dict | None:
            try:
                t = yf.Ticker(ticker_symbol)
                hist = t.history(period="5d")
                if hist is None or hist.empty:
                    return None
                last = hist["Close"].iloc[-1]
                prev = hist["Close"].iloc[-2] if len(hist) > 1 else last
                change = float(last - prev)
                change_pct = (change / prev * 100) if prev else 0.0
                return {
                    "value": round(float(last), 4),
                    "change": round(change, 4),
                    "change_pct": round(change_pct, 2),
                    "source": "yfinance",
                    "last_updated": datetime.now().isoformat(),
                }
            except Exception as exc:
                indicators["errors"].append(f"{label}: {exc}")
                return None

        # Global market indices
        index_map = {
            "^GSPC": "sp_500",
            "^IXIC": "nasdaq_composite",
            "^N225": "nikkei_225",
            "^FTSE": "ftse_100",
            "^VIX": "vix",
            "DX-Y.NYB": "dxy",
        }
        for ticker, key in index_map.items():
            data = _fetch(ticker, key)
            if data:
                indicators["data"][key] = data

        # Commodities
        commodity_map = {
            "CL=F": "crude_oil_wti",
            "GC=F": "gold_futures",
            "HG=F": "copper_futures",
            "NG=F": "natural_gas",
            "BZ=F": "global_oil_prices",
        }
        for ticker, key in commodity_map.items():
            data = _fetch(ticker, key)
            if data:
                indicators["data"][key] = data

        # India-specific indices
        india_map = {
            "^NSEI": "nifty_50",
            "^NSEBANK": "bank_nifty",
        }
        for ticker, key in india_map.items():
            data = _fetch(ticker, key)
            if data:
                indicators["data"][key] = data

        # USD/INR
        usdinr = _fetch("USDINR=X", "inr_usd")
        if usdinr:
            indicators["data"]["inr_usd"] = usdinr

        # India 10Y bond yield (use短期国债 proxy — yfinance has limited bond data)
        # Skip if not available; mark as unavailable
        try:
            bond = yf.Ticker("^TNX")  # US 10Y — closest proxy available
            bond_hist = bond.history(period="5d")
            if bond_hist is not None and not bond_hist.empty:
                us_10y = float(bond_hist["Close"].iloc[-1])
                indicators["data"]["us_10y_yield"] = {
                    "value": round(us_10y, 4),
                    "unit": "percent",
                    "source": "yfinance (US proxy)",
                    "last_updated": datetime.now().isoformat(),
                }
        except Exception:
            logger.debug("Failed to fetch US 10Y bond yield via proxy", exc_info=True)

        # US economic indicators — try FRED API, fallback to None
        fred_data = fetch_fred_indicators()
        for key in ["us_gdp", "us_cpi", "fed_funds_rate"]:
            indicators["data"][key] = fred_data.get(key, {
                "value": None,
                "unit": "percent",
                "source": "FRED API — not configured",
                "last_updated": "unavailable",
            })

        # Cache results
        _macro_cache.set(cache_key, indicators)
        
    except ImportError:
        indicators["errors"].append("yfinance not installed — using empty data")
        _macro_cache.set(cache_key, indicators)
    except Exception as e:
        indicators["errors"].append(f"Error fetching macro indicators: {str(e)}")
        _macro_cache.set(cache_key, indicators)
    
    return indicators


# ================== CORRELATION ANALYSIS ==================

def analyze_correlations(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze correlations between global indicators and Indian markets.
    
    Args:
        indicators: Dictionary of macro indicator values
        
    Returns:
        Dictionary with correlation analysis results
    """
    correlations = {
        "timestamp": datetime.now().isoformat(),
        "indian_market_correlations": {},
        "global_market_correlations": {},
        "commodity_correlations": {},
    }
    
    try:
        data = indicators.get("data", {})
        
        # Calculate Nifty 50 correlations
        nifty_50 = data.get("nifty_50", {})
        nifty_value = nifty_50.get("value", 0)
        
        # US Fed Rate vs Nifty (Inverse correlation expected)
        fed_rate = data.get("fed_funds_rate", {})
        fed_rate_change = fed_rate.get("change", 0)
        
        if fed_rate_change > 0:
            correlations["indian_market_correlations"]["fed_rate_impact"] = {
                "direction": "negative",
                "strength": "medium",
                "reasoning": "Higher interest rates typically lead to capital outflows from Indian markets",
            }
        elif fed_rate_change < 0:
            correlations["indian_market_correlations"]["fed_rate_impact"] = {
                "direction": "positive",
                "strength": "medium",
                "reasoning": "Lower interest rates may lead to capital inflows into Indian markets",
            }
        
        # USD Index vs Nifty (Inverse correlation expected)
        dxy = data.get("dxy", {})
        dxy_change = dxy.get("change", 0)
        
        if dxy_change > 0:
            correlations["indian_market_correlations"]["dxy_impact"] = {
                "direction": "negative",
                "strength": "medium",
                "reasoning": "Stronger USD may lead to INR depreciation and outflows",
            }
        elif dxy_change < 0:
            correlations["indian_market_correlations"]["dxy_impact"] = {
                "direction": "positive",
                "strength": "medium",
                "reasoning": "Weaker USD may support INR and inflows",
            }
        
        # Crude Oil vs Nifty (Negative for import-dependent India)
        crude_oil = data.get("crude_oil_wti", {})
        oil_price = crude_oil.get("value", 0)
        oil_change = crude_oil.get("change_pct", 0)
        
        if oil_price > 100:
            correlations["commodity_correlations"]["crude_oil_impact"] = {
                "direction": "negative",
                "strength": "high",
                "reasoning": "High crude oil prices hurt India's trade deficit and inflation",
            }
        elif oil_change > 5:
            correlations["commodity_correlations"]["crude_oil_impact"] = {
                "direction": "negative",
                "strength": "medium",
                "reasoning": "Rapid crude price increases negatively impact Indian markets",
            }
        else:
            correlations["commodity_correlations"]["crude_oil_impact"] = {
                "direction": "neutral",
                "strength": "low",
                "reasoning": "Crude oil at manageable levels",
            }
        
        # INR/USD vs Nifty
        inr_usd = data.get("inr_usd", {})
        inr_change = inr_usd.get("change", 0)
        
        if inr_change < 0:  # INR appreciating
            correlations["indian_market_correlations"]["inr_impact"] = {
                "direction": "positive",
                "strength": "medium",
                "reasoning": "INR appreciation attracts foreign investment",
            }
        else:  # INR depreciating
            correlations["indian_market_correlations"]["inr_impact"] = {
                "direction": "negative",
                "strength": "medium",
                "reasoning": "INR depreciation may cause capital outflows",
            }
        
        # Global Market Correlations
        sp_500 = data.get("sp_500", {})
        sp_500_change = sp_500.get("change_pct", 0)
        
        correlations["global_market_correlations"]["sp500_impact"] = {
            "direction": "positive",
            "strength": "high",
            "reasoning": "Strong correlation with global risk sentiment",
            "sp500_change_pct": sp_500_change,
        }
        
        # VIX vs Nifty (Inverse correlation)
        vix = data.get("vix", {})
        vix_change = vix.get("change_pct", 0)
        
        if vix_change > 10:
            correlations["indian_market_correlations"]["vix_impact"] = {
                "direction": "negative",
                "strength": "high",
                "reasoning": "Higher volatility usually leads to risk-off behavior",
            }
        elif vix_change < -10:
            correlations["indian_market_correlations"]["vix_impact"] = {
                "direction": "positive",
                "strength": "high",
                "reasoning": "Lower volatility supports risk-on sentiment",
            }
        
    except Exception as e:
        correlations["error"] = f"Error analyzing correlations: {str(e)}"
    
    return correlations


# ================== IMPACT SCORING ==================

def calculate_impact_score(analysis: Dict[str, Any]) -> str:
    """
    Calculate overall impact score on Indian markets.
    
    Impact levels:
    - high_positive: Positive impact > 2%
    - medium_positive: Positive impact 1-2%
    - neutral: Neutral impact -1% to +1%
    - medium_negative: Negative impact -1% to -2%
    - high_negative: Negative impact < -2%
    
    Args:
        analysis: Dictionary with correlation analysis results
        
    Returns:
        Impact score string
    """
    impact_score = 0.0
    impact_factors = []
    
    try:
        correlations = analysis.get("indian_market_correlations", {})
        global_correlations = analysis.get("global_market_correlations", {})
        commodity_correlations = analysis.get("commodity_correlations", {})
        
        # Process Indian market correlations
        for key, value in correlations.items():
            if isinstance(value, dict) and "direction" in value:
                direction = 1 if value["direction"] == "positive" else -1
                strength = value.get("strength", "low")
                
                # Assign weights based on strength
                weight_map = {
                    "high": 2.0,
                    "medium": 1.0,
                    "low": 0.5,
                }
                
                weight = weight_map.get(strength, 0.5)
                impact_score += direction * weight
                impact_factors.append({
                    "factor": key,
                    "direction": value["direction"],
                    "strength": strength,
                    "impact": direction * weight,
                })
        
        # Process global market correlations
        for key, value in global_correlations.items():
            if isinstance(value, dict) and "direction" in value:
                direction = 1 if value["direction"] == "positive" else -1
                strength = value.get("strength", "low")
                
                weight_map = {
                    "high": 1.5,
                    "medium": 0.75,
                    "low": 0.375,
                }
                
                weight = weight_map.get(strength, 0.375)
                impact_score += direction * weight
                impact_factors.append({
                    "factor": key,
                    "direction": value["direction"],
                    "strength": strength,
                    "impact": direction * weight,
                })
        
        # Process commodity correlations
        for key, value in commodity_correlations.items():
            if isinstance(value, dict) and "direction" in value:
                direction = 1 if value["direction"] == "positive" else -1
                strength = value.get("strength", "low")
                
                weight_map = {
                    "high": 1.5,
                    "medium": 0.75,
                    "low": 0.375,
                }
                
                weight = weight_map.get(strength, 0.375)
                impact_score += direction * weight
                impact_factors.append({
                    "factor": key,
                    "direction": value["direction"],
                    "strength": strength,
                    "impact": direction * weight,
                })
        
    except Exception as e:
        # Return neutral on error
        return "neutral"
    
    # Determine impact category
    if impact_score >= 2.0:
        return "high_positive"
    elif impact_score >= 1.0:
        return "medium_positive"
    elif impact_score >= -1.0:
        return "neutral"
    elif impact_score >= -2.0:
        return "medium_negative"
    else:
        return "high_negative"


def analyze_india_impact(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze impact of global indicators on Indian markets.
    
    Args:
        indicators: Dictionary of macro indicator values
        
    Returns:
        Dictionary with impact analysis results
    """
    # Get correlations
    correlations = analyze_correlations(indicators)
    
    # Calculate impact score
    impact_score = calculate_impact_score(correlations)
    
    # Build comprehensive analysis
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "indicators_summary": {
            "total_indicators": len(indicators.get("data", {})),
            "indicators_with_data": len([k for k, v in indicators.get("data", {}).items() if v]),
            "indicators_missing": len(indicators.get("data", {})) - len([k for k, v in indicators.get("data", {}).items() if v]),
        },
        "correlations": correlations,
        "impact_score": impact_score,
        "impact_factors": correlations.get("impact_factors", []),
        "recommendations": [],
    }
    
    # Generate recommendations based on impact
    if impact_score == "high_positive":
        analysis["recommendations"] = [
            "Consider increasing equity exposure",
            "Monitor macro trends for sustainability",
            "Watch for potential overheating signals",
        ]
    elif impact_score == "medium_positive":
        analysis["recommendations"] = [
            "Maintain moderate equity exposure",
            "Diversify across asset classes",
            "Keep an eye on global developments",
        ]
    elif impact_score == "neutral":
        analysis["recommendations"] = [
            "Hold current allocation",
            "Wait for clearer signals",
            "Focus on stock-specific fundamentals",
        ]
    elif impact_score == "medium_negative":
        analysis["recommendations"] = [
            "Reduce equity exposure",
            "Increase cash buffer",
            "Hedge currency risk if applicable",
        ]
    else:  # high_negative
        analysis["recommendations"] = [
            "Significantly reduce equity exposure",
            "Prioritize capital preservation",
            "Consider defensive sectors",
            "Monitor for market correction",
        ]
    
    return analysis


# ================== SECTOR IMPACT MATRIX ==================

# FR-053: Sector-specific recommendations based on macro indicators.
# Maps macro conditions → sector impact direction and strength.
SECTOR_IMPACT_MATRIX: Dict[str, Dict[str, Dict[str, str]]] = {
    # Oil up: bad for importers (airlines, OMCs, textiles), good for O&G
    "crude_oil_wti": {
        "positive": {
            "direction": "negative",
            "strength": "high",
            "affected_sectors": ["airlines", "OMCs", "textiles", "chemicals"],
            "reasoning": "Higher crude increases input costs for oil-importing sectors",
        },
        "negative": {
            "direction": "positive",
            "strength": "medium",
            "affected_sectors": ["O&G", "refineries"],
            "reasoning": "Lower crude reduces input costs and may boost margins",
        },
    },
    # USD/INR up (INR weak): bad for IT (import costs), good for exporters
    "inr_usd": {
        "positive": {
            "direction": "mixed",
            "strength": "medium",
            "affected_sectors": ["IT", "pharma", "autos"],
            "reasoning": "Weaker INR benefits IT/pharma exporters but raises import costs",
        },
        "negative": {
            "direction": "mixed",
            "strength": "low",
            "affected_sectors": ["metals", "FMCG"],
            "reasoning": "Stronger INR reduces export earnings but lowers import costs",
        },
    },
    # Fed rate up: risk-off, FII outflows → negative for financials
    "fed_funds_rate": {
        "positive": {
            "direction": "negative",
            "strength": "medium",
            "affected_sectors": ["banks", "NBFCs", "realty"],
            "reasoning": "Higher US rates trigger FII outflows from Indian financials",
        },
        "negative": {
            "direction": "positive",
            "strength": "medium",
            "affected_sectors": ["banks", "NBFCs", "realty"],
            "reasoning": "Lower US rates attract FII inflows into Indian financials",
        },
    },
    # Gold up: risk-off → positive for gold stocks, negative for consumer
    "gold_futures": {
        "positive": {
            "direction": "positive",
            "strength": "low",
            "affected_sectors": ["gold", "jewellery"],
            "reasoning": "Higher gold benefits gold miners and jewellers",
        },
        "negative": {
            "direction": "neutral",
            "strength": "low",
            "affected_sectors": ["gold"],
            "reasoning": "Lower gold may impact gold stocks but minimal broader effect",
        },
    },
    # VIX up: fear → negative for banks, autos; positive for defensives
    "vix": {
        "positive": {
            "direction": "negative",
            "strength": "high",
            "affected_sectors": ["banks", "autos", "metals", "realty"],
            "reasoning": "High volatility triggers risk-off, hits cyclical sectors hardest",
        },
        "negative": {
            "direction": "positive",
            "strength": "medium",
            "affected_sectors": ["banks", "autos", "metals"],
            "reasoning": "Low volatility supports risk-on and cyclical recovery",
        },
    },
    # DXY up: risk-off → negative for emerging markets
    "dxy": {
        "positive": {
            "direction": "negative",
            "strength": "medium",
            "affected_sectors": ["banks", "IT", "metals"],
            "reasoning": "Strong USD triggers FII outflows from Indian markets",
        },
        "negative": {
            "direction": "positive",
            "strength": "medium",
            "affected_sectors": ["banks", "IT", "metals"],
            "reasoning": "Weak USD supports EM flows into Indian equities",
        },
    },
    # Nifty up: bullish → positive for banks, autos, cyclicals
    "nifty_50": {
        "positive": {
            "direction": "positive",
            "strength": "high",
            "affected_sectors": ["banks", "autos", "metals", "realty"],
            "reasoning": "Rising Nifty signals broad market strength, benefits cyclicals",
        },
        "negative": {
            "direction": "negative",
            "strength": "medium",
            "affected_sectors": ["banks", "autos", "metals"],
            "reasoning": "Falling Nifty signals risk-off, hits cyclical sectors",
        },
    },
    # US 10Y yield up: hawkish → negative for growth/tech
    "us_10y_yield": {
        "positive": {
            "direction": "negative",
            "strength": "medium",
            "affected_sectors": ["IT", "pharma", "FMCG"],
            "reasoning": "Rising yields hurt growth stocks via higher discount rates",
        },
        "negative": {
            "direction": "positive",
            "strength": "low",
            "affected_sectors": ["IT", "pharma"],
            "reasoning": "Falling yields support growth stock valuations",
        },
    },
}


def compute_sector_impact(
    indicators: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Compute sector-specific impact based on macro indicator directions.

    Args:
        indicators: Raw macro indicator data (from fetch_macro_indicators).

    Returns:
        Dict mapping sector name → {impact_direction, strength, reasoning, drivers}.
    """
    data = indicators.get("data", {})
    sector_scores: Dict[str, Dict[str, Any]] = {}

    for indicator_key, sector_map in SECTOR_IMPACT_MATRIX.items():
        indicator = data.get(indicator_key, {})
        change_pct = indicator.get("change_pct", 0)

        if change_pct == 0:
            continue

        # Determine if indicator moved "up" or "down"
        direction = "positive" if change_pct > 0 else "negative"
        impacts = sector_map.get(direction, {})

        for sector in impacts.get("affected_sectors", []):
            if sector not in sector_scores:
                sector_scores[sector] = {
                    "impact_direction": "neutral",
                    "strength": "low",
                    "drivers": [],
                    "reasoning": "",
                }

            entry = sector_scores[sector]
            entry["drivers"].append(
                f"{indicator_key}: {change_pct:+.2f}%"
            )

            # If indicator has high impact, override to high
            if impacts.get("strength") == "high":
                entry["strength"] = "high"
                entry["impact_direction"] = impacts.get("direction", "neutral")
            elif entry["strength"] == "low":
                entry["strength"] = impacts.get("strength", "low")
                entry["impact_direction"] = impacts.get("direction", "neutral")

    # Add reasoning per sector
    for sector, info in sector_scores.items():
        if info["impact_direction"] == "positive":
            info["reasoning"] = (
                f"Bullish macro backdrop for {sector}. "
                f"Drivers: {', '.join(info['drivers'])}"
            )
        elif info["impact_direction"] == "negative":
            info["reasoning"] = (
                f"Bearish macro headwinds for {sector}. "
                f"Drivers: {', '.join(info['drivers'])}"
            )
        else:
            info["reasoning"] = (
                f"Mixed signals for {sector}. "
                f"Drivers: {', '.join(info['drivers'])}"
            )

    return sector_scores


# ================== GLOBAL MACRO AGENT ==================

class GlobalMacroAgent:
    """
    Analyzes global macroeconomic indicators and their impact on Indian markets.
    
    Collects data on 20+ indicators across US economy, commodities,
    global markets, and India-specific metrics.
    """
    
    def __init__(self):
        """Initialize the Global Macro Agent."""
        pass
    
    async def fetch_indicators(self) -> Dict[str, Any]:
        """
        Fetch all macroeconomic indicators.
        
        Returns:
            Dictionary with indicator values, changes, and metadata
        """
        return await fetch_macro_indicators()
    
    async def analyze_impact(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze impact of indicators on Indian markets.
        
        Args:
            indicators: Dictionary of macro indicator values
            
        Returns:
            Dictionary with impact analysis results
        """
        return analyze_india_impact(indicators)
    
    async def calculate_impact_score(self, analysis: Dict[str, Any]) -> str:
        """
        Calculate overall impact score.
        
        Args:
            analysis: Dictionary with impact analysis results
            
        Returns:
            Impact score string
        """
        return calculate_impact_score(analysis)

    async def _agentic_analyze(self) -> Optional[Dict[str, Any]]:
        """Tool-calling agentic macro analysis: LLM decides which indicators to query."""
        llm = get_llm_client()
        if not llm.is_available:
            return None

        system_prompt = (
            "You are a global macro strategist focused on Indian market impact.\n"
            "Use your tools to fetch key macro indicators: ^GSPC, ^VIX, CL=F, GC=F,\n"
            "^TNX, USDINR=X, ^NSEI, plus FRED data (GDP, CPIAUCSL, FEDFUNDS, UNRATE, DGS10).\n"
            "Then produce a COMPLETE JSON with: indicators (dict of fetched values),\n"
            "india_impact_summary (2-3 sentence narrative),\n"
            "overall_sentiment (positive/neutral/caution/negative),\n"
            "sector_impacts (dict: sector → impact score -5 to +5),\n"
            "risk_level (low/medium/high), conviction (0.0-1.0).\n"
            "Output ONLY the JSON — no markdown, no explanation."
        )

        raw = await llm.generate_with_tools(
            system_prompt=system_prompt,
            tools=MACRO_TOOLS,
            tool_callables=MACRO_CALLABLES,
            model=llm.deep_model,
            max_tokens=2048,
        )
        if not raw:
            return None

        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Macro agentic output not JSON")
            return None

    async def global_macro_agent(self, state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
        """
        Orchestrate macro analysis workflow.
        
        Fetches macro indicators, analyzes impact on Indian markets,
        and updates state with macro signals.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with macro signals
        """
        errors = state.get("errors", [])
        
        # Try agentic path first
        result = await self._agentic_analyze()
        if result:
            india_impact = (
                result.get("india_impact")
                or result.get("india_impact_summary")
                or result.get("narrative", "")
            )
            sector_impact = (
                result.get("sector_impact")
                or result.get("sector_impacts")
                or {}
            )
            return {
                "macro_signals": {
                    "indicators": result.get("indicators", {}),
                    "india_impact": india_impact,
                    "sector_impact": sector_impact,
                    "narrative": india_impact,
                    "overall_sentiment": result.get("overall_sentiment", "neutral"),
                    "risk_level": result.get("risk_level", "medium"),
                    "conviction": result.get("conviction", 0.5),
                    "source": "agentic",
                }
            }

        try:
            # Step 1: Fetch macro indicators
            indicators = await fetch_macro_indicators()
            
            if indicators.get("errors"):
                errors.extend(indicators.get("errors", []))
            
            # Step 2: Analyze impact on Indian markets
            analysis = await analyze_india_impact(indicators)
            
            # Step 3: Compute sector-specific impacts (FR-053)
            sector_impact = compute_sector_impact(indicators)
            
            # Step 4: Generate macro narrative (2-3 sentence LLM summary)
            narrative = await generate_macro_narrative(indicators, analysis)
            
            # Step 5: Update state with macro signals
            updated_state = {
                "macro_signals": {
                    "timestamp": datetime.now().isoformat(),
                    "indicators": indicators.get("data", {}),
                    "impact_analysis": analysis,
                    "impact_score": analysis.get("impact_score", "neutral"),
                    "recommendations": analysis.get("recommendations", []),
                    "sector_impact": sector_impact,
                    "narrative": narrative,
                },
                "agent": "global_macro",
                "next": None,  # Will be set by supervisor
                "errors": errors,
            }
            
            return updated_state
            
        except Exception as e:
            errors.append(f"Global macro agent failed: {str(e)}")
            
            updated_state = {
                "macro_signals": {
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                },
                "agent": "global_macro",
                "next": None,
                "errors": errors,
            }
            
            return updated_state


# ================== UTILITY FUNCTIONS ==================

def clear_cache() -> None:
    """Clear the macro data cache."""
    _macro_cache.clear()


def get_cache_status() -> Dict[str, Any]:
    """Get cache status for debugging."""
    return {
        "cache_size": len(_macro_cache._cache),
        "cache_keys": list(_macro_cache._cache.keys()),
    }


def get_indicator_list() -> Dict[str, Dict[str, str]]:
    """Get categorized list of all supported indicators."""
    return {
        "us_economy": US_INDICATORS,
        "commodities": COMMODITY_INDICATORS,
        "global_markets": GLOBAL_MARKET_INDICATORS,
        "india_specific": INDIA_INDICATORS,
        "other": OTHER_INDICATORS,
    }


# Global agent instance
_global_macro_agent: Optional[GlobalMacroAgent] = None


def create_global_macro_agent() -> GlobalMacroAgent:
    """Create or get global macro agent instance."""
    global _global_macro_agent
    
    if _global_macro_agent is None:
        _global_macro_agent = GlobalMacroAgent()
    
    return _global_macro_agent


async def global_macro_agent(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """
    Convenience function to run global macro agent.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with macro signals
    """
    agent = create_global_macro_agent()
    return await agent.global_macro_agent(state, config)
