"""
Regulatory Analyst — Indian market regulatory event scanner.

Scans NSE/BSE corporate announcements, SEBI orders, RBI circulars,
and DPCO notifications. Classifies impact on sectors and holdings.

Deterministic — keyword-based classification with optional LLM for
ambiguous cases (not implemented yet, flagged for future).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sector-specific regulatory sensitivity
# ---------------------------------------------------------------------------

SECTOR_SENSITIVITY = {
    "NBFC": ["RBI", "PFRDA", "NBFC", "MUDRA", "NHB"],
    "Banks": ["RBI", "SEBI", "IBA", "DICGC"],
    "Pharma": ["DPCO", "NPPA", "CDSCO", "MCI", "DCGI"],
    "IT": ["MEITY", "TRAI", "DATA_PROTECTION"],
    "Telecom": ["TRAI", "DOT", "SPECTRUM"],
    "Auto": ["AIS", "SIAM", "EMISSIONS", "FAME"],
    "Energy": ["PNGRB", "CERC", "OPEC", "MINING"],
    "FMCG": ["FSSAI", "BIS"],
    "Real Estate": ["RERA", "NCLT", "IBC"],
    "Metals": ["DGFT", "MINING", "ENV_CLEARANCE"],
}

# Keywords that signal regulatory impact
REGULATORY_KEYWORDS = {
    "high_impact": [
        "SEBI order", "RBI circular", "DPCO", "NPPA", "penalty",
        "investigation", "fraud", "insolvency", "NCLT", "winding up",
        "licence revoked", "licence cancelled", "show cause",
        "search and seizure", "ED raid", "SFIO", "CBI",
    ],
    "medium_impact": [
        "compliance", "disclosure", "filing", "regulatory",
        "audit observation", "qualified opinion", "CARO",
        "board meeting", "AGM", "EGM", "rights issue",
        "buyback", "delisting", "scheme of arrangement",
    ],
    "low_impact": [
        "annual report", "quarterly results", "dividend",
        "investor presentation", "credit rating", "reaffirmation",
    ],
}

# SEBI-specific keywords
SEBI_KEYWORDS = [
    "SEBI", "SAST", "PIT", "insider trading", "front running",
    "front-running", "circular trading", "market manipulation",
    "algo trading", "co-location", "short selling",
]

# RBI-specific keywords
RBI_KEYWORDS = [
    "RBI", "MPC", "repo rate", "reverse repo", "CRR", "SLR",
    "priority sector", "NPA", "provisioning", "Basel III",
    "digital lending", "NBFC regulation", "payment aggregator",
]

# DPCO/NPPA for pharma
DPCO_KEYWORDS = [
    "DPCO", "NPPA", "drug price control", "price ceiling",
    "scheduled formulation", "non-scheduled", "trade margin",
    "MRP revision",
]


# ---------------------------------------------------------------------------
# Detector functions
# ---------------------------------------------------------------------------


def classify_regulatory_event(announcement: Dict[str, Any]) -> Dict[str, Any]:
    """Classify a regulatory announcement by impact and sector.

    Args:
        announcement: {title, body, source, date, symbol}

    Returns:
        {impact_level, sector_impact, direction, urgency, matched_keywords}
    """
    # Accept the field names producers actually send. Announcements shaped as
    # {headline, description} used to scan an empty string and classify as
    # "low" impact — a SEBI order read as nothing.
    def _field(*keys: str) -> str:
        for key in keys:
            value = announcement.get(key)
            if value:
                return str(value).upper()
        return ""

    title = _field("title", "subject", "headline")
    body = _field("body", "details", "description", "summary", "text")
    text = f"{title} {body}"

    if not text.strip():
        logger.debug("regulatory announcement had no scoreable text: %r", announcement)

    matched_keywords = []

    # Check high-impact keywords
    for kw in REGULATORY_KEYWORDS["high_impact"]:
        if kw.upper() in text:
            matched_keywords.append(kw)
    high_count = len(matched_keywords)

    # Check medium-impact keywords
    for kw in REGULATORY_KEYWORDS["medium_impact"]:
        if kw.upper() in text:
            matched_keywords.append(kw)
    medium_count = len(matched_keywords) - high_count

    # Determine impact level
    if high_count > 0:
        impact_level = "high"
    elif medium_count > 0:
        impact_level = "medium"
    else:
        impact_level = "low"

    # Detect sector
    sector_impact = []
    for sector, keywords in SECTOR_SENSITIVITY.items():
        for kw in keywords:
            if kw.upper() in text:
                sector_impact.append(sector)
                break

    # Detect direction
    negative_signals = ["penalty", "fraud", "investigation", "insolvency", "winding up",
                        "revoked", "cancelled", "raid", "seizure"]
    positive_signals = ["approval", "granted", "awarded", "order in favour", "cleared"]

    negative_count = sum(1 for s in negative_signals if s in text.lower())
    positive_count = sum(1 for s in positive_signals if s in text.lower())

    if negative_count > positive_count:
        direction = "negative"
    elif positive_count > negative_count:
        direction = "positive"
    else:
        direction = "neutral"

    # Determine urgency
    date_str = announcement.get("date", "")
    try:
        event_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        days_old = (datetime.now() - event_date.replace(tzinfo=None)).days
        if days_old <= 1:
            urgency = "immediate"
        elif days_old <= 7:
            urgency = "1-week"
        else:
            urgency = "historical"
    except Exception:
        urgency = "unknown"

    return {
        "impact_level": impact_level,
        "sector_impact": sector_impact,
        "direction": direction,
        "urgency": urgency,
        "matched_keywords": matched_keywords,
        "high_count": high_count,
        "medium_count": medium_count,
    }


def scan_nse_announcements(symbol: Optional[str] = None, days: int = 7) -> List[Dict[str, Any]]:
    """Scan NSE corporate announcements for regulatory events.

    Uses nselib to fetch corporate announcements.

    Args:
        symbol: Filter by symbol (None = all)
        days: Number of days to look back

    Returns:
        List of announcement dicts
    """
    announcements = []

    try:
        from nselib import corporate as nse_corporate

        # Fetch corporate announcements
        data = nse_corporate.announcements(symbol=symbol, from_date=(datetime.now() - timedelta(days=days)).strftime("%d-%m-%Y"))
        if data is not None and not data.empty:
            for _, row in data.iterrows():
                announcements.append({
                    "title": str(row.get("Subject", row.get("subject", ""))),
                    "body": str(row.get("Details", row.get("details", ""))),
                    "source": "NSE",
                    "date": str(row.get("Date", row.get("BroadcastDateTime", ""))),
                    "symbol": str(row.get("Symbol", row.get("symbol", symbol or ""))),
                })
    except ImportError:
        logger.debug("nselib not available for NSE announcements")
    except Exception as e:
        logger.warning("NSE announcement scan failed: %s", e)

    return announcements


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------


async def regulatory_audit(
    symbol: str,
    depth: str = "standard",
) -> Dict[str, Any]:
    """Run regulatory analysis for a stock.

    Scans NSE announcements and classifies regulatory impact.

    Args:
        symbol: Stock symbol (e.g. 'RELIANCE')
        depth: quick (last 3 days), standard (last 7 days), deep (last 30 days)

    Returns:
        {status, symbol, events: [...], high_impact_count, summary}
    """
    try:
        days_map = {"quick": 3, "standard": 7, "deep": 30}
        days = days_map.get(depth, 7)

        # Fetch announcements
        announcements = scan_nse_announcements(symbol=symbol, days=days)

        if not announcements:
            return {
                "status": "success",
                "symbol": symbol,
                "events": [],
                "high_impact_count": 0,
                "medium_impact_count": 0,
                "summary": f"No regulatory events found for {symbol} in last {days} days",
                "depth": depth,
                "timestamp": datetime.now().isoformat(),
            }

        # Classify each announcement
        classified_events = []
        for ann in announcements:
            classification = classify_regulatory_event(ann)
            classified_events.append({
                "title": ann["title"],
                "source": ann["source"],
                "date": ann["date"],
                **classification,
            })

        # Sort by impact level
        impact_order = {"high": 0, "medium": 1, "low": 2}
        classified_events.sort(key=lambda e: impact_order.get(e.get("impact_level", "low"), 3))

        high_impact = [e for e in classified_events if e.get("impact_level") == "high"]
        medium_impact = [e for e in classified_events if e.get("impact_level") == "medium"]

        if high_impact:
            summary = f"WARNING: {len(high_impact)} high-impact regulatory event(s) for {symbol}"
        elif medium_impact:
            summary = f"CAUTION: {len(medium_impact)} medium-impact regulatory event(s) for {symbol}"
        else:
            summary = f"No significant regulatory events for {symbol}"

        return {
            "status": "success",
            "symbol": symbol,
            "events": classified_events,
            "high_impact_count": len(high_impact),
            "medium_impact_count": len(medium_impact),
            "total_events": len(classified_events),
            "summary": summary,
            "depth": depth,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.warning("regulatory_audit failed for %s: %s", symbol, e)
        return {"status": "error", "symbol": symbol, "message": str(e)}
