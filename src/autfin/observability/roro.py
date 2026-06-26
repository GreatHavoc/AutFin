"""Return on Research Output (RORO) — measures if research effort correlates with returns.

Computes a research-depth weighted P&L ratio. Stocks with deeper analysis
(full fundamentals + DCF) should outperform those with quick screens.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "autfin.db"

WEIGHT_MAP = {
    "quick_screen": 1,
    "standard": 2,
    "deep_dive": 3,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_db():
    """Get a database connection from the pool."""
    from src.autfin.memory.connection_pool import get_db_pool
    pool = get_db_pool(DB_PATH)
    return pool.connection()


def record_research_output(
    symbol: str,
    depth: str,
    decision_id: str,
    scores: Dict[str, float],
    research_duration_ms: float = 0.0,
):
    weight = WEIGHT_MAP.get(depth, 1)
    try:
        from src.autfin.memory.connection_pool import get_db_pool
        pool = get_db_pool(DB_PATH)
        with pool.connection() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS research_output (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    depth TEXT NOT NULL,
                    research_weight INTEGER NOT NULL,
                    fundamentals_score REAL,
                    technical_score REAL,
                    sentiment_score REAL,
                    macro_score REAL,
                    valuation_score REAL,
                    research_duration_ms REAL,
                    created_at TEXT DEFAULT (datetime('now'))
                )""",
            )
            conn.execute(
                """INSERT INTO research_output
                (decision_id, symbol, depth, research_weight, fundamentals_score,
                 technical_score, sentiment_score, macro_score, valuation_score, research_duration_ms)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    decision_id,
                    symbol,
                    depth,
                    weight,
                    scores.get("fundamentals", 0),
                    scores.get("technical", 0),
                    scores.get("sentiment", 0),
                    scores.get("macro", 0),
                    scores.get("valuation", 0),
                    research_duration_ms,
                ),
            )
    except Exception as e:
        logger.debug(f"RORO record failed: {e}")


def compute_roro(lookback_days: int = 90) -> Dict[str, Any]:
    """Compute Return on Research Output (RORO) score.
    
    Args:
        lookback_days: Only include research from the last N days (default: 90).
    """
    result = {
        "by_depth": {},
        "total_decisions": 0,
        "total_trades": 0,
        "roro_score": 0.0,
        "interpretation": "Insufficient data",
    }
    try:
        from src.autfin.memory.connection_pool import get_db_pool
        pool = get_db_pool(DB_PATH)
        with pool.connection() as conn:
            rows = conn.execute(
                """SELECT r.decision_id, r.symbol, r.depth, r.research_weight, r.created_at
                FROM research_output r
                WHERE r.created_at >= datetime('now', ?)
                ORDER BY r.created_at DESC""",
                (f"-{lookback_days} days",),
            ).fetchall()

        if not rows:
            return result

        depth_pnl: Dict[str, Tuple[float, int]] = {}
        total_pnl = 0.0
        total_weight = 0

        for row in rows:
            decision_id, symbol, depth, weight, created_at = row
            trade_pnl = _get_trade_pnl_for_symbol(symbol)
            depth_pnl.setdefault(depth, (0.0, 0))
            pnl, count = depth_pnl[depth]
            depth_pnl[depth] = (pnl + trade_pnl, count + 1)
            total_pnl += trade_pnl * weight
            total_weight += weight

        result["total_decisions"] = len(rows)

        for depth, (pnl, count) in depth_pnl.items():
            result["by_depth"][depth] = {
                "count": count,
                "total_pnl": round(pnl, 2),
                "avg_pnl_per_trade": round(pnl / count, 2) if count else 0,
                "weight": WEIGHT_MAP.get(depth, 1),
            }
            result["total_trades"] += count

        if total_weight > 0:
            result["roro_score"] = round(total_pnl / total_weight, 2)

        if result["roro_score"] > 50:
            result["interpretation"] = "Research strongly correlates with returns"
        elif result["roro_score"] > 0:
            result["interpretation"] = "Research weakly correlates with returns"
        elif result["roro_score"] > -50:
            result["interpretation"] = "Research does not correlate with returns"
        else:
            result["interpretation"] = "Research may be counter-productive — review methodology"

    except Exception as e:
        logger.warning(f"RORO computation failed: {e}")

    return result


def _get_trade_pnl_for_symbol(symbol: str) -> float:
    try:
        from src.autfin.memory.connection_pool import get_db_pool
        pool = get_db_pool(DB_PATH)
        with pool.connection() as conn:
            row = conn.execute(
                "SELECT pnl_pct FROM learning_journal WHERE symbol=? ORDER BY created_at DESC LIMIT 1",
                (symbol.upper(),),
            ).fetchone()
        return float(row[0]) if row and row[0] else 0.0
    except Exception:
        return 0.0


def get_research_efficiency() -> Dict[str, Any]:
    try:
        from src.autfin.memory.connection_pool import get_db_pool
        pool = get_db_pool(DB_PATH)
        with pool.connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM research_output").fetchone()[0]
    except Exception:
        total = 0

    roro = compute_roro()

    return {
        "decisions_recorded": total,
        "roro_score": roro["roro_score"],
        "roro_interpretation": roro["interpretation"],
        "by_depth": roro["by_depth"],
    }
