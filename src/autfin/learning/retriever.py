"""Learning retriever — semantic similarity search over learning journal for similar trade setups.

Uses sentence-transformers embeddings with cosine similarity for semantic retrieval.
Falls back to keyword-overlap scoring if embeddings unavailable.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def retrieve_learnings(
    query: str,
    filters: Optional[dict] = None,
    limit: int = 5,
) -> dict:
    """
    Search learning journal for similar past setups and their outcomes.

    Uses semantic embeddings (sentence-transformers) with cosine similarity.
    Falls back to keyword-overlap if embeddings unavailable.

    Args:
        query: Natural language description of current setup
        filters: Optional {sector: str, symbol: str, action: str} filters
        limit: Max results (default 5)

    Returns:
        {
            similar_setups: [{setup_description, action, outcome, pnl_pct, conviction, date, similarity_score}],
            calibration: {high_conv: {expected_win_rate, actual_win_rate, delta, sample_size}, ...},
            patterns: [{pattern_description, frequency, win_rate, severity}],
        }
    """
    try:
        from src.autfin.memory.database import get_connection
        db = get_connection()

        where = "WHERE 1=1"
        params: list = []
        if filters:
            if filters.get("sector"):
                where += " AND sector = ?"
                params.append(filters["sector"])
            if filters.get("symbol"):
                where += " AND symbol = ?"
                params.append(filters["symbol"])
            if filters.get("action"):
                where += " AND action = ?"
                params.append(filters["action"])

        rows = db.execute(
            f"SELECT symbol, action, conviction, pnl_pct, thesis_summary, outcome, "
            f"created_at, sector FROM learning_journal {where} "
            f"ORDER BY created_at DESC LIMIT ?",
            [*params, limit * 5]
        ).fetchall()

        # Try semantic embeddings first, fall back to keyword overlap
        from src.autfin.memory.embeddings import cosine_similarity, generate_embedding

        query_embedding = None
        try:
            query_embedding = generate_embedding(query)
        except Exception:
            query_embedding = None

        similar = []
        for row in rows:
            thesis = (row[4] or "")
            if not thesis:
                continue

            if query_embedding is not None:
                # Semantic similarity via embeddings
                row_embedding = generate_embedding(thesis)
                similarity = cosine_similarity(query_embedding, row_embedding)
            else:
                # Fallback: keyword overlap
                query_words = set(query.lower().split())
                thesis_lower = thesis.lower()
                overlap = len(query_words & set(thesis_lower.split()))
                similarity = overlap / max(len(query_words), 1)

            if similarity > 0.1 or not filters:
                similar.append({
                    "symbol": row[0],
                    "action": row[1],
                    "conviction": row[2],
                    "pnl_pct": row[3],
                    "thesis_summary": row[4],
                    "outcome": row[5],
                    "date": row[6],
                    "sector": row[7],
                    "similarity_score": round(similarity, 3),
                })

        similar = sorted(similar, key=lambda x: x["similarity_score"], reverse=True)[:limit]

        calibration = _compute_calibration(db)

        patterns = _detect_recent_patterns(db)

        return {
            "similar_setups": similar,
            "calibration": calibration,
            "patterns": patterns,
        }
    except Exception as e:
        logger.debug("Failed to retrieve learnings: %s", e, exc_info=True)
        return {"similar_setups": [], "calibration": {}, "patterns": [], "error": str(e)}


async def search_similar_setups(symbol: str, action: str, limit: int = 5) -> list[dict]:
    """Quick search for similar past trades by symbol and action."""
    result = await retrieve_learnings(
        query=f"{action} {symbol}",
        filters={"symbol": symbol, "action": action},
        limit=limit,
    )
    return result.get("similar_setups", [])


def _compute_calibration(db) -> dict:
    """Compute conviction tier calibration from learning_journal."""
    tiers = {"high_conv": (0.7, 1.01), "medium_conv": (0.4, 0.7), "low_conv": (0.0, 0.4)}
    calibration = {}

    for tier_name, (lo, hi) in tiers.items():
        rows = db.execute(
            "SELECT COUNT(*), SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) "
            "FROM learning_journal WHERE conviction >= ? AND conviction < ?",
            [lo, hi]
        ).fetchone()

        total, wins = rows[0], rows[1] or 0
        if total > 0:
            actual = wins / total
            expected = {"high_conv": 0.75, "medium_conv": 0.55, "low_conv": 0.35}[tier_name]
            calibration[tier_name] = {
                "sample_size": total,
                "expected_win_rate": expected,
                "actual_win_rate": round(actual, 3),
                "delta": round(actual - expected, 3),
                "status": _calibration_status(actual, expected),
            }

    return calibration


def _calibration_status(actual: float, expected: float) -> str:
    if actual < expected - 0.1:
        return "overconfident"
    if abs(actual - expected) < 0.1:
        return "calibrated"
    return "underconfident"


def _detect_recent_patterns(db) -> list[dict]:
    """Detect patterns from recent trades in learning_journal."""
    patterns: list[dict] = []

    loss_rows = db.execute(
        "SELECT symbol, sector, conviction, pnl_pct, thesis_summary "
        "FROM learning_journal WHERE pnl_pct < 0 AND outcome='completed' "
        "ORDER BY created_at DESC LIMIT 10"
    ).fetchall()

    if loss_rows:
        sectors: dict[str, int] = {}
        for r in loss_rows:
            s = r[1] or "Unknown"
            sectors[s] = sectors.get(s, 0) + 1
        for sector, count in sectors.items():
            if count >= 3:
                patterns.append({
                    "pattern": f"Loss concentration in {sector} sector",
                    "frequency": count,
                    "severity": "HIGH",
                    "recommendation": f"Review {sector} sector exposure. {count}/{len(loss_rows)} recent losses.",
                })

    high_conv_losses = [r for r in loss_rows if r[2] and r[2] >= 0.7]
    if len(high_conv_losses) >= 3:
        patterns.append({
            "pattern": "High-conviction loss streak",
            "frequency": len(high_conv_losses),
            "severity": "CRITICAL",
            "recommendation": "Conviction scores may be overconfident. Apply -0.15 calibration factor.",
        })

    total = db.execute("SELECT COUNT(*) FROM learning_journal WHERE outcome='completed'").fetchone()[0]
    wins = db.execute("SELECT COUNT(*) FROM learning_journal WHERE pnl_pct > 0 AND outcome='completed'").fetchone()[0]
    if total >= 10:
        win_rate = wins / max(total, 1)
        patterns.append({
            "pattern": f"Overall win rate: {win_rate:.0%}",
            "frequency": f"{wins}/{total}",
            "severity": "INFO",
            "recommendation": "Healthy" if win_rate >= 0.5 else "Needs improvement",
        })

    return patterns
