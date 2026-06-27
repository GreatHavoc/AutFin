"""
Self-critique loop — weekly hindsight review.

Periodically reviews past decisions and asks: "Would I make the same decision today?"
Captures hindsight lessons to improve future calibration.

Design: O(1) per critique — processes a batch of decisions, not one-by-one.
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def get_decisions_needing_review(
    conn: sqlite3.Connection,
    max_age_days: int = 30,
    min_age_days: int = 7,
) -> List[Dict[str, Any]]:
    """
    Fetch decisions that are old enough for hindsight review but not yet reviewed.

    Args:
        conn: SQLite connection
        max_age_days: Skip decisions newer than this
        min_age_days: Only review decisions older than this

    Returns:
        List of decision dicts with id, symbol, conviction, thesis, outcome
    """
    cutoff_old = (datetime.now() - timedelta(days=max_age_days)).isoformat()
    cutoff_new = (datetime.now() - timedelta(days=min_age_days)).isoformat()

    rows = conn.execute(
        """SELECT id, symbol, conviction, thesis_summary, outcome, pnl_pct, created_at
           FROM learning_journal
           WHERE created_at < ? AND created_at > ?
             AND outcome != 'open'
           ORDER BY created_at ASC
           LIMIT 50""",
        (cutoff_new, cutoff_old),
    ).fetchall()

    return [
        {
            "id": r[0],
            "symbol": r[1],
            "conviction": r[2],
            "thesis": r[3],
            "outcome": r[4],
            "pnl_pct": r[5],
            "created_at": r[6],
        }
        for r in rows
    ]


def build_critique_prompt(decision: Dict[str, Any]) -> str:
    """
    Build a hindsight review prompt for a single decision.

    Args:
        decision: Dict with symbol, conviction, thesis, outcome, pnl_pct

    Returns:
        Prompt string for LLM to critique
    """
    outcome_label = "WIN" if (decision.get("pnl_pct") or 0) > 0 else "LOSS"

    return f"""HINDSIGHT REVIEW — {decision['symbol']} ({outcome_label})

Decision made: {decision.get('thesis', 'N/A')}
Stated conviction: {decision.get('conviction', 'N/A')}
Actual outcome: {decision.get('outcome', 'N/A')} (PnL: {decision.get('pnl_pct', 0):.1f}%)

Review this past decision. Consider:
1. Was the conviction well-calibrated given the outcome?
2. What information was available at the time that was missed?
3. What was over-weighted or under-weighted?
4. What specific lesson should guide future similar decisions?

Respond in JSON:
{{
    "calibration_score": <0.0 to 1.0, how well-calibrated was the original conviction>,
    "overweighted_factors": ["factor1", "factor2"],
    "underweighted_factors": ["factor1"],
    "lesson": "<one-sentence actionable lesson>",
    "would_repeat": <true/false — would you make the same decision?>
}}"""


def parse_critique_response(response: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse LLM critique response into structured dict."""
    if not response:
        return None

    try:
        import json
        # Try to extract JSON from response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(response[start:end])
            return {
                "calibration_score": float(data.get("calibration_score", 0.5)),
                "overweighted_factors": data.get("overweighted_factors", []),
                "underweighted_factors": data.get("underweighted_factors", []),
                "lesson": data.get("lesson", ""),
                "would_repeat": bool(data.get("would_repeat", True)),
            }
    except Exception as e:
        logger.debug("Failed to parse critique response: %s", e)

    return None


def store_critique(
    conn: sqlite3.Connection,
    decision_id: int,
    critique: Dict[str, Any],
) -> None:
    """
    Store critique results in the episodic memory table.

    Args:
        conn: SQLite connection
        decision_id: ID from learning_journal
        critique: Parsed critique dict
    """
    import json

    # Store as episodic memory entry
    try:
        conn.execute(
            """INSERT INTO episodic_memory
               (decision_id, symbol, outcome, pnl_pct, lesson, context, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                decision_id,
                f"critique-{decision_id}",
                "critique",
                critique.get("calibration_score", 0.5),
                critique.get("lesson", ""),
                json.dumps({
                    "overweighted": critique.get("overweighted_factors", []),
                    "underweighted": critique.get("underweighted_factors", []),
                    "would_repeat": critique.get("would_repeat", True),
                }),
                datetime.now().isoformat(),
            ),
        )
    except Exception as e:
        logger.debug("Could not store critique (episodic_memory may not exist): %s", e)

    # Update learning journal with calibration note
    try:
        conn.execute(
            """UPDATE learning_journal
               SET thesis_summary = thesis_summary || ' [CALIBRATED: ' || ? || ']'
               WHERE id = ?""",
            (
                f"cal={critique.get('calibration_score', 0.5):.2f}",
                decision_id,
            ),
        )
    except Exception as e:
        logger.debug("Could not update learning_journal: %s", e)


def run_weekly_critique(
    conn: sqlite3.Connection,
    llm_generate_fn=None,
    max_decisions: int = 10,
) -> Dict[str, Any]:
    """
    Run the weekly hindsight review cycle.

    Args:
        conn: SQLite connection
        llm_generate_fn: Async callable (prompt) -> str for LLM generation
        max_decisions: Max decisions to review per cycle

    Returns:
        Summary dict with reviewed count, avg calibration, lessons found
    """
    decisions = get_decisions_needing_review(conn)
    if not decisions:
        return {"reviewed": 0, "avg_calibration": 0.0, "lessons": [], "decisions_available": 0}

    reviewed = 0
    total_calibration = 0.0
    lessons = []

    for decision in decisions[:max_decisions]:
        prompt = build_critique_prompt(decision)

        # If no LLM available, skip but count as needing review
        if llm_generate_fn is None:
            continue

        try:
            import asyncio
            import inspect

            # Handle both sync and async LLM functions
            if inspect.iscoroutinefunction(llm_generate_fn):
                response = asyncio.run(llm_generate_fn(prompt))
            else:
                response = llm_generate_fn(prompt)

            critique = parse_critique_response(response)

            if critique:
                store_critique(conn, decision["id"], critique)
                reviewed += 1
                total_calibration += critique["calibration_score"]
                if critique["lesson"]:
                    lessons.append({
                        "symbol": decision["symbol"],
                        "lesson": critique["lesson"],
                        "calibration": critique["calibration_score"],
                    })
        except Exception as e:
            logger.debug("Critique failed for %s: %s", decision["symbol"], e)

    avg_calibration = total_calibration / reviewed if reviewed > 0 else 0.0

    return {
        "reviewed": reviewed,
        "avg_calibration": avg_calibration,
        "lessons": lessons,
        "decisions_available": len(decisions),
    }
