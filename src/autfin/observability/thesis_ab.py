"""A/B testing framework for LLM vs rule-based thesis checks.

Compares outcomes of LLM-driven thesis validation (THESIS_CHECK_PROMPT)
against the rule-based fallback (_rule_based_thesis_check) over time
to determine if LLM judgments are worth the API cost.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "autfin.db"


@dataclass
class ThesisCheckResult:
    symbol: str
    llm_verdict: Optional[str]
    rule_verdict: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    llm_reasoning: Optional[str] = None
    price_change_pct: float = 0.0
    days_held: int = 0
    subsequent_outcome: Optional[str] = None
    subsequent_pnl_pct: Optional[float] = None


def _get_db() -> sqlite3.Connection:
    return sqlite3.connect(str(DB_PATH), isolation_level=None)


def _ensure_table():
    try:
        conn = _get_db()
        conn.execute(
            """CREATE TABLE IF NOT EXISTS thesis_check_ab (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                llm_verdict TEXT,
                rule_verdict TEXT NOT NULL,
                llm_reasoning TEXT,
                price_change_pct REAL DEFAULT 0,
                days_held INTEGER DEFAULT 0,
                subsequent_outcome TEXT,
                subsequent_pnl_pct REAL,
                timestamp TEXT NOT NULL
            )"""
        )
        conn.close()
    except Exception:
        pass


def record_thesis_check(result: ThesisCheckResult):
    _ensure_table()
    try:
        conn = _get_db()
        conn.execute(
            """INSERT INTO thesis_check_ab
            (symbol, llm_verdict, rule_verdict, llm_reasoning, price_change_pct,
             days_held, subsequent_outcome, subsequent_pnl_pct, timestamp)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                result.symbol,
                result.llm_verdict,
                result.rule_verdict,
                result.llm_reasoning,
                result.price_change_pct,
                result.days_held,
                result.subsequent_outcome,
                result.subsequent_pnl_pct,
                result.timestamp,
            ),
        )
        conn.close()
    except Exception as e:
        logger.debug(f"Thesis check record failed: {e}")


def compute_ab_stats(lookback_days: int = 90) -> Dict[str, Any]:
    """Compare LLM vs rule-based accuracy by whether subsequent price moved correctly."""
    _ensure_table()
    try:
        conn = _get_db()
        rows = conn.execute(
            """SELECT llm_verdict, rule_verdict, subsequent_outcome, price_change_pct
            FROM thesis_check_ab
            ORDER BY timestamp DESC LIMIT 1000"""
        ).fetchall()
        conn.close()
    except Exception:
        return {"error": "DB unavailable", "total_comparisons": 0}

    total = len(rows)
    if total == 0:
        return {"total_comparisons": 0, "message": "No A/B data yet"}

    verdicts = {"llm": {"correct": 0, "wrong": 0, "total": 0}, "rule": {"correct": 0, "wrong": 0, "total": 0}}
    agreement = 0

    for row in rows:
        llm_v, rule_v, outcome, change = row

        llm_correct = _is_correct(llm_v, outcome, change)
        rule_correct = _is_correct(rule_v, outcome, change)

        if llm_v:
            verdicts["llm"]["total"] += 1
            if llm_correct:
                verdicts["llm"]["correct"] += 1
            else:
                verdicts["llm"]["wrong"] += 1

        verdicts["rule"]["total"] += 1
        if rule_correct:
            verdicts["rule"]["correct"] += 1
        else:
            verdicts["rule"]["wrong"] += 1

        if llm_v and llm_v == rule_v:
            agreement += 1

    return {
        "total_comparisons": total,
        "agreement_rate": round(agreement / total, 3) if total else 0,
        "llm_accuracy": round(
            verdicts["llm"]["correct"] / verdicts["llm"]["total"], 3
        ) if verdicts["llm"]["total"] else None,
        "rule_accuracy": round(
            verdicts["rule"]["correct"] / verdicts["rule"]["total"], 3
        ) if verdicts["rule"]["total"] else 0,
        "llm_total": verdicts["llm"]["total"],
        "rule_total": verdicts["rule"]["total"],
        "llm_better": (
            "LLM"
            if (
                verdicts["llm"]["total"] > 0
                and (verdicts["llm"]["correct"] / verdicts["llm"]["total"])
                > (verdicts["rule"]["correct"] / verdicts["rule"]["total"])
            )
            else "Rule-based"
        ),
    }


def _is_correct(verdict: Optional[str], subsequent_outcome: Optional[str], change_pct: Optional[float]) -> bool:
    if not verdict or not subsequent_outcome:
        return False

    if verdict == "BROKEN" and subsequent_outcome == "DOWN":
        return True
    if verdict == "INTACT" and subsequent_outcome in ("UP", "FLAT"):
        return True
    return False


def record_outcome(symbol: str, verdict: str, pnl_pct: float):
    try:
        conn = _get_db()
        outcome = "UP" if pnl_pct > 1 else "DOWN" if pnl_pct < -1 else "FLAT"
        conn.execute(
            """UPDATE thesis_check_ab SET subsequent_outcome=?, subsequent_pnl_pct=?
            WHERE symbol=? AND id=(SELECT MAX(id) FROM thesis_check_ab WHERE symbol=? AND subsequent_outcome IS NULL)""",
            (outcome, round(pnl_pct, 2), symbol, symbol),
        )
        conn.close()
    except Exception:
        pass
