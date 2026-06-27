"""Episodic memory for AutFin.

Stores and retrieves past decisions with their outcomes and lessons.
Enables "Last time you bought INFY after 8% drop..." style retrieval.

Schema in SQLite:
  episodic_memory(
    id TEXT PRIMARY KEY,
    symbol TEXT,
    action TEXT,
    entry_date TEXT,
    entry_price REAL,
    exit_date TEXT,
    exit_price REAL,
    pnl_pct REAL,
    conviction REAL,
    thesis TEXT,
    market_context TEXT,
    lesson TEXT,
    outcome TEXT,
    created_at TEXT
  )
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "autfin.db"

_CREATE_SQL = """CREATE TABLE IF NOT EXISTS episodic_memory (
    id TEXT PRIMARY KEY,
    symbol TEXT,
    action TEXT,
    entry_date TEXT,
    entry_price REAL,
    exit_date TEXT,
    exit_price REAL,
    pnl_pct REAL,
    conviction REAL,
    thesis TEXT,
    market_context TEXT,
    lesson TEXT,
    outcome TEXT,
    created_at TEXT
)"""


class EpisodicMemory:
    """Store and retrieve past trading decisions with outcomes and lessons."""

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = Path(db_path) if db_path else _DB_PATH
        self._ensure_table()

    def _ensure_table(self):
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(_CREATE_SQL)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("EpisodicMemory table init failed: %s", e)

    def record_decision(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        conviction: float,
        thesis: str = "",
        market_context: str = "",
    ) -> str:
        """Record a new decision. Returns the decision ID."""
        decision_id = f"EP-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now(timezone.utc).isoformat()
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                "INSERT INTO episodic_memory (id, symbol, action, entry_date, entry_price, "
                "conviction, thesis, market_context, outcome, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)",
                (decision_id, symbol, action.upper(), now, entry_price,
                 conviction, thesis, market_context, now),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("record_decision failed: %s", e)
        return decision_id

    def close_decision(
        self,
        decision_id: str,
        exit_price: float,
        lesson: str = "",
    ) -> bool:
        """Close an open decision with exit price and lesson. Returns success."""
        try:
            conn = sqlite3.connect(self._db_path)
            row = conn.execute(
                "SELECT entry_price, action FROM episodic_memory WHERE id = ?",
                (decision_id,),
            ).fetchone()
            if not row:
                return False
            entry_price, action = row
            if entry_price <= 0:
                return False
            pnl_pct = ((exit_price - entry_price) / entry_price * 100
                       if action == "BUY"
                       else (entry_price - exit_price) / entry_price * 100)
            outcome = "profit" if pnl_pct > 0 else "loss"
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "UPDATE episodic_memory SET exit_date = ?, exit_price = ?, "
                "pnl_pct = ?, lesson = ?, outcome = ? WHERE id = ?",
                (now, exit_price, round(pnl_pct, 2), lesson, outcome, decision_id),
            )
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.warning("close_decision failed: %s", e)
            return False

    def retrieve_similar(
        self,
        symbol: str,
        action: str = "",
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve past decisions for a symbol, most recent first."""
        results = []
        try:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            where = "WHERE symbol = ?"
            params: list = [symbol]
            if action:
                where += " AND action = ?"
                params.append(action.upper())
            rows = conn.execute(
                f"SELECT * FROM episodic_memory {where} ORDER BY created_at DESC LIMIT ?",
                [*params, limit],
            ).fetchall()
            for r in rows:
                results.append(dict(r))
            conn.close()
        except Exception as e:
            logger.warning("retrieve_similar failed: %s", e)
        return results

    def get_lessons(self, symbol: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Get decisions that have lessons attached."""
        results = []
        try:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            where = "WHERE lesson IS NOT NULL AND lesson != ''"
            params: list = []
            if symbol:
                where += " AND symbol = ?"
                params.append(symbol)
            rows = conn.execute(
                f"SELECT * FROM episodic_memory {where} ORDER BY created_at DESC LIMIT ?",
                [*params, limit],
            ).fetchall()
            for r in rows:
                results.append(dict(r))
            conn.close()
        except Exception as e:
            logger.warning("get_lessons failed: %s", e)
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get summary stats of episodic memory."""
        try:
            conn = sqlite3.connect(self._db_path)
            total = conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0]
            open_count = conn.execute(
                "SELECT COUNT(*) FROM episodic_memory WHERE outcome = 'open'"
            ).fetchone()[0]
            profit = conn.execute(
                "SELECT COUNT(*) FROM episodic_memory WHERE outcome = 'profit'"
            ).fetchone()[0]
            loss = conn.execute(
                "SELECT COUNT(*) FROM episodic_memory WHERE outcome = 'loss'"
            ).fetchone()[0]
            avg_pnl = conn.execute(
                "SELECT AVG(pnl_pct) FROM episodic_memory WHERE outcome IN ('profit', 'loss')"
            ).fetchone()[0] or 0.0
            conn.close()
            closed = profit + loss
            return {
                "total": total,
                "open": open_count,
                "closed": closed,
                "profit": profit,
                "loss": loss,
                "win_rate": round(profit / max(closed, 1) * 100, 1),
                "avg_pnl_pct": round(avg_pnl, 2),
            }
        except Exception as e:
            logger.warning("get_stats failed: %s", e)
            return {"total": 0, "open": 0, "closed": 0, "profit": 0, "loss": 0, "win_rate": 0, "avg_pnl_pct": 0}
