"""
Performance tracking for AutFin.

Provides win/loss ratio tracking, strategy comparison, and learning journal
CRUD operations. Computes metrics from SQLite tables (trades, portfolio_snapshots,
learning_journal) and integrates with Pydantic AI Memory for cross-thread memory.

Design references:
- technical_spec.md §7: learning_journal table schema (v3.0)
- technical_spec.md §3.2: DecisionRecord with outcome fields
- PRD §2.4: Win/loss ratio tracking by strategy type
- PRD §8.4: Monthly review — P&L, win rate, strategy performance
- autfin-design.md §9: Store namespace constants
"""

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .database import get_connection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# learning_journal CRUD operations (v3.0 schema)
# ---------------------------------------------------------------------------

def record_learning(
    symbol: str,
    action: str = "BUY",
    conviction: float = 0.5,
    pnl_pct: Optional[float] = None,
    thesis_summary: Optional[str] = None,
    outcome: str = "open",
    sector: Optional[str] = None,
) -> int:
    """Insert a learning journal entry using v3.0 schema."""
    conn = get_connection()
    try:
        created_at = datetime.now(timezone.utc).isoformat()
        cursor = conn.execute(
            """
            INSERT INTO learning_journal
                (symbol, action, conviction, pnl_pct, thesis_summary, outcome, sector, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (symbol.upper(), action.upper(), conviction, pnl_pct,
             thesis_summary, outcome, sector, created_at),
        )
        row_id = cursor.lastrowid
        logger.info(
            "Recorded learning for %s (pnl_pct=%s, outcome=%s)",
            symbol, pnl_pct, outcome,
        )
        return row_id
    finally:
        conn.close()


def get_learnings_for_symbol(symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieve learning journal entries for a specific symbol.

    Args:
        symbol: NSE symbol
        limit: Maximum entries to return

    Returns:
        List of learning entry dicts
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM learning_journal WHERE symbol = ? ORDER BY created_at DESC LIMIT ?",
            (symbol.upper(), limit),
        ).fetchall()
        columns = conn.execute("PRAGMA table_info(learning_journal)").fetchall()
        return [_row_to_dict(row, columns) for row in rows]
    finally:
        conn.close()


def get_recent_learnings(days: int = 30, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Retrieve recent learning journal entries.

    Args:
        days: Look back N days
        limit: Maximum entries to return

    Returns:
        List of learning entry dicts
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM learning_journal WHERE created_at >= ? ORDER BY created_at DESC LIMIT ?",
            (cutoff, limit),
        ).fetchall()
        columns = conn.execute("PRAGMA table_info(learning_journal)").fetchall()
        return [_row_to_dict(row, columns) for row in rows]
    finally:
        conn.close()


def update_learning_outcome(
    row_id: int,
    pnl_pct: float,
    outcome: str,
    thesis_summary: Optional[str] = None,
) -> bool:
    """Update outcome fields of a learning entry using v3.0 schema."""
    conn = get_connection()
    try:
        if thesis_summary is not None:
            cursor = conn.execute(
                "UPDATE learning_journal SET pnl_pct = ?, outcome = ?, thesis_summary = ? WHERE id = ?",
                (pnl_pct, outcome, thesis_summary, row_id),
            )
        else:
            cursor = conn.execute(
                "UPDATE learning_journal SET pnl_pct = ?, outcome = ? WHERE id = ?",
                (pnl_pct, outcome, row_id),
            )
        updated = cursor.rowcount > 0
        if updated:
            logger.info("Updated learning outcome for row %s", row_id)
        return updated
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Performance metrics computation
# ---------------------------------------------------------------------------

def get_win_rate(days: Optional[int] = None) -> Dict[str, Any]:
    """
    Compute overall win rate from learning_journal.

    Args:
        days: Optional lookback period (None = all time)

    Returns:
        Dict with total_trades, wins, losses, win_rate, total_pnl,
        avg_win, avg_loss
    """
    conn = get_connection()
    try:
        where_clause = "WHERE outcome IS NOT NULL"
        params: list = []
        if days is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
            where_clause += " AND created_at >= ?"
            params.append(cutoff)

        rows = conn.execute(
            f"SELECT pnl_pct, outcome FROM learning_journal {where_clause}",
            params,
        ).fetchall()

        if not rows:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
            }

        total = len(rows)
        # "completed" means a closed trade, not a winning one; deciding wins by
        # outcome counted every completed trade as a win regardless of pnl.
        # Wins/losses now come from realised pnl, matching learning/calibrator.py
        # and portfolio/tax_engine.py.
        wins = sum(1 for pnl, _ in rows if pnl is not None and pnl > 0)
        losses = sum(1 for pnl, _ in rows if pnl is not None and pnl <= 0)
        pnls = [pnl for pnl, _ in rows if pnl is not None]
        total_pnl = sum(pnls)
        win_pnls = [pnl for pnl, _ in rows if pnl is not None and pnl > 0]
        loss_pnls = [pnl for pnl, _ in rows if pnl is not None and pnl <= 0]

        return {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total if total > 0 else 0.0,
            "total_pnl": total_pnl,
            "avg_win": sum(win_pnls) / len(win_pnls) if win_pnls else 0.0,
            "avg_loss": sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0,
        }
    finally:
        conn.close()


def get_profit_factor(days: Optional[int] = None) -> Dict[str, Any]:
    """
    Compute profit factor (sum of wins / abs(sum of losses)).

    Args:
        days: Optional lookback period

    Returns:
        Dict with profit_factor, gross_profit, gross_loss
    """
    conn = get_connection()
    try:
        where_clause = "WHERE pnl_pct IS NOT NULL"
        params: list = []
        if days is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
            where_clause += " AND created_at >= ?"
            params.append(cutoff)

        rows = conn.execute(
            f"SELECT pnl_pct FROM learning_journal {where_clause}",
            params,
        ).fetchall()

        pnls = [row[0] for row in rows]
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

        return {
            "profit_factor": profit_factor,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
        }
    finally:
        conn.close()


def get_symbol_performance(
    symbol: Optional[str] = None,
    days: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Compute per-symbol performance metrics.

    Args:
        symbol: Optional specific symbol (None = all symbols)
        days: Optional lookback period

    Returns:
        List of dicts, one per symbol, with win/loss/P&L metrics
    """
    conn = get_connection()
    try:
        where_parts = ["WHERE outcome IS NOT NULL"]
        params: list = []
        if symbol:
            where_parts.append("AND symbol = ?")
            params.append(symbol.upper())
        if days is not None:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
            where_parts.append("AND created_at >= ?")
            params.append(cutoff)

        where_clause = " ".join(where_parts)
        rows = conn.execute(
            f"""
            SELECT symbol, pnl_pct, outcome
            FROM learning_journal
            {where_clause}
            """,
            params,
        ).fetchall()

        # Group by symbol
        by_symbol: Dict[str, List] = {}
        for sym, pnl, oc in rows:
            if sym not in by_symbol:
                by_symbol[sym] = []
            by_symbol[sym].append((pnl, oc))

        results = []
        for sym, trades in sorted(by_symbol.items()):
            total = len(trades)
            wins = sum(1 for _, o in trades if o == "completed")
            pnls = [p for p, _ in trades if p is not None]
            total_pnl = sum(pnls)
            results.append({
                "symbol": sym,
                "total_trades": total,
                "wins": wins,
                "losses": total - wins,
                "win_rate": wins / total if total > 0 else 0.0,
                "total_pnl": total_pnl,
                "avg_pnl": total_pnl / len(pnls) if pnls else 0.0,
            })

        return results
    finally:
        conn.close()


def get_monthly_summary(year: int, month: int) -> Dict[str, Any]:
    """
    Compute monthly performance summary.

    Args:
        year: Year (e.g. 2026)
        month: Month (1-12)

    Returns:
        Dict with monthly metrics
    """
    month_str = f"{year}-{month:02d}"
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT pnl_pct, outcome
            FROM learning_journal
            WHERE created_at LIKE ? AND outcome IS NOT NULL
            """,
            (f"{month_str}%",),
        ).fetchall()

        if not rows:
            return {
                "year": year,
                "month": month,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
            }

        total = len(rows)
        wins = sum(1 for _, o in rows if o == "completed")
        pnls = [p for p, _ in rows if p is not None]
        total_pnl = sum(pnls)

        return {
            "year": year,
            "month": month,
            "total_trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": wins / total if total > 0 else 0.0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(pnls) if pnls else 0.0,
        }
    finally:
        conn.close()


def get_cumulative_pnl() -> List[Dict[str, Any]]:
    """
    Get cumulative P&L from portfolio_snapshots ordered by date.

    Returns:
        List of dicts with date, total_pnl, total_pnl_pct, available_cash
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT date, total_pnl, total_pnl_pct, available_cash
            FROM portfolio_snapshots
            ORDER BY date ASC
            """,
        ).fetchall()

        return [
            {
                "date": row[0],
                "total_pnl": row[1],
                "total_pnl_pct": row[2],
                "available_cash": row[3],
            }
            for row in rows
        ]
    finally:
        conn.close()


def get_drawdown_stats() -> Dict[str, Any]:
    """
    Compute drawdown statistics from cumulative equity curve.

    Returns:
        Dict with max_drawdown, max_drawdown_pct, current_drawdown,
        current_drawdown_pct, peak_date, trough_date
    """
    snapshots = get_cumulative_pnl()

    if len(snapshots) < 2:
        return {
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "current_drawdown": 0.0,
            "current_drawdown_pct": 0.0,
            "peak_date": None,
            "trough_date": None,
        }

    # Compute equity as available_cash + deployed (approximate from P&L)
    # Use total_pnl as equity change proxy
    peak_pnl = 0.0
    peak_date = None
    max_dd = 0.0
    max_dd_pct = 0.0
    dd_peak_date = None
    dd_trough_date = None

    for snap in snapshots:
        pnl = snap["total_pnl"]
        if pnl > peak_pnl:
            peak_pnl = pnl
            peak_date = snap["date"]

        dd = peak_pnl - pnl
        dd_pct = dd / abs(peak_pnl) if peak_pnl != 0 else 0.0
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd_pct
            dd_peak_date = peak_date
            dd_trough_date = snap["date"]

    # Current drawdown from latest peak
    latest_pnl = snapshots[-1]["total_pnl"]
    current_dd = peak_pnl - latest_pnl
    current_dd_pct = current_dd / abs(peak_pnl) if peak_pnl != 0 else 0.0

    return {
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct,
        "current_drawdown": current_dd,
        "current_drawdown_pct": current_dd_pct,
        "peak_date": dd_peak_date,
        "trough_date": dd_trough_date,
    }


def get_all_performance_metrics(days: Optional[int] = None) -> Dict[str, Any]:
    """
    Aggregate all performance metrics into a single dict.

    Args:
        days: Optional lookback period

    Returns:
        Complete performance snapshot
    """
    win_rate = get_win_rate(days)
    profit_factor = get_profit_factor(days)
    symbol_perf = get_symbol_performance(days=days)
    drawdown = get_drawdown_stats()

    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "symbol_performance": symbol_perf,
        "drawdown": drawdown,
    }


# ---------------------------------------------------------------------------
# Outcome backfill — record realized P&L when a position closes
# ---------------------------------------------------------------------------

def record_trade_outcome(
    symbol: str,
    entry_price: float,
    exit_price: float,
    quantity: int,
    action: str = "BUY",
    thesis_summary: Optional[str] = None,
) -> Dict[str, Any]:
    """Record completed trade outcome using v3.0 schema."""
    if action.upper() == "BUY":
        pnl = (exit_price - entry_price) * quantity
    else:
        pnl = (entry_price - exit_price) * quantity

    outcome = "completed" if pnl > 0 else "loss"

    record_learning(
        symbol=symbol,
        action=action,
        pnl_pct=pnl,
        thesis_summary=thesis_summary,
        outcome=outcome,
    )

    logger.info(
        "Recorded trade outcome: %s %s ₹%.2f (%s)",
        symbol, "profit" if outcome == "completed" else "loss", abs(pnl),
        outcome,
    )

    return {
        "symbol": symbol,
        "pnl_pct": pnl,
        "outcome": outcome,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_dict(row: sqlite3.Row, columns: list) -> Dict[str, Any]:
    """Convert a sqlite3.Row to a dict using column info."""
    if hasattr(row, "keys"):
        return dict(row)
    return {col[1]: row[idx] for idx, col in enumerate(columns)}
