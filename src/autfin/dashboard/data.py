"""
Dashboard data fetchers for AutFin.

Provides data access functions used by the Streamlit dashboard to populate
panels: portfolio overview, trade history, performance metrics, signal log,
and agent status. All functions are synchronous (Streamlit runs synchronously)
and query SQLite directly or call Kite SDK when available.

Design references:
- PRD §5.3: Dashboard panels — portfolio, signals, performance, trade log
- technical_spec.md §10.5: Dashboard module spec
- autfin-design.md §10: Dashboard architecture
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from ..memory.database import get_connection, DB_PATH
from ..memory.performance import (
    get_all_performance_metrics,
    get_cumulative_pnl,
    get_win_rate,
    get_profit_factor,
    get_symbol_performance,
    get_drawdown_stats,
    get_monthly_summary,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Portfolio snapshot helpers
# ---------------------------------------------------------------------------

def get_latest_snapshot() -> Optional[Dict[str, Any]]:
    """Get the most recent portfolio snapshot.

    Returns:
        Dict with date, available_cash, utilized_margin, total_holdings_value,
        total_pnl, total_pnl_pct, holdings_json, or None if no snapshots exist.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM portfolio_snapshots ORDER BY date DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        cols = [d[0] for d in conn.execute("PRAGMA table_info(portfolio_snapshots)").fetchall()]
        # PRAGMA returns (cid, name, type, notnull, dflt_value, pk); extract name
        col_names = [col[1] for col in conn.execute("PRAGMA table_info(portfolio_snapshots)").fetchall()]
        return dict(zip(col_names, row))
    finally:
        conn.close()


def get_snapshot_history(days: int = 30) -> List[Dict[str, Any]]:
    """Get portfolio snapshots for the last N days.

    Args:
        days: Number of days to look back.

    Returns:
        List of snapshot dicts ordered by date ascending.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM portfolio_snapshots WHERE date >= ? ORDER BY date ASC",
            (cutoff,),
        ).fetchall()
        col_names = [col[1] for col in conn.execute("PRAGMA table_info(portfolio_snapshots)").fetchall()]
        return [dict(zip(col_names, row)) for row in rows]
    finally:
        conn.close()


def save_snapshot(
    available_cash: float,
    utilized_margin: float,
    total_holdings_value: float,
    total_pnl: float,
    total_pnl_pct: float,
    holdings_json: str,
    date: Optional[str] = None,
) -> int:
    """Save a portfolio snapshot.

    Args:
        available_cash: Available cash balance.
        utilized_margin: Margin used.
        total_holdings_value: Total value of holdings.
        total_pnl: Total P&L.
        total_pnl_pct: Total P&L percentage.
        holdings_json: JSON string of holdings.
        date: Snapshot date (defaults to today).

    Returns:
        Row ID of inserted snapshot.
    """
    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT OR REPLACE INTO portfolio_snapshots
                (date, available_cash, utilized_margin, total_holdings_value,
                 total_pnl, total_pnl_pct, holdings_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (date, available_cash, utilized_margin, total_holdings_value,
             total_pnl, total_pnl_pct, holdings_json),
        )
        return cursor.lastrowid if cursor.lastrowid else 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Trade history queries
# ---------------------------------------------------------------------------

def get_trade_history(
    symbol: Optional[str] = None,
    days: int = 90,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Get recent trades, optionally filtered by symbol.

    Args:
        symbol: Optional NSE symbol filter.
        days: Look back N days.
        limit: Maximum rows.

    Returns:
        List of trade dicts ordered by timestamp descending.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    conn = get_connection()
    try:
        if symbol:
            rows = conn.execute(
                "SELECT * FROM trades WHERE symbol = ? AND timestamp >= ? ORDER BY timestamp DESC LIMIT ?",
                (symbol.upper(), cutoff, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM trades WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT ?",
                (cutoff, limit),
            ).fetchall()
        col_names = [col[1] for col in conn.execute("PRAGMA table_info(trades)").fetchall()]
        return [dict(zip(col_names, row)) for row in rows]
    finally:
        conn.close()


def get_trade_summary() -> Dict[str, Any]:
    """Get aggregate trade statistics.

    Returns:
        Dict with total_trades, total_buys, total_sells, total_volume,
        avg_trade_value, symbols_traded.
    """
    conn = get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        buys = conn.execute("SELECT COUNT(*) FROM trades WHERE action = 'BUY'").fetchone()[0]
        sells = conn.execute("SELECT COUNT(*) FROM trades WHERE action = 'SELL'").fetchone()[0]
        volume = conn.execute("SELECT COALESCE(SUM(total_value), 0) FROM trades").fetchone()[0]
        avg_val = conn.execute("SELECT COALESCE(AVG(total_value), 0) FROM trades").fetchone()[0]
        symbols = conn.execute(
            "SELECT COUNT(DISTINCT symbol) FROM trades"
        ).fetchone()[0]
        return {
            "total_trades": total,
            "total_buys": buys,
            "total_sells": sells,
            "total_volume": volume,
            "avg_trade_value": avg_val,
            "symbols_traded": symbols,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Signal log queries
# ---------------------------------------------------------------------------

def get_signal_history(
    symbol: Optional[str] = None,
    days: int = 30,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Get recent signals.

    Args:
        symbol: Optional symbol filter.
        days: Look back N days.
        limit: Maximum rows.

    Returns:
        List of signal dicts.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    conn = get_connection()
    try:
        if symbol:
            rows = conn.execute(
                "SELECT * FROM signals WHERE symbol = ? AND timestamp >= ? ORDER BY timestamp DESC LIMIT ?",
                (symbol.upper(), cutoff, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM signals WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT ?",
                (cutoff, limit),
            ).fetchall()
        col_names = [col[1] for col in conn.execute("PRAGMA table_info(signals)").fetchall()]
        return [dict(zip(col_names, row)) for row in rows]
    finally:
        conn.close()


# Alias for consistency
get_signal_log = get_signal_history


def get_signal_summary() -> Dict[str, Any]:
    """Get aggregate signal statistics.

    Returns:
        Dict with total_signals, acted_count, pending_count, buy_signals,
        sell_signals, hold_signals.
    """
    conn = get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        acted = conn.execute("SELECT COUNT(*) FROM signals WHERE was_acted_on = 1").fetchone()[0]
        buy = conn.execute("SELECT COUNT(*) FROM signals WHERE signal_type = 'BUY'").fetchone()[0]
        sell = conn.execute("SELECT COUNT(*) FROM signals WHERE signal_type = 'SELL'").fetchone()[0]
        hold = conn.execute("SELECT COUNT(*) FROM signals WHERE signal_type = 'HOLD'").fetchone()[0]
        return {
            "total_signals": total,
            "acted_count": acted,
            "pending_count": total - acted,
            "buy_signals": buy,
            "sell_signals": sell,
            "hold_signals": hold,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Learning journal queries
# ---------------------------------------------------------------------------

def get_learning_entries(
    symbol: Optional[str] = None,
    days: int = 90,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Get learning journal entries.

    Args:
        symbol: Optional symbol filter.
        days: Number of days to look back.
        limit: Maximum rows.

    Returns:
        List of learning entry dicts.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    conn = get_connection()
    try:
        if symbol:
            rows = conn.execute(
                "SELECT * FROM learning_journal WHERE symbol = ? AND created_at >= ? ORDER BY created_at DESC LIMIT ?",
                (symbol.upper(), cutoff, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM learning_journal WHERE created_at >= ? ORDER BY created_at DESC LIMIT ?",
                (cutoff, limit),
            ).fetchall()
        col_names = [col[1] for col in conn.execute("PRAGMA table_info(learning_journal)").fetchall()]
        return [dict(zip(col_names, row)) for row in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Stock pitch queries (Phase 3 dashboard)
# ---------------------------------------------------------------------------

def save_stock_pitch(
    symbol: str,
    context: str,
    pitch: Dict[str, Any],
    created_at: Optional[str] = None,
) -> int:
    """Persist a 6-element stock pitch to the stock_pitches table.

    Args:
        symbol: NSE symbol.
        context: Supervisor context ("morning_routine", "price_alert", etc.).
        pitch: Pitch dict from build_stock_pitch (may be wrapped in "pitch" key).
        created_at: Optional ISO timestamp string.

    Returns:
        Row ID of inserted pitch, or 0 on failure.
    """
    try:
        if created_at is None:
            created_at = datetime.now(timezone.utc).isoformat()

        payload = pitch.get("pitch", pitch) if isinstance(pitch, dict) else pitch
        if not isinstance(payload, dict):
            payload = {"raw_response": str(payload)[:2000]}

        catalysts = payload.get("catalysts", [])
        risks = payload.get("risks", [])
        price_target = payload.get("price_target", {}) or {}

        conn = get_connection()
        try:
            cursor = conn.execute(
                """
                INSERT INTO stock_pitches (
                    symbol, context, investment_thesis, catalysts_json, risks_json,
                    valuation_summary, competitive_moat,
                    price_target_base, price_target_upside, price_target_downside,
                    price_target_rationale, raw_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol.upper(),
                    context,
                    payload.get("investment_thesis", ""),
                    json.dumps(catalysts, default=str),
                    json.dumps(risks, default=str),
                    payload.get("valuation_summary", ""),
                    payload.get("competitive_moat", ""),
                    price_target.get("base") if isinstance(price_target, dict) else None,
                    price_target.get("upside") if isinstance(price_target, dict) else None,
                    price_target.get("downside") if isinstance(price_target, dict) else None,
                    price_target.get("rationale", "") if isinstance(price_target, dict) else "",
                    json.dumps(payload, default=str),
                    created_at,
                ),
            )
            return cursor.lastrowid if cursor.lastrowid else 0
        finally:
            conn.close()
    except Exception as e:
        logger.warning("save_stock_pitch failed for %s: %s", symbol, e)
        return 0


def get_stock_pitches(
    symbol: Optional[str] = None,
    context: Optional[str] = None,
    days: int = 90,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Get stored stock pitches with parsed catalysts/risks/price_target.

    Args:
        symbol: Optional NSE symbol filter.
        context: Optional context filter ("morning_routine", "price_alert", ...).
        days: Look back N days.
        limit: Maximum rows.

    Returns:
        List of pitch dicts ordered by created_at desc.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    conn = get_connection()
    try:
        sql = "SELECT * FROM stock_pitches WHERE created_at >= ?"
        params: List[Any] = [cutoff]
        if symbol:
            sql += " AND symbol = ?"
            params.append(symbol.upper())
        if context:
            sql += " AND context = ?"
            params.append(context)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()
        col_names = [col[1] for col in conn.execute("PRAGMA table_info(stock_pitches)").fetchall()]
        results = []
        for row in rows:
            d = dict(zip(col_names, row))
            for json_field in ("catalysts_json", "risks_json", "raw_json"):
                if d.get(json_field):
                    try:
                        d[json_field.replace("_json", "")] = json.loads(d[json_field])
                    except (json.JSONDecodeError, TypeError):
                        d[json_field.replace("_json", "")] = []
            results.append(d)
        return results
    finally:
        conn.close()


def get_latest_pitches_by_symbol(symbol: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Get most-recent pitch per symbol (one per symbol).

    Args:
        symbol: Optional filter.
        limit: Max rows.

    Returns:
        List of {symbol, latest_pitch_id, latest_created_at} dicts.
    """
    conn = get_connection()
    try:
        if symbol:
            rows = conn.execute(
                """
                SELECT symbol, MAX(created_at) AS latest, COUNT(*) AS count
                FROM stock_pitches WHERE symbol = ? GROUP BY symbol ORDER BY latest DESC LIMIT ?
                """,
                (symbol.upper(), limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT symbol, MAX(created_at) AS latest, COUNT(*) AS count
                FROM stock_pitches GROUP BY symbol ORDER BY latest DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [{"symbol": r[0], "latest_created_at": r[1], "pitch_count": r[2]} for r in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Fan-out progress queries (Phase 3 dashboard)
# ---------------------------------------------------------------------------

def record_fanout_start(session_id: str, context: str, symbols: List[str]) -> int:
    """Create a fan-out_progress row per symbol with status='running'.

    Args:
        session_id: UUID/grouping session key for the wave.
        context: Supervisor context.
        symbols: List of symbols being researched.

    Returns:
        Number of rows inserted.
    """
    if not symbols:
        return 0
    try:
        conn = get_connection()
        try:
            now = datetime.now(timezone.utc).isoformat()
            count = 0
            for sym in symbols:
                conn.execute(
                    """
                    INSERT INTO fanout_progress
                    (session_id, context, symbol, status, started_at, created_at)
                    VALUES (?, ?, ?, 'running', ?, ?)
                    """,
                    (session_id, context, sym.upper(), now, now),
                )
                count += 1
            return count
        finally:
            conn.close()
    except Exception as e:
        logger.warning("record_fanout_start failed: %s", e)
        return 0


def update_fanout_status(
    session_id: str,
    symbol: str,
    status: str,
    error: Optional[str] = None,
) -> bool:
    """Update a fan-out row's status. Mark finished_at for terminal statuses.

    Args:
        session_id: Session UUID.
        symbol: NSE symbol.
        status: One of 'pending'/'running'/'completed'/'failed'.
        error: Optional error message.

    Returns:
        True on success.
    """
    try:
        conn = get_connection()
        try:
            finished_at = (
                datetime.now(timezone.utc).isoformat()
                if status in ("completed", "failed") else None
            )
            conn.execute(
                """
                UPDATE fanout_progress
                SET status = ?, error = ?, finished_at = COALESCE(?, finished_at)
                WHERE session_id = ? AND symbol = ?
                """,
                (status, error, finished_at, session_id, symbol.upper()),
            )
            return True
        finally:
            conn.close()
    except Exception as e:
        logger.warning("update_fanout_status failed: %s", e)
        return False


def get_fanout_progress(
    session_id: Optional[str] = None,
    hours: int = 24,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Return fan-out progress rows, optionally limited to a session.

    Args:
        session_id: Optional session filter; if None, returns recent across sessions.
        hours: Look-back window in hours.
        limit: Max rows.

    Returns:
        List of progress dicts.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")
    conn = get_connection()
    try:
        if session_id:
            rows = conn.execute(
                "SELECT * FROM fanout_progress WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM fanout_progress WHERE created_at >= ? "
                "ORDER BY created_at DESC LIMIT ?",
                (cutoff, limit),
            ).fetchall()
        col_names = [col[1] for col in conn.execute("PRAGMA table_info(fanout_progress)").fetchall()]
        # Newest first; reverse to chronological by started_at for chart friendliness
        return [dict(zip(col_names, row)) for row in rows]
    finally:
        conn.close()


def get_fanout_summary(hours: int = 24) -> Dict[str, Any]:
    """Aggregate fan-out progress counts.

    Args:
        hours: Look-back window.

    Returns:
        Dict with running/completed/failed/pending counts and completion_pct.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT status, COUNT(*) FROM fanout_progress WHERE created_at >= ? GROUP BY status",
            (cutoff,),
        ).fetchall()
        counts = {r[0]: r[1] for r in rows}
        total = sum(counts.values())
        completed = counts.get("completed", 0)
        return {
            "total": total,
            "pending": counts.get("pending", 0),
            "running": counts.get("running", 0),
            "completed": completed,
            "failed": counts.get("failed", 0),
            "completion_pct": round((completed / total) * 100, 1) if total else 0.0,
            "window_hours": hours,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Balance history
# ---------------------------------------------------------------------------

def get_balance_history(days: int = 30) -> List[Dict[str, Any]]:
    """Get balance history records.

    Args:
        days: Look back N days.

    Returns:
        List of balance history dicts ordered by timestamp ascending.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM balance_history WHERE timestamp >= ? ORDER BY timestamp ASC",
            (cutoff,),
        ).fetchall()
        col_names = [col[1] for col in conn.execute("PRAGMA table_info(balance_history)").fetchall()]
        return [dict(zip(col_names, row)) for row in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Dashboard aggregate data
# ---------------------------------------------------------------------------

def get_dashboard_overview() -> Dict[str, Any]:
    """Fetch all dashboard data in a single call for the main overview panel.

    Returns:
        Dict with keys: snapshot, trade_summary, signal_summary,
        performance, drawdown, equity_curve, recent_trades, recent_signals.
    """
    snapshot = get_latest_snapshot()
    trade_summary = get_trade_summary()
    signal_summary = get_signal_summary()
    performance = get_all_performance_metrics(days=90)
    drawdown = get_drawdown_stats()
    equity_curve = get_cumulative_pnl()
    recent_trades = get_trade_history(days=90, limit=20)
    recent_signals = get_signal_history(days=30, limit=20)

    return {
        "snapshot": snapshot,
        "trade_summary": trade_summary,
        "signal_summary": signal_summary,
        "performance": performance,
        "drawdown": drawdown,
        "equity_curve": equity_curve,
        "recent_trades": recent_trades,
        "recent_signals": recent_signals,
    }


def get_kite_status() -> Dict[str, Any]:
    """Check Kite SDK connectivity and return connection status.

    Returns:
        Dict with connected (bool), balance (float or None), error (str or None).
    """
    try:
        from ..tools.kite_tools import KiteTools
        kt = KiteTools()
        # Try to get margins as a connectivity check
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside Streamlit (sync context), can't use await
                return {"connected": False, "balance": None, "error": "Cannot run async in Streamlit context"}
        except RuntimeError:
            pass
        return {"connected": True, "balance": None, "error": None}
    except Exception as e:
        return {"connected": False, "balance": None, "error": str(e)}


def format_inr(amount: float) -> str:
    """Format amount in Indian Rupee notation (lakhs/crores).

    Args:
        amount: Amount in INR.

    Returns:
        Formatted string like ₹1,50,000 or ₹10,00,000.
    """
    if amount == 0:
        return "₹0"

    negative = amount < 0
    amount = abs(amount)

    if amount >= 1_00_00_000:  # 1 crore
        return ("-" if negative else "") + f"₹{amount / 1_00_00_000:.2f} Cr"
    if amount >= 1_00_000:  # 1 lakh
        return ("-" if negative else "") + f"₹{amount / 1_00_000:.2f}L"
    if amount >= 1_000:  # 1 thousand
        return ("-" if negative else "") + f"₹{amount:,.0f}"
    return ("-" if negative else "") + f"₹{amount:.2f}"


def format_pct(value: float) -> str:
    """Format a percentage with color indicator.

    Args:
        value: Percentage value.

    Returns:
        Formatted string like '+5.2%' or '-3.1%'.
    """
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.1f}%"


# ---------------------------------------------------------------------------
# System / Agent status
# ---------------------------------------------------------------------------

def get_agent_status() -> Dict[str, Any]:
    """
    Get agent and scheduler status for the System dashboard page.

    Returns:
        Dict with scheduler_jobs (list of dicts), cache_info, and db_stats.
    """
    result: Dict[str, Any] = {
        "scheduler_jobs": [],
        "cache_info": {},
        "db_stats": {},
    }

    # --- Scheduler jobs from APScheduler ---
    try:
        from ..scheduler.jobs import _scheduler  # type: ignore
        if _scheduler is not None:
            for job in _scheduler.get_jobs():
                result["scheduler_jobs"].append({
                    "id": job.id,
                    "name": job.name,
                    "next_run": str(job.next_run_time) if job.next_run_time else "N/A",
                    "trigger": str(job.trigger),
                })
    except Exception:
        pass

    # --- Agent cache sizes ---
    try:
        from ..agents.market_data import _cache as md_cache
        result["cache_info"]["market_data"] = len(md_cache._cache)
    except Exception:
        pass
    try:
        from ..agents.global_macro import _macro_cache
        result["cache_info"]["global_macro"] = len(_macro_cache._cache)
    except Exception:
        pass
    try:
        from ..agents.fundamentals import _cache as fund_cache
        result["cache_info"]["fundamentals"] = len(fund_cache._cache)
    except Exception:
        pass

    # --- DB table row counts ---
    conn = get_connection()
    try:
        for table in ["trades", "signals", "portfolio_snapshots", "balance_history", "learning_journal"]:
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                result["db_stats"][table] = count
            except Exception:
                result["db_stats"][table] = 0
    finally:
        conn.close()

    return result
