"""
Scheduled jobs for AutFin automated workflows.

Four recurring jobs driven by APScheduler AsyncIOScheduler:
  1. Daily Morning  — Mon-Fri 9:00 AM IST  (full Track A + Track B pipeline)
  2. Midday Guardian — Mon-Fri 1:00 PM IST  (position health check only)
  3. Weekly Review   — Sunday 10:00 AM IST   (portfolio review + watchlist refresh)
  4. Monthly Deep Dive — 1st of month 8:00 AM IST (strategy review + fundamentals refresh)

Each job creates a unique thread_id, builds an initial state, and streams through the
compiled LangGraph daily_morning graph (or subset for guardian-only runs).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

TIMEZONE_IST = "Asia/Kolkata"

# Indian stock market holidays (NSE/BSE) — 2026 major dates
# Source: NSE holiday calendar (https://www.nse.co.in/holiday-calendar)
INDIAN_MARKET_HOLIDAYS_2026 = {
    "2026-01-26",  # Republic Day
    "2026-03-10",  # Holi
    "2026-04-02",  # Ram Navami
    "2026-04-14",  # Dr. Ambedkar Jayanti
    "2026-04-15",  # Vishu / Tamil New Year
    "2026-05-01",  # Maharashtra Day
    "2026-08-15",  # Independence Day
    "2026-08-27",  # Ganesh Chaturthi
    "2026-10-02",  # Mahatma Gandhi Jayanti
    "2026-10-21",  # Dussehra
    "2026-11-01",  # Diwali (Laxmi Pujan)
    "2026-11-03",  # Diwali Balipratipada
    "2026-11-26",  # Guru Nanak Jayanti
    "2026-12-25",  # Christmas
}


def is_market_holiday(date_str: str | None = None) -> bool:
    """
    Check if today (or given date) is an Indian market holiday.

    Args:
        date_str: Optional date string in YYYY-MM-DD format. Uses today if None.

    Returns:
        True if the date is a market holiday.
    """
    if date_str is None:
        from datetime import date
        date_str = date.today().isoformat()
    return date_str in INDIAN_MARKET_HOLIDAYS_2026


async def _send_alert_safe(settings: Any, alert_type: str, title: str, message: str) -> None:
    """Send an alert without failing the job if the alert system errors."""
    try:
        from src.autfin.alerts.notifier import get_alert_manager, AlertType, Severity
        manager = get_alert_manager(settings)
        type_map = {
            "price": AlertType.PRICE,
            "stop_loss": AlertType.STOP_LOSS,
            "trade": AlertType.TRADE,
            "summary": AlertType.SUMMARY,
            "error": AlertType.ERROR,
            "info": AlertType.INFO,
        }
        severity = Severity.CRITICAL if alert_type == "error" else Severity.MEDIUM
        await manager.send(
            alert_type=type_map.get(alert_type, AlertType.INFO),
            title=title,
            message=message,
            severity=severity,
        )
    except Exception as exc:
        logger.debug("Alert send failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Job implementations
# ---------------------------------------------------------------------------


async def daily_morning_routine(
    graph: Any,
    store: Any,
    settings: Any,
) -> dict[str, Any]:
    """
    FR-120: Run the full daily morning pipeline at 9:00 AM IST (Mon-Fri).

    Executes both Track A (new buys when cash >= min_trade_amount) and
    Track B (portfolio guardian, always runs).

    Returns a summary dict with thread_id and status.
    """
    if is_market_holiday():
        logger.info("Skipping daily morning routine — market holiday")
        return {"thread_id": "holiday-skip", "status": "skipped", "reason": "market_holiday"}

    thread_id = f"morning-{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "thread_id": thread_id,
        "context": "morning_routine",     # v3.0 context-driven
        "depth": "standard",
        "run_mode": "daily",
        "available_cash": 0.0,
        "holdings": [],
        "positions": [],
        "messages": [HumanMessage(content="Run the daily morning routine")],
        "errors": [],
    }

    logger.info("[%s] Starting daily morning routine", thread_id)

    final_state: dict[str, Any] = {}
    try:
        async for event in graph.astream(initial_state, config, stream_mode="values"):
            final_state = event
            logger.debug("[%s] State update keys: %s", thread_id, list(event.keys()))
    except Exception as exc:
        logger.error("[%s] Morning routine failed: %s", thread_id, exc)
        await _send_alert_safe(settings, "error", "Morning routine failed", str(exc))
        return {
            "thread_id": thread_id,
            "status": "failed",
            "error": str(exc),
        }

    logger.info("[%s] Daily morning routine complete", thread_id)
    trades_today = len(final_state.get("execution_reports", []))
    errors_count = len(final_state.get("errors", []))
    await _send_alert_safe(
        settings, "summary", "Daily morning routine complete",
        f"Trades: {trades_today}, Errors: {errors_count}",
    )
    return {
        "thread_id": thread_id,
        "status": "completed",
        "state": final_state,
    }


async def portfolio_guardian_check(
    graph: Any,
    store: Any,
    settings: Any,
) -> dict[str, Any]:
    """
    FR-121: Mid-day position health check at 1:00 PM IST (Mon-Fri).

    Always runs regardless of cash level. Monitors existing positions for
    stop-loss triggers, profit targets, and risk assessment sell signals.

    Returns a summary dict with thread_id and status.
    """
    if is_market_holiday():
        logger.info("Skipping guardian check — market holiday")
        return {"thread_id": "holiday-skip", "status": "skipped", "reason": "market_holiday"}

    thread_id = f"guardian-{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d-%H')}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "thread_id": thread_id,
        "context": "price_alert",          # v3.0 context-driven
        "run_mode": "guardian",
        "available_cash": 0.0,
        "holdings": [],
        "positions": [],
        "messages": [HumanMessage(content="Run portfolio guardian check")],
        "errors": [],
    }

    logger.info("[%s] Starting midday guardian check", thread_id)

    final_state: dict[str, Any] = {}
    try:
        async for event in graph.astream(initial_state, config, stream_mode="values"):
            final_state = event
    except Exception as exc:
        logger.error("[%s] Guardian check failed: %s", thread_id, exc)
        return {
            "thread_id": thread_id,
            "status": "failed",
            "error": str(exc),
        }

    logger.info("[%s] Midday guardian check complete", thread_id)
    return {
        "thread_id": thread_id,
        "status": "completed",
        "state": final_state,
    }


async def weekly_portfolio_review(
    graph: Any,
    store: Any,
    settings: Any,
) -> dict[str, Any]:
    """
    FR-122: Weekly portfolio review on Sunday at 10:00 AM IST.

    Full portfolio review, watchlist refresh, pacing check (deploying too
    fast?), Screener.in fundamentals refresh.

    Returns a summary dict with thread_id and status.
    """
    thread_id = f"weekly-{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "thread_id": thread_id,
        "context": "weekly_review",
        "depth": "standard",
        "run_mode": "weekly",
        "available_cash": 0.0,
        "holdings": [],
        "positions": [],
        "messages": [HumanMessage(content="Run weekly portfolio review")],
        "errors": [],
    }

    logger.info("[%s] Starting weekly portfolio review", thread_id)

    final_state: dict[str, Any] = {}
    try:
        async for event in graph.astream(initial_state, config, stream_mode="values"):
            final_state = event
    except Exception as exc:
        logger.error("[%s] Weekly review failed: %s", thread_id, exc)
        return {
            "thread_id": thread_id,
            "status": "failed",
            "error": str(exc),
        }

    logger.info("[%s] Weekly portfolio review complete", thread_id)
    return {
        "thread_id": thread_id,
        "status": "completed",
        "state": final_state,
    }


async def monthly_deep_dive(
    graph: Any,
    store: Any,
    settings: Any,
) -> dict[str, Any]:
    """
    FR-123: Monthly deep dive on 1st of month at 8:00 AM IST.

    Review last month (deployments, P&L, win rate), deep fundamental review
    of all holdings, rebalancing recommendations, strategy performance review,
    knowledge consolidation (vector store summary generation).

    Returns a summary dict with thread_id and status.
    """
    thread_id = f"monthly-{datetime.now(tz=timezone.utc).strftime('%Y-%m')}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "thread_id": thread_id,
        "context": "quarterly_review",
        "depth": "deep",
        "run_mode": "quarterly",
        "available_cash": 0.0,
        "holdings": [],
        "positions": [],
        "messages": [HumanMessage(content="Run monthly deep dive")],
        "errors": [],
    }

    logger.info("[%s] Starting monthly deep dive", thread_id)

    final_state: dict[str, Any] = {}
    try:
        async for event in graph.astream(initial_state, config, stream_mode="values"):
            final_state = event
    except Exception as exc:
        logger.error("[%s] Monthly deep dive failed: %s", thread_id, exc)
        return {
            "thread_id": thread_id,
            "status": "failed",
            "error": str(exc),
        }

    logger.info("[%s] Monthly deep dive complete", thread_id)
    return {
        "thread_id": thread_id,
        "status": "completed",
        "state": final_state,
    }


# ---------------------------------------------------------------------------
# Scheduler registration
# ---------------------------------------------------------------------------


def register_jobs(
    scheduler: Any,
    graph: Any,
    store: Any,
    settings: Any,
) -> None:
    """
    Register all four scheduled jobs with an APScheduler AsyncIOScheduler.

    Call this once during startup after the graph is compiled and the
    scheduler is created but before ``scheduler.start()``.

    Parameters
    ----------
    scheduler : AsyncIOScheduler
        The APScheduler scheduler instance.
    graph : compiled LangGraph
        The compiled daily_morning graph (or equivalent).
    store : AutFinStore
        The memory store instance.
    settings : AutFinSettings
        Application settings (used for future per-job config overrides).
    """
    from apscheduler.triggers.cron import CronTrigger

    # Job 1: Daily Morning Routine — Mon-Fri 9:00 AM IST
    scheduler.add_job(
        daily_morning_routine,
        CronTrigger(
            day_of_week="mon-fri",
            hour=9,
            minute=0,
            timezone=TIMEZONE_IST,
        ),
        args=[graph, store, settings],
        id="daily_morning",
        name="Daily Market Scan + Portfolio Guardian",
        replace_existing=True,
    )

    # Job 2: Midday Guardian Check — Mon-Fri 1:00 PM IST
    scheduler.add_job(
        portfolio_guardian_check,
        CronTrigger(
            day_of_week="mon-fri",
            hour=13,
            minute=0,
            timezone=TIMEZONE_IST,
        ),
        args=[graph, store, settings],
        id="midday_guardian",
        name="Mid-Day Portfolio Health Check",
        replace_existing=True,
    )

    # Job 3: Weekly Portfolio Review — Sunday 10:00 AM IST
    scheduler.add_job(
        weekly_portfolio_review,
        CronTrigger(
            day_of_week="sun",
            hour=10,
            minute=0,
            timezone=TIMEZONE_IST,
        ),
        args=[graph, store, settings],
        id="weekly_review",
        name="Weekly Portfolio Review + Watchlist Refresh",
        replace_existing=True,
    )

    # Job 4: Monthly Deep Dive — 1st of month, 8:00 AM IST
    scheduler.add_job(
        monthly_deep_dive,
        CronTrigger(
            day=1,
            hour=8,
            minute=0,
            timezone=TIMEZONE_IST,
        ),
        args=[graph, store, settings],
        id="monthly_deep_dive",
        name="Monthly Strategy Review + Full Fundamental Refresh",
        replace_existing=True,
    )

    logger.info(
        "Registered 4 scheduled jobs: daily_morning, midday_guardian, "
        "weekly_review, monthly_deep_dive"
    )


# ---------------------------------------------------------------------------
# Manual trigger helpers (FR-124)
# ---------------------------------------------------------------------------


async def trigger_job(
    scheduler: Any,
    job_id: str,
    graph: Any,
    store: Any,
    settings: Any,
) -> dict[str, Any]:
    """
    FR-124: Manually trigger any scheduled job by its ID.

    Parameters
    ----------
    scheduler : AsyncIOScheduler
    job_id : str
        One of: daily_morning, midday_guardian, weekly_review, monthly_deep_dive
    graph, store, settings : required by the job functions

    Returns
    -------
    dict with thread_id and status from the triggered job.
    """
    job_map: dict[str, Callable] = {
        "daily_morning": daily_morning_routine,
        "midday_guardian": portfolio_guardian_check,
        "weekly_review": weekly_portfolio_review,
        "monthly_deep_dive": monthly_deep_dive,
    }

    job_fn = job_map.get(job_id)
    if job_fn is None:
        available = ", ".join(sorted(job_map.keys()))
        raise ValueError(
            f"Unknown job_id '{job_id}'. Available jobs: {available}"
        )

    logger.info("Manually triggering job: %s", job_id)
    return await job_fn(graph, store, settings)
