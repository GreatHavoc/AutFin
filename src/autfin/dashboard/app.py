"""
AutFin Streamlit Dashboard.

Main entry point for the portfolio visualization dashboard. Run with:
    streamlit run src/autfin/dashboard/app.py

Panels:
1. Portfolio Overview — holdings, cash, P&L summary with KPI cards
2. Equity Curve — cumulative P&L chart over time
3. Trade History — recent trades with P&L coloring
4. Signal Log — recent signals with strength indicators
5. Performance Metrics — win rate, Sharpe, drawdown, per-symbol breakdown
6. Stock Pitches — 6-element buy-side pitches (Phase 3.1)
7. Fan-Out Progress — live status of concurrent research waves (Phase 3.2)
8. Learning Insights — calibration curves, pattern library, sector heatmap (Phase 3.3)
9. Agent Status — last scan timestamp, scheduler status

Design references:
- PRD §5.3: Dashboard panels
- technical_spec.md §10.5: Dashboard module spec
"""

import json
import sys
import time
from datetime import datetime, timezone

# Lazy import of streamlit to allow import of data.py without streamlit installed
try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Handle both direct execution (streamlit run) and package import
try:
    from .data import (
        get_dashboard_overview,
        get_latest_snapshot,
        get_snapshot_history,
        get_trade_history,
        get_trade_summary,
        get_signal_history,
        get_signal_summary,
        get_learning_entries,
        get_balance_history,
        get_kite_status,
        get_agent_status,
        get_stock_pitches,
        get_latest_pitches_by_symbol,
        get_fanout_progress,
        get_fanout_summary,
        format_inr,
        format_pct,
    )
    from ..memory.performance import (
        get_all_performance_metrics,
        get_cumulative_pnl,
        get_win_rate,
        get_profit_factor,
        get_symbol_performance,
        get_drawdown_stats,
        get_monthly_summary,
    )
except ImportError:
    # Fallback: running as a standalone script (streamlit run)
    # Add project root to path so src.autfin imports work
    import os as _os
    _project_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    
    from src.autfin.dashboard.data import (
        get_dashboard_overview,
        get_latest_snapshot,
        get_snapshot_history,
        get_trade_history,
        get_trade_summary,
        get_signal_history,
        get_signal_summary,
        get_learning_entries,
        get_balance_history,
        get_kite_status,
        get_agent_status,
        get_stock_pitches,
        get_latest_pitches_by_symbol,
        get_fanout_progress,
        get_fanout_summary,
        format_inr,
        format_pct,
    )
    from src.autfin.memory.performance import (
        get_all_performance_metrics,
        get_cumulative_pnl,
        get_win_rate,
        get_profit_factor,
        get_symbol_performance,
        get_drawdown_stats,
        get_monthly_summary,
    )


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

def configure_page():
    """Set Streamlit page configuration."""
    st.set_page_config(
        page_title="AutFin Dashboard",
        page_icon="chart_with_upwards_trend",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("AutFin — Portfolio Dashboard")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    """Render sidebar with navigation and status. Returns (page, auto_refresh, refresh_interval)."""
    with st.sidebar:
        st.header("AutFin")
        st.caption("AI-Powered Portfolio Management")

        st.divider()

        # Kite connection status
        kite = get_kite_status()
        if kite["connected"]:
            st.success("Kite SDK: Connected")
        else:
            st.warning("Kite SDK: Disconnected")
            if kite.get("error"):
                st.caption(kite["error"])

        st.divider()

        # Quick stats from latest snapshot
        snapshot = get_latest_snapshot()
        if snapshot:
            st.metric(
                "Available Cash",
                format_inr(snapshot.get("available_cash", 0)),
            )
            st.metric(
                "Total P&L",
                format_inr(snapshot.get("total_pnl", 0)),
                delta=format_pct(snapshot.get("total_pnl_pct", 0)),
            )

        st.divider()

        # Navigation
        page = st.radio(
            "Navigate",
            [
                "Overview", "Trades", "Signals", "Performance",
                "Stock Pitches", "Fan-Out Progress",
                "Learning Insights", "Paper Trading", "System",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.divider()

        # Auto-refresh controls
        auto_refresh = st.checkbox("Auto-refresh", value=False)
        refresh_interval = st.select_slider(
            "Refresh interval (seconds)",
            options=[10, 30, 60, 120, 300],
            value=60,
            disabled=not auto_refresh,
        )

        if st.button("Refresh Now"):
            st.rerun()

        st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%H:%M UTC')}")

    return page, auto_refresh, refresh_interval


# ---------------------------------------------------------------------------
# Panel: Portfolio Overview
# ---------------------------------------------------------------------------

def render_overview():
    """Render portfolio overview panel with KPI cards."""
    st.header("Portfolio Overview")

    overview = get_dashboard_overview()
    snapshot = overview["snapshot"]
    trade_sum = overview["trade_summary"]
    signal_sum = overview["signal_summary"]
    perf = overview["performance"]

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        cash = snapshot.get("available_cash", 0) if snapshot else 0
        st.metric("Cash", format_inr(cash))

    with col2:
        holdings = snapshot.get("total_holdings_value", 0) if snapshot else 0
        st.metric("Holdings", format_inr(holdings))

    with col3:
        pnl = snapshot.get("total_pnl", 0) if snapshot else 0
        pnl_pct = snapshot.get("total_pnl_pct", 0) if snapshot else 0
        st.metric("Total P&L", format_inr(pnl), delta=format_pct(pnl_pct))

    with col4:
        win_rate = perf.get("win_rate", {}).get("win_rate", 0)
        st.metric("Win Rate", f"{win_rate * 100:.1f}%")

    with col5:
        pf = perf.get("profit_factor", {}).get("profit_factor", 0)
        st.metric("Profit Factor", f"{pf:.2f}" if pf != float("inf") else "∞")

    st.divider()

    # Trade & Signal summary row
    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.metric("Total Trades", trade_sum.get("total_trades", 0))

    with col_b:
        st.metric("Symbols Traded", trade_sum.get("symbols_traded", 0))

    with col_c:
        st.metric("Signals (30d)", signal_sum.get("total_signals", 0))

    with col_d:
        st.metric("Pending Signals", signal_sum.get("pending_count", 0))

    # Holdings table
    if snapshot and snapshot.get("holdings_json"):
        st.subheader("Current Holdings")
        try:
            holdings = json.loads(snapshot["holdings_json"])
            if isinstance(holdings, list) and holdings:
                import pandas as pd
                df = pd.DataFrame(holdings)
                st.dataframe(df, use_container_width=True)
            elif isinstance(holdings, dict) and holdings:
                import pandas as pd
                df = pd.DataFrame([holdings])
                st.dataframe(df, use_container_width=True)
        except (json.JSONDecodeError, TypeError):
            st.info("Holdings data not available")


# ---------------------------------------------------------------------------
# Panel: Equity Curve
# ---------------------------------------------------------------------------

def render_equity_curve():
    """Render equity curve chart."""
    st.subheader("Equity Curve")

    equity = get_cumulative_pnl()
    if not equity:
        st.info("No portfolio snapshots yet. Equity curve will appear after the first scan.")
        return

    import pandas as pd

    df = pd.DataFrame(equity)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    st.line_chart(df[["total_pnl"]], use_container_width=True)

    # Cash over time
    if "available_cash" in df.columns:
        st.line_chart(df[["available_cash"]], use_container_width=True)


# ---------------------------------------------------------------------------
# Panel: Trade History
# ---------------------------------------------------------------------------

def render_trades():
    """Render trade history panel."""
    st.header("Trade History")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        symbol_filter = st.text_input("Filter by Symbol", "")
    with col2:
        days = st.selectbox("Period", [7, 30, 90, 180, 365], index=2)

    trades = get_trade_history(
        symbol=symbol_filter if symbol_filter else None,
        days=days,
        limit=200,
    )

    if not trades:
        st.info("No trades found for the selected period.")
        return

    import pandas as pd

    df = pd.DataFrame(trades)

    # Summary metrics
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Trades", len(df))
    with col_b:
        buys = len(df[df["action"] == "BUY"]) if "action" in df.columns else 0
        st.metric("Buys", buys)
    with col_c:
        sells = len(df[df["action"] == "SELL"]) if "action" in df.columns else 0
        st.metric("Sells", sells)

    # Trade table
    display_cols = [c for c in ["timestamp", "symbol", "action", "quantity", "price", "total_value", "order_status"] if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True)


# ---------------------------------------------------------------------------
# Panel: Signals
# ---------------------------------------------------------------------------

def render_signals():
    """Render signal log panel."""
    st.header("Signal Log")

    col1, col2 = st.columns(2)
    with col1:
        symbol_filter = st.text_input("Filter by Symbol", "", key="sig_filter")
    with col2:
        days = st.selectbox("Period", [7, 30, 90], index=1, key="sig_period")

    signals = get_signal_history(
        symbol=symbol_filter if symbol_filter else None,
        days=days,
        limit=200,
    )

    if not signals:
        st.info("No signals found for the selected period.")
        return

    import pandas as pd

    df = pd.DataFrame(signals)

    # Summary
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        buys = len(df[df["signal_type"] == "BUY"]) if "signal_type" in df.columns else 0
        st.metric("Buy Signals", buys)
    with col_b:
        sells = len(df[df["signal_type"] == "SELL"]) if "signal_type" in df.columns else 0
        st.metric("Sell Signals", sells)
    with col_c:
        acted = len(df[df["was_acted_on"] == True]) if "was_acted_on" in df.columns else 0
        st.metric("Acted On", acted)

    # Signal table
    display_cols = [c for c in ["timestamp", "symbol", "signal_type", "strength", "fundamentals_score", "technical_score", "sentiment_score", "was_acted_on"] if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True)


# ---------------------------------------------------------------------------
# Panel: Performance Metrics
# ---------------------------------------------------------------------------

def render_performance():
    """Render performance metrics panel."""
    st.header("Performance Metrics")

    # Period selector
    days = st.selectbox("Period", [30, 90, 180, 365, None], index=1, format_func=lambda x: f"{x} days" if x else "All Time", key="perf_period")

    # Overview metrics
    overview = get_dashboard_overview()
    drawdown = overview["drawdown"]
    perf = get_all_performance_metrics(days=days)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        wr = perf.get("win_rate", {}).get("win_rate", 0)
        st.metric("Win Rate", f"{wr * 100:.1f}%")
    with col2:
        pf = perf.get("profit_factor", {}).get("profit_factor", 0)
        pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
        st.metric("Profit Factor", pf_str)
    with col3:
        st.metric("Max Drawdown", f"{drawdown.get('max_drawdown_pct', 0) * 100:.1f}%")
    with col4:
        total_pnl = perf.get("win_rate", {}).get("total_pnl", 0)
        st.metric("Total P&L", format_inr(total_pnl))

    st.divider()

    # Per-symbol breakdown
    symbols = perf.get("symbol_performance", [])
    if symbols:
        st.subheader("Per-Symbol Performance")
        import pandas as pd
        df = pd.DataFrame(symbols)
        display_cols = [c for c in ["symbol", "total_trades", "wins", "losses", "win_rate", "total_pnl", "avg_pnl"] if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True)
    else:
        st.info("No per-symbol performance data yet.")

    # Equity curve
    render_equity_curve()


# ---------------------------------------------------------------------------
# Panel: Learning Journal
# ---------------------------------------------------------------------------

def render_learnings():
    """Render learning journal panel (legacy; superseded by render_learning_insights)."""
    st.header("Learning Journal")

    col1, col2 = st.columns(2)
    with col1:
        symbol_filter = st.text_input("Filter by Symbol", "", key="learn_filter")
    with col2:
        days = st.selectbox("Period", [30, 90, 180, 365], index=1, key="learn_period")

    entries = get_learning_entries(
        symbol=symbol_filter if symbol_filter else None,
        days=days,
        limit=100,
    )

    if not entries:
        st.info("No learning entries found.")
        return

    for entry in entries:
        with st.expander(f"{entry.get('symbol', '?')} — {entry.get('created_at', entry.get('trade_date', '?'))} (P&L: {entry.get('pnl_pct', 0) or 0:.1%})"):
            st.write(f"**Outcome:** {entry.get('outcome', entry.get('was_correct', '?'))}")
            if entry.get("lesson"):
                st.write(f"**Lesson:** {entry['lesson']}")


# ---------------------------------------------------------------------------
# Panel: Stock Pitches (Phase 3.1) — 6-element buy-side pitches
# ---------------------------------------------------------------------------

def render_stock_pitches():
    """Render the 6-element stock pitch dashboard panel (Phase 3.1)."""
    st.header("Stock Pitches")
    st.caption("6-element buy-side pitches generated by `build_stock_pitch` (Phase 3.1).")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol_filter = st.text_input("Filter by Symbol", "", key="pitch_filter")
    with col2:
        context_filter = st.selectbox(
            "Context",
            ["All", "morning_routine", "price_alert", "quarterly_review", "manual"],
            index=0,
            key="pitch_context",
        )
    with col3:
        days = st.selectbox(
            "Period (days)",
            [7, 30, 90, 180, 365],
            index=2,
            key="pitch_days",
        )

    pitches = get_stock_pitches(
        symbol=symbol_filter or None,
        context=None if context_filter == "All" else context_filter,
        days=days,
        limit=50,
    )

    if not pitches:
        st.info("No stock pitches generated yet. The supervisor calls `build_stock_pitch` per candidate.")
        return

    summary = get_latest_pitches_by_symbol(
        symbol=symbol_filter or None,
        limit=50,
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Total Pitches", len(pitches))
    with col_b:
        st.metric("Unique Symbols", len(summary))

    st.divider()
    st.subheader("Pitch Library")
    for p in pitches:
        title = f"{p.get('symbol', '?')} — {p.get('context', 'manual')} ({p.get('created_at', '?')[:19]})"
        with st.expander(title):
            # 6 elements
            st.markdown("**1. Investment Thesis**")
            st.write(p.get("investment_thesis") or "(missing)")

            st.markdown("**2. Catalysts**")
            catalysts = p.get("catalysts") or []
            if isinstance(catalysts, list) and catalysts:
                for c in catalysts:
                    st.write(f"- {c}")
            else:
                st.write("_(none)_")

            st.markdown("**3. Risks**")
            risks = p.get("risks") or []
            if isinstance(risks, list) and risks:
                for r in risks:
                    st.write(f"- {r}")
            else:
                st.write("_(none)_")

            st.markdown("**4. Valuation Summary**")
            st.write(p.get("valuation_summary") or "(missing)")

            st.markdown("**5. Competitive Moat**")
            st.write(p.get("competitive_moat") or "(missing)")

            # 6. Price target
            st.markdown("**6. Price Target**")
            pt_base = p.get("price_target_base")
            pt_up = p.get("price_target_upside")
            pt_down = p.get("price_target_downside")
            pt_rationale = p.get("price_target_rationale")
            if any(x is not None for x in (pt_base, pt_up, pt_down)):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Base", f"₹{pt_base:,.2f}" if pt_base else "—")
                with c2:
                    st.metric("Upside", f"₹{pt_up:,.2f}" if pt_up else "—")
                with c3:
                    st.metric("Downside", f"₹{pt_down:,.2f}" if pt_down else "—")
                if pt_rationale:
                    st.write(pt_rationale)
            else:
                st.write("_(no quantified target)_")


# ---------------------------------------------------------------------------
# Panel: Fan-Out Progress (Phase 3.2) — live concurrent-research status
# ---------------------------------------------------------------------------

def render_fanout_progress():
    """Render the fan-out progress dashboard (Phase 3.2)."""
    st.header("Fan-Out Progress")
    st.caption("Live status of concurrent research waves written by `research_batch` (Phase 3.2).")

    col1, col2 = st.columns(2)
    with col1:
        hours = st.selectbox(
            "Window (hours)",
            [1, 6, 12, 24, 72, 168],
            index=3,
            key="fanout_hours",
        )
    with col2:
        session_filter = st.text_input(
            "Session ID (optional)",
            "",
            key="fanout_session",
        )

    summary = get_fanout_summary(hours=hours)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Total", summary["total"])
    with c2:
        st.metric("Running", summary["running"])
    with c3:
        st.metric("Completed", summary["completed"])
    with c4:
        st.metric("Failed", summary["failed"])
    with c5:
        st.metric("Completion %", f"{summary['completion_pct']}%")

    if summary["pending"]:
        st.metric("Pending", summary["pending"])

    st.divider()

    rows = get_fanout_progress(
        session_id=session_filter or None,
        hours=hours,
        limit=200,
    )
    if not rows:
        st.info("No fan-out rows in this window. The supervisor calls `research_batch` per wave.")
        return

    import pandas as pd

    df = pd.DataFrame([
        {
            "Symbol": r["symbol"],
            "Context": r["context"],
            "Session": r["session_id"][:12] + "...",
            "Status": r["status"],
            "Started": (r["started_at"] or "")[:19],
            "Finished": (r["finished_at"] or "")[:19],
            "Error": (r["error"] or "")[:60],
        }
        for r in rows
    ])
    st.dataframe(df, use_container_width=True)

    # Per-status counts bar chart
    st.divider()
    st.subheader("Status Distribution")
    sdf = pd.DataFrame([
        {"Status": k, "Count": v}
        for k, v in (("running", summary["running"]),
                      ("completed", summary["completed"]),
                      ("failed", summary["failed"]),
                      ("pending", summary["pending"])) if v
    ])
    if not sdf.empty:
        st.bar_chart(sdf.set_index("Status"))


# ---------------------------------------------------------------------------
# Panel: Learning Insights (Phase 3.3) — calibration, patterns, sector heat
# ---------------------------------------------------------------------------

def render_learning_insights():
    """Render the Learning Insights dashboard (Phase 3.3).

    Three blocks:
      A. Calibration curves — actual vs expected win-rate per conviction tier
      B. Pattern library — recurring patterns from losing trades
      C. Sector performance heatmap — win-rate per sector
    """
    st.header("Learning Insights")
    st.caption("Calibration curves, pattern library, and sector heatmap (Phase 3.3).")

    col1, col2 = st.columns(2)
    with col1:
        days = st.selectbox(
            "Period (days)",
            [30, 90, 180, 365, 0],
            index=2,
            format_func=lambda x: "All Time" if x == 0 else f"{x} days",
            key="insight_days",
        )
    with col2:
        min_samples = st.slider("Min samples per slice", 1, 50, 5, key="insight_min")

    st.divider()

    # ---- A. Calibration curves ----
    st.subheader("A. Conviction-Tier Calibration")
    try:
        from src.autfin.learning.retriever import _compute_calibration
        from src.autfin.memory.database import get_connection
        db = get_connection()
        cal = _compute_calibration(db)
    except Exception as e:
        st.error(f"Calibration fetch failed: {e}")
        cal = {}

    if not cal:
        st.info("No calibration data yet — at least 1 completed trade is needed per tier.")
    else:
        import pandas as pd

        rows = []
        for tier, payload in cal.items():
            rows.append({
                "Tier": tier,
                "Sample Size": payload.get("sample_size", 0),
                "Expected Win Rate": payload.get("expected_win_rate", 0),
                "Actual Win Rate": payload.get("actual_win_rate", 0),
                "Delta": payload.get("delta", 0),
                "Status": payload.get("status", "unknown"),
            })
        df = pd.DataFrame(rows).sort_values("Expected Win Rate")
        if not df.empty:
            # Side-by-side: numeric table + bar of actual vs expected
            st.dataframe(df, use_container_width=True)
            chart_df = df.set_index("Tier")[["Expected Win Rate", "Actual Win Rate"]]
            st.bar_chart(chart_df)

    st.divider()

    # ---- B. Pattern library ----
    st.subheader("B. Pattern Library (loss concentration)")
    try:
        import asyncio
        from src.autfin.learning.calibrator import detect_patterns
        patterns = asyncio.run(detect_patterns(min_samples=min_samples))
    except Exception as e:
        patterns = []
        st.warning(f"Pattern detection failed: {e}")

    if not patterns:
        st.info("No patterns above the minimum sample threshold yet.")
    else:
        import pandas as pd
        df = pd.DataFrame(patterns)
        st.dataframe(df, use_container_width=True)

    st.divider()

    # ---- C. Sector performance heatmap ----
    st.subheader("C. Sector Performance")
    try:
        from src.autfin.learning.calibrator import detect_patterns
        import asyncio

        # Pattern detection returns sector_performance entries too
        secs = asyncio.run(detect_patterns(min_samples=1))
        sector_rows = [p for p in secs if p.get("type") == "sector_performance"]
    except Exception:
        sector_rows = []

    if not sector_rows:
        st.info("No sector data in learning_journal yet.")
    else:
        import pandas as pd
        sdf = pd.DataFrame(sector_rows).sort_values("win_rate", ascending=False)
        st.dataframe(sdf, use_container_width=True)
        if not sdf.empty:
            st.bar_chart(sdf.set_index("sector")[["win_rate"]])


# ---------------------------------------------------------------------------
# Panel: System Status
# ---------------------------------------------------------------------------

def render_system():
    """Render system status panel — scheduler, agent caches, DB stats, observability."""
    st.header("System Status")

    status = get_agent_status()

    # --- Scheduler jobs ---
    st.subheader("Scheduler Jobs")
    jobs = status.get("scheduler_jobs", [])
    if jobs:
        import pandas as pd
        df = pd.DataFrame(jobs)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Scheduler not running in this process (started via `autfin start` daemon).")

    # --- Agent cache sizes ---
    st.subheader("Agent Cache Entries")
    cache = status.get("cache_info", {})
    if cache:
        cols = st.columns(len(cache))
        for i, (agent, size) in enumerate(cache.items()):
            with cols[i]:
                st.metric(agent.replace("_", " ").title(), size)
    else:
        st.info("No agent caches populated yet.")

    # --- DB table row counts ---
    st.subheader("Database Table Row Counts")
    db_stats = status.get("db_stats", {})
    if db_stats:
        import pandas as pd
        df = pd.DataFrame([
            {"Table": t, "Rows": c} for t, c in db_stats.items()
        ])
        st.dataframe(df, use_container_width=True)

    # --- Observability ---
    st.divider()
    st.subheader("Observability")

    try:
        from src.autfin.observability.metrics import get_metrics_snapshot
        obs = get_metrics_snapshot()

        col1, col2, col3, col4 = st.columns(4)
        traces = obs.get("traces", {})
        graphs = obs.get("graphs", {})
        llm = obs.get("llm", {})

        with col1:
            st.metric("Graph Runs", graphs.get("runs", 0))
        with col2:
            st.metric("Trace Error Rate", f"{traces.get('error_rate', 0) * 100:.1f}%")
        with col3:
            st.metric("Graph Error Rate", f"{graphs.get('error_rate', 0) * 100:.1f}%")
        with col4:
            st.metric("LLM Tokens", f"{llm.get('total_tokens', 0):,}")

        st.caption(f"LLM Cost: ${llm.get('total_cost', 0):.6f} | Updated: {obs.get('last_updated', 'N/A')}")

        # Agent performance
        agent_metrics = obs.get("agent_metrics", {})
        if agent_metrics:
            st.write("**Agent Performance**")
            import pandas as pd
            am_df = pd.DataFrame([
                {
                    "Agent": name,
                    "Calls": m["calls"],
                    "Errors": m["errors"],
                    "Avg Latency (ms)": m["avg_duration_ms"],
                    "Success Rate": f"{m['success_rate'] * 100:.1f}%",
                }
                for name, m in agent_metrics.items()
            ])
            st.dataframe(am_df, use_container_width=True)
    except Exception:
        st.info("Observability not available (Langfuse not configured).")

    # --- RORO ---
    st.divider()
    st.subheader("RORO — Return on Research Output")
    try:
        from src.autfin.observability.roro import get_research_efficiency
        roro = get_research_efficiency()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Decisions Recorded", roro["decisions_recorded"])
            st.metric("RORO Score", f"₹{roro['roro_score']:.2f}")
        with col2:
            st.write(f"**Interpretation:** {roro['roro_interpretation']}")
            by_depth = roro.get("by_depth", {})
            if by_depth:
                import pandas as pd
                dd = pd.DataFrame([
                    {
                        "Depth": d,
                        "Trades": v["count"],
                        "Total P&L": format_inr(v["total_pnl"]),
                        "Avg P&L/Trade": format_inr(v["avg_pnl_per_trade"]),
                    }
                    for d, v in by_depth.items()
                ])
                st.dataframe(dd, use_container_width=True)
    except Exception:
        st.info("RORO metrics not available.")

    # --- A/B Thesis Check ---
    st.divider()
    st.subheader("A/B Thesis Check — LLM vs Rule-based")
    try:
        from src.autfin.observability.thesis_ab import compute_ab_stats
        ab = compute_ab_stats()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Comparisons", ab["total_comparisons"])
            st.metric("Agreement Rate", f"{ab['agreement_rate'] * 100:.1f}%")
        with col2:
            llm_acc = ab.get("llm_accuracy")
            llm_str = f"{llm_acc * 100:.1f}%" if llm_acc is not None else "N/A"
            st.metric("LLM Accuracy", llm_str, delta=f"({ab.get('llm_total', 0)} checks)")
        with col3:
            st.metric("Rule Accuracy", f"{ab['rule_accuracy'] * 100:.1f}%")
            st.metric("Better", ab["llm_better"])
    except Exception:
        st.info("A/B thesis check data not available.")


# ---------------------------------------------------------------------------
# Panel: Paper Trading
# ---------------------------------------------------------------------------

def render_paper_trading():
    """Render paper trading simulation dashboard."""
    st.header("Paper Trading")

    try:
        from src.autfin.paper_trading.engine import get_paper_engine
        paper = get_paper_engine()
        paper_status = paper.get_status()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Available Cash", format_inr(paper_status["available_cash"]))
        with col2:
            st.metric("Market Value", format_inr(paper_status["market_value"]))
        with col3:
            realized = paper_status["realized_pnl"]
            unrealized = paper_status["unrealized_pnl"]
            st.metric("Realized P&L", format_inr(realized))
        with col4:
            st.metric("Total Return",
                      f"{paper_status['total_return_pct']}%",
                      delta=f"Unrealized: {format_inr(unrealized)}")

        st.divider()
        st.subheader(f"Positions ({paper_status['open_positions']})")

        positions = paper_status.get("positions", {})
        if positions:
            import pandas as pd
            pos_df = pd.DataFrame([
                {
                    "Symbol": p["symbol"],
                    "Qty": p["quantity"],
                    "Avg Price": format_inr(p["avg_price"]),
                    "LTP": format_inr(p["last_price"]),
                    "Market Value": format_inr(p["market_value"]),
                    "Unrealized P&L": format_inr(p["unrealized_pnl"]),
                    "P&L %": f"{p['unrealized_pnl_pct']:.2f}%",
                }
                for p in positions.values()
            ])
            st.dataframe(pos_df, use_container_width=True)
        else:
            st.info("No open positions.")

        st.divider()
        st.subheader("Recent Trades")
        trades = paper.trade_history[-20:] if paper.trade_history else []
        if trades:
            import pandas as pd
            t_df = pd.DataFrame([
                {
                    "Symbol": t["symbol"],
                    "Action": t["action"],
                    "Qty": t["quantity"],
                    "Price": format_inr(t["price"]),
                    "Value": format_inr(t["value"]),
                    "P&L": format_inr(t["pnl"]),
                    "Time": t.get("filled_at", "")[:19],
                }
                for t in reversed(trades)
            ])
            st.dataframe(t_df, use_container_width=True)

            # Simple equity curve
            cum_pnl = 0
            equity = [paper_status["initial_cash"]]
            for t in trades:
                cum_pnl += t.get("pnl", 0)
                equity.append(paper_status["initial_cash"] + cum_pnl)
            if len(equity) > 1:
                st.subheader("Equity Curve")
                import pandas as pd
                eq_df = pd.DataFrame({
                    "Trade": list(range(len(equity))),
                    "Equity": equity,
                })
                st.line_chart(eq_df, x="Trade", y="Equity")
        else:
            st.info("No trades yet. Place orders to see results.")

        if st.button("Reset Paper Trading"):
            from src.autfin.paper_trading.engine import reset_paper_engine
            reset_paper_engine()
            st.rerun()

    except Exception as e:
        st.error(f"Paper trading engine unavailable: {e}")

def main():
    """Run the Streamlit dashboard."""
    if not STREAMLIT_AVAILABLE:
        print("Error: streamlit is required for the dashboard.")
        print("Install with: pip install 'autfin[dashboard]'")
        sys.exit(1)

    configure_page()
    page, auto_refresh, refresh_interval = render_sidebar()

    if page == "Overview":
        render_overview()
        render_equity_curve()
    elif page == "Trades":
        render_trades()
    elif page == "Signals":
        render_signals()
    elif page == "Performance":
        render_performance()
    elif page == "Stock Pitches":
        render_stock_pitches()
    elif page == "Fan-Out Progress":
        render_fanout_progress()
    elif page == "Learning Insights":
        render_learning_insights()
    elif page == "Learnings":
        render_learnings()
    elif page == "Paper Trading":
        render_paper_trading()
    elif page == "System":
        render_system()

    # Auto-refresh: sleep and rerun
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
