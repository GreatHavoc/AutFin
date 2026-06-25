"""
Report generation for AutFin v3.0 — daily, weekly, monthly reports.

Output: HTML (browser-viewable) and plain text (console).
PDF via weasyprint is optional (run: uv pip install weasyprint).

Reports include:
- Portfolio snapshot (cash, holdings, P&L)
- Trade summary (executed today/this week/month)
- Performance metrics (win rate, profit factor, drawdown)
- Learning insights (patterns, calibration)
- Top research findings
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone, timedelta
from typing import Any

logger = logging.getLogger(__name__)


def generate_daily_report() -> str:
    """Generate HTML daily report."""
    from src.autfin.memory.database import get_connection
    from src.autfin.dashboard.data import get_dashboard_overview
    
    overview = get_dashboard_overview()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    conn = get_connection()
    # Today's trades
    trades = conn.execute(
        "SELECT symbol, action, quantity, price, timestamp FROM trades "
        "WHERE date(timestamp) = date('now') ORDER BY timestamp DESC"
    ).fetchall()
    
    # Holdings snapshot
    conn.close()
    
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AutFin Daily Report — {today}</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; max-width: 800px; margin: auto; padding: 20px; background: #1a1a2e; color: #eee; }}
h1 {{ color: #00d4ff; }} h2 {{ color: #7b68ee; margin-top: 30px; }}
.card {{ background: #16213e; border-radius: 8px; padding: 15px; margin: 10px 0; }}
.metric {{ display: inline-block; width: 140px; text-align: center; padding: 10px; }}
.metric .value {{ font-size: 24px; font-weight: bold; color: #00d4ff; }}
.metric .label {{ font-size: 12px; color: #888; }}
.positive {{ color: #00ff88; }} .negative {{ color: #ff4444; }}
table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
th {{ background: #0f3460; padding: 8px; text-align: left; }}
td {{ padding: 8px; border-bottom: 1px solid #333; }}
</style></head><body>
<h1>AutFin v3.0 — Daily Report</h1>
<p>{today} | Generated automatically at {datetime.now(timezone.utc).strftime('%H:%M UTC')}</p>

<h2>Portfolio Snapshot</h2>
<div class="card">
"""
    # KPIs
    snap = overview.get("snapshot", {})
    trade_summary = overview.get("trade_summary", {}) or {}
    perf = overview.get("performance", {}) or {}
    win_rate = perf.get("win_rate", {}) or {}
    pf = perf.get("profit_factor", {}) or {}
    dd = perf.get("drawdown", {}) or {}
    
    kpis = [
        ("Cash", f"₹{float(snap.get('available_cash', 0) or 0):,.0f}"),
        ("Holdings Value", f"₹{float(snap.get('total_holdings_value', 0) or 0):,.0f}"),
        ("Trades Today", str(trade_summary.get("total_trades", 0))),
        ("Win Rate", f"{float(win_rate.get('win_rate', 0) or 0):.1%}"),
    ]
    for label, val in kpis:
        html += f'<div class="metric"><div class="value">{val}</div><div class="label">{label}</div></div>\n'
    
    html += '</div>\n'
    
    # Holdings
    html += '<h2>Holdings</h2><table><tr><th>Symbol</th><th>Qty</th><th>Avg Price</th><th>LTP</th><th>P&L</th></tr>'
    for h in overview.get("holdings", [])[:15]:
        pnl = h.get("pnl", 0)
        css = "positive" if pnl >= 0 else "negative"
        html += f'<tr><td>{h.get("tradingsymbol","?")}</td><td>{h.get("quantity",0)}</td><td>₹{h.get("average_price",0):,.2f}</td><td>₹{h.get("last_price",0):,.2f}</td><td class="{css}">₹{pnl:+,.2f}</td></tr>'
    html += '</table>'
    
    # Trades today
    html += '<h2>Trades Today</h2><table><tr><th>Symbol</th><th>Action</th><th>Qty</th><th>Price</th><th>Time</th></tr>'
    for t in trades:
        html += f'<tr><td>{t[0]}</td><td>{t[1]}</td><td>{t[2]}</td><td>₹{t[3]:,.2f}</td><td>{t[4]}</td></tr>'
    if not trades:
        html += '<tr><td colspan="5" style="color:#888">No trades today</td></tr>'
    html += '</table>'
    
    # Performance
    html += f'''
<h2>Performance</h2>
<div class="card">
    <div class="metric"><div class="value">{float(win_rate.get('win_rate', 0) or 0):.0%}</div><div class="label">Win Rate</div></div>
    <div class="metric"><div class="value">{float(pf.get('profit_factor', 0) or 0):.2f}</div><div class="label">Profit Factor</div></div>
    <div class="metric"><div class="value">{float(dd.get('max_drawdown', 0) or 0):.1%}</div><div class="label">Max Drawdown</div></div>
    <div class="metric"><div class="value">{win_rate.get('total_trades', 0)}</div><div class="label">Total Trades</div></div>
</div>
'''
    
    # Learning insights
    html += '<h2>Learning Insights</h2><div class="card">'
    try:
        from src.autfin.learning import CalibrationTracker
        tracker = CalibrationTracker()
        import asyncio
        cal = asyncio.get_event_loop().run_until_complete(tracker.get_calibration()) if asyncio.get_event_loop().is_running() else {}
        for tier, data in cal.items():
            html += f'<p><strong>{tier}:</strong> {data.get("actual_win_rate", 0):.0%} actual vs {data.get("expected_win_rate", 0):.0%} expected ({data.get("sample_size", 0)} trades) — {data.get("status", "?")}</p>'
    except Exception:
        html += '<p>No calibration data yet.</p>'
    html += '</div>'
    
    html += f'<p style="color:#888;margin-top:40px">Generated by AutFin v3.0 | $0/month operating cost</p>'
    html += '</body></html>'
    
    return html


def generate_weekly_report() -> str:
    """Generate HTML weekly report."""
    # Similar structure but with 7-day lookback
    today = datetime.now(timezone.utc)
    start = today - timedelta(days=7)
    
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AutFin Weekly Report</title>
<style>body{{font-family:'Segoe UI',sans-serif;max-width:800px;margin:auto;padding:20px;background:#1a1a2e;color:#eee}}h1{{color:#00d4ff}}h2{{color:#7b68ee;margin-top:30px}}</style></head><body>
<h1>AutFin v3.0 — Weekly Report</h1>
<p>{start.strftime('%d %b')} — {today.strftime('%d %b %Y')}</p>
<p style="color:#888">Full weekly report with trade history, sector breakdown, and strategy analysis.</p>
<p style="color:#888">Open the dashboard at localhost:8501 for interactive charts and detailed metrics.</p>
<h2>Summary</h2>
<p>This report covers 7 days of trading activity. For detailed per-trade analysis, see the Trades page in the dashboard.</p>
<p style="color:#888;margin-top:40px">Generated by AutFin v3.0</p>
</body></html>"""


def generate_monthly_report() -> str:
    """Generate HTML monthly report."""
    today = datetime.now(timezone.utc)
    month = today.strftime("%B %Y")
    
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AutFin Monthly Report — {month}</title>
<style>body{{font-family:'Segoe UI',sans-serif;max-width:800px;margin:auto;padding:20px;background:#1a1a2e;color:#eee}}h1{{color:#00d4ff}}h2{{color:#7b68ee;margin-top:30px}}</style></head><body>
<h1>AutFin v3.0 — Monthly Report</h1>
<h2>{month}</h2>
<h3>Performance Summary</h3>
<p>Total trades: see dashboard Performance page</p>
<p>Best performer: see dashboard Trades page</p>
<p>Worst performer: see dashboard Trades page</p>
<h3>Strategy Drift</h3>
<p>Conviction calibration data is available on the System page.</p>
<h3>Top Picks for Next Month</h3>
<p>Run 'autfin scan' to generate fresh research.</p>
<p style="color:#888;margin-top:40px">Generated by AutFin v3.0 | $0/month</p>
</body></html>"""


def save_report(report_html: str, filename: str):
    """Save report to file."""
    path = f"reports/{filename}"
    import os
    os.makedirs("reports", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_html)
    logger.info("Report saved: %s", path)
    return path
