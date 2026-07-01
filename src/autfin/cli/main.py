"""
AutFin CLI — Command Line Interface

Commands:
  autfin status                          Show account status
  autfin analyze <SYMBOL>                Full research report for a stock
  autfin scan                            Run daily morning routine manually
  autfin start                           Start as daemon with scheduler
  autfin history --symbol <S> --days N   View trade history
  autfin config set <KEY> <VALUE>        Modify configuration
  autfin config get <KEY>                Show configuration value
  autfin backtest --strategy S --period P --symbol SYM  Backtest a strategy
"""
from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..config.settings import AutFinSettings

logger = logging.getLogger(__name__)

console = Console()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_settings() -> AutFinSettings:
    """Load settings (reads .env)."""
    return AutFinSettings()


def _get_db_path() -> Path:
    """Path to the SQLite database."""
    return Path(__file__).parent.parent.parent.parent / "data" / "autfin.db"


def _get_db_connection() -> sqlite3.Connection:
    """Get a read-only SQLite connection."""
    db_path = _get_db_path()
    if not db_path.exists():
        console.print("[yellow]No database found. Run 'autfin scan' first.[/yellow]")
        return None
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _format_inr(amount: float) -> str:
    """Format amount in Indian Rupee style (lakhs/crores)."""
    sign = "-" if amount < 0 else ""
    amount = abs(amount)

    if amount >= 1_00_00_000:
        # Crores
        formatted = f"{amount / 1_00_00_000:,.2f} Cr"
    elif amount >= 1_00_000:
        # Lakhs — Indian comma style: first 3 digits from right, then groups of 2
        s = f"{int(amount):,}"  # Western: 150,000
        parts = s.split(",")
        if len(parts) >= 2:
            # Re-format: keep last group as-is, preceding groups as-is
            # 1,50,000 format: last 3 digits, then groups of 2
            integer_part = str(int(amount))
            last3 = integer_part[-3:]
            rest = integer_part[:-3]
            groups = []
            while rest:
                groups.append(rest[-2:])
                rest = rest[:-2]
            groups.reverse()
            formatted = ",".join(groups + [last3])
        else:
            formatted = s
        return f"{sign}₹{formatted}"
    else:
        formatted = f"{amount:,.2f}"
    return f"{sign}₹{formatted}"


# ---------------------------------------------------------------------------
# Command: autfin status
# ---------------------------------------------------------------------------

async def status_command():
    """Show current status: available cash, positions, P&L (FR-130)."""
    settings = _get_settings()

    console.print(Panel("[bold]AutFin Status[/bold]", border_style="green"))

    # Try to fetch real data from Kite SDK
    available = None
    holdings_value = None
    positions_value = None
    holdings_list = []

    try:
        from ..tools.kite_tools import KiteTools
        kite = KiteTools()
        margins_resp = await kite.get_margins()
        if margins_resp.get("status") == "success":
            equity = margins_resp["data"].get("equity", {})
            available = float(equity.get("available", {}).get("cash", 0))
            positions_value = float(equity.get("used", 0))
    except Exception:
        pass

    # Try to get holdings from Kite
    try:
        from ..tools.kite_tools import KiteTools
        kite = KiteTools()
        holdings_resp = await kite.get_holdings()
        if holdings_resp.get("status") == "success":
            holdings_list = holdings_resp["data"]
            holdings_value = sum(
                float(h.get("last_price", 0)) * int(h.get("quantity", 0))
                for h in holdings_list
            )
    except Exception:
        pass

    # Build the status table
    table = Table(title="Account Summary", border_style="green")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green", justify="right")

    if available is not None:
        table.add_row("Available Cash", _format_inr(available))
    else:
        table.add_row("Available Cash", "[dim]Connect Kite to view[/dim]")

    if holdings_value is not None:
        table.add_row("Holdings Value", _format_inr(holdings_value))
    else:
        table.add_row("Holdings Value", "[dim]Connect Kite to view[/dim]")

    if positions_value is not None:
        table.add_row("Used Margin", _format_inr(positions_value))

    # Try to get P&L from latest portfolio snapshot
    conn = _get_db_connection()
    if conn:
        try:
            row = conn.execute(
                "SELECT total_pnl, total_pnl_pct FROM portfolio_snapshots "
                "ORDER BY date DESC LIMIT 1"
            ).fetchone()
            if row:
                pnl = float(row["total_pnl"])
                pnl_pct = float(row["total_pnl_pct"])
                pnl_str = f"{'+'if pnl>=0 else ''}{_format_inr(pnl)} ({pnl_pct:+.1f}%)"
                table.add_row("P&L", pnl_str)
        except Exception:
            pass
        finally:
            conn.close()

    # Execution mode
    table.add_row("Execution Mode", f"[cyan]{settings.execution_mode}[/cyan]")
    table.add_row("Deployment Pacing", f"[cyan]{settings.deployment_pacing}[/cyan]")

    console.print(table)

    # Show holdings if available
    if holdings_list:
        console.print()
        htable = Table(title="Holdings", border_style="dim")
        htable.add_column("Symbol", style="cyan")
        htable.add_column("Qty", justify="right")
        htable.add_column("Avg Price", justify="right")
        htable.add_column("LTP", justify="right")
        htable.add_column("P&L", justify="right")

        for h in holdings_list:
            symbol = h.get("tradingsymbol", h.get("symbol", "?"))
            qty = int(h.get("quantity", 0))
            avg = float(h.get("average_price", 0))
            ltp = float(h.get("last_price", 0))
            pnl = (ltp - avg) * qty
            pnl_style = "green" if pnl >= 0 else "red"
            htable.add_row(
                symbol,
                str(qty),
                _format_inr(avg),
                _format_inr(ltp),
                f"[{pnl_style}]{_format_inr(pnl)}[/{pnl_style}]",
            )
        console.print(htable)

    if available is not None:
        console.print(f"\n[green]Ready to trade![/green]")
    else:
        console.print(
            "\n[yellow]Set KITE_API_KEY and KITE_API_SECRET in .env "
            "to connect to Zerodha.[/yellow]"
        )


# ---------------------------------------------------------------------------
# Command: autfin analyze <SYMBOL>
# ---------------------------------------------------------------------------

async def analyze_command(symbol: str):
    """Full research report for a stock (FR-131)."""
    console.print(Panel(f"[bold]Analyzing {symbol}[/bold]", border_style="green"))

    settings = _get_settings()
    symbol = symbol.upper()

    # Build initial state for the graph
    from langchain_core.messages import HumanMessage
    from ..memory.store import get_store
    from ..memory.setup import create_checkpointer

    store = get_store()
    checkpointer = await create_checkpointer()

    initial_state = {
        "thread_id": f"analyze-{symbol.lower()}",
        "run_mode": "analyze",
        "available_cash": 0.0,
        "holdings": [],
        "positions": [],
        "messages": [HumanMessage(content=f"Analyze {symbol}")],
        "errors": [],
    }
    config = {"configurable": {"thread_id": f"analyze-{symbol.lower()}"}}

    console.print("[dim]Running research agents...[/dim]")

    # Compile and run a minimal analysis graph
    try:
        from ..graph.supervisor import compile_supervisor_graph  # v5.3

        graph = compile_supervisor_graph(checkpointer=checkpointer, store=store)
        final_state = {}
        async for event in graph.astream(initial_state, config, stream_mode="values"):
            final_state = event

        # Display results from final state
        _display_analysis_results(symbol, final_state)
    except Exception as exc:
        console.print(f"[red]Analysis failed: {exc}[/red]")
        logger.exception("analyze command failed")


def _display_analysis_results(symbol: str, state: dict):
    """Display analysis results from graph state."""
    # Research debates
    debates = state.get("research_debates", {})
    if debates and symbol in debates:
        debate = debates[symbol]
        if isinstance(debate, dict):
            bull_case = debate.get("bull_case", "") or str(debate.get("bull", {}).get("signals", ""))[:500]
            bear_case = debate.get("bear_case", "") or str(debate.get("bear", {}).get("signals", ""))[:500]
            synthesis = debate.get("synthesis", "") or debate.get("recommendation", "")
            if not synthesis:
                summary = debate.get("summary", {})
                synthesis = summary.get("llm_rationale", "") if isinstance(summary, dict) else ""
            console.print(Panel(
                f"[bold]Bull Case:[/bold]\n{bull_case or 'N/A'}\n\n"
                f"[bold]Bear Case:[/bold]\n{bear_case or 'N/A'}\n\n"
                f"[bold]Synthesis:[/bold]\n{synthesis or 'N/A'}",
                title=f"{symbol} — Bull/Bear Debate",
                border_style="cyan",
            ))

    # Risk assessments
    risk = state.get("risk_assessments", {})
    if risk and symbol in risk:
        assessment = risk[symbol]
        verdict = assessment.get("verdict", "UNKNOWN")
        style = "green" if verdict == "APPROVED" else "red"
        console.print(Panel(
            f"[bold]Verdict: [{style}]{verdict}[/{style}][/bold]\n"
            f"{assessment.get('rationale', '')}",
            title=f"{symbol} — Risk Assessment",
            border_style=style,
        ))

    # Trade queue
    trades = state.get("trade_queue", [])
    if trades:
        ttable = Table(title="Proposed Trades", border_style="yellow")
        ttable.add_column("Symbol", style="cyan")
        ttable.add_column("Action", style="bold")
        ttable.add_column("Qty", justify="right")
        ttable.add_column("Price", justify="right")
        ttable.add_column("Value", justify="right")

        for t in trades:
            action_style = "green" if t.get("action") == "BUY" else "red"
            ttable.add_row(
                t.get("symbol", symbol),
                f"[{action_style}]{t.get('action', '?')}[/{action_style}]",
                str(t.get("qty", 0)),
                _format_inr(t.get("price", 0)),
                _format_inr(t.get("qty", 0) * t.get("price", 0)),
            )
        console.print(ttable)

    # Messages
    messages = state.get("messages", [])
    if messages:
        console.print("\n[bold]Analysis Log:[/bold]")
        for msg in messages[-5:]:
            if hasattr(msg, "content"):
                console.print(f"  [dim]{msg.content[:200]}[/dim]")

    # Errors
    errors = state.get("errors", [])
    if errors:
        console.print("\n[bold red]Errors:[/bold red]")
        for err in errors[-3:]:
            console.print(f"  [red]{err}[/red]")


# ---------------------------------------------------------------------------
# Command: autfin scan
# ---------------------------------------------------------------------------

async def scan_command():
    """Run the daily morning routine manually (FR-132)."""
    console.print(Panel("[bold]Running Daily Morning Routine[/bold]", border_style="green"))

    settings = _get_settings()
    from ..memory.store import get_store
    from ..memory.setup import create_checkpointer

    store = get_store()
    checkpointer = await create_checkpointer()

    try:
        from ..graph.supervisor import compile_supervisor_graph  # v5.3
        from langchain_core.messages import HumanMessage

        graph = compile_supervisor_graph(checkpointer=checkpointer, store=store)
        thread_id = f"scan-{datetime.now(tz=timezone.utc).strftime('%Y-%m-%d-%H%M%S')}"
        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "thread_id": thread_id,
            "context": "morning_routine",
            "depth": "standard",
            "run_mode": "daily",
            "available_cash": 0.0,
            "holdings": [],
            "positions": [],
            "messages": [HumanMessage(content="Manual scan triggered via CLI")],
            "errors": [],
        }

        console.print(f"[dim]Thread: {thread_id}[/dim]")
        console.print("[dim]Running pipeline...[/dim]\n")

        final_state = {}
        step = 0
        async for event in graph.astream(initial_state, config, stream_mode="values"):
            final_state = event
            step += 1
            # Show progress for key steps
            msgs = event.get("messages", [])
            if msgs:
                last_msg = msgs[-1]
                if hasattr(last_msg, "content") and last_msg.content:
                    console.print(f"  [dim]Step {step}: {last_msg.content[:120]}[/dim]")

        # Summary
        errors = final_state.get("errors", [])
        trades = final_state.get("execution_reports", [])
        console.print(f"\n[green]Scan complete![/green]")
        console.print(f"  Thread: {thread_id}")
        console.print(f"  Steps executed: {step}")
        if trades:
            console.print(f"  Trades/Recommendations: {len(trades)}")
        if errors:
            console.print(f"  [yellow]Errors: {len(errors)}[/yellow]")

    except Exception as exc:
        console.print(f"[red]Scan failed: {exc}[/red]")
        logger.exception("scan command failed")


# ---------------------------------------------------------------------------
# Command: autfin start
# ---------------------------------------------------------------------------

async def start_command():
    """Start AutFin as a daemon with scheduled jobs (FR-133)."""
    console.print(Panel(
        "[bold]Starting AutFin Daemon[/bold]\n\n"
        "Scheduled Jobs:\n"
        "  • Daily morning scan  — Mon-Fri 9:00 AM IST\n"
        "  • Midday guardian     — Mon-Fri 1:00 PM IST\n"
        "  • Weekly review       — Sunday 10:00 AM IST\n"
        "  • Monthly deep dive   — 1st of month 8:00 AM IST\n\n"
        "Press [bold]Ctrl+C[/bold] to stop.",
        border_style="green",
    ))

    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
    except ImportError:
        console.print("[red]APScheduler not installed. Run: pip install apscheduler[/red]")
        return

    from ..graph.supervisor import compile_supervisor_graph  # v5.3 Subagents
    from ..memory.store import get_store
    from ..memory.setup import create_checkpointer
    from ..scheduler.jobs import register_jobs

    settings = _get_settings()
    store = get_store()
    checkpointer = await create_checkpointer()
    graph = compile_supervisor_graph(checkpointer=checkpointer, store=store)  # v5.3

    scheduler = AsyncIOScheduler(timezone="Asia/Kolkata")
    register_jobs(scheduler, graph, store, settings)

    scheduler.start()
    console.print("[green]Scheduler started.[/green]")

    try:
        # Keep running until Ctrl+C
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        console.print("\n[yellow]Shutting down...[/yellow]")
        scheduler.shutdown(wait=False)
        console.print("[green]Daemon stopped.[/green]")


# ---------------------------------------------------------------------------
# Command: autfin history
# ---------------------------------------------------------------------------

async def history_command(symbol: str, days: int = 90):
    """Show trade history for a symbol (FR-135)."""
    symbol = symbol.upper()
    console.print(Panel(
        f"[bold]Trade History: {symbol}[/bold] (last {days} days)",
        border_style="green",
    ))

    conn = _get_db_connection()
    if not conn:
        return

    try:
        from datetime import timedelta
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=days)).isoformat()

        rows = conn.execute(
            "SELECT * FROM trades WHERE symbol = ? AND timestamp >= ? "
            "ORDER BY timestamp DESC",
            (symbol, cutoff),
        ).fetchall()

        if not rows:
            console.print(f"[dim]No trades found for {symbol} in the last {days} days.[/dim]")
            return

        table = Table(title=f"{symbol} Trades", border_style="cyan")
        table.add_column("Date", style="dim")
        table.add_column("Action", style="bold")
        table.add_column("Qty", justify="right")
        table.add_column("Price", justify="right")
        table.add_column("Value", justify="right")
        table.add_column("Status", style="dim")
        table.add_column("Order ID", style="dim")

        for row in rows:
            action_style = "green" if row["action"] == "BUY" else "red"
            table.add_row(
                row["timestamp"][:10],
                f"[{action_style}]{row['action']}[/{action_style}]",
                str(row["quantity"]),
                _format_inr(row["price"]),
                _format_inr(row["total_value"]),
                row["order_status"] or "-",
                row["order_id"] or "-",
            )

        console.print(table)

        # Summary stats
        total_buys = sum(r["total_value"] for r in rows if r["action"] == "BUY")
        total_sells = sum(r["total_value"] for r in rows if r["action"] == "SELL")
        console.print(f"\n  Total buys:  {_format_inr(total_buys)}")
        console.print(f"  Total sells: {_format_inr(total_sells)}")

    except Exception as exc:
        console.print(f"[red]Error querying history: {exc}[/red]")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Command: autfin config
# ---------------------------------------------------------------------------

# Valid config keys and their types
CONFIG_SCHEMA = {
    "execution_mode": {"type": str, "choices": ["recommendation", "auto-small", "paper"]},
    "deployment_pacing": {"type": str, "choices": ["spread", "aggressive", "conservative"]},
    "auto_execute_max_amount": {"type": float},
    "min_trade_amount": {"type": float},
    "cash_buffer_pct": {"type": float},
    "max_single_position_pct": {"type": float},
    "max_sector_exposure_pct": {"type": float},
    "default_stop_loss_pct": {"type": float},
    "trailing_stop_loss_pct": {"type": float},
    "profit_target_pct": {"type": float},
    "portfolio_drawdown_halt_pct": {"type": float},
    "max_daily_trades": {"type": int},
}


async def config_command(action: str, key: str = "", value: str = ""):
    """Modify or view configuration settings (FR-136)."""
    if action == "get" and key:
        await _config_get(key)
    elif action == "set" and key and value:
        await _config_set(key, value)
    elif action == "list" or (not action):
        await _config_list()
    else:
        console.print("[red]Usage: autfin config set <KEY> <VALUE> | autfin config get <KEY> | autfin config list[/red]")


async def _config_get(key: str):
    """Show a single config value."""
    if key not in CONFIG_SCHEMA:
        console.print(f"[red]Unknown config key: {key}[/red]")
        console.print(f"[dim]Valid keys: {', '.join(CONFIG_SCHEMA.keys())}[/dim]")
        return
    settings = _get_settings()
    current = getattr(settings, key, None)
    console.print(f"  {key} = [cyan]{current}[/cyan]")


async def _config_set(key: str, value: str):
    """Set a config value by updating the .env file."""
    if key not in CONFIG_SCHEMA:
        console.print(f"[red]Unknown config key: {key}[/red]")
        console.print(f"[dim]Valid keys: {', '.join(CONFIG_SCHEMA.keys())}[/dim]")
        return

    schema = CONFIG_SCHEMA[key]
    choices = schema.get("choices")
    if choices and value not in choices:
        console.print(f"[red]Invalid value '{value}' for {key}.[/red]")
        console.print(f"[dim]Valid choices: {', '.join(choices)}[/dim]")
        return

    # Convert to correct type
    try:
        if schema["type"] == float:
            value_typed = float(value)
        elif schema["type"] == int:
            value_typed = int(value)
        else:
            value_typed = value
    except (ValueError, TypeError):
        console.print(f"[red]Cannot convert '{value}' to {schema['type'].__name__}[/red]")
        return

    # Write to .env file
    env_path = Path(".env")
    env_key = key.upper()

    lines = []
    found = False
    if env_path.exists():
        lines = env_path.read_text().splitlines()
        for i, line in enumerate(lines):
            if line.startswith(f"{env_key}="):
                lines[i] = f"{env_key}={value_typed}"
                found = True
                break

    if not found:
        lines.append(f"{env_key}={value_typed}")

    env_path.write_text("\n".join(lines) + "\n")

    console.print(f"  [green]{key} = {value_typed}[/green]")
    console.print(f"  [dim]Updated .env — changes take effect on next run.[/dim]")


async def _config_list():
    """List all config values."""
    settings = _get_settings()
    table = Table(title="Configuration", border_style="green")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Type", style="dim")

    for key, schema in CONFIG_SCHEMA.items():
        val = getattr(settings, key, "N/A")
        table.add_row(key, str(val), schema["type"].__name__)

    console.print(table)


# ---------------------------------------------------------------------------
# Command: autfin backtest (Task 29 — full implementation)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Commands: autfin modify / autfin cancel (Phase B4 — order management)
# ---------------------------------------------------------------------------

async def modify_order_command(order_id: str, qty: str = "", price: str = ""):
    """Modify an existing order via Kite SDK."""
    from ..tools.kite_tools import KiteTools

    if not order_id:
        console.print("[red]Error: provide an order ID, e.g. autfin modify ORDER123 --qty 10[/red]")
        return

    new_qty = int(qty) if qty else None
    new_price = float(price) if price else None

    if not new_qty and not new_price:
        console.print("[red]Error: provide at least --qty or --price to modify[/red]")
        return

    settings = _get_settings()
    kite_auto_login = bool(settings.kite_user_id and settings.kite_password and settings.kite_totp_secret)

    if not kite_auto_login:
        console.print("[red]Kite auto-login not configured. Set KITE_USER_ID, KITE_PASSWORD, KITE_TOTP_SECRET in .env[/red]")
        return

    try:
        kite = KiteTools()
        console.print(f"[cyan]Modifying order {order_id}...[/cyan]")

        result = await kite.modify_order(order_id=order_id, quantity=new_qty, price=new_price)

        if result.get("status") == "success":
            console.print(f"[green]Order {order_id} modified successfully.[/green]")
        else:
            msg = result.get("message", "Unknown error")
            console.print(f"[red]Failed to modify order: {msg}[/red]")
    except Exception as e:
        console.print(f"[red]Error modifying order: {e}[/red]")


async def cancel_order_command(order_id: str):
    """Cancel an existing order via Kite SDK."""
    from ..tools.kite_tools import KiteTools

    if not order_id:
        console.print("[red]Error: provide an order ID, e.g. autfin cancel ORDER123[/red]")
        return

    settings = _get_settings()
    kite_auto_login = bool(settings.kite_user_id and settings.kite_password and settings.kite_totp_secret)

    if not kite_auto_login:
        console.print("[red]Kite auto-login not configured. Set KITE_USER_ID, KITE_PASSWORD, KITE_TOTP_SECRET in .env[/red]")
        return

    try:
        kite = KiteTools()
        console.print(f"[cyan]Cancelling order {order_id}...[/cyan]")

        result = await kite.cancel_order(order_id=order_id)

        if result.get("status") == "success":
            console.print(f"[green]Order {order_id} cancelled successfully.[/green]")
        else:
            msg = result.get("message", "Unknown error")
            console.print(f"[red]Failed to cancel order: {msg}[/red]")
    except Exception as e:
        console.print(f"[red]Error cancelling order: {e}[/red]")


async def backtest_command(strategy: str = "momentum", period: str = "1y", symbol: str = ""):
    """Strategy backtesting with real engine."""
    from ..backtest.engine import BacktestEngine, BacktestConfig, fetch_historical_data
    from ..backtest.strategies import STRATEGY_MAP

    # Validate strategy
    if strategy not in STRATEGY_MAP:
        available = ", ".join(STRATEGY_MAP.keys())
        console.print(f"[red]Unknown strategy: {strategy}[/red]")
        console.print(f"[yellow]Available: {available}[/yellow]")
        return

    # Default symbol if not provided
    if not symbol:
        symbol = "RELIANCE"

    # Map period strings to yfinance periods
    period_map = {
        "6m": "6mo",
        "1y": "1y",
        "2y": "2y",
        "3y": "3y",
        "5y": "5y",
    }
    yf_period = period_map.get(period, period)

    console.print(f"[bold]Running backtest: {strategy} on {symbol} ({period})[/bold]\n")

    try:
        # Fetch historical data
        console.print("[cyan]Fetching historical data...[/cyan]")
        data = fetch_historical_data(symbol, yf_period)
        console.print(f"[green]Loaded {len(data)} bars of data[/green]\n")

        # Run backtest
        config = BacktestConfig()
        engine = BacktestEngine(config)
        result = engine.run(data, strategy, symbol)

        # Display results
        metrics = result.metrics

        # Summary table
        summary = Table(title="Backtest Results", border_style="green")
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="white")

        summary.add_row("Symbol", result.symbol)
        summary.add_row("Strategy", result.strategy_name)
        summary.add_row("Period", period)
        summary.add_row("Data Points", str(len(data)))
        summary.add_row("Total Trades", str(metrics.total_trades))
        summary.add_row("Win Rate", f"{metrics.win_rate * 100:.1f}%")
        summary.add_row("Total Return", f"{metrics.total_return * 100:.2f}%")
        summary.add_row("Annualized Return", f"{metrics.annualized_return * 100:.2f}%")
        summary.add_row("Sharpe Ratio", f"{metrics.sharpe_ratio:.3f}")
        summary.add_row("Sortino Ratio", f"{metrics.sortino_ratio:.3f}")
        summary.add_row("Max Drawdown", f"{metrics.max_drawdown * 100:.2f}%")
        summary.add_row("Calmar Ratio", f"{metrics.calmar_ratio:.3f}")
        summary.add_row("Profit Factor", f"{metrics.profit_factor:.3f}")
        summary.add_row("Avg Win", f"{metrics.avg_win * 100:.2f}%")
        summary.add_row("Avg Loss", f"{metrics.avg_loss * 100:.2f}%")
        summary.add_row("Max Consecutive Losses", str(metrics.max_consecutive_losses))
        summary.add_row("Avg Holding Period", f"{metrics.avg_holding_period_days:.1f} days")
        summary.add_row("Recovery Factor", f"{metrics.recovery_factor:.3f}")

        console.print(summary)

        # Trade log (last 10 trades)
        if result.trades:
            console.print()
            trade_table = Table(title="Recent Trades (last 10)", border_style="blue")
            trade_table.add_column("Entry", style="cyan")
            trade_table.add_column("Exit", style="cyan")
            trade_table.add_column("Entry Price", justify="right")
            trade_table.add_column("Exit Price", justify="right")
            trade_table.add_column("P&L", justify="right")
            trade_table.add_column("P&L %", justify="right")
            trade_table.add_column("Days", justify="right")

            for trade in result.trades[-10:]:
                pnl_style = "green" if trade.pnl >= 0 else "red"
                trade_table.add_row(
                    trade.entry_date,
                    trade.exit_date,
                    f"₹{trade.entry_price:,.2f}",
                    f"₹{trade.exit_price:,.2f}",
                    f"[{pnl_style}]₹{trade.pnl:,.2f}[/{pnl_style}]",
                    f"[{pnl_style}]{trade.pnl_pct * 100:.2f}%[/{pnl_style}]",
                    str(trade.holding_days),
                )

            console.print(trade_table)

    except ImportError:
        console.print("[red]Error: yfinance is required for backtesting.[/red]")
        console.print("[yellow]Install with: pip install yfinance[/yellow]")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
    except Exception as e:
        console.print(f"[red]Backtest failed: {e}[/red]")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> tuple[str, list[str], dict[str, str]]:
    """Parse sys.argv into (command, positional_args, options)."""
    args = sys.argv[1:]
    if not args:
        return "", [], {}

    command = args[0]
    positional = []
    options = {}

    i = 1
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            key = arg[2:]
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                options[key] = args[i + 1]
                i += 2
            else:
                options[key] = "true"
                i += 1
        else:
            positional.append(arg)
            i += 1

    return command, positional, options


async def main():
    """CLI entry point."""
    command, positional, options = _parse_args()

    if not command:
        console.print(Panel(
            "AutFin — Autonomous AI-Powered Financial Research & Portfolio Management\n\n"
            "Usage: autfin <command> [options]\n\n"
            "Commands:\n"
            "  status                          Show account status\n"
            "  analyze <symbol>                Full research report for a stock\n"
            "  scan                            Run daily morning routine\n"
            "  start                           Start as daemon with scheduler\n"
            "  history --symbol <S> --days N   View trade history\n"
            "  modify <ORDER_ID> --qty N --price P  Modify an existing order\n"
            "  cancel <ORDER_ID>              Cancel an existing order\n"
            "  config set <KEY> <VALUE>        Modify configuration\n"
            "  config get <KEY>                Show config value\n"
            "  config list                     List all config values\n"
            "  backtest --strategy S --period P --symbol SYM  Backtest a strategy\n\n"
            "Examples:\n"
            "  autfin status\n"
            "  autfin analyze RELIANCE\n"
            "  autfin scan\n"
            "  autfin start\n"
            "  autfin history --symbol RELIANCE --days 90\n"
            "  autfin modify 2025011512345 --qty 10 --price 2500\n"
            "  autfin cancel 2025011512345\n"
            "  autfin config set execution_mode auto-small\n"
            "  autfin config get execution_mode",
            title="AutFin CLI",
            border_style="green",
        ))
        return

    if command == "status":
        await status_command()

    elif command == "analyze":
        if not positional:
            console.print("[red]Error: provide a stock symbol, e.g. autfin analyze RELIANCE[/red]")
            return
        await analyze_command(positional[0])

    elif command == "scan":
        await scan_command()

    elif command == "start":
        await start_command()

    elif command == "history":
        symbol = options.get("symbol", "")
        if not symbol and positional:
            symbol = positional[0]
        if not symbol:
            console.print("[red]Error: provide --symbol, e.g. autfin history --symbol RELIANCE[/red]")
            return
        days = int(options.get("days", "90"))
        await history_command(symbol, days)

    elif command == "config":
        action = positional[0] if positional else "list"
        key = positional[1] if len(positional) > 1 else options.get("key", "")
        value = positional[2] if len(positional) > 2 else options.get("value", "")
        await config_command(action, key, value)

    elif command == "modify":
        if not positional:
            console.print("[red]Error: provide an order ID, e.g. autfin modify ORDER123 --qty 10[/red]")
            return
        await modify_order_command(
            positional[0],
            qty=options.get("qty", ""),
            price=options.get("price", ""),
        )

    elif command == "cancel":
        if not positional:
            console.print("[red]Error: provide an order ID, e.g. autfin cancel ORDER123[/red]")
            return
        await cancel_order_command(positional[0])

    elif command == "backtest":
        strategy = options.get("strategy", "momentum")
        period = options.get("period", "1y")
        symbol = options.get("symbol", "")
        await backtest_command(strategy, period, symbol)

    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        console.print("[yellow]Run 'autfin' without arguments for help[/yellow]")


if __name__ == "__main__":
    asyncio.run(main())
