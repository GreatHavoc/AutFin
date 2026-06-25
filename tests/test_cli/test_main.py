"""
Tests for AutFin CLI (src/autfin/cli/main.py).

Covers all commands: status, analyze, scan, start, history, config, backtest.
"""
import asyncio
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.autfin.cli.main import (
    _parse_args,
    _format_inr,
    _get_db_path,
    CONFIG_SCHEMA,
    main,
    status_command,
    analyze_command,
    scan_command,
    start_command,
    history_command,
    config_command,
    backtest_command,
    _config_get,
    _config_set,
    _config_list,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_parse(args_list):
    """Run _parse_args with a custom argv list."""
    original_argv = sys.argv
    try:
        sys.argv = ["autfin"] + args_list
        return _parse_args()
    finally:
        sys.argv = original_argv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_settings():
    """Return a mock AutFinSettings."""
    with patch("src.autfin.cli.main.AutFinSettings") as MockSettings:
        settings = MagicMock()
        settings.execution_mode = "recommendation"
        settings.deployment_pacing = "spread"
        settings.auto_execute_max_amount = 5000.0
        settings.min_trade_amount = 1000.0
        settings.cash_buffer_pct = 0.10
        settings.max_single_position_pct = 0.30
        settings.max_sector_exposure_pct = 0.30
        settings.default_stop_loss_pct = 0.08
        settings.trailing_stop_loss_pct = 0.05
        settings.profit_target_pct = 0.15
        settings.portfolio_drawdown_halt_pct = 0.10
        settings.max_daily_trades = 5
        MockSettings.return_value = settings
        yield settings


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary SQLite database with required schema."""
    db_path = tmp_path / "autfin.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL DEFAULT 'NSE',
            action TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price REAL NOT NULL,
            total_value REAL NOT NULL,
            order_id TEXT,
            order_status TEXT,
            decision_id TEXT,
            rationale_summary TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            available_cash REAL NOT NULL,
            utilized_margin REAL NOT NULL,
            total_holdings_value REAL NOT NULL,
            total_pnl REAL NOT NULL,
            total_pnl_pct REAL NOT NULL,
            holdings_json TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS learning_journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            pnl_pct REAL,
            outcome TEXT,
            thesis_summary TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)

    conn.commit()

    now = datetime.now(tz=timezone.utc).isoformat()
    sample_trades = [
        (now, "RELIANCE", "NSE", "BUY", 10, 2500.0, 25000.0, "ORDER001", "COMPLETE"),
        (now, "RELIANCE", "NSE", "SELL", 5, 2600.0, 13000.0, "ORDER002", "COMPLETE"),
        (now, "INFY", "NSE", "BUY", 20, 1500.0, 30000.0, "ORDER003", "COMPLETE"),
    ]
    cursor.executemany(
        "INSERT INTO trades (timestamp, symbol, exchange, action, quantity, price, "
        "total_value, order_id, order_status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        sample_trades,
    )

    cursor.execute(
        "INSERT INTO portfolio_snapshots (date, available_cash, utilized_margin, "
        "total_holdings_value, total_pnl, total_pnl_pct, holdings_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("2026-06-22", 48230.0, 0.0, 38000.0, 1240.0, 2.5, "[]"),
    )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def mock_db_path(temp_db):
    """Patch _get_db_path to return temp database."""
    with patch("src.autfin.cli.main._get_db_path", return_value=temp_db):
        yield temp_db


# ===========================================================================
# Test: _parse_args
# ===========================================================================

class TestParseArgs:
    def test_no_args(self):
        cmd, pos, opts = _run_parse([])
        assert cmd == ""
        assert pos == []
        assert opts == {}

    def test_command_only(self):
        cmd, pos, opts = _run_parse(["status"])
        assert cmd == "status"
        assert pos == []
        assert opts == {}

    def test_command_with_positional(self):
        cmd, pos, opts = _run_parse(["analyze", "RELIANCE"])
        assert cmd == "analyze"
        assert pos == ["RELIANCE"]
        assert opts == {}

    def test_command_with_options(self):
        cmd, pos, opts = _run_parse(["history", "--symbol", "RELIANCE", "--days", "30"])
        assert cmd == "history"
        assert pos == []
        assert opts == {"symbol": "RELIANCE", "days": "30"}

    def test_boolean_option(self):
        cmd, pos, opts = _run_parse(["scan", "--verbose"])
        assert cmd == "scan"
        assert opts == {"verbose": "true"}

    def test_backtest_options(self):
        cmd, pos, opts = _run_parse(["backtest", "--strategy", "momentum", "--period", "1y"])
        assert cmd == "backtest"
        assert opts == {"strategy": "momentum", "period": "1y"}


# ===========================================================================
# Test: _format_inr
# ===========================================================================

class TestFormatInr:
    def test_positive_small(self):
        assert _format_inr(1234.56) == "₹1,234.56"

    def test_zero(self):
        assert _format_inr(0) == "₹0.00"

    def test_negative(self):
        assert _format_inr(-500.0) == "-₹500.00"

    def test_lakh(self):
        result = _format_inr(150000)
        assert result == "₹1,50,000"

    def test_large_negative(self):
        result = _format_inr(-250000)
        assert result == "-₹2,50,000"

    def test_very_small(self):
        assert _format_inr(0.5) == "₹0.50"

    def test_one_lakh(self):
        result = _format_inr(100000)
        assert result == "₹1,00,000"

    def test_ten_lakh(self):
        result = _format_inr(1000000)
        assert result == "₹10,00,000"

    def test_one_crore(self):
        result = _format_inr(10000000)
        assert "Cr" in result


# ===========================================================================
# Test: CONFIG_SCHEMA
# ===========================================================================

class TestConfigSchema:
    def test_all_keys_present(self):
        expected = {
            "execution_mode", "deployment_pacing", "auto_execute_max_amount",
            "min_trade_amount", "cash_buffer_pct", "max_single_position_pct",
            "max_sector_exposure_pct", "default_stop_loss_pct",
            "trailing_stop_loss_pct", "profit_target_pct",
            "portfolio_drawdown_halt_pct", "max_daily_trades",
        }
        assert expected == set(CONFIG_SCHEMA.keys())

    def test_execution_mode_choices(self):
        assert CONFIG_SCHEMA["execution_mode"]["choices"] == [
            "recommendation", "auto-small", "paper"
        ]

    def test_auto_execute_max_is_float(self):
        assert CONFIG_SCHEMA["auto_execute_max_amount"]["type"] == float

    def test_max_daily_trades_is_int(self):
        assert CONFIG_SCHEMA["max_daily_trades"]["type"] == int


# ===========================================================================
# Test: status_command
# ===========================================================================

class TestStatusCommand:
    @pytest.mark.asyncio
    async def test_status_runs_without_error(self, mock_settings, mock_db_path, capsys):
        with patch("src.autfin.tools.kite_tools.KiteTools", side_effect=Exception("No Kite")):
            await status_command()
        output = capsys.readouterr().out
        assert "Account Summary" in output or "Status" in output

    @pytest.mark.asyncio
    async def test_status_shows_config_values(self, mock_settings, mock_db_path, capsys):
        with patch("src.autfin.tools.kite_tools.KiteTools", side_effect=Exception("No Kite")):
            await status_command()
        output = capsys.readouterr().out
        assert "recommendation" in output
        assert "spread" in output

    @pytest.mark.asyncio
    async def test_status_with_kite_data(self, mock_settings, mock_db_path, capsys):
        mock_margins = MagicMock()
        mock_margins.get_margins = AsyncMock(return_value={
            "status": "success",
            "data": {"equity": {"available": {"cash": 50000.0}, "used": 10000.0}},
        })
        mock_margins.get_holdings = AsyncMock(return_value={
            "status": "success",
            "data": [{"tradingsymbol": "RELIANCE", "quantity": 10,
                       "average_price": 2500.0, "last_price": 2600.0}],
        })

        with patch("src.autfin.tools.kite_tools.KiteTools", return_value=mock_margins):
            await status_command()

        output = capsys.readouterr().out
        assert "50,000" in output
        assert "RELIANCE" in output

    @pytest.mark.asyncio
    async def test_status_portfolio_snapshot_pnl(self, mock_settings, mock_db_path, capsys):
        with patch("src.autfin.tools.kite_tools.KiteTools", side_effect=Exception("No Kite")):
            await status_command()
        output = capsys.readouterr().out
        assert "1,240" in output or "P&L" in output


# ===========================================================================
# Test: analyze_command
# ===========================================================================

class TestAnalyzeCommand:
    @pytest.mark.asyncio
    async def test_analyze_displays_symbol(self, mock_settings, capsys):
        async def fake_stream(state, config, stream_mode=None):
            yield {}

        with patch("src.autfin.graph.supervisor.compile_supervisor_graph") as mock_compile:
            mock_graph = MagicMock()
            mock_graph.astream = fake_stream
            mock_compile.return_value = mock_graph
            await analyze_command("RELIANCE")
        output = capsys.readouterr().out
        assert "RELIANCE" in output

    @pytest.mark.asyncio
    async def test_analyze_handles_failure(self, mock_settings, capsys):
        with patch("src.autfin.graph.supervisor.compile_supervisor_graph",
                    side_effect=Exception("Graph error")):
            await analyze_command("RELIANCE")
        output = capsys.readouterr().out
        assert "failed" in output.lower() or "error" in output.lower()

    @pytest.mark.asyncio
    async def test_analyze_display_results_with_debate(self, mock_settings, capsys):
        with patch("src.autfin.graph.supervisor.compile_supervisor_graph") as mock_compile:
            mock_graph = MagicMock()

            async def fake_stream(state, config, stream_mode=None):
                yield {
                    "research_debates": {
                        "RELIANCE": {
                            "bull_case": "Strong refining margins",
                            "bear_case": "High debt levels",
                            "synthesis": "Moderate buy",
                        }
                    },
                    "messages": [],
                    "errors": [],
                }

            mock_graph.astream = fake_stream
            mock_compile.return_value = mock_graph
            await analyze_command("RELIANCE")

        output = capsys.readouterr().out
        assert "Bull Case" in output
        assert "Bear Case" in output
        assert "Strong refining margins" in output

    @pytest.mark.asyncio
    async def test_analyze_display_results_with_risk(self, mock_settings, capsys):
        with patch("src.autfin.graph.supervisor.compile_supervisor_graph") as mock_compile:
            mock_graph = MagicMock()

            async def fake_stream(state, config, stream_mode=None):
                yield {
                    "risk_assessments": {
                        "RELIANCE": {"verdict": "APPROVED", "rationale": "Within limits"}
                    },
                    "messages": [],
                    "errors": [],
                }

            mock_graph.astream = fake_stream
            mock_compile.return_value = mock_graph
            await analyze_command("RELIANCE")

        output = capsys.readouterr().out
        assert "APPROVED" in output

    @pytest.mark.asyncio
    async def test_analyze_display_results_with_trades(self, mock_settings, capsys):
        with patch("src.autfin.graph.supervisor.compile_supervisor_graph") as mock_compile:
            mock_graph = MagicMock()

            async def fake_stream(state, config, stream_mode=None):
                yield {
                    "trade_queue": [
                        {"symbol": "RELIANCE", "action": "BUY", "qty": 5, "price": 2500.0}
                    ],
                    "messages": [],
                    "errors": [],
                }

            mock_graph.astream = fake_stream
            mock_compile.return_value = mock_graph
            await analyze_command("RELIANCE")

        output = capsys.readouterr().out
        assert "Proposed Trades" in output
        assert "BUY" in output

    @pytest.mark.asyncio
    async def test_analyze_display_errors(self, mock_settings, capsys):
        with patch("src.autfin.graph.supervisor.compile_supervisor_graph") as mock_compile:
            mock_graph = MagicMock()

            async def fake_stream(state, config, stream_mode=None):
                yield {
                    "messages": [],
                    "errors": ["Kite connection failed"],
                }

            mock_graph.astream = fake_stream
            mock_compile.return_value = mock_graph
            await analyze_command("RELIANCE")

        output = capsys.readouterr().out
        assert "Kite connection failed" in output


# ===========================================================================
# Test: scan_command
# ===========================================================================

class TestScanCommand:
    @pytest.mark.asyncio
    async def test_scan_runs_pipeline(self, mock_settings, capsys):
        with patch("src.autfin.graph.supervisor.compile_supervisor_graph") as mock_compile:
            mock_graph = MagicMock()

            async def fake_stream(state, config, stream_mode=None):
                yield {"messages": [MagicMock(content="Running capital check")], "errors": []}
                yield {"messages": [MagicMock(content="Research complete")], "errors": []}

            mock_graph.astream = fake_stream
            mock_compile.return_value = mock_graph
            await scan_command()

        output = capsys.readouterr().out
        assert "Scan complete" in output

    @pytest.mark.asyncio
    async def test_scan_shows_thread_id(self, mock_settings, capsys):
        async def fake_stream(state, config, stream_mode=None):
            yield {"messages": [], "errors": []}

        with patch("src.autfin.graph.supervisor.compile_supervisor_graph") as mock_compile:
            mock_graph = MagicMock()
            mock_graph.astream = fake_stream
            mock_compile.return_value = mock_graph
            await scan_command()

        output = capsys.readouterr().out
        assert "Thread:" in output

    @pytest.mark.asyncio
    async def test_scan_handles_failure(self, mock_settings, capsys):
        with patch("src.autfin.graph.supervisor.compile_supervisor_graph",
                    side_effect=Exception("Graph error")):
            await scan_command()
        output = capsys.readouterr().out
        assert "failed" in output.lower() or "error" in output.lower()

    @pytest.mark.asyncio
    async def test_scan_reports_trade_count(self, mock_settings, capsys):
        with patch("src.autfin.graph.supervisor.compile_supervisor_graph") as mock_compile:
            mock_graph = MagicMock()

            async def fake_stream(state, config, stream_mode=None):
                yield {
                    "messages": [],
                    "execution_reports": [{"status": "executed"}, {"status": "recommended"}],
                    "errors": [],
                }

            mock_graph.astream = fake_stream
            mock_compile.return_value = mock_graph
            await scan_command()

        output = capsys.readouterr().out
        assert "2" in output  # 2 trades


# ===========================================================================
# Test: start_command
# ===========================================================================

class TestStartCommand:
    @pytest.mark.asyncio
    async def test_start_function_exists(self):
        assert callable(start_command)


# ===========================================================================
# Test: history_command
# ===========================================================================

class TestHistoryCommand:
    @pytest.mark.asyncio
    async def test_history_shows_trades(self, mock_settings, mock_db_path, capsys):
        await history_command("RELIANCE", days=90)
        output = capsys.readouterr().out
        assert "RELIANCE" in output
        assert "BUY" in output or "SELL" in output

    @pytest.mark.asyncio
    async def test_history_no_trades(self, mock_settings, mock_db_path, capsys):
        await history_command("TCS", days=90)
        output = capsys.readouterr().out
        assert "No trades found" in output

    @pytest.mark.asyncio
    async def test_history_shows_summary(self, mock_settings, mock_db_path, capsys):
        await history_command("RELIANCE", days=90)
        output = capsys.readouterr().out
        assert "Total buys" in output
        assert "Total sells" in output

    @pytest.mark.asyncio
    async def test_history_different_symbol(self, mock_settings, mock_db_path, capsys):
        await history_command("INFY", days=90)
        output = capsys.readouterr().out
        assert "INFY" in output
        assert "30,000" in output

    @pytest.mark.asyncio
    async def test_history_date_filter(self, mock_settings, mock_db_path, capsys):
        await history_command("RELIANCE", days=1)
        output = capsys.readouterr().out
        assert "RELIANCE" in output

    @pytest.mark.asyncio
    async def test_history_no_database(self, mock_settings, capsys):
        with patch("src.autfin.cli.main._get_db_path", return_value=Path("/nonexistent/db.sqlite")):
            await history_command("RELIANCE")
        output = capsys.readouterr().out
        assert "No trades" in output or "Error" in output or "database" in output.lower()


# ===========================================================================
# Test: config_command
# ===========================================================================

class TestConfigCommand:
    @pytest.mark.asyncio
    async def test_config_set_valid(self, mock_settings, capsys):
        with patch("pathlib.Path.exists", return_value=False), \
             patch("pathlib.Path.write_text") as mock_write:
            await _config_set("execution_mode", "paper")
        output = capsys.readouterr().out
        assert "paper" in output
        assert "Updated" in output

    @pytest.mark.asyncio
    async def test_config_set_invalid_key(self, mock_settings, capsys):
        await _config_set("unknown_key", "value")
        output = capsys.readouterr().out
        assert "Unknown" in output

    @pytest.mark.asyncio
    async def test_config_set_invalid_choice(self, mock_settings, capsys):
        await _config_set("execution_mode", "invalid_mode")
        output = capsys.readouterr().out
        assert "Invalid" in output

    @pytest.mark.asyncio
    async def test_config_set_float_value(self, mock_settings, capsys):
        with patch("pathlib.Path.exists", return_value=False), \
             patch("pathlib.Path.write_text"):
            await _config_set("auto_execute_max_amount", "10000.0")
        output = capsys.readouterr().out
        assert "10000.0" in output

    @pytest.mark.asyncio
    async def test_config_set_int_value(self, mock_settings, capsys):
        with patch("pathlib.Path.exists", return_value=False), \
             patch("pathlib.Path.write_text"):
            await _config_set("max_daily_trades", "10")
        output = capsys.readouterr().out
        assert "10" in output

    @pytest.mark.asyncio
    async def test_config_set_invalid_float(self, mock_settings, capsys):
        await _config_set("auto_execute_max_amount", "not_a_number")
        output = capsys.readouterr().out
        assert "Cannot convert" in output

    @pytest.mark.asyncio
    async def test_config_get(self, mock_settings, capsys):
        await _config_get("execution_mode")
        output = capsys.readouterr().out
        assert "recommendation" in output

    @pytest.mark.asyncio
    async def test_config_get_unknown(self, capsys):
        await _config_get("nonexistent_key")
        output = capsys.readouterr().out
        assert "Unknown" in output

    @pytest.mark.asyncio
    async def test_config_list(self, mock_settings, capsys):
        await _config_list()
        output = capsys.readouterr().out
        assert "Configuration" in output
        assert "execution_mode" in output

    @pytest.mark.asyncio
    async def test_config_command_dispatch_list(self, mock_settings, capsys):
        await config_command("list")
        output = capsys.readouterr().out
        assert "Configuration" in output

    @pytest.mark.asyncio
    async def test_config_command_dispatch_set(self, mock_settings, capsys):
        with patch("pathlib.Path.exists", return_value=False), \
             patch("pathlib.Path.write_text"):
            await config_command("set", "execution_mode", "paper")
        output = capsys.readouterr().out
        assert "paper" in output

    @pytest.mark.asyncio
    async def test_config_command_dispatch_get(self, mock_settings, capsys):
        await config_command("get", "execution_mode")
        output = capsys.readouterr().out
        assert "recommendation" in output

    @pytest.mark.asyncio
    async def test_config_set_updates_existing_env_line(self, mock_settings, capsys):
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.read_text",
                   return_value="EXECUTION_MODE=recommendation\nOTHER=value\n"), \
             patch("pathlib.Path.write_text") as mock_write:
            await _config_set("execution_mode", "paper")
        written = mock_write.call_args[0][0]
        assert "EXECUTION_MODE=paper" in written
        assert "OTHER=value" in written


# ===========================================================================
# Test: backtest_command
# ===========================================================================

class TestBacktestCommand:
    @pytest.mark.asyncio
    async def test_backtest_shows_placeholder(self, mock_settings, capsys):
        await backtest_command("momentum", "1y")
        output = capsys.readouterr().out
        assert "Backtest" in output
        assert "momentum" in output
        assert "1y" in output


# ===========================================================================
# Test: main() entry point
# ===========================================================================

class TestMainEntryPoint:
    @pytest.mark.asyncio
    async def test_main_no_args_shows_help(self, capsys):
        with patch("sys.argv", ["autfin"]):
            await main()
        output = capsys.readouterr().out
        assert "AutFin" in output
        assert "Commands:" in output

    @pytest.mark.asyncio
    async def test_main_unknown_command(self, capsys):
        with patch("sys.argv", ["autfin", "unknown"]):
            await main()
        output = capsys.readouterr().out
        assert "Unknown command" in output

    @pytest.mark.asyncio
    async def test_main_status_dispatch(self, mock_settings, mock_db_path, capsys):
        with patch("sys.argv", ["autfin", "status"]), \
             patch("src.autfin.tools.kite_tools.KiteTools", side_effect=Exception("No Kite")):
            await main()
        output = capsys.readouterr().out
        assert "Account Summary" in output or "Status" in output

    @pytest.mark.asyncio
    async def test_main_analyze_dispatch(self, mock_settings, capsys):
        async def fake_stream(state, config, stream_mode=None):
            yield {}

        with patch("sys.argv", ["autfin", "analyze", "RELIANCE"]), \
             patch("src.autfin.graph.supervisor.compile_supervisor_graph") as mock_compile:
            mock_graph = MagicMock()
            mock_graph.astream = fake_stream
            mock_compile.return_value = mock_graph
            await main()
        output = capsys.readouterr().out
        assert "RELIANCE" in output

    @pytest.mark.asyncio
    async def test_main_analyze_no_symbol(self, capsys):
        with patch("sys.argv", ["autfin", "analyze"]):
            await main()
        output = capsys.readouterr().out
        assert "Error" in output

    @pytest.mark.asyncio
    async def test_main_history_dispatch(self, mock_settings, mock_db_path, capsys):
        with patch("sys.argv", ["autfin", "history", "--symbol", "RELIANCE"]):
            await main()
        output = capsys.readouterr().out
        assert "RELIANCE" in output

    @pytest.mark.asyncio
    async def test_main_history_no_symbol(self, capsys):
        with patch("sys.argv", ["autfin", "history"]):
            await main()
        output = capsys.readouterr().out
        assert "Error" in output

    @pytest.mark.asyncio
    async def test_main_backtest_dispatch(self, mock_settings, capsys):
        with patch("sys.argv", ["autfin", "backtest", "--strategy", "momentum", "--period", "1y"]):
            await main()
        output = capsys.readouterr().out
        assert "Backtest" in output
        assert "momentum" in output

    @pytest.mark.asyncio
    async def test_main_config_list(self, mock_settings, capsys):
        with patch("sys.argv", ["autfin", "config"]):
            await main()
        output = capsys.readouterr().out
        assert "Configuration" in output

    @pytest.mark.asyncio
    async def test_main_config_set(self, mock_settings, capsys):
        with patch("sys.argv", ["autfin", "config", "set", "execution_mode", "paper"]), \
             patch("pathlib.Path.exists", return_value=False), \
             patch("pathlib.Path.write_text"):
            await main()
        output = capsys.readouterr().out
        assert "paper" in output

    @pytest.mark.asyncio
    async def test_main_config_get(self, mock_settings, capsys):
        with patch("sys.argv", ["autfin", "config", "get", "execution_mode"]):
            await main()
        output = capsys.readouterr().out
        assert "recommendation" in output

    @pytest.mark.asyncio
    async def test_main_scan_dispatch(self, mock_settings, capsys):
        async def fake_stream(state, config, stream_mode=None):
            yield {"messages": [], "errors": []}

        with patch("sys.argv", ["autfin", "scan"]), \
             patch("src.autfin.graph.supervisor.compile_supervisor_graph") as mock_compile:
            mock_graph = MagicMock()
            mock_graph.astream = fake_stream
            mock_compile.return_value = mock_graph
            await main()
        output = capsys.readouterr().out
        assert "Scan complete" in output


# ===========================================================================
# Test: Database integration
# ===========================================================================

class TestDatabaseIntegration:
    def test_get_db_path_returns_path(self):
        result = _get_db_path()
        assert isinstance(result, Path)

    def test_get_db_path_contains_autfin_db(self):
        result = _get_db_path()
        assert result.name == "autfin.db"

    def test_trades_table_schema(self, temp_db):
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.execute("PRAGMA table_info(trades)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()
        expected = {
            "id", "timestamp", "symbol", "exchange", "action",
            "quantity", "price", "total_value", "order_id",
            "order_status", "decision_id", "rationale_summary", "created_at",
        }
        assert expected == columns

    def test_portfolio_snapshots_table_schema(self, temp_db):
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.execute("PRAGMA table_info(portfolio_snapshots)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()
        expected = {
            "id", "date", "available_cash", "utilized_margin",
            "total_holdings_value", "total_pnl", "total_pnl_pct",
            "holdings_json", "created_at",
        }
        assert expected == columns

    def test_learning_journal_table_schema(self, temp_db):
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.execute("PRAGMA table_info(learning_journal)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()
        expected = {
            "id", "decision_id", "symbol", "trade_date",
            "pnl_pct", "outcome", "thesis_summary", "created_at",
        }
        assert expected == columns
