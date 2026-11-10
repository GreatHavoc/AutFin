"""Tests for guardrails and the browser-MCP endpoint wiring."""
from __future__ import annotations

from types import SimpleNamespace

import pytest


# ── guardrails: behaviour ───────────────────────────────────────────────────
def _call(name, **args):
    return SimpleNamespace(name=name, args=args)


def test_trade_guard_accepts_real_nse_symbols():
    """isalnum() rejected BAJAJ-AUTO and M&M, both in the curated universe, so
    a legitimate limit order was blocked."""
    from autfin.agents.guardrails import validate_trade_args

    for symbol in ("RELIANCE", "BAJAJ-AUTO", "M&M", "HDFCBANK.NS"):
        result = validate_trade_args(
            _call("place_order", tradingsymbol=symbol, quantity=1, order_type="LIMIT", price=2500.0)
        )
        assert result.action == "allow", symbol


@pytest.mark.parametrize("symbol", ["", "   ", "DROP;TABLE", "a" * 25])
def test_trade_guard_rejects_garbage_symbols(symbol):
    from autfin.agents.guardrails import validate_trade_args

    result = validate_trade_args(
        _call("place_order", tradingsymbol=symbol, quantity=1, order_type="MARKET")
    )
    assert result.action == "block"


def test_trade_guard_rejects_non_positive_quantity():
    from autfin.agents.guardrails import validate_trade_args

    result = validate_trade_args(_call("place_order", quantity=0, tradingsymbol="TCS"))
    assert result.action == "block"
    assert "quantity" in result.message.lower()


def test_trade_guard_requires_a_price_for_limit_orders():
    from autfin.agents.guardrails import validate_trade_args

    result = validate_trade_args(
        _call("place_order", quantity=10, tradingsymbol="TCS", order_type="SL", price=None)
    )
    assert result.action == "block"


def test_workspace_guard_blocks_paths_outside_the_workspace(tmp_path, monkeypatch):
    from autfin.agents import guardrails

    monkeypatch.setattr(guardrails, "WORKSPACE", tmp_path.resolve())

    inside = (tmp_path / "notes.md").as_posix()
    outside = "C:/Windows/system32/drivers/etc/hosts"

    assert guardrails.stay_in_workspace(_call("edit_file", path=inside)).action == "allow"
    assert guardrails.stay_in_workspace(_call("edit_file", path=outside)).action == "block"


def test_workspace_guard_ignores_non_file_tools():
    from autfin.agents.guardrails import stay_in_workspace

    assert stay_in_workspace(_call("place_order", path="C:/Windows/x")).action == "allow"


def test_risk_reward_output_guard_retries_below_the_floor():
    from autfin.agents.guardrails import validate_risk_reward

    assert validate_risk_reward({"risk_reward": 1.2}).action == "retry"
    assert validate_risk_reward({"conviction": 0.2}).action == "retry"
    assert validate_risk_reward({"risk_reward": 3.0, "conviction": 0.8}).action == "allow"
    assert validate_risk_reward("plain text").action == "allow"


def test_citation_guard_retries_long_uncited_output():
    from autfin.agents.guardrails import enforce_citations

    long_text = "x" * 400
    assert enforce_citations(long_text).action == "retry"
    assert enforce_citations(f"Source: NSE\n{long_text}").action == "allow"
    assert enforce_citations("short").action == "allow"


def test_guardrail_map_matches_the_agent_kinds():
    from autfin.agents.guardrails import (
        get_guardrails_for_agent,
        input_guardrails,
        output_guardrails,
        tool_guardrails,
    )

    assert get_guardrails_for_agent("market_data") == [input_guardrails]
    assert get_guardrails_for_agent("risk") == [input_guardrails, output_guardrails, tool_guardrails]
    assert get_guardrails_for_agent("supervisor") == [input_guardrails, output_guardrails, tool_guardrails]
    assert get_guardrails_for_agent("nothing_known") == [input_guardrails, output_guardrails]


def test_tool_guard_guards_both_argument_shape_and_containment():
    from autfin.agents.guardrails import tool_guardrails

    assert tool_guardrails.tools == ["place_order", "execute_trade", "write_file", "edit_file"]
    # `guard` accepts a chain; the workspace guard was defined but never registered.
    assert tool_guardrails.guard is not None
    guards = tool_guardrails.guard if isinstance(tool_guardrails.guard, (list, tuple)) else [tool_guardrails.guard]
    names = {getattr(f, "__name__", str(f)) for f in guards}
    assert {"validate_trade_args", "stay_in_workspace"} <= names


# ── guardrails: actually attached ───────────────────────────────────────────
@pytest.mark.parametrize(
    "agent_name,expected",
    [
        ("market_data_agent", ["InputGuardrail"]),
        ("analysis_agent", ["InputGuardrail", "OutputGuardrail"]),
        ("risk_agent", ["InputGuardrail", "OutputGuardrail", "ToolGuardrail"]),
        ("supervisor_agent", ["InputGuardrail", "OutputGuardrail", "ToolGuardrail"]),
        ("memory_agent", ["InputGuardrail", "ToolGuardrail"]),
        ("quant_agent", ["InputGuardrail", "OutputGuardrail", "ToolGuardrail"]),
    ],
)
def test_agents_carry_their_guardrails(agent_name, expected):
    from pydantic_ai_harness.guardrails import InputGuardrail, OutputGuardrail, ToolGuardrail

    import autfin.agents as agents

    agent = getattr(agents, agent_name)
    kinds = [
        type(c).__name__
        for c in agent.root_capability.capabilities
        if isinstance(c, (InputGuardrail, OutputGuardrail, ToolGuardrail))
    ]
    assert sorted(kinds) == sorted(expected), agent_name


def test_unknown_agent_gets_the_default_guardrails():
    from pydantic_ai_harness.guardrails import InputGuardrail, OutputGuardrail

    from autfin.agents.base import create_agent

    agent = create_agent(name="mystery_agent", output_type=dict, instructions="test")
    kinds = {
        type(c).__name__
        for c in agent.root_capability.capabilities
        if isinstance(c, (InputGuardrail, OutputGuardrail))
    }
    assert kinds == {"InputGuardrail", "OutputGuardrail"}


# ── browser MCP endpoint wiring ─────────────────────────────────────────────
def test_browser_toolset_defaults_to_the_documented_settings(monkeypatch):
    monkeypatch.delenv("OBSCURA_MCP_URL", raising=False)
    monkeypatch.setenv("BROWSER_MCP_PORT", "9333")

    from autfin.tools.browser_agent import obscura_endpoint

    assert obscura_endpoint() == "http://127.0.0.1:9333/mcp"


def test_named_url_beats_the_settings_module(monkeypatch):
    monkeypatch.setenv("OBSCURA_MCP_URL", "http://localhost:7777/mcp")
    monkeypatch.setenv("BROWSER_MCP_PORT", "9333")

    from autfin.tools.browser_agent import obscura_endpoint

    assert obscura_endpoint() == "http://localhost:7777/mcp"


def test_daemon_binds_the_port_the_client_dials(monkeypatch):
    """run.py starts Obscura on OBSCURA_PORT, so the client must dial that same
    port — it used to default to 8080/9333 independently."""
    monkeypatch.setenv("BROWSER_MCP_TRANSPORT", "streamable_http")
    monkeypatch.delenv("BROWSER_MCP_PORT", raising=False)

    import run

    monkeypatch.setattr(run, "OBSCURA_PORT", 9222)
    # the module-level line the daemon runs, executed with the test's env
    monkeypatch.setenv("BROWSER_MCP_PORT", str(run.OBSCURA_PORT))

    from autfin.tools.browser_agent import BrowserMCPSettings

    assert BrowserMCPSettings.from_env().port == 9222
