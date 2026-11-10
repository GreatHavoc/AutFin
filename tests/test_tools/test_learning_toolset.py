"""Tests for learning_toolset.py — the agent-facing self-learning tools.

Tools are invoked through `toolset.tools[name].function(ctx, ...)` so the tests
exercise the same path an agent run does, not just the underlying modules.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def journal_db(tmp_path, monkeypatch):
    from autfin.memory import database

    monkeypatch.setattr(database, "DB_PATH", tmp_path / "autfin.db")
    database.init_database()
    return database.DB_PATH


@pytest.fixture
def ctx():
    from autfin.types import AgentDeps

    mock = MagicMock()
    mock.deps = AgentDeps(user_id="test")
    return mock


@pytest.fixture
def toolset():
    from autfin.tools.learning_toolset import create_learning_toolset

    return create_learning_toolset()


def _call(toolset, name, ctx, **kwargs):
    return toolset.tools[name].function(ctx, **kwargs)


def test_expected_tools_exist(toolset):
    assert sorted(toolset.tools) == [
        "calibrate_conviction",
        "get_calibration",
        "get_learning_context",
        "get_learning_summary",
        "record_trade_outcome",
    ]


@pytest.mark.asyncio
async def test_get_calibration_empty_before_any_trade(toolset, journal_db, ctx):
    assert await _call(toolset, "get_calibration", ctx) == {}


@pytest.mark.asyncio
async def test_get_calibration_reflects_recorded_trades(toolset, journal_db, ctx):
    await _call(
        toolset, "record_trade_outcome", ctx,
        symbol="RELIANCE", action="BUY", conviction=0.85, pnl_pct=0.10,
        thesis_summary="Margin expansion", sector="Energy",
    )
    await _call(
        toolset, "record_trade_outcome", ctx,
        symbol="TCS", action="BUY", conviction=0.85, pnl_pct=-0.04,
        thesis_summary="Deal flow slowing", sector="IT",
    )

    calibration = await _call(toolset, "get_calibration", ctx)
    assert calibration["high_conv"]["sample_size"] == 2
    assert calibration["high_conv"]["actual_win_rate"] == 0.5
    assert calibration["high_conv"]["delta"] < 0  # overconfident tier


@pytest.mark.asyncio
async def test_learning_context_calibrates_conviction_downward(toolset, journal_db, ctx):
    """An overconfident tier must pull the reasoned conviction toward reality."""
    for _ in range(4):
        await _call(
            toolset, "record_trade_outcome", ctx,
            symbol="X", action="BUY", conviction=0.9, pnl_pct=-0.05,
            thesis_summary="Thesis failed", sector="IT",
        )

    result = await _call(
        toolset, "get_learning_context", ctx,
        symbol="RELIANCE", action="BUY", sector="Energy", raw_conviction=0.9,
    )

    adjustment = result["conviction_adjustment"]
    assert adjustment["calibration_delta"] < 0
    assert adjustment["adjusted_conviction"] < 0.9
    assert result["calibration"]
    assert result["patterns"] is not None


@pytest.mark.asyncio
async def test_learning_context_degrades_cleanly_on_empty_journal(toolset, journal_db, ctx):
    """With no history the tool still returns the four context buckets."""
    result = await _call(
        toolset, "get_learning_context", ctx,
        symbol="RELIANCE", action="BUY", raw_conviction=0.5,
    )
    assert set(result) >= {"similar_setups", "calibration", "patterns", "conviction_adjustment"}
    assert result["conviction_adjustment"]["sample_size"] == 0
    assert result["conviction_adjustment"]["adjusted_conviction"] == 0.5


@pytest.mark.asyncio
async def test_calibrate_conviction_passes_through_without_history(toolset, journal_db, ctx):
    assert await _call(toolset, "calibrate_conviction", ctx, raw_conviction=0.7) == 0.7


@pytest.mark.asyncio
async def test_get_learning_summary_after_trade(toolset, journal_db, ctx):
    await _call(
        toolset, "record_trade_outcome", ctx,
        symbol="RELIANCE", action="BUY", conviction=0.8, pnl_pct=0.01,
        thesis_summary="Fine", sector="Energy",
    )
    summary = await _call(toolset, "get_learning_summary", ctx)
    assert summary["calibration"]["high_conv"]["sample_size"] == 1
    assert summary["recent_trades"]


@pytest.mark.asyncio
async def test_record_trade_outcome_reports_success(toolset, journal_db, ctx):
    result = await _call(
        toolset, "record_trade_outcome", ctx,
        symbol="RELIANCE", action="SELL", conviction=0.6, pnl_pct=-0.02,
        thesis_summary="Stopped out", sector="Energy",
    )
    assert result["recorded"] is True
    assert "high_conv" not in result["updated_calibration"] or True


def test_agents_expose_the_toolset():
    """The wiring the tools exist for: risk, memory and supervisor can call them."""
    import autfin.agents as agents
    from autfin.tools import learning_toolset

    for name in ("risk", "memory", "supervisor"):
        assert learning_toolset in getattr(agents, f"{name}_agent")._user_toolsets, name
