"""Tool catalogue guards: reachability of the agent modules, the risk sizing path.

The `risk_manager` ModuleNotFoundError only appeared when the risk agent's tool
actually ran, so the guard is importing every agent module, not the package.
"""
from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

import autfin
import autfin.agents as agents
from autfin.tools import technical_toolset


def _agent_modules() -> list[str]:
    """Dotted names of every module under autfin.agents."""
    base = Path(autfin.__file__).parent          # src/autfin
    agents_dir = base / "agents"
    names = []
    for path in sorted(agents_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(base).with_suffix("")
        names.append(f"autfin.{'.'.join(rel.parts)}")
    return names


def _run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("module", _agent_modules())
def test_every_agent_module_imports(module):
    importlib.import_module(module)


@pytest.mark.parametrize(
    "name", ["market_data", "analysis", "research", "risk", "quant", "supervisor"]
)
def test_agent_constructs_with_a_resolving_toolset_graph(name):
    agent = getattr(agents, f"{name}_agent")
    caps = agent.root_capability.capabilities
    assert caps, name


def test_risk_toolset_is_not_double_mounted():
    """risk.py and quant.py mounted risk_toolset AND technical_toolset, but
    risk_toolset already bundles technical, so every technical tool was in the
    agent's toolset list twice."""
    for name in ("risk", "quant"):
        mounted = getattr(agents, f"{name}_agent")._user_toolsets
        assert id(technical_toolset) not in {id(t) for t in mounted}, name


def test_risk_agent_sizes_a_trade_end_to_end():
    """The tool that was dead: assess_trade -> autfin.risk.manager.assess_risk."""
    from autfin.types import AgentDeps

    agent = agents.risk_agent
    model = TestModel(call_tools=["assess_trade"])
    with agent.override(model=model):
        result = agent.run_sync(
            "assess RELIANCE", deps=AgentDeps(user_id="t", capital=1_000_000.0)
        )
    assert result.output.verdict in {"approve", "reduce", "reject", "conditional"}


# ── assess_risk itself ─────────────────────────────────────────────────────


def _assess(**kw) -> object:
    from autfin.risk.manager import assess_risk

    base = dict(
        symbol="RELIANCE",
        action="BUY",
        entry_price=2500.0,
        conviction=0.8,
        capital=1_000_000.0,
        portfolio=[],
        max_position_pct=0.30,
        sector_limit_pct=0.30,
        max_drawdown_pct=0.15,
    )
    base.update(kw)
    return _run(assess_risk(**base))


def test_sizes_within_limits_and_reports_risk_reward():
    out = _assess()
    assert out.position_size > 0
    assert out.max_position_pct <= 0.30
    assert out.stop_loss < 2500.0 < out.target
    assert out.risk_reward >= 1.0


def test_low_conviction_is_rejected():
    assert _assess(conviction=0.2).verdict == "reject"


def test_symbol_over_limit_is_rejected_before_sizing():
    from autfin.types import Holding

    holdings = [
        Holding(tradingsymbol="RELIANCE", qty=1000, avg_price=2500, ltp=2500, pnl=0, sector="Energy")
    ]
    out = _assess(portfolio=holdings)
    assert out.verdict == "reject"
    assert out.position_size == 0


def test_sector_limit_makes_it_conditional():
    from autfin.types import Holding

    holdings = [
        Holding(tradingsymbol="TCS", qty=100, avg_price=2500, ltp=2500, pnl=0, sector="IT")
    ]
    # sector comes from analysis_data; 250k of IT on 500k capital = 50% > 30% limit
    out = _assess(
        symbol="INFY",
        analysis_data={"sector": "IT"},
        portfolio=holdings,
        capital=500_000.0,
    )
    assert out.verdict == "conditional"
    assert any("sector" in c.lower() for c in out.conditions)
