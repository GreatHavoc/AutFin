"""Tests for B2: shared spend ceilings."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from decimal import Decimal

import pytest


def _agent(limits, deps_type=None):
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    return Agent(TestModel(), deps_type=deps_type or object, capabilities=[limits])


def test_one_shared_instance_across_every_agent():
    """The default store is per capability: one limiter per agent would give
    every agent its own daily ceiling — nine agents, nine times the budget."""
    from pydantic_ai_harness import SpendLimits

    import autfin.agents as agents
    from autfin.agents.spend import get_spend_limits

    shared = get_spend_limits()
    for name in (
        "market_data", "analysis", "research", "risk", "debate",
        "conference", "memory", "quant", "supervisor",
    ):
        found = [
            c for c in getattr(agents, f"{name}_agent").root_capability.capabilities
            if isinstance(c, SpendLimits)
        ]
        assert len(found) == 1, name
        assert found[0] is shared, name


def test_daily_budget_refuses_the_next_request():
    from pydantic_ai_harness.spend import Budget, SpendLimitExceeded, SpendLimits

    limits = SpendLimits(
        budgets=[Budget(usd=Decimal("0.01"), window="day", name="daily")],
        price=lambda response: Decimal("0.01"),
    )
    _agent(limits).run_sync("first")

    with pytest.raises(SpendLimitExceeded):
        _agent(limits).run_sync("second")


def test_tenant_budgets_count_separately():
    """A per-user_id partition: one tenant exhausting the ceiling does not stop
    another, which a single shared counter would do."""
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel
    from pydantic_ai_harness.spend import Budget, SpendLimitExceeded, SpendLimits

    @dataclass
    class Deps:
        tenant_id: str

    limits = SpendLimits[Deps](
        budgets=[
            Budget(
                usd=Decimal("0.01"),
                window="day",
                scope=lambda ctx: ctx.deps.tenant_id,
                name="tenant",
            )
        ],
        price=lambda response: Decimal("0.01"),
    )
    agent = Agent(TestModel(), deps_type=Deps, capabilities=[limits])

    agent.run_sync("tick", deps=Deps(tenant_id="acme"))
    with pytest.raises(SpendLimitExceeded):
        agent.run_sync("tick", deps=Deps(tenant_id="acme"))
    agent.run_sync("tick", deps=Deps(tenant_id="globex"))  # own counter, still runs


def test_exhausted_is_an_admission_check():
    from pydantic_ai_harness.spend import Budget, SpendLimits

    limits = SpendLimits(
        budgets=[Budget(usd=Decimal("0.01"), window="day", name="daily")],
        price=lambda response: Decimal("0.01"),
    )
    agent = _agent(limits)

    assert asyncio.run(limits.exhausted()) is False
    agent.run_sync("spend it")
    assert asyncio.run(limits.exhausted()) is True


def test_ceilings_are_env_overridable(monkeypatch):
    from decimal import Decimal as D

    monkeypatch.setenv("AUTFIN_SPEND_USD_DAY", "0.25")
    monkeypatch.setenv("AUTFIN_SPEND_USD_REQUEST", "0")

    import importlib

    import autfin.agents.spend as spend

    importlib.reload(spend)
    budgets = {b.name: b for b in spend.get_spend_limits().budgets}
    assert budgets["autfin-day"].usd == D("0.25")

    importlib.reload(spend)  # restore module state for other tests
