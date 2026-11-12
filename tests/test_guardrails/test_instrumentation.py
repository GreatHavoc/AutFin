"""Tests for the per-agent Instrumentation capability (B7).

`run.py` calls `logfire.configure()` + `logfire.instrument_pydantic_ai()` at
import, so the daemon emits spans — and nothing else does. The capability now
rides on every agent from `create_agent()`, so any entrypoint (or a test) gets
the same spans once a provider is configured.

`capfire` is logfire's own test fixture: it configures the global provider with
an in-memory exporter and returns the spans as dicts. A hand-rolled
`TracerProvider` cannot be used here — installing a global tracer provider is
refused once one exists, and a test in this repo imports `run.py`, which
configures logfire at import time.
"""
from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.models.test import TestModel


def _capabilities_of(agent):
    caps = list(agent.root_capability.capabilities)
    flat: list[object] = []
    while caps:
        cap = caps.pop()
        flat.append(cap)
        caps.extend(getattr(cap, "capabilities", []) or [])
    return flat


def _span_names(capfire):
    return {span["name"] for span in capfire.exporter.exported_spans_as_dict()}


def test_every_agent_carries_an_instrumentation_capability():
    import autfin.agents as agents

    names = (
        "market_data", "analysis", "research", "risk", "debate",
        "conference", "memory", "quant", "supervisor",
    )
    for name in names:
        agent = getattr(agents, f"{name}_agent")
        assert any(isinstance(c, Instrumentation) for c in _capabilities_of(agent)), name


def test_spans_are_emitted_on_an_agent_run(capfire):
    agent = Agent(TestModel(), name="probe", capabilities=[Instrumentation()])

    agent.run_sync("hello")

    names = _span_names(capfire)
    assert any(n.startswith("invoke_agent probe") for n in names), names
    assert any(n.startswith("chat") for n in names), names


def test_a_factory_built_agent_emits_spans_too(capfire):
    """The capability is attached in create_agent(), not bolted on by callers."""
    from pydantic import BaseModel

    from autfin.agents.base import create_agent

    class Probe(BaseModel):
        ok: bool = True

    agent = create_agent(
        name="probe_agent", output_type=Probe, instructions="probe", model_type="fast"
    )
    with agent.override(model=TestModel()):
        agent.run_sync("hello")

    names = _span_names(capfire)
    assert any(n.startswith("invoke_agent probe_agent") for n in names), names
