"""Observability for AutFin.

The hand-rolled metrics recorder, SQLite trace store and RORO SQL derivation
were removed: `RunUsage` on every AgentRunResult already carries cost, tokens,
requests and tool_calls, and the `Instrumentation` capability emits the OTel
spans Logfire stores. What is left is the LangSmith callback configuration.
"""
from __future__ import annotations

from autfin.observability.handler import (
    get_callback_handler,
    is_langsmith_configured,
)

__all__ = [
    "get_callback_handler",
    "is_langsmith_configured",
]
