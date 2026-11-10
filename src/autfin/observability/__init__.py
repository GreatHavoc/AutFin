"""Observability module for AutFin — LangSmith tracing, metrics aggregation."""

from src.autfin.observability.handler import (
    get_callback_handler,
    is_langsmith_configured,
)
from src.autfin.observability.metrics import (
    ObservabilityMetrics,
    get_metrics_snapshot,
)
from src.autfin.observability.tracer import (
    AutFinTracer,
    TraceContext,
    get_tracer,
    reset_tracer,
    trace_agent_node,
    trace_llm_call,
)

__all__ = [
    "AutFinTracer",
    "get_tracer",
    "reset_tracer",
    "trace_llm_call",
    "trace_agent_node",
    "TraceContext",
    "get_callback_handler",
    "is_langsmith_configured",
    "ObservabilityMetrics",
    "get_metrics_snapshot",
]
