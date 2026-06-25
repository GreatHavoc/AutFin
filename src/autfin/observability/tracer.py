"""Langfuse tracer for AutFin — LLM call and agent node tracing.

Uses the official Langfuse Python SDK (v3/v4) with get_client() singleton.
For LangGraph auto-tracing, see handler.py (uses langfuse.langchain.CallbackHandler).
This module handles manual tracing for direct LLM calls and custom spans.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

_LANGFUSE_AVAILABLE = False
_get_client = None
_propagate_attributes = None

try:
    from langfuse import get_client as _get_client_fn
    from langfuse import propagate_attributes as _propagate_attributes_fn
    _get_client = _get_client_fn
    _propagate_attributes = _propagate_attributes_fn
    _LANGFUSE_AVAILABLE = True
except ImportError:
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TraceContext:
    trace_id: str
    name: str = ""
    model: str = ""
    total_tokens: int = 0
    total_cost: float = 0.0
    errors: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    started_at: float = field(default_factory=time.monotonic)

    @property
    def duration_ms(self) -> float:
        return (time.monotonic() - self.started_at) * 1000


class AutFinTracer:
    """Tracer using official Langfuse SDK with in-memory fallback."""

    def __init__(self):
        self._use_langfuse = False
        self._client = None
        self._active_traces: dict[str, TraceContext] = {}
        self._trace_history: list[TraceContext] = []
        self._active_spans: dict[str, Any] = {}

        if _LANGFUSE_AVAILABLE:
            try:
                self._client = _get_client()
                if self._client.auth_check():
                    self._use_langfuse = True
                    logger.info("Langfuse tracer initialized via get_client()")
                else:
                    logger.debug("Langfuse auth failed — using in-memory fallback")
            except Exception as e:
                logger.debug("Langfuse init failed: %s — using in-memory fallback", e)

    def start_trace(self, name: str, metadata: dict | None = None) -> str:
        import uuid
        tid = uuid.uuid4().hex
        ctx = TraceContext(trace_id=tid, name=name)
        ctx.metadata.update(metadata or {})
        self._active_traces[tid] = ctx

        if self._use_langfuse and self._client:
            try:
                span = self._client.start_observation(
                    name=name,
                    as_type="span",
                    metadata=metadata or {},
                )
                self._active_spans[tid] = span
            except Exception as e:
                logger.debug("Langfuse start_trace failed: %s", e)

        return tid

    def end_trace(self, tid: str):
        ctx = self._active_traces.pop(tid, None)
        if ctx is None:
            return

        self._trace_history.append(ctx)
        if len(self._trace_history) > 500:
            self._trace_history = self._trace_history[-500:]

        span = self._active_spans.pop(tid, None)
        if span:
            try:
                span.update(output={"tokens": ctx.total_tokens, "cost": ctx.total_cost})
                span.end()
            except Exception:
                pass

        if self._use_langfuse and self._client:
            try:
                self._client.flush()
            except Exception:
                pass

    def log_llm_call(
        self, tid: str, model: str,
        prompt_tokens: int, completion_tokens: int,
        latency_ms: float, cost: float = 0.0,
        error: str | None = None,
    ):
        ctx = self._active_traces.get(tid)
        if ctx:
            ctx.model = model
            ctx.total_tokens += prompt_tokens + completion_tokens
            ctx.total_cost += cost
            if error:
                ctx.errors.append(error)

        if self._use_langfuse and self._client:
            try:
                gen = self._client.start_observation(
                    trace_context={"trace_id": tid} if tid else None,
                    name=f"llm-{model}",
                    as_type="generation",
                    model=model,
                    usage_details={
                        "input": prompt_tokens,
                        "output": completion_tokens,
                    } if prompt_tokens or completion_tokens else None,
                    metadata={"latency_ms": latency_ms, "error": error} if error else None,
                )
                gen.end()
            except Exception as e:
                logger.debug("Langfuse log_llm_call failed: %s", e)

    def log_agent_step(
        self, tid: str, agent_name: str,
        duration_ms: float, success: bool = True,
        metadata: dict | None = None,
    ):
        ctx = self._active_traces.get(tid)
        if ctx and not success:
            ctx.errors.append(agent_name)

        if self._use_langfuse and self._client:
            try:
                span = self._client.start_observation(
                    trace_context={"trace_id": tid} if tid else None,
                    name=f"agent-{agent_name}",
                    as_type="span",
                    metadata={"duration_ms": duration_ms, "success": success, **(metadata or {})},
                )
                span.update(output={"success": success})
                span.end()
            except Exception as e:
                logger.debug("Langfuse log_agent_step failed: %s", e)

    def get_recent_traces(self, limit: int = 20) -> list[dict]:
        return [
            {
                "trace_id": t.trace_id,
                "name": t.name,
                "model": t.model,
                "total_tokens": t.total_tokens,
                "total_cost": t.total_cost,
                "duration_ms": t.duration_ms,
                "error_count": len(t.errors),
            }
            for t in reversed(self._trace_history[-limit:])
        ]

    def get_aggregate_stats(self) -> dict:
        traces = self._trace_history[-100:]
        if not traces:
            return {"total_traces": 0}
        return {
            "total_traces": len(traces),
            "error_traces": sum(1 for t in traces if t.errors),
            "total_tokens": sum(t.total_tokens for t in traces),
            "total_cost": round(sum(t.total_cost for t in traces), 4),
            "avg_duration_ms": round(sum(t.duration_ms for t in traces) / len(traces), 1),
            "models_used": list(set(t.model for t in traces if t.model)),
        }

    def flush(self):
        if self._use_langfuse and self._client:
            try:
                self._client.flush()
            except Exception:
                pass


_tracer_instance: AutFinTracer | None = None


def get_tracer() -> AutFinTracer:
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = AutFinTracer()
    return _tracer_instance


def reset_tracer():
    global _tracer_instance
    _tracer_instance = None


@asynccontextmanager
async def trace_llm_call(model: str, trace_id: str | None = None):
    tracer = get_tracer()
    tid = trace_id or tracer.start_trace(f"llm-{model}")
    start = time.monotonic()
    error_msg = None
    try:
        yield tid
    except Exception as e:
        error_msg = str(e)
        raise
    finally:
        elapsed = (time.monotonic() - start) * 1000
        tracer.log_llm_call(tid, model, 0, 0, elapsed, 0.0, error_msg)


@asynccontextmanager
async def trace_agent_node(agent_name: str, trace_id: str | None = None):
    tracer = get_tracer()
    tid = trace_id or tracer.start_trace(agent_name)
    start = time.monotonic()
    success = True
    try:
        yield tid
    except Exception:
        success = False
        raise
    finally:
        elapsed = (time.monotonic() - start) * 1000
        tracer.log_agent_step(tid, agent_name, elapsed, success=success)
