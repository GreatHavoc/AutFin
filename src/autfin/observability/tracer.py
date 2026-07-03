"""LangSmith tracer for AutFin — LLM call and agent node tracing.

Uses @traceable decorator and langsmith.Client for explicit trace management.
For LangGraph auto-tracing, see handler.py (LANGSMITH_TRACING=true handles it).
This module handles manual tracing for direct LLM calls and custom spans.
Persists traces to SQLite for durability beyond in-memory cap.

Key tracing setup:
- wrap_openai() in llm/client.py wraps the OpenAI client for automatic LLM tracing
- @traceable decorator on tool functions creates nested spans
- LANGSMITH_TRACING=true enables LangGraph auto-tracing for node executions
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_LANGSMITH_AVAILABLE = False
_langsmith_client = None
_traceable = None
_wrap_openai = None


def _init_langsmith():
    """Initialize LangSmith client lazily (after dotenv loads)."""
    global _LANGSMITH_AVAILABLE, _langsmith_client, _traceable, _wrap_openai
    if _langsmith_client is not None:
        return

    try:
        import os
        if not os.environ.get("LANGSMITH_API_KEY"):
            return

        from langsmith import Client as _LangsmithClient
        from langsmith import traceable as _traceable_fn

        _langsmith_client = _LangsmithClient()
        _traceable = _traceable_fn
        _LANGSMITH_AVAILABLE = True

        # Import wrap_openai for LLM call tracing
        try:
            from langsmith.wrappers import wrap_openai as _wrap_openai_fn
            _wrap_openai = _wrap_openai_fn
        except ImportError:
            pass

    except ImportError:
        pass
    except Exception:
        pass

# SQLite persistence path
_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "autfin.db"


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
    ended_at: Optional[float] = None

    @property
    def duration_ms(self) -> float:
        end = self.ended_at or time.monotonic()
        return (end - self.started_at) * 1000

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "model": self.model,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "errors": self.errors,
            "metadata": self.metadata,
            "duration_ms": self.duration_ms,
        }


class AutFinTracer:
    """Tracer using LangSmith SDK with in-memory + SQLite persistence."""

    def __init__(self, persist_to_db: bool = True):
        _init_langsmith()
        self._use_langsmith = _LANGSMITH_AVAILABLE
        self._client = _langsmith_client
        self._active_traces: dict[str, TraceContext] = {}
        self._trace_history: list[TraceContext] = []
        self._active_runs: dict[str, Any] = {}
        self._persist_to_db = persist_to_db
        self._max_in_memory = 500

        if self._use_langsmith:
            logger.info("LangSmith tracer initialized")

        # Ensure trace_history table exists
        if self._persist_to_db:
            self._ensure_table()

    def _ensure_table(self):
        """Create trace_history table if it doesn't exist."""
        try:
            from src.autfin.memory.connection_pool import get_db_pool
            pool = get_db_pool(_DB_PATH)
            with pool.connection() as conn:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS trace_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trace_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        model TEXT,
                        total_tokens INTEGER DEFAULT 0,
                        total_cost REAL DEFAULT 0.0,
                        errors_json TEXT,
                        metadata_json TEXT,
                        duration_ms REAL DEFAULT 0.0,
                        created_at TEXT DEFAULT (datetime('now'))
                    )"""
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trace_id ON trace_history(trace_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_trace_created ON trace_history(created_at)"
                )
        except Exception as e:
            logger.debug("Failed to create trace_history table: %s", e)

    def _persist_trace(self, ctx: TraceContext):
        """Persist a completed trace to SQLite."""
        if not self._persist_to_db:
            return
        try:
            from src.autfin.memory.connection_pool import get_db_pool
            pool = get_db_pool(_DB_PATH)
            with pool.connection() as conn:
                conn.execute(
                    """INSERT INTO trace_history
                    (trace_id, name, model, total_tokens, total_cost, errors_json, metadata_json, duration_ms)
                    VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        ctx.trace_id,
                        ctx.name,
                        ctx.model,
                        ctx.total_tokens,
                        ctx.total_cost,
                        json.dumps(ctx.errors) if ctx.errors else None,
                        json.dumps(ctx.metadata) if ctx.metadata else None,
                        ctx.duration_ms,
                    ),
                )
        except Exception as e:
            logger.debug("Failed to persist trace: %s", e)

    def _load_recent_traces_from_db(self, limit: int = 100):
        """Load recent traces from SQLite on startup."""
        if not self._persist_to_db:
            return []
        try:
            from src.autfin.memory.connection_pool import get_db_pool
            pool = get_db_pool(_DB_PATH)
            with pool.connection() as conn:
                rows = conn.execute(
                    """SELECT trace_id, name, model, total_tokens, total_cost,
                    errors_json, metadata_json, duration_ms, created_at
                    FROM trace_history
                    ORDER BY created_at DESC LIMIT ?""",
                    (limit,),
                ).fetchall()

            traces = []
            for row in rows:
                trace_id, name, model, tokens, cost, errors_json, meta_json, dur, created = row
                traces.append({
                    "trace_id": trace_id,
                    "name": name,
                    "model": model or "",
                    "total_tokens": tokens or 0,
                    "total_cost": cost or 0.0,
                    "duration_ms": dur or 0.0,
                    "error_count": len(json.loads(errors_json)) if errors_json else 0,
                })
            return traces
        except Exception as e:
            logger.debug("Failed to load traces from DB: %s", e)
            return []

    def start_trace(self, name: str, metadata: dict | None = None) -> str:
        import uuid
        tid = uuid.uuid4().hex
        ctx = TraceContext(trace_id=tid, name=name)
        ctx.metadata.update(metadata or {})
        self._active_traces[tid] = ctx

        if self._use_langsmith and self._client:
            try:
                run = self._client.create_run(
                    name=name,
                    run_type="chain",
                    extra={"metadata": metadata or {}},
                )
                self._active_runs[tid] = run
            except Exception as e:
                logger.debug("LangSmith start_trace failed: %s", e)

        return tid

    def end_trace(self, tid: str):
        ctx = self._active_traces.pop(tid, None)
        if ctx is None:
            return

        ctx.ended_at = time.monotonic()

        self._trace_history.append(ctx)
        if len(self._trace_history) > self._max_in_memory:
            self._trace_history = self._trace_history[-self._max_in_memory:]

        # Persist to SQLite for durability
        self._persist_trace(ctx)

        run = self._active_runs.pop(tid, None)
        if run and self._client:
            try:
                self._client.update_run(
                    run_id=run.id,
                    extra={
                        "tokens": ctx.total_tokens,
                        "cost": ctx.total_cost,
                    },
                    end_time=datetime.now(timezone.utc),
                )
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

        # Note: LLM calls are automatically traced by wrap_openai() in llm/client.py
        # This method only records metrics for local persistence
        if self._use_langsmith and self._client:
            try:
                # Create a child run under the active trace
                parent_run = self._active_runs.get(tid)
                self._client.create_run(
                    name=f"llm-{model}",
                    run_type="llm",
                    parent_run_id=parent_run.id if parent_run else None,
                    inputs={"model": model},
                    outputs={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "latency_ms": latency_ms,
                        "error": error,
                    },
                )
            except Exception as e:
                logger.debug("LangSmith log_llm_call failed: %s", e)

    def log_agent_step(
        self, tid: str, agent_name: str,
        duration_ms: float, success: bool = True,
        metadata: dict | None = None,
    ):
        ctx = self._active_traces.get(tid)
        if ctx and not success:
            ctx.errors.append(agent_name)

        # Note: Agent tool calls are automatically traced by @traceable decorator
        # This method only records metrics for local persistence
        if self._use_langsmith and self._client:
            try:
                # Create a child run under the active trace
                parent_run = self._active_runs.get(tid)
                self._client.create_run(
                    name=f"agent-{agent_name}",
                    run_type="chain",
                    parent_run_id=parent_run.id if parent_run else None,
                    inputs={"agent": agent_name},
                    outputs={
                        "duration_ms": duration_ms,
                        "success": success,
                        **(metadata or {}),
                    },
                )
            except Exception as e:
                logger.debug("LangSmith log_agent_step failed: %s", e)

    def get_recent_traces(self, limit: int = 20) -> list[dict]:
        # Combine in-memory and DB traces
        in_memory = [
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

        if len(in_memory) >= limit:
            return in_memory[:limit]

        # Supplement from DB if needed
        db_traces = self._load_recent_traces_from_db(limit - len(in_memory))
        return in_memory + db_traces

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
        if self._use_langsmith and self._client:
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
