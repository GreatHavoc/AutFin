"""Aggregated observability metrics for the dashboard."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class AgentMetrics:
    name: str
    total_calls: int = 0
    total_errors: int = 0
    total_duration_ms: float = 0.0
    last_error: Optional[str] = None
    last_success_at: Optional[str] = None

    @property
    def avg_duration_ms(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_duration_ms / self.total_calls

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 1.0
        return (self.total_calls - self.total_errors) / self.total_calls


@dataclass
class ObservabilityMetrics:
    agents: Dict[str, AgentMetrics] = field(default_factory=dict)
    total_llm_tokens: int = 0
    total_llm_cost: float = 0.0
    total_traces: int = 0
    error_traces: int = 0
    graph_runs: int = 0
    graph_errors: int = 0
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def record_agent_call(self, name: str, duration_ms: float, success: bool, error: Optional[str] = None):
        if name not in self.agents:
            self.agents[name] = AgentMetrics(name=name)
        m = self.agents[name]
        m.total_calls += 1
        m.total_duration_ms += duration_ms
        if not success:
            m.total_errors += 1
            m.last_error = error
        else:
            m.last_success_at = datetime.now(timezone.utc).isoformat()

    def record_llm_usage(self, tokens: int, cost: float = 0.0):
        self.total_llm_tokens += tokens
        self.total_llm_cost += cost

    def record_trace(self, has_errors: bool):
        self.total_traces += 1
        if has_errors:
            self.error_traces += 1

    def record_graph_run(self, success: bool):
        self.graph_runs += 1
        if not success:
            self.graph_errors += 1

    @property
    def error_rate(self) -> float:
        if self.total_traces == 0:
            return 0.0
        return round(self.error_traces / self.total_traces, 4)

    @property
    def graph_error_rate(self) -> float:
        if self.graph_runs == 0:
            return 0.0
        return round(self.graph_errors / self.graph_runs, 4)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "agent_metrics": {
                name: {
                    "calls": m.total_calls,
                    "errors": m.total_errors,
                    "avg_duration_ms": round(m.avg_duration_ms, 1),
                    "success_rate": round(m.success_rate, 4),
                    "last_error": m.last_error,
                }
                for name, m in self.agents.items()
            },
            "llm": {
                "total_tokens": self.total_llm_tokens,
                "total_cost": round(self.total_llm_cost, 6),
            },
            "traces": {
                "total": self.total_traces,
                "errors": self.error_traces,
                "error_rate": self.error_rate,
            },
            "graphs": {
                "runs": self.graph_runs,
                "errors": self.graph_errors,
                "error_rate": self.graph_error_rate,
            },
            "last_updated": self.last_updated,
        }


_metrics = ObservabilityMetrics()


def get_metrics_snapshot() -> Dict[str, Any]:
    return _metrics.snapshot()


def record_agent(name: str, duration_ms: float, success: bool, error: Optional[str] = None):
    _metrics.record_agent_call(name, duration_ms, success, error)


def record_llm(tokens: int, cost: float = 0.0):
    _metrics.record_llm_usage(tokens, cost)


def record_graph_run(success: bool):
    _metrics.record_graph_run(success)


def record_trace(has_errors: bool):
    _metrics.record_trace(has_errors)
