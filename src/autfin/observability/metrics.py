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
class TradingMetrics:
    """Trading performance quality metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    max_consecutive_losses: int = 0
    conviction_accuracy_sum: float = 0.0
    conviction_samples: int = 0

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def avg_pnl_per_trade(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl_pct / self.total_trades

    @property
    def profit_factor(self) -> float:
        if self.losing_trades == 0:
            return 1.0 if self.winning_trades > 0 else 0.0
        gross_wins = self.total_pnl_pct if self.total_pnl_pct > 0 else 0
        gross_losses = abs(self.total_pnl_pct) if self.total_pnl_pct < 0 else 1
        return round(gross_wins / gross_losses, 2) if gross_losses > 0 else 0.0

    @property
    def avg_conviction_accuracy(self) -> float:
        if self.conviction_samples == 0:
            return 0.0
        return round(self.conviction_accuracy_sum / self.conviction_samples, 4)


@dataclass
class LearningMetrics:
    """Self-learning quality metrics."""
    calibration_accuracy: float = 0.0
    pattern_detection_count: int = 0
    similar_setup_hit_rate: float = 0.0
    conviction_overconfidence: float = 0.0
    recent_win_rate_trend: str = "stable"


@dataclass
class ObservabilityMetrics:
    agents: Dict[str, AgentMetrics] = field(default_factory=dict)
    total_llm_tokens: int = 0
    total_llm_cost: float = 0.0
    total_traces: int = 0
    error_traces: int = 0
    graph_runs: int = 0
    graph_errors: int = 0
    trading: TradingMetrics = field(default_factory=TradingMetrics)
    learning: LearningMetrics = field(default_factory=LearningMetrics)
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

    def record_trade(self, pnl_pct: float, conviction: float = 0.5):
        """Record a completed trade for quality metrics."""
        self.trading.total_trades += 1
        self.trading.total_pnl_pct += pnl_pct
        self.trading.conviction_accuracy_sum += abs(pnl_pct)
        self.trading.conviction_samples += 1

        if pnl_pct > 0:
            self.trading.winning_trades += 1
        elif pnl_pct < 0:
            self.trading.losing_trades += 1

        # Update max drawdown
        if pnl_pct < 0 and abs(pnl_pct) > self.trading.max_drawdown_pct:
            self.trading.max_drawdown_pct = abs(pnl_pct)

    def record_learning_quality(self, calibration: dict, patterns: list):
        """Record learning system quality."""
        if calibration:
            deltas = [abs(v.get("delta", 0)) for v in calibration.values() if isinstance(v, dict) and v.get("sample_size", 0) > 0]
            if deltas:
                self.learning.calibration_accuracy = 1.0 - (sum(deltas) / len(deltas))
                self.learning.conviction_overconfidence = max(deltas) if deltas else 0.0
        self.learning.pattern_detection_count = len(patterns)

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
            "trading": {
                "total_trades": self.trading.total_trades,
                "win_rate": round(self.trading.win_rate, 4),
                "avg_pnl_pct": round(self.trading.avg_pnl_per_trade, 4),
                "profit_factor": self.trading.profit_factor,
                "max_drawdown_pct": round(self.trading.max_drawdown_pct, 2),
            },
            "learning": {
                "calibration_accuracy": round(self.learning.calibration_accuracy, 3),
                "pattern_count": self.learning.pattern_detection_count,
                "win_rate_trend": self.learning.recent_win_rate_trend,
            },
            "last_updated": self.last_updated,
        }


_metrics = ObservabilityMetrics()


def get_metrics_snapshot() -> Dict[str, Any]:
    return _metrics.snapshot()


def get_metrics():
    """Get raw metrics object (for advanced callers)."""
    return _metrics


def record_agent(name: str, duration_ms: float, success: bool, error: Optional[str] = None):
    _metrics.record_agent_call(name, duration_ms, success, error)


def record_llm(tokens: int, cost: float = 0.0):
    _metrics.record_llm_usage(tokens, cost)


def record_graph_run(success: bool):
    _metrics.record_graph_run(success)


def record_trace(has_errors: bool):
    _metrics.record_trace(has_errors)


def record_trade(pnl_pct: float, conviction: float = 0.5):
    """Record a completed trade for quality metrics."""
    _metrics.record_trade(pnl_pct, conviction)


def record_learning_quality(calibration: dict, patterns: list):
    """Record learning system quality."""
    _metrics.record_learning_quality(calibration, patterns)
