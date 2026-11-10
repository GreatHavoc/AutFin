"""A/B testing framework for trading strategies.

Provides experiment management with deterministic variant assignment,
outcome tracking, and statistical result computation.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    """An A/B test experiment."""
    experiment_id: str
    name: str
    variant_a: str
    variant_b: str
    start_date: str = ""
    end_date: str = ""
    sample_size_a: int = 0
    sample_size_b: int = 0
    outcome_sum_a: float = 0.0
    outcome_sum_b: float = 0.0
    is_active: bool = True
    description: str = ""

    @property
    def conversion_a(self) -> float:
        return self.outcome_sum_a / max(self.sample_size_a, 1)

    @property
    def conversion_b(self) -> float:
        return self.outcome_sum_b / max(self.sample_size_b, 1)


@dataclass
class ExperimentResult:
    """Statistical result of an A/B test."""
    experiment_id: str
    winner: str          # "A", "B", or "inconclusive"
    confidence: float    # 0-1
    lift: float          # % improvement of winner over loser
    sample_size: int
    sample_size_a: int
    sample_size_b: int
    recommendation: str


class ABTestingFramework:
    """Manages A/B testing for trading strategies."""

    MIN_SAMPLE_SIZE = 20

    def __init__(self):
        self._experiments: Dict[str, Experiment] = {}

    def create_experiment(
        self,
        name: str,
        variant_a: str,
        variant_b: str,
        experiment_id: Optional[str] = None,
        description: str = "",
    ) -> Experiment:
        """Create a new A/B test experiment.

        Args:
            name: Human-readable name
            variant_a: Control variant name
            variant_b: Treatment variant name
            experiment_id: Optional ID (auto-generated if None)
            description: Optional description

        Returns:
            Created experiment
        """
        if experiment_id is None:
            seed = f"{name}{datetime.now(timezone.utc).isoformat()}"
            experiment_id = hashlib.md5(seed.encode()).hexdigest()[:8]

        exp = Experiment(
            experiment_id=experiment_id,
            name=name,
            variant_a=variant_a,
            variant_b=variant_b,
            start_date=datetime.now(timezone.utc).isoformat(),
            description=description,
        )
        self._experiments[experiment_id] = exp
        logger.info("Created experiment %s: %s vs %s", experiment_id, variant_a, variant_b)
        return exp

    def assign_variant(self, experiment_id: str, subject: str) -> str:
        """Deterministically assign a subject to variant A or B.

        Uses hash-based assignment for consistency — same subject always
        gets the same variant for a given experiment.

        Args:
            experiment_id: Experiment ID
            subject: Unique subject identifier (e.g., stock symbol)

        Returns:
            "A" or "B"
        """
        exp = self._experiments.get(experiment_id)
        if not exp:
            return "A"

        # Use sha256 and take last byte (better distribution than md5 first byte)
        seed = f"{experiment_id}:{subject}".encode()
        h = hashlib.sha256(seed).digest()
        # Use last byte for better distribution
        return "A" if h[-1] % 2 == 0 else "B"

    def record_outcome(
        self,
        experiment_id: str,
        subject: str,
        outcome: float,
    ) -> str:
        """Record the outcome for a variant assignment.

        Args:
            experiment_id: Experiment ID
            subject: Subject that was assigned to a variant
            outcome: Outcome value (e.g., P&L percentage)

        Returns:
            The variant that was assigned ("A" or "B")
        """
        exp = self._experiments.get(experiment_id)
        if not exp:
            return "A"

        variant = self.assign_variant(experiment_id, subject)

        if variant == "A":
            exp.sample_size_a += 1
            exp.outcome_sum_a += outcome
        else:
            exp.sample_size_b += 1
            exp.outcome_sum_b += outcome

        return variant

    def compute_results(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Compute statistical results for an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            ExperimentResult or None if experiment not found
        """
        exp = self._experiments.get(experiment_id)
        if not exp:
            return None

        total = exp.sample_size_a + exp.sample_size_b
        if total < self.MIN_SAMPLE_SIZE:
            return ExperimentResult(
                experiment_id=experiment_id,
                winner="inconclusive",
                confidence=0.0,
                lift=0.0,
                sample_size=total,
                sample_size_a=exp.sample_size_a,
                sample_size_b=exp.sample_size_b,
                recommendation=f"Need at least {self.MIN_SAMPLE_SIZE} samples, have {total}",
            )

        a, b = exp.conversion_a, exp.conversion_b

        if a > b:
            winner = "A"
            loser_val = b if b != 0 else 0.01
            lift = ((a - b) / abs(loser_val)) * 100
        elif b > a:
            winner = "B"
            loser_val = a if a != 0 else 0.01
            lift = ((b - a) / abs(loser_val)) * 100
        else:
            winner = "inconclusive"
            lift = 0.0

        # Confidence based on sample size (simplified)
        confidence = min(0.95, 0.5 + (total / 200))

        recommendation = (
            f"Variant {winner} wins with {confidence:.0%} confidence (lift: {lift:.1f}%)"
            if winner != "inconclusive"
            else "Insufficient difference between variants"
        )

        return ExperimentResult(
            experiment_id=experiment_id,
            winner=winner,
            confidence=round(confidence, 3),
            lift=round(lift, 2),
            sample_size=total,
            sample_size_a=exp.sample_size_a,
            sample_size_b=exp.sample_size_b,
            recommendation=recommendation,
        )

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID."""
        return self._experiments.get(experiment_id)

    def list_experiments(self) -> List[Experiment]:
        """List all experiments."""
        return list(self._experiments.values())


# Singleton instance
_framework: Optional[ABTestingFramework] = None


def get_ab_framework() -> ABTestingFramework:
    """Get or create the singleton A/B testing framework."""
    global _framework
    if _framework is None:
        _framework = ABTestingFramework()
    return _framework


def reset_ab_framework():
    """Reset the singleton (for testing)."""
    global _framework
    _framework = None
