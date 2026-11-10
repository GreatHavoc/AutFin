"""Evaluation suite: datasets, evaluators and the pydantic-evals runner.

Consumed by `autfin.validation` (gate 3) via `run_evaluations`. Runs are
deterministic except where LLM judges are enabled.
"""
from __future__ import annotations

from autfin.evals.dataset import (
    create_agent_specific_dataset,
    create_autfin_dataset,
    create_test_deps,
    load_dataset,
    run_evaluations,
    save_dataset,
)

__all__ = [
    "create_agent_specific_dataset",
    "create_autfin_dataset",
    "create_test_deps",
    "load_dataset",
    "run_evaluations",
    "save_dataset",
]
