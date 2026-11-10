"""Evaluation suite: datasets, evaluators and the pydantic-evals runner.

Run an experiment with the native API:
    report = dataset.evaluate_sync(task_fn)
Runs are deterministic except where LLM judges are enabled.
"""
from __future__ import annotations

from autfin.evals.dataset import (
    create_agent_specific_dataset,
    create_autfin_dataset,
    create_test_deps,
    load_dataset,
    save_dataset,
)

__all__ = [
    "create_agent_specific_dataset",
    "create_autfin_dataset",
    "create_test_deps",
    "load_dataset",
    "save_dataset",
]
