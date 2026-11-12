"""Evaluation suite: datasets, evaluators and the pydantic-evals runner.

Serialise and reload with the native API:
    Dataset.to_file(path) / Dataset.from_file(path)
Run an experiment with the native API:
    report = dataset.evaluate_sync(task_fn)
Runs are deterministic except where LLM judges are enabled.
"""
from __future__ import annotations

from autfin.evals.dataset import (
    create_agent_specific_dataset,
    create_autfin_dataset,
    create_test_deps,
)

__all__ = [
    "create_agent_specific_dataset",
    "create_autfin_dataset",
    "create_test_deps",
]
