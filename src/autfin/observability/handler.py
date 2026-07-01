"""LangSmith tracing handler for AutFin.

LangSmith auto-traces LangGraph node executions when LANGSMITH_TRACING=true
is set in the environment. This module provides configuration checks and
explicit tracing utilities.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _check_langsmith() -> bool:
    """Check if LangSmith tracing is properly configured via env vars."""
    api_key = os.environ.get("LANGSMITH_API_KEY", "")
    tracing = os.environ.get("LANGSMITH_TRACING", "").lower()
    return bool(api_key and tracing in ("true", "1", "yes"))


def is_langsmith_configured() -> bool:
    """Check if LangSmith tracing is properly configured via env vars."""
    return _check_langsmith()


def get_callback_handler(trace_id: Optional[str] = None) -> Optional[object]:
    """Get LangSmith callback handler for LangGraph.

    With LANGSMITH_TRACING=true, LangGraph auto-traces all node executions.
    This function returns None to signal that auto-tracing is active.

    Returns:
        None — LangSmith auto-tracing handles LangGraph via env vars.
    """
    if not _check_langsmith():
        logger.debug("LangSmith not configured — tracing disabled")
        return None

    # LangGraph auto-traces when LANGSMITH_TRACING=true is set.
    # No explicit callback handler needed.
    return None
