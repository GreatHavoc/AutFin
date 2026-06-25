"""Langfuse callback handler for AutFin — wraps the official langfuse.langchain.CallbackHandler.

The official CallbackHandler auto-traces all LangGraph node executions,
LLM calls, tool calls, and retriever invocations. No custom reimplementation needed.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_langfuse_handler_class = None

try:
    from langfuse.langchain import CallbackHandler as _LangfuseCallbackHandler
    _langfuse_handler_class = _LangfuseCallbackHandler
except ImportError:
    logger.debug("langfuse.langchain.CallbackHandler not available — tracing disabled")


def get_callback_handler(trace_id: Optional[str] = None) -> Optional[object]:
    """Get the official Langfuse CallbackHandler.

    Args:
        trace_id: Optional trace_id to link this handler to a specific trace.
                  If provided, all observations will be grouped under this trace.

    Returns:
        Langfuse CallbackHandler instance, or None if langfuse is not configured.
    """
    if _langfuse_handler_class is None:
        return None

    try:
        from langfuse import get_client
        langfuse = get_client()

        if not langfuse.auth_check():
            logger.debug("Langfuse auth failed — tracing disabled")
            return None

        handler = _langfuse_handler_class()
        return handler
    except Exception as e:
        logger.debug("Failed to create Langfuse CallbackHandler: %s", e)
        return None
