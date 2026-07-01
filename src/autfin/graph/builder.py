"""
Graph builder — clean compilation with checkpointer and store support.

v5.3: Accepts optional LangGraph Store for long-term memory.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def build_and_compile_graph(
    checkpointer=None,
    store=None,
    attach_observability: bool = True,
) -> Any:
    """
    Build and compile the supervisor graph.

    Args:
        checkpointer: Optional LangGraph checkpointer for crash recovery.
        store: Optional LangGraph Store for long-term memory.
        attach_observability: Whether to attach LangSmith callback handler.

    Returns:
        Compiled runnable graph.
    """
    from src.autfin.graph.supervisor import build_supervisor_graph

    builder = build_supervisor_graph()

    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if store is not None:
        compile_kwargs["store"] = store

    compiled = builder.compile(**compile_kwargs)

    # Attach observability handler (LangSmith auto-tracing)
    if attach_observability:
        try:
            from src.autfin.observability.handler import get_callback_handler

            handler = get_callback_handler()
            if handler is not None:
                compiled = compiled.with_config({"callbacks": [handler]})
        except Exception:
            logger.debug("Failed to attach observability handler", exc_info=True)

    return compiled
