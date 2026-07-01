"""
LangGraph Store — cross-session long-term memory.

The store passed to ``graph.compile(store=...)`` is the raw LangGraph
``BaseStore`` (InMemoryStore or AsyncSqliteStore). Nodes access it via
``config["store"]``.

Namespaces:
- ("agent_instructions",)  — procedural memory (agent prefs, rules)
- ("trade_history",)        — episodic memory (past trades, outcomes)
- ("user_preferences",)     — semantic memory (risk tolerance, sectors)
- ("research_cache",)       — research results with TTL
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Namespace constants
# ---------------------------------------------------------------------------

NS_AGENT_INSTRUCTIONS = ("agent_instructions",)
NS_TRADE_HISTORY = ("trade_history",)
NS_USER_PREFERENCES = ("user_preferences",)
NS_RESEARCH_CACHE = ("research_cache",)

# ---------------------------------------------------------------------------
# Store creation
# ---------------------------------------------------------------------------


def create_langgraph_store(mode: Optional[str] = None):
    """
    Create the LangGraph Store instance for graph-level long-term memory.

    This is the store passed to ``graph.compile(store=...)``.

    Args:
        mode: "in-memory" (default) or "persistent" (SQLite)

    Returns:
        LangGraph BaseStore instance
    """
    from langgraph.store.memory import InMemoryStore

    if mode is None:
        mode = os.getenv("AUTFIN_STORE_MODE", "in-memory")

    if mode == "persistent":
        try:
            from langgraph.store.sqlite import AsyncSqliteStore
            import aiosqlite

            db_path = os.getenv("AUTFIN_STORE_DB_PATH", "data/autfin_store.db")
            from pathlib import Path
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

            # Must be called within a running event loop
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # We're inside an async context — use connection factory
                import functools
                conn_factory = functools.partial(
                    aiosqlite.connect, db_path, isolation_level=None
                )
                store = AsyncSqliteStore.from_conn_string(db_path)
                logger.info("Created AsyncSqliteStore (persistent): %s", db_path)
                return store
            else:
                logger.info("No running event loop — using InMemoryStore")
                return InMemoryStore()
        except Exception as e:
            logger.warning(
                "AsyncSqliteStore unavailable (%s) — falling back to InMemoryStore", e
            )
            return InMemoryStore()

    logger.info("Created InMemoryStore (in-memory mode)")
    return InMemoryStore()


# ---------------------------------------------------------------------------
# Convenience functions for node use (access store via config)
# ---------------------------------------------------------------------------


async def store_put(
    store,
    namespace: tuple,
    key: str,
    value: Any,
) -> None:
    """Put a value into the graph store."""
    if store is None:
        return
    await store.aput(namespace=namespace, key=key, value=value)


async def store_get(store, namespace: tuple, key: str) -> Optional[Any]:
    """Get a value from the graph store."""
    if store is None:
        return None
    result = await store.aget(namespace=namespace, key=key)
    return result.value if result else None


async def store_search(
    store,
    namespace: tuple,
    query: str,
    limit: int = 10,
) -> list:
    """Search the graph store."""
    if store is None:
        return []
    results = await store.asearch(namespace=namespace, query=query, limit=limit)
    return [{"key": r.key, "value": r.value} for r in results]
