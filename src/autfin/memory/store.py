"""LangGraph store for AutFin cross-thread memory."""

import asyncio
import os
from typing import Any, Dict, List, Optional
from pathlib import Path
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from ..config.settings import AutFinSettings, get_settings

try:
    from langgraph.store.sqlite import AsyncSqliteStore
    _SQLITE_STORE_AVAILABLE = True
except ImportError:
    _SQLITE_STORE_AVAILABLE = False


def get_store_mode() -> str:
    """
    Get the store mode from environment or config.

    Returns:
        "in-memory" or "persistent" (SQLite-based)
    """
    env_mode = os.getenv("AUTFIN_STORE_MODE")
    if env_mode in ("in-memory", "persistent"):
        return env_mode

    try:
        settings = get_settings()
        if settings.store_mode == "persistent":
            return "persistent"
        return "in-memory"
    except Exception:
        return "in-memory"


async def create_store(mode: Optional[str] = None) -> BaseStore:
    """
    Create a LangGraph store instance (async — handles aiosqlite connection).

    Args:
        mode: Either "in-memory" or "persistent" (defaults to auto-detect)

    Returns:
        LangGraph BaseStore instance
    """
    if mode is None:
        mode = get_store_mode()

    if mode == "in-memory":
        return InMemoryStore()

    if mode == "persistent":
        if _SQLITE_STORE_AVAILABLE:
            import logging
            logger = logging.getLogger(__name__)
            db_path = os.getenv("AUTFIN_STORE_DB_PATH", "data/autfin_store.db")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            logger.info("Using AsyncSqliteStore for persistent memory (%s)", db_path)
            try:
                import aiosqlite
                conn = await aiosqlite.connect(db_path, isolation_level=None)
                return AsyncSqliteStore(conn=conn)
            except Exception as e:
                logger.warning(
                    "AsyncSqliteStore init failed (%s) — falling back to InMemoryStore", e
                )
                return InMemoryStore()
        else:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "AsyncSqliteStore not available (langgraph >= 0.4.0 required). "
                "Falling back to InMemoryStore."
            )
            return InMemoryStore()

    return InMemoryStore()


def get_namespace(store: BaseStore, namespace: str = "default") -> str:
    """
    Get a namespaced key prefix.

    Args:
        store: The BaseStore instance
        namespace: Namespace name (e.g., "agent_state", "shared_memory")

    Returns:
        Namespace string
    """
    return namespace


class AutFinStore:
    """
    Wrapper around LangGraph BaseStore for AutFin.

    Provides convenience methods for agent state persistence and shared memory.
    """

    def __init__(self, store: Optional[BaseStore] = None, mode: Optional[str] = None):
        """
        Initialize the AutFinStore.

        Args:
            store: Existing BaseStore instance (uses create_store if None)
            mode: Store mode ("in-memory" or "persistent")
        """
        self._store = store
        self._mode = mode or get_store_mode()
        self._initialized = store is not None
        self._init_lock = asyncio.Lock() if not store else None

    async def _ensure_store(self):
        """Lazily initialize the store on first async call."""
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            self._store = await create_store(mode=self._mode)
            self._initialized = True
            self._init_lock = None

    @property
    def mode(self) -> str:
        """Get the current store mode."""
        return self._mode

    async def aput(
        self,
        namespace: str,
        key: str,
        value: Any,
    ) -> None:
        """
        Asynchronously store a value.

        Args:
            namespace: Namespace for the key (single string, will be converted to tuple)
            key: Key to store
            value: Value to store
        """
        await self._ensure_store()
        await self._store.aput(namespace=(namespace,), key=key, value=value)

    async def aget(
        self,
        namespace: str,
        key: str,
    ) -> Optional[Any]:
        """
        Asynchronously retrieve a value.

        Args:
            namespace: Namespace for the key (single string, will be converted to tuple)
            key: Key to retrieve

        Returns:
            Stored value or None if not found
        """
        await self._ensure_store()
        result = await self._store.aget(
            namespace=(namespace,),
            key=key,
        )
        return result.value if result else None

    async def adelete(
        self,
        namespace: str,
        key: str,
    ) -> bool:
        """
        Asynchronously delete a value.

        Args:
            namespace: Namespace for the key (single string, will be converted to tuple)
            key: Key to delete

        Returns:
            True if deleted, False if not found
        """
        await self._ensure_store()
        try:
            await self._store.adelete(
                namespace=(namespace,),
                key=key,
            )
            return True
        except Exception:
            return False

    async def asearch(
        self,
        namespace: str,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously search for values.

        Args:
            namespace: Namespace to search in
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching values
        """
        await self._ensure_store()
        results = await self._store.asearch(
            namespace=(namespace,),
            query=query,
            limit=limit,
        )
        return [{"key": r.key, "value": r.value} for r in results]

    def put(
        self,
        namespace: str,
        key: str,
        value: Any,
    ) -> None:
        """
        Synchronously store a value.

        Args:
            namespace: Namespace for the key (single string, will be converted to tuple)
            key: Key to store
            value: Value to store
        """
        self._store.put(namespace=(namespace,), key=key, value=value)

    def get(
        self,
        namespace: str,
        key: str,
    ) -> Optional[Any]:
        """
        Synchronously retrieve a value.

        Args:
            namespace: Namespace for the key (single string, will be converted to tuple)
            key: Key to retrieve

        Returns:
            Stored value or None if not found
        """
        result = self._store.get(
            namespace=(namespace,),
            key=key,
        )
        return result.value if result else None

    def delete(
        self,
        namespace: str,
        key: str,
    ) -> bool:
        """
        Synchronously delete a value.

        Args:
            namespace: Namespace for the key (single string, will be converted to tuple)
            key: Key to delete

        Returns:
            True if deleted, False if not found
        """
        try:
            self._store.delete(
                namespace=(namespace,),
                key=key,
            )
            return True
        except Exception:
            return False


# Default store singleton
_default_store: Optional[AutFinStore] = None


def _get_default_store() -> AutFinStore:
    """
    Get or create the default AutFinStore instance.

    Returns:
        Default AutFinStore instance
    """
    global _default_store
    if _default_store is None:
        _default_store = AutFinStore()
    return _default_store


# Public alias for backward compatibility
get_default_store = _get_default_store


# Re-implement get_store as a proper singleton
def get_store(mode: Optional[str] = None) -> AutFinStore:
    """
    Factory function to get a configured store instance.
    Uses singleton pattern — same instance returned for same mode.

    Defaults to persistent SQLite store for production. Falls back to
    InMemoryStore if SQLite store is unavailable.

    Args:
        mode: Store mode ("in-memory" or "persistent"). Defaults to env var or persistent.

    Returns:
        Configured AutFinStore instance
    """
    global _default_store

    if _default_store is None:
        store_mode = mode or os.getenv("AUTFIN_STORE_MODE", "persistent")
        try:
            _default_store = AutFinStore(mode=store_mode)
            import logging
            logging.getLogger(__name__).info(
                "Initialized store in '%s' mode", store_mode
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "Failed to init '%s' store (%s), falling back to in-memory", store_mode, e
            )
            _default_store = AutFinStore(mode="in-memory")
    return _default_store
