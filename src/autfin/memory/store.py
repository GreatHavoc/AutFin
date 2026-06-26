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
    # Environment variable takes precedence
    env_mode = os.getenv("AUTFIN_STORE_MODE")
    if env_mode in ("in-memory", "persistent"):
        return env_mode
    
    # Fallback to config
    try:
        settings = get_settings()
        if settings.store_mode == "persistent":
            return "persistent"
        return "in-memory"
    except Exception:
        return "in-memory"


def create_store(mode: Optional[str] = None) -> BaseStore:
    """
    Create a LangGraph InMemoryStore instance.
    
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
            db_path = os.getenv("AUTFIN_STORE_DB_PATH", "autfin_store.db")
            logger.info("Using AsyncSqliteStore for persistent memory (%s)", db_path)
            try:
                return AsyncSqliteStore(conn=db_path)
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
        Full namespace path
    """
    return namespace


class AutFinStore:
    """
    wrapper around LangGraph InMemoryStore with AutFin-specific operations.
    
    Provides convenience methods for agent state persistence and shared memory.
    """
    
    def __init__(self, store: Optional[BaseStore] = None, mode: Optional[str] = None):
        """
        Initialize the AutFinStore.
        
        Args:
            store: Existing BaseStore instance (uses create_store if None)
            mode: Store mode ("in-memory" or "persistent")
        """
        self._store = store or create_store(mode=mode)
        self._mode = mode or get_store_mode()
    
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
            key: Key to store under
            value: Value to store
        """
        await self._store.aput(
            namespace=(namespace,),
            key=key,
            value=value,
        )
    
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
        exists = await self._store.aget(
            namespace=(namespace,),
            key=key,
        )
        if exists:
            await self._store.adelete(
                namespace=(namespace,),
                key=key,
            )
            return True
        return False
    
    async def asearch(
        self,
        namespace: str,
        metadata: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously search for values in a namespace.
        
        Args:
            namespace: Namespace to search in (single string, will be converted to tuple)
            metadata: Filter by metadata (optional)
            limit: Maximum number of results
        
        Returns:
            List of matching key-value pairs
        """
        results = await self._store.asearch(
            namespace=(namespace,),
            filter=metadata or {},
            limit=limit,
        )
        return [
            {"key": doc.key, "value": doc.value, "metadata": doc.metadata}
            for doc in results
        ]

    async def _list_ns_keys(self, namespace: str) -> List[str]:
        """List all keys in a namespace.

        Args:
            namespace: Namespace to list keys from

        Returns:
            List of keys in the namespace
        """
        results = await self._store.asearch(
            namespace=(namespace,),
            filter={},
            limit=10000,
        )
        return [doc.key for doc in results]
    
    # Synchronous wrappers for convenience
    def put(
        self,
        namespace: str,
        key: str,
        value: Any,
    ) -> None:
        """Synchronously store a value.

        Raises RuntimeError if called from within an already-running event loop.
        Use the async methods (aput/aget/adelete/asearch) when inside async code.
        """
        asyncio.run(self.aput(namespace, key, value))
    
    def get(
        self,
        namespace: str,
        key: str,
    ) -> Optional[Any]:
        """Synchronously retrieve a value.

        Raises RuntimeError if called from within an already-running event loop.
        Use the async methods (aput/aget/adelete/asearch) when inside async code.
        """
        return asyncio.run(self.aget(namespace, key))
    
    def delete(
        self,
        namespace: str,
        key: str,
    ) -> bool:
        """Synchronously delete a value.

        Raises RuntimeError if called from within an already-running event loop.
        Use the async methods (aput/aget/adelete/asearch) when inside async code.
        """
        return asyncio.run(self.adelete(namespace, key))
    
    def search(
        self,
        namespace: str,
        metadata: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Synchronously search for values.

        Raises RuntimeError if called from within an already-running event loop.
        Use the async methods (aput/aget/adelete/asearch) when inside async code.
        """
        return asyncio.run(self.asearch(namespace, metadata, limit))





# Convenience singleton for in-memory operations
_default_store: Optional[AutFinStore] = None


def get_default_store() -> AutFinStore:
    """
    Get or create the default store instance.
    
    Returns:
        Default AutFinStore instance
    """
    global _default_store
    if _default_store is None:
        _default_store = get_store()
    return _default_store


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
