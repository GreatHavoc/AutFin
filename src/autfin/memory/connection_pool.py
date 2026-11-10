"""SQLite connection pool for AutFin — thread-safe, reusable connections.

Provides a centralized connection pool to avoid creating new connections
for every database operation. Uses context manager pattern for safe handling.

Usage:
    from autfin.memory.connection_pool import get_db_pool

    pool = get_db_pool()
    with pool.connection() as conn:
        conn.execute("SELECT ...")
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

# Default database path
_DEFAULT_DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "autfin.db"


class SQLiteConnectionPool:
    """Thread-safe SQLite connection pool with WAL mode.

    Maintains a single connection per thread, creating new connections
    as needed. All connections use WAL journal mode for better
    concurrent read performance.
    """

    def __init__(self, db_path: str | Path, max_connections: int = 5):
        self._db_path = str(db_path)
        self._max_connections = max_connections
        self._local = threading.local()
        self._lock = threading.Lock()
        self._connections: dict[int, sqlite3.Connection] = {}
        self._ensure_db_directory()

    def _ensure_db_directory(self):
        """Create database directory if it doesn't exist."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

    def _get_thread_id(self) -> int:
        return threading.get_ident()

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with WAL mode."""
        conn = sqlite3.connect(self._db_path, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    @contextmanager
    def connection(self):
        """Get a database connection from the pool.

        Yields a sqlite3.Connection that is automatically returned
        to the pool when the context exits.

        Usage:
            with pool.connection() as conn:
                conn.execute("SELECT ...")
        """
        thread_id = self._get_thread_id()
        conn = None

        with self._lock:
            if thread_id in self._connections:
                conn = self._connections[thread_id]

        if conn is None:
            with self._lock:
                if len(self._connections) >= self._max_connections:
                    logger.warning("Connection pool at max capacity, creating temporary connection")
                    conn = self._create_connection()
                    try:
                        yield conn
                    finally:
                        conn.close()
                    return

            conn = self._create_connection()
            with self._lock:
                self._connections[thread_id] = conn

        try:
            yield conn
        except Exception:
            # On error, remove connection from pool (it may be in bad state)
            with self._lock:
                self._connections.pop(thread_id, None)
            try:
                conn.close()
            except Exception:
                pass
            raise

    def close_all(self):
        """Close all connections in the pool."""
        with self._lock:
            for conn in self._connections.values():
                try:
                    conn.close()
                except Exception:
                    pass
            self._connections.clear()
            logger.info("All connections in pool closed")

    def get_stats(self) -> dict:
        """Get pool statistics."""
        with self._lock:
            return {
                "active_connections": len(self._connections),
                "max_connections": self._max_connections,
                "db_path": self._db_path,
            }


# Global pool instances
_pools: dict[str, SQLiteConnectionPool] = {}
_pools_lock = threading.Lock()


def get_db_pool(
    db_path: str | Path | None = None,
    max_connections: int = 5,
) -> SQLiteConnectionPool:
    """Get or create a SQLite connection pool for the given database path.

    Args:
        db_path: Path to the SQLite database. Defaults to data/autfin.db.
        max_connections: Maximum number of connections in the pool.

    Returns:
        SQLiteConnectionPool instance.
    """
    if db_path is None:
        db_path = str(_DEFAULT_DB_PATH)
    else:
        db_path = str(db_path)

    with _pools_lock:
        if db_path not in _pools:
            _pools[db_path] = SQLiteConnectionPool(db_path, max_connections)
        return _pools[db_path]


def close_all_pools():
    """Close all connection pools. Call on application shutdown."""
    with _pools_lock:
        for pool in _pools.values():
            pool.close_all()
        _pools.clear()
