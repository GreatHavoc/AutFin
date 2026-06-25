"""Memory setup for AutFin."""

from typing import Optional
from pathlib import Path
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.memory import MemorySaver
from ..config.settings import AutFinSettings


def get_checkpoint_db_path() -> Path:
    """Get the checkpoint database path."""
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    return data_dir / "checkpoints.db"


def create_checkpointer(
    mode: str = "persistent", db_path: Optional[Path] = None
):
    """
    Create a LangGraph checkpointer for node-level persistence.
    
    v3.0: Uses AsyncSqliteSaver with aiosqlite for async compatibility.
    
    Args:
        mode: Either "persistent" (SQLite) or "in-memory" (MemorySaver)
        db_path: Optional custom database path (defaults to data/checkpoints.db)
    
    Returns:
        AsyncSqliteSaver for persistent mode, MemorySaver for in-memory mode
    """
    if mode == "in-memory":
        return MemorySaver()
    
    if db_path is None:
        db_path = get_checkpoint_db_path()
    
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # AsyncSqliteSaver requires aiosqlite connection
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop is None:
        conn = asyncio.run(aiosqlite.connect(str(db_path)))
    else:
        # Running loop exists — create connection in a new thread
        import concurrent.futures
        def _connect():
            return asyncio.run(aiosqlite.connect(str(db_path)))
        with concurrent.futures.ThreadPoolExecutor() as pool:
            conn = pool.submit(_connect).result(timeout=10)
    
    return AsyncSqliteSaver(conn)


async def setup_memory(checkpoint_mode: str = "persistent") -> dict:
    """
    Set up memory infrastructure including checkpointers.
    
    Args:
        checkpoint_mode: "persistent" or "in-memory"
    
    Returns:
        Dictionary with initialized memory components
    """
    checkpointer = create_checkpointer(mode=checkpoint_mode)
    
    return {
        "checkpointer": checkpointer,
        "checkpoint_mode": checkpoint_mode,
    }
