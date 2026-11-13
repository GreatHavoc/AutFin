"""The Memory capability's store: SQLite, because a file store cannot run here.

`Memory(FileStore("./data/memory"))` resolves its directory from the *run's
workspace* and raises `UserError` in `before_run` when none is attached. The only
local workspace backend, `LocalWorkspaceBackend`, raises `NotImplementedError` on
non-POSIX platforms (its timeout contract kills a process group), and
`DockerSandboxBackend` cannot help: it invokes the podman/docker CLI on the host
through that same `LocalWorkspaceBackend` runner. So the supervisor and the
memory agent both failed at the start of every live run.

`SqliteMemoryStore` needs no workspace and lands memory in the SQLite file the
rest of AutFin's state already uses. One process-wide instance so the two agents
share rows instead of each holding a private copy.
"""
from __future__ import annotations

from pydantic_ai_harness.memory import SqliteMemoryStore

from autfin.memory import database

_store: SqliteMemoryStore | None = None


def get_memory_store() -> SqliteMemoryStore:
    """Process-wide store on data/autfin.db (gitignored, local-only)."""
    global _store
    if _store is None:
        _store = SqliteMemoryStore(database=database.DB_PATH)
    return _store