"""Cross-run conversation persistence.

A ConversationStore keeps conversations for you: each message stored once,
conflicting saves caught, forking built in. Core defines the protocol and the
in-memory implementation; this module supplies a SQLite backend on the existing
data/autfin.db, so a daily run continues yesterday's thread instead of starting
cold.

The five primitives below are what core asks a backend to implement
(read_record / write_record / read_records / read_entry / delete_record);
load, save, branch and fork are built on top of them by core. All are async
because core awaits them, and their signatures are core's, not ours: entry ids
are global, and revision/scope are keyword-only on the paths core calls.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any

from pydantic_ai.conversation_store import ConversationEntry, ConversationRecord, ConversationStore
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.usage import RunUsage
from pydantic_core import to_jsonable_python

from autfin.memory import database

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id   TEXT NOT NULL,
    scope             TEXT NOT NULL DEFAULT '',
    revision          INTEGER NOT NULL DEFAULT 0,
    record_json       TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    PRIMARY KEY (conversation_id)
);
CREATE TABLE IF NOT EXISTS conversation_entries (
    entry_id          TEXT NOT NULL,
    conversation_id   TEXT NOT NULL,
    parent_id         TEXT,
    ordinal           INTEGER NOT NULL DEFAULT 0,
    message_json      TEXT NOT NULL,
    PRIMARY KEY (entry_id)
);
"""


def _record_to_json(record: ConversationRecord) -> dict[str, Any]:
    """ConversationRecord is a plain dataclass: its nested usage and dates are
    not JSON types, so normalise with pydantic-core instead of a hand-rolled
    default=."""
    return to_jsonable_python(dataclasses.asdict(record))


def _record_from_json(data: dict[str, Any]) -> ConversationRecord:
    return ConversationRecord(usage=RunUsage(**data.pop("usage", {})), **data)


class SqliteConversationStore(ConversationStore):
    """SQLite ConversationStore on the AutFin data directory."""

    def __init__(self, db_path: Any = None):
        self._db_path = db_path or database.DB_PATH

    def _connect(self):
        parent = getattr(self._db_path, "parent", None)
        if parent is not None:
            parent.mkdir(parents=True, exist_ok=True)
        conn = database.sqlite3.connect(self._db_path, isolation_level=None)
        conn.row_factory = database.sqlite3.Row
        conn.executescript(_SCHEMA)
        return conn

    # ── the five primitives core builds on ────────────────────────────────

    async def read_record(self, conversation_id: str) -> ConversationRecord | None:
        import json

        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT record_json FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            return _record_from_json(json.loads(row["record_json"])) if row else None
        finally:
            conn.close()

    async def write_record(
        self,
        record: ConversationRecord,
        *,
        scope: str | None = None,
        new_entries: list[ConversationEntry] | None = None,
        expected_revision: int | None = None,
    ) -> list[ConversationEntry]:
        """Persist the record and the entries that arrived with it.

        `expected_revision` is core's conflict guard: a mismatch raises
        ConversationConflict instead of overwriting another writer's save.
        """
        import json

        from pydantic_ai.conversation_store import ConversationConflict

        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT revision FROM conversations WHERE conversation_id = ?",
                (record.conversation_id,),
            ).fetchone()
            current = row["revision"] if row else 0
            if expected_revision is not None and expected_revision != current:
                raise ConversationConflict(
                    f"conversation {record.conversation_id!r} is at revision {current}, "
                    f"expected {expected_revision}; load it again and retry"
                )

            conn.execute(
                "INSERT INTO conversations"
                " (conversation_id, scope, revision, record_json, updated_at)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(conversation_id) DO UPDATE SET"
                "   revision = excluded.revision, record_json = excluded.record_json,"
                "   scope = excluded.scope, updated_at = excluded.updated_at",
                (
                    record.conversation_id,
                    (scope if scope is not None else record.scope) or "",
                    record.revision,
                    json.dumps(_record_to_json(record)),
                    database.datetime.now(database.timezone.utc).isoformat(),
                ),
            )

            stored = list(new_entries or [])
            for ordinal, entry in enumerate(stored):
                conn.execute(
                    "INSERT OR REPLACE INTO conversation_entries"
                    " (entry_id, conversation_id, parent_id, ordinal, message_json)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (
                        entry.id,
                        record.conversation_id,
                        entry.parent_id,
                        ordinal,
                        ModelMessagesTypeAdapter.dump_json([entry.message]).decode(),
                    ),
                )
            return stored
        finally:
            conn.close()

    async def read_records(self, *, scope: str | None = None, limit: int | None = None):
        import json

        conn = self._connect()
        try:
            sql = "SELECT record_json FROM conversations WHERE scope = ? ORDER BY updated_at DESC"
            rows = conn.execute(sql, (scope or "",)).fetchall()
            return [_record_from_json(json.loads(r["record_json"])) for r in rows[:limit]]
        finally:
            conn.close()

    async def read_entry(self, entry_id: str) -> ConversationEntry:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT entry_id, conversation_id, parent_id, message_json FROM conversation_entries"
                " WHERE entry_id = ?",
                (entry_id,),
            ).fetchone()
            if row is None:
                raise KeyError(entry_id)
            return ConversationEntry(
                id=row["entry_id"],
                parent_id=row["parent_id"],
                message=ModelMessagesTypeAdapter.validate_json(row["message_json"])[0],
            )
        finally:
            conn.close()

    async def delete_record(self, conversation_id: str, *, expected_revision: int) -> None:
        conn = self._connect()
        try:
            conn.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM conversation_entries WHERE conversation_id = ?", (conversation_id,))
        finally:
            conn.close()


_store: SqliteConversationStore | None = None


def get_conversation_store() -> SqliteConversationStore:
    """Process-wide store; the SQLite file is data/autfin.db, gitignored."""
    global _store
    if _store is None:
        _store = SqliteConversationStore()
    return _store
