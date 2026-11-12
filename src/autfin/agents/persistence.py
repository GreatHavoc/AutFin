"""Crash recovery for the supervisor run — the harness StepPersistence layer.

`memory/database.py` used to carry a `runs` table recording a run's start and
end. Nothing ever wrote or read it (its two writers had zero callers), so it
provided no recovery: after a crash you could see that a run had happened, not
how far it got or which tool side effects landed. That table and its writers are
deleted; this module supplies the real thing on the same SQLite file.

`StepPersistence` snapshots the message history at every settled step, appends
an event trail, and records a tool-effect ledger keyed by
`(run_id, tool_call_id)`. After a crash, `continue_run`/`fork_run` load the last
complete snapshot's messages and hand them back to a new `Agent.run`, and a tool
effect still marked `started` reads as `unknown_after_crash` — the side effect
may or may not have happened, which is exactly the question that matters before
re-running a graph whose only side-effecting tool is `execute_trade`.

The store gets its **own** file (`data/autfin_steps.db`), not `data/autfin.db`:
the harness's SQLite schema creates a table named `runs`, and the AutFin data
directory already carried a table of that name from the hand-rolled run log.
`CREATE TABLE IF NOT EXISTS` then silently kept the old shape and every write
failed with `no such column: conversation_id` — a collision no test caught,
because tests use a fresh database each time.

Only the supervisor gets this: it is the run the daemon actually performs, and
snapshotting every sub-agent's internal steps would fill the store with noise.
The store itself is shared, so a sub-agent's run can be inspected by the same
file if one is ever attached.
"""
from __future__ import annotations

from pydantic_ai_harness import StepPersistence
from pydantic_ai_harness.step_persistence import SqliteStepStore

from autfin.memory import database

AGENT_NAME = "supervisor"

_store: SqliteStepStore | None = None
_capability: StepPersistence | None = None


def get_step_store() -> SqliteStepStore:
    """Process-wide store on data/autfin_steps.db.

    Its own file, not data/autfin.db: the harness's schema creates a `runs`
    table and the AutFin data directory already had one from the deleted
    hand-rolled run log, so a shared file left `CREATE TABLE IF NOT EXISTS`
    keeping the old shape and every write failed with `no such column:
    conversation_id`.
    """
    global _store
    if _store is None:
        _store = SqliteStepStore(database=database.DB_PATH.with_name("autfin_steps.db"))
    return _store


def get_step_persistence() -> StepPersistence:
    """One shared capability instance: the store is built per capability, so a
    fresh one here would scatter each agent's snapshots into its own store."""
    global _capability
    if _capability is None:
        _capability = StepPersistence(store=get_step_store(), agent_name=AGENT_NAME)
    return _capability
