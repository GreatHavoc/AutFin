"""
Database schema and CRUD operations for AutFin
"""
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent.parent / "data" / "autfin.db"


def init_database():
    """Initialize the SQLite database with schema."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()

        # Enable WAL mode for durability
        cursor.execute("PRAGMA journal_mode=WAL")

        # Create trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                exchange TEXT NOT NULL DEFAULT 'NSE',
                action TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                total_value REAL NOT NULL,
                order_id TEXT,
                order_status TEXT,
                decision_id TEXT,
                rationale_summary TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")

        # Portfolio snapshots
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                available_cash REAL NOT NULL,
                utilized_margin REAL NOT NULL,
                total_holdings_value REAL NOT NULL,
                total_pnl REAL NOT NULL,
                total_pnl_pct REAL NOT NULL,
                holdings_json TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_snapshots_date ON portfolio_snapshots(date)")

        # Signal log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                strength REAL NOT NULL,
                fundamentals_score REAL,
                technical_score REAL,
                sentiment_score REAL,
                macro_score REAL,
                risk_score REAL,
                rationale TEXT,
                was_acted_on BOOLEAN DEFAULT FALSE,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_acted ON signals(was_acted_on)")

        # Balance history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS balance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                available_cash REAL NOT NULL,
                utilized_margin REAL NOT NULL,
                source TEXT DEFAULT 'kite_sdk'
            )
        """)

        # Learning journal (current schema)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL DEFAULT 'BUY',
                conviction REAL NOT NULL DEFAULT 0.5,
                pnl_pct REAL,
                thesis_summary TEXT,
                outcome TEXT DEFAULT 'open',
                sector TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_learning_symbol ON learning_journal(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_learning_sector ON learning_journal(sector)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_learning_outcome ON learning_journal(outcome)")

        # Per-invocation structured trace
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL UNIQUE,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                status TEXT NOT NULL DEFAULT 'running',
                query TEXT,
                mode TEXT,
                model_deep TEXT,
                model_fast TEXT,
                total_tokens INTEGER DEFAULT 0,
                total_cost_inr REAL DEFAULT 0.0,
                nodes_visited TEXT DEFAULT '[]',
                decisions_made INTEGER DEFAULT 0,
                trades_executed INTEGER DEFAULT 0,
                error_message TEXT,
                duration_ms REAL,
                metadata_json TEXT DEFAULT '{}'
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)")

        # SEBI audit trail — Strategy ID tags per SEBI Apr 2026 circular
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sebi_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                strategy_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                quantity INTEGER,
                price REAL,
                order_type TEXT,
                exchange TEXT DEFAULT 'NSE',
                timestamp TEXT NOT NULL,
                compliance_notes TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sebi_strategy ON sebi_audit(strategy_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sebi_run ON sebi_audit(run_id)")

        _migrate_learning_journal(cursor)

        conn.commit()
    finally:
        conn.close()


def _migrate_learning_journal(cursor):
    """Copy values from legacy v2 columns into v3.0 columns when both exist."""
    try:
        existing = {r[1] for r in cursor.execute("PRAGMA table_info(learning_journal)").fetchall()}

        if "outcome_pnl" in existing and "pnl_pct" in existing:
            cursor.execute("UPDATE learning_journal SET pnl_pct = outcome_pnl WHERE pnl_pct IS NULL AND outcome_pnl IS NOT NULL")
        if "lesson" in existing and "thesis_summary" in existing:
            cursor.execute("UPDATE learning_journal SET thesis_summary = lesson WHERE thesis_summary IS NULL AND lesson IS NOT NULL")
        if "was_correct" in existing and "outcome" in existing:
            cursor.execute(
                "UPDATE learning_journal SET outcome = CASE WHEN was_correct=1 THEN 'completed' WHEN was_correct=0 THEN 'loss' ELSE 'open' END "
                "WHERE outcome='open' AND was_correct IS NOT NULL"
            )
    except Exception:
        pass


def _is_healthy(conn) -> bool:
    """True if the SQLite file opens and passes a structural check."""
    try:
        return conn.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    except sqlite3.DatabaseError:
        return False


def _quarantine_corrupt_db() -> None:
    """Move an unreadable database aside instead of deleting it.

    Trade history is local-only and gitignored, so the corrupt file is the only
    copy: rename with a timestamp and re-init rather than dropping it.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    broken = DB_PATH.with_name(f"{DB_PATH.name}.malformed-{stamp}")
    try:
        DB_PATH.rename(broken)
        logger.error(
            "Database %s was unreadable — quarantined as %s and re-initialized "
            "empty. Restore from that file once it has been repaired.",
            DB_PATH, broken,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Cannot quarantine corrupt database {DB_PATH}: {exc}. "
            "Move or delete it manually, then re-run."
        ) from exc


def get_connection():
    """Get a connection, recovering from a corrupt database file.

    A corrupt file used to be handed straight to callers, and every learning
    function wraps its DB work in `except Exception` — so the whole learning
    engine returned empty results with only a debug log.
    """
    if not DB_PATH.exists():
        init_database()

    conn = sqlite3.connect(DB_PATH, isolation_level=None)
    if _is_healthy(conn):
        return conn

    conn.close()
    _quarantine_corrupt_db()
    init_database()
    return sqlite3.connect(DB_PATH, isolation_level=None)


def init_database_if_not_exists():
    """Initialize database if it doesn't exist."""
    if not DB_PATH.exists():
        init_database()


if __name__ == "__main__":
    init_database()
    print("Database initialized successfully at", DB_PATH)
