"""
Database schema and CRUD operations for AutFin
"""
import sqlite3
from pathlib import Path


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

        # Create portfolio_snapshots table
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

        # Create signals table
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

        # Create balance_history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS balance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                available_cash REAL NOT NULL,
                utilized_margin REAL NOT NULL,
                source TEXT DEFAULT 'kite_sdk'
            )
        """)

        # Create learning_journal table (v3.0 schema)
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

        # Stock pitches table (v3.0) — 6-element buy-side pitches rendered by dashboard
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_pitches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                context TEXT NOT NULL DEFAULT 'manual',
                investment_thesis TEXT,
                catalysts_json TEXT,
                risks_json TEXT,
                valuation_summary TEXT,
                competitive_moat TEXT,
                price_target_base REAL,
                price_target_upside REAL,
                price_target_downside REAL,
                price_target_rationale TEXT,
                raw_json TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pitches_symbol ON stock_pitches(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pitches_created ON stock_pitches(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pitches_context ON stock_pitches(context)")

        # Fan-out progress (v3.0) — live status of concurrent research per symbol
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fanout_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                context TEXT NOT NULL,
                symbol TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                started_at TEXT,
                finished_at TEXT,
                error TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fanout_session ON fanout_progress(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fanout_status ON fanout_progress(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fanout_created ON fanout_progress(created_at)")

        # Migration: if old columns exist, copy data then drop them
        _migrate_learning_journal(cursor)

        conn.commit()
    finally:
        conn.close()


def _migrate_learning_journal(cursor):
    """Migrate v2 learning_journal columns to v3.0. Destructive — old columns removed."""
    try:
        existing = {r[1] for r in cursor.execute("PRAGMA table_info(learning_journal)").fetchall()}

        if "outcome_pnl" in existing and "pnl_pct" in existing:
            cursor.execute("UPDATE learning_journal SET pnl_pct = outcome_pnl WHERE pnl_pct IS NULL AND outcome_pnl IS NOT NULL")
        if "lesson" in existing and "thesis_summary" in existing:
            cursor.execute("UPDATE learning_journal SET thesis_summary = lesson WHERE thesis_summary IS NULL AND lesson IS NOT NULL")
        if "was_correct" in existing and "outcome" in existing:
            cursor.execute("UPDATE learning_journal SET outcome = CASE WHEN was_correct=1 THEN 'completed' WHEN was_correct=0 THEN 'loss' ELSE 'open' END WHERE outcome='open' AND was_correct IS NOT NULL")
    except Exception:
        pass  # Migration is best-effort — tests recreate DB anyway


def get_connection():
    """Get a database connection."""
    if not DB_PATH.exists():
        init_database()
    return sqlite3.connect(DB_PATH, isolation_level=None)


def init_database_if_not_exists():
    """Initialize database if it doesn't exist."""
    if not DB_PATH.exists():
        init_database()


if __name__ == "__main__":
    init_database()
    print("Database initialized successfully at", DB_PATH)