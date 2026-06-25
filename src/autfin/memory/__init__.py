"""
AutFin Memory Layer
"""
from .setup import create_checkpointer, setup_memory, get_checkpoint_db_path
from .store import (
    AutFinStore,
    get_store,
    get_default_store,
    get_store_mode,
    create_store,
)
from .vector_store import (
    VectorStore,
    get_chroma_path,
    create_chroma_client,
    get_or_create_collection,
    initialize_collections,
    REQUIRED_COLLECTIONS,
)
from .retrieval import (
    # Store write helpers
    store_research_report,
    store_research,
    store_trade_decision,
    store_trade_history,
    store_capital_state,
    store_portfolio_state,
    # Store read helpers
    get_research_report,
    get_latest_research,
    get_trade_history,
    get_capital_state,
    get_portfolio_state,
    get_decision,
    # Composite retrieval
    build_research_context,
    load_agent_context,
    # Vector Store write helpers
    store_strategy_learning,
    store_company_knowledge,
    store_market_knowledge,
    # Constants
    NS_CAPITAL_STATE,
    NS_PORTFOLIO_STATE,
    NS_TRADE_HISTORY,
    NS_RESEARCH_REPORTS,
    NS_DECISIONS,
    COLLECTION_MARKET_KNOWLEDGE,
    COLLECTION_STRATEGY_LEARNINGS,
    COLLECTION_COMPANY_KNOWLEDGE,
)
from .performance import (
    # learning_journal CRUD (v3.0)
    record_learning,
    get_learnings_for_symbol,
    get_recent_learnings,
    update_learning_outcome,
    # Performance metrics
    get_win_rate,
    get_profit_factor,
    get_symbol_performance,
    get_monthly_summary,
    get_cumulative_pnl,
    get_drawdown_stats,
    get_all_performance_metrics,
    # Outcome backfill
    record_trade_outcome,
)

__all__ = [
    # Setup
    "create_checkpointer",
    "setup_memory",
    "get_checkpoint_db_path",
    # Store
    "AutFinStore",
    "get_store",
    "get_default_store",
    "get_store_mode",
    "create_store",
    # Vector Store
    "VectorStore",
    "get_chroma_path",
    "create_chroma_client",
    "get_or_create_collection",
    "initialize_collections",
    "REQUIRED_COLLECTIONS",
    # Retrieval — write
    "store_research_report",
    "store_research",
    "store_trade_decision",
    "store_trade_history",
    "store_capital_state",
    "store_portfolio_state",
    # Retrieval — read
    "get_research_report",
    "get_latest_research",
    "get_trade_history",
    "get_capital_state",
    "get_portfolio_state",
    "get_decision",
    # Composite
    "build_research_context",
    "load_agent_context",
    # Vector Store helpers
    "store_strategy_learning",
    "store_company_knowledge",
    "store_market_knowledge",
    # Constants
    "NS_CAPITAL_STATE",
    "NS_PORTFOLIO_STATE",
    "NS_TRADE_HISTORY",
    "NS_RESEARCH_REPORTS",
    "NS_DECISIONS",
    "COLLECTION_MARKET_KNOWLEDGE",
    "COLLECTION_STRATEGY_LEARNINGS",
    "COLLECTION_COMPANY_KNOWLEDGE",
    # Performance (v3.0)
    "record_learning",
    "get_learnings_for_symbol",
    "get_recent_learnings",
    "update_learning_outcome",
    "get_win_rate",
    "get_profit_factor",
    "get_symbol_performance",
    "get_monthly_summary",
    "get_cumulative_pnl",
    "get_drawdown_stats",
    "get_all_performance_metrics",
    "record_trade_outcome",
]
