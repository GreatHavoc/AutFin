"""
Memory tools for AutFin agents.

Provides LangChain-compatible tools that agents can call to read/write
long-term memory via the Store and Vector Store.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence

from ..memory.retrieval import (
    build_research_context,
    get_latest_research,
    get_research_report,
    get_trade_history,
    get_capital_state,
    get_portfolio_state,
    get_decision,
    store_research,
    store_research_report,
    store_trade_history,
    store_trade_decision,
    store_capital_state,
    store_portfolio_state,
    store_strategy_learning,
    store_company_knowledge,
    store_market_knowledge,
)
from ..memory.performance import (
    get_win_rate,
    get_symbol_performance,
    get_monthly_summary,
    get_drawdown_stats,
    get_all_performance_metrics,
    record_trade_outcome,
)
# # from ..memory.store import AutFinStore, get_store  # REMOVED  # REMOVED - use Pydantic AI Memory
from ..memory.vector_store import VectorStore

logger = logging.getLogger(__name__)


class MemoryTools:
    """
    Tool set for agent memory operations.

    Wraps retrieval and storage functions into a callable interface
    that agents can use during their execution loops.
    """

    def __init__(
        self,
        store: Optional[AutFinStore] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        """
        Initialize MemoryTools.

        Args:
            store: AutFinStore instance (creates default if None)
            vector_store: VectorStore instance (optional, for semantic search)
        """
        self._store = store or get_store()
        self._vector_store = vector_store

    @property
    def store(self) -> AutFinStore:
        return self._store

    @property
    def vector_store(self) -> Optional[VectorStore]:
        return self._vector_store

    # ------------------------------------------------------------------
    # Research memory
    # ------------------------------------------------------------------

    async def get_cached_research(
        self,
        symbols: Sequence[str],
        research_type: str,
        max_age_days: int = 7,
    ) -> Dict[str, Any]:
        """
        Check which symbols have recent research of a given type.

        Args:
            symbols: NSE symbols to check
            research_type: e.g. "fundamentals", "technical", "sentiment", "macro"
            max_age_days: Maximum age in days to consider fresh

        Returns:
            Dict mapping symbol → report for fresh entries
        """
        return await get_latest_research(
            symbols, research_type, max_age_days=max_age_days, store=self._store
        )

    async def save_research(
        self,
        symbols: Sequence[str],
        research_type: str,
        reports: Dict[str, Any],
    ) -> None:
        """
        Cache research results by type.

        Args:
            symbols: NSE symbols
            research_type: e.g. "fundamentals"
            reports: Dict mapping symbol → report data
        """
        await store_research(symbols, research_type, reports, store=self._store)

    async def get_research(
        self, stock: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get latest research report for a single stock.

        Args:
            stock: NSE symbol

        Returns:
            Research report dict or None
        """
        return await get_research_report(stock, store=self._store)

    async def save_research_report(
        self, stock: str, report: Dict[str, Any]
    ) -> None:
        """
        Save a full research report for a stock.

        Args:
            stock: NSE symbol
            report: Research report dict
        """
        await store_research_report(stock, report, store=self._store)

    # ------------------------------------------------------------------
    # Trade history
    # ------------------------------------------------------------------

    async def get_trades(self, stock: str) -> List[Dict[str, Any]]:
        """
        Get trade history for a stock.

        Args:
            stock: NSE symbol

        Returns:
            List of trade records
        """
        return await get_trade_history(stock, store=self._store)

    async def save_trade(
        self, stock: str, trade: Dict[str, Any]
    ) -> None:
        """
        Record a trade in history.

        Args:
            stock: NSE symbol
            trade: Trade record dict
        """
        await store_trade_history(stock, trade, store=self._store)

    # ------------------------------------------------------------------
    # Decision traces
    # ------------------------------------------------------------------

    async def get_decision(
        self, decision_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a decision trace by ID.

        Args:
            decision_id: Decision identifier

        Returns:
            Decision trace dict or None
        """
        return await get_decision(decision_id, store=self._store)

    async def save_decision(
        self, decision_id: str, decision: Dict[str, Any]
    ) -> None:
        """
        Persist a decision trace.

        Args:
            decision_id: Unique decision identifier
            decision: Full decision trace dict
        """
        await store_trade_decision(decision_id, decision, store=self._store)

    # ------------------------------------------------------------------
    # Capital & portfolio state
    # ------------------------------------------------------------------

    async def get_capital(self) -> Optional[Dict[str, Any]]:
        """Get latest capital state snapshot."""
        return await get_capital_state(store=self._store)

    async def save_capital(self, state: Dict[str, Any]) -> None:
        """Persist capital state snapshot."""
        await store_capital_state(state, store=self._store)

    async def get_portfolio(self) -> Optional[Dict[str, Any]]:
        """Get latest portfolio state snapshot."""
        return await get_portfolio_state(store=self._store)

    async def save_portfolio(self, state: Dict[str, Any]) -> None:
        """Persist portfolio state snapshot."""
        await store_portfolio_state(state, store=self._store)

    # ------------------------------------------------------------------
    # Composite context building
    # ------------------------------------------------------------------

    async def build_context(
        self,
        stock: str,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a rich research context from all memory sources.

        Args:
            stock: NSE symbol
            thread_id: Optional thread ID

        Returns:
            Dict with past_research, trade_history, learnings, etc.
        """
        return await build_research_context(
            stock,
            store=self._store,
            vector_store=self._vector_store,
            thread_id=thread_id,
        )

    # ------------------------------------------------------------------
    # Vector Store operations
    # ------------------------------------------------------------------

    def save_strategy_learning(
        self,
        content: str,
        metadata: Dict[str, Any],
        doc_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Store a strategy learning in the vector store.

        Args:
            content: Text to embed
            metadata: Metadata dict
            doc_id: Optional explicit ID

        Returns:
            Stored document ID
        """
        if self._vector_store is None:
            logger.warning("No vector store configured — cannot save strategy learning")
            return None
        return store_strategy_learning(
            self._vector_store, content, metadata, doc_id=doc_id
        )

    def save_company_knowledge(
        self,
        content: str,
        metadata: Dict[str, Any],
        doc_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Store company/sector knowledge in the vector store.

        Args:
            content: Text to embed
            metadata: Metadata dict
            doc_id: Optional explicit ID

        Returns:
            Stored document ID
        """
        if self._vector_store is None:
            logger.warning("No vector store configured — cannot save company knowledge")
            return None
        return store_company_knowledge(
            self._vector_store, content, metadata, doc_id=doc_id
        )

    def save_market_knowledge(
        self,
        content: str,
        metadata: Dict[str, Any],
        doc_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Store market knowledge in the vector store.

        Args:
            content: Text to embed
            metadata: Metadata dict
            doc_id: Optional explicit ID

        Returns:
            Stored document ID
        """
        if self._vector_store is None:
            logger.warning("No vector store configured — cannot save market knowledge")
            return None
        return store_market_knowledge(
            self._vector_store, content, metadata, doc_id=doc_id
        )

    # ── Performance tracking ──────────────────────────────────

    def get_performance_metrics(self, days: Optional[int] = None) -> Dict[str, Any]:
        """
        Get all performance metrics (win rate, profit factor, symbol breakdown).

        Args:
            days: Optional lookback period in days

        Returns:
            Complete performance snapshot dict
        """
        return get_all_performance_metrics(days=days)

    def get_win_rate(self, days: Optional[int] = None) -> Dict[str, Any]:
        """
        Get win/loss ratio and related metrics.

        Args:
            days: Optional lookback period

        Returns:
            Dict with total_trades, wins, losses, win_rate, total_pnl
        """
        return get_win_rate(days=days)

    def get_strategy_comparison(self, days: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get per-symbol performance breakdown for strategy comparison.

        Args:
            days: Optional lookback period

        Returns:
            List of per-symbol performance dicts
        """
        return get_symbol_performance(days=days)

    def record_trade_outcome(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        quantity: int,
        action: str = "BUY",
        thesis_summary: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Record the outcome of a completed trade cycle.

        Args:
            symbol: NSE symbol
            entry_price: Price at entry
            exit_price: Price at exit
            quantity: Number of shares
            action: "BUY" or "SELL"
            thesis_summary: Optional thesis summary text

        Returns:
            Dict with symbol, pnl_pct, outcome
        """
        result = record_trade_outcome(
            symbol=symbol,
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            action=action,
            thesis_summary=thesis_summary,
        )

        # Also store strategy learning in vector store if available
        if self._vector_store is not None and thesis_summary:
            try:
                store_strategy_learning(
                    self._vector_store,
                    content=thesis_summary,
                    metadata={
                        "symbol": symbol,
                        "pnl": result["pnl_pct"],
                        "outcome": result["outcome"],
                        "date": result["trade_date"],
                    },
                )
            except Exception as e:
                logger.warning("Failed to store strategy learning: %s", e)

        return result

    def get_monthly_summary(self, year: int, month: int) -> Dict[str, Any]:
        """
        Get monthly performance summary.

        Args:
            year: Year (e.g. 2026)
            month: Month (1-12)

        Returns:
            Dict with monthly metrics
        """
        return get_monthly_summary(year=year, month=month)

    def get_drawdown_stats(self) -> Dict[str, Any]:
        """
        Get drawdown statistics from equity curve.

        Returns:
            Dict with max_drawdown, current_drawdown, etc.
        """
        return get_drawdown_stats()
