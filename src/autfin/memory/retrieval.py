"""
Memory retrieval strategies for AutFin.

Provides cross-thread memory via Pydantic AI Memory and semantic search
via ChromaDB Vector Store. Used by agents to build context from
past research, trade history, and learnings.

Design references:
- technical_spec.md §8: build_research_context(symbol, store, vector_store, thread_id)
- PRD §4.2-4.3: Memory namespaces and retrieval strategies
- autfin-design.md §9: Store namespace constants
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Sequence

# from .store import AutFinStore, get_store  # REMOVED - use Pydantic AI Memory
from .vector_store import VectorStore, get_or_create_collection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Store namespace constants (PRD §4.2 / autfin-design.md §9)
# ---------------------------------------------------------------------------
NS_CAPITAL_STATE = "capital_state"
NS_PORTFOLIO_STATE = "portfolio_state"
NS_TRADE_HISTORY = "trade_history"       # key = symbol
NS_RESEARCH_REPORTS = "research_reports"  # key = symbol
NS_DECISIONS = "decisions"               # key = decision_id

# Vector Store collection constants
COLLECTION_MARKET_KNOWLEDGE = "market_data"
COLLECTION_STRATEGY_LEARNINGS = "learning_journal"
COLLECTION_COMPANY_KNOWLEDGE = "research_notes"


# ---------------------------------------------------------------------------
# Store helpers — write
# ---------------------------------------------------------------------------

async def store_research_report(
    stock: str,
    report: Dict[str, Any],
    store: Optional[AutFinStore] = None,
) -> None:
    """
    Persist a research report for a stock in the Pydantic AI Store.

    Args:
        stock: NSE symbol (e.g. "RELIANCE")
        report: Research output dict (fundamentals, technical, sentiment, etc.)
        store: Optional pre-configured AutFinStore instance
    """
    store = store or get_store()
    key = stock.upper()
    value = {
        "stock": key,
        "report": report,
        "stored_at": datetime.now(timezone.utc).isoformat(),
    }
    await store.aput(namespace=NS_RESEARCH_REPORTS, key=key, value=value)
    logger.info("Stored research report for %s", key)


async def store_research(
    symbols: Sequence[str],
    research_type: str,
    reports: Dict[str, Any],
    store: Optional[AutFinStore] = None,
) -> None:
    """
    Cache research results by type for one or more symbols.

    Called by agents (e.g. fundamentals_agent) after analysis to persist
    results.  Matches the pattern from technical_spec.md §4.3:
        await retrieval.store_research(symbols, "fundamentals", analyzed)

    Args:
        symbols: NSE symbols to store for
        research_type: e.g. "fundamentals", "technical", "sentiment", "macro"
        reports: Dict mapping symbol → report data
        store: Optional pre-configured AutFinStore instance
    """
    store = store or get_store()
    now_iso = datetime.now(timezone.utc).isoformat()

    for symbol in symbols:
        sym = symbol.upper()
        # Load existing research for this symbol
        existing = await store.aget(namespace=NS_RESEARCH_REPORTS, key=sym)
        value: Dict[str, Any] = {}
        if existing and isinstance(existing, dict):
            value = {k: v for k, v in existing.items() if k not in ("stock", "stored_at")}

        # Merge new research type
        if sym in reports:
            value[research_type] = reports[sym]
            value[f"{research_type}_stored_at"] = now_iso

        value["stock"] = sym
        value["stored_at"] = now_iso

        await store.aput(namespace=NS_RESEARCH_REPORTS, key=sym, value=value)
        logger.info("Stored %s research for %s", research_type, sym)


async def store_trade_decision(
    decision_id: str,
    decision: Dict[str, Any],
    store: Optional[AutFinStore] = None,
) -> None:
    """
    Persist a full decision trace (input → debate → risk → execution → outcome).

    Args:
        decision_id: Unique decision identifier
        decision: Full decision trace dict
        store: Optional pre-configured AutFinStore instance
    """
    store = store or get_store()
    value = {
        "decision_id": decision_id,
        "decision": decision,
        "stored_at": datetime.now(timezone.utc).isoformat(),
    }
    await store.aput(namespace=NS_DECISIONS, key=decision_id, value=value)
    logger.info("Stored decision %s", decision_id)


async def store_trade_history(
    stock: str,
    trade: Dict[str, Any],
    store: Optional[AutFinStore] = None,
) -> None:
    """
    Append a trade record to the stock's trade history.

    Existing history is loaded, the new trade appended, and the
    combined list re-stored.

    Args:
        stock: NSE symbol
        trade: Trade record dict (entry, exit, rationale, outcome, …)
        store: Optional pre-configured AutFinStore instance
    """
    store = store or get_store()
    key = stock.upper()

    # Load existing history
    existing = await store.aget(namespace=NS_TRADE_HISTORY, key=key)
    history: List[Dict[str, Any]] = []
    if existing and isinstance(existing, dict):
        history = existing.get("trades", [])
    elif existing and isinstance(existing, list):
        history = existing

    trade_with_ts = {
        **trade,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    history.append(trade_with_ts)

    value = {
        "stock": key,
        "trades": history,
        "trade_count": len(history),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    await store.aput(namespace=NS_TRADE_HISTORY, key=key, value=value)
    logger.info("Stored trade for %s (total: %d)", key, len(history))


async def store_capital_state(
    state: Dict[str, Any],
    store: Optional[AutFinStore] = None,
) -> None:
    """
    Persist a capital state snapshot (available_cash, deployment log).

    Args:
        state: Capital state dict
        store: Optional pre-configured AutFinStore instance
    """
    store = store or get_store()
    value = {
        **state,
        "snapshot_at": datetime.now(timezone.utc).isoformat(),
    }
    await store.aput(namespace=NS_CAPITAL_STATE, key="current", value=value)
    logger.info("Stored capital state snapshot")


async def store_portfolio_state(
    state: Dict[str, Any],
    store: Optional[AutFinStore] = None,
) -> None:
    """
    Persist portfolio state (positions, allocation, P&L, margins).

    Args:
        state: Portfolio state dict
        store: Optional pre-configured AutFinStore instance
    """
    store = store or get_store()
    value = {
        **state,
        "snapshot_at": datetime.now(timezone.utc).isoformat(),
    }
    await store.aput(namespace=NS_PORTFOLIO_STATE, key="current", value=value)
    logger.info("Stored portfolio state snapshot")


# ---------------------------------------------------------------------------
# Store helpers — read
# ---------------------------------------------------------------------------

async def get_research_report(
    stock: str,
    store: Optional[AutFinStore] = None,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve the latest research report for a stock.

    Args:
        stock: NSE symbol
        store: Optional pre-configured AutFinStore instance

    Returns:
        Stored research report dict or None
    """
    store = store or get_store()
    result = await store.aget(namespace=NS_RESEARCH_REPORTS, key=stock.upper())
    if result and isinstance(result, dict):
        # Handle both wrapped ("report" key) and unwrapped storage formats
        if "report" in result:
            return result.get("report")
        # If the result has research-type keys directly, return it
        research_keys = {"fundamentals", "technical", "sentiment", "macro", "debate"}
        if any(k in result for k in research_keys):
            return {k: v for k, v in result.items() if k not in ("stock", "stored_at")}
        return result
    return None


async def get_latest_research(
    symbols: Sequence[str],
    research_type: str,
    max_age_days: int = 7,
    store: Optional[AutFinStore] = None,
) -> Dict[str, Any]:
    """
    Check which symbols have recent research of a given type.

    Called by agents to decide which stocks need fresh analysis.
    Matches the pattern from technical_spec.md §4.3:
        cached = await retrieval.get_latest_research(symbols, "fundamentals", max_age_days=7)

    Args:
        symbols: NSE symbols to check
        research_type: e.g. "fundamentals", "technical", "sentiment", "macro"
        max_age_days: Maximum age in days to consider fresh
        store: Optional pre-configured AutFinStore instance

    Returns:
        Dict mapping symbol → report data for symbols with fresh research
    """
    store = store or get_store()
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    cached: Dict[str, Any] = {}

    for symbol in symbols:
        sym = symbol.upper()
        result = await store.aget(namespace=NS_RESEARCH_REPORTS, key=sym)
        if not result or not isinstance(result, dict):
            continue

        # Check if the specific research type exists and is fresh
        stored_at_str = result.get(f"{research_type}_stored_at")
        if stored_at_str:
            try:
                stored_at = datetime.fromisoformat(stored_at_str)
                if stored_at > cutoff:
                    cached[sym] = result.get(research_type)
            except (ValueError, TypeError):
                logger.debug("Invalid stored_at timestamp format", exc_info=True)

    return cached


async def get_trade_history(
    stock: str,
    store: Optional[AutFinStore] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve trade history for a stock.

    Args:
        stock: NSE symbol
        store: Optional pre-configured AutFinStore instance

    Returns:
        List of trade records (may be empty)
    """
    store = store or get_store()
    result = await store.aget(namespace=NS_TRADE_HISTORY, key=stock.upper())
    if result and isinstance(result, dict):
        return result.get("trades", [])
    if result and isinstance(result, list):
        return result
    return []


async def get_capital_state(
    store: Optional[AutFinStore] = None,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve the latest capital state snapshot.

    Returns:
        Capital state dict or None
    """
    store = store or get_store()
    return await store.aget(namespace=NS_CAPITAL_STATE, key="current")


async def get_portfolio_state(
    store: Optional[AutFinStore] = None,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve the latest portfolio state snapshot.

    Returns:
        Portfolio state dict or None
    """
    store = store or get_store()
    return await store.aget(namespace=NS_PORTFOLIO_STATE, key="current")


async def get_decision(
    decision_id: str,
    store: Optional[AutFinStore] = None,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve a full decision trace by ID.

    Args:
        decision_id: Decision identifier
        store: Optional pre-configured AutFinStore instance

    Returns:
        Decision trace dict or None
    """
    store = store or get_store()
    return await store.aget(namespace=NS_DECISIONS, key=decision_id)


# ---------------------------------------------------------------------------
# Composite retrieval — build_research_context
# (technical_spec.md §8)
# ---------------------------------------------------------------------------

async def load_agent_context(
    symbol: str,
    agent_name: str = "unknown",
    store: Optional[AutFinStore] = None,
    vector_store: Optional[VectorStore] = None,
) -> Dict[str, Any]:
    """
    Pillar 7: Per-agent memory context loader.

    Each agent can call this at the start of analysis to independently pull
    episodic, semantic, and procedural memory for the stock being analyzed.

    Wraps ``build_research_context()`` with agent-awareness — tags retrievals
    with agent_name for scoped memory queries.

    Args:
        symbol: NSE symbol (e.g. 'INFY')
        agent_name: Which agent is requesting context
        store: Optional pre-configured AutFinStore
        vector_store: Optional pre-configured VectorStore

    Returns:
        Same structure as ``build_research_context()`` with additional
        ``agent_name`` key.
    """
    context = await build_research_context(
        stock=symbol,
        store=store,
        vector_store=vector_store,
    )
    context["agent_name"] = agent_name
    context["loaded_by"] = agent_name
    return context


# ---------------------------------------------------------------------------
# build_research_context — full memory retrieval pipeline

async def build_research_context(
    stock: str,
    store: Optional[AutFinStore] = None,
    vector_store: Optional[VectorStore] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a rich research context for an agent by pulling from multiple
    memory sources (technical_spec.md §8):

    1. Past research on this stock (Store)
    2. Similar past trade rationales (Vector Store — strategy_learnings)
    3. Sector knowledge (Vector Store — company_knowledge)
    4. Recent performance / past outcomes (Store — trade history)

    Args:
        stock: NSE symbol
        store: Optional pre-configured AutFinStore instance
        vector_store: Optional pre-configured VectorStore instance
        thread_id: Optional thread ID for context

    Returns:
        Dict with keys: past_research, trade_history, past_decisions,
        similar_trades_learnings, sector_context, recent_90d_pnl,
        context_summary
    """
    stock = stock.upper()
    store = store or get_store()

    # 1. Cross-thread memory via Store
    past_reports = await store.aget(namespace=NS_RESEARCH_REPORTS, key=stock)
    past_trades = await store.aget(namespace=NS_TRADE_HISTORY, key=stock)

    # Extract trade list for trade_history field
    trade_history: List[Dict[str, Any]] = []
    if past_trades and isinstance(past_trades, dict):
        trade_history = past_trades.get("trades", [])
    elif past_trades and isinstance(past_trades, list):
        trade_history = past_trades

    # 2. Semantic memory via Vector Store
    similar_trades_learnings: List[Dict[str, Any]] = []
    sector_context: List[Dict[str, Any]] = []

    if vector_store is not None:
        # Strategy learnings — similar past decisions
        try:
            learnings_col = vector_store.get_collection(COLLECTION_STRATEGY_LEARNINGS)
            if learnings_col is not None and learnings_col.count() > 0:
                results = learnings_col.query(
                    query_texts=[f"Past decisions and learnings for {stock}"],
                    n_results=5,
                )
                if results and results.get("documents"):
                    for i, doc in enumerate(results["documents"][0]):
                        meta = {}
                        if results.get("metadatas") and results["metadatas"][0]:
                            meta = results["metadatas"][0][i]
                        similar_trades_learnings.append({"content": doc, "metadata": meta})
        except Exception as e:
            logger.warning("Failed to query strategy learnings: %s", e)

        # Company / sector knowledge
        try:
            company_col = vector_store.get_collection(COLLECTION_COMPANY_KNOWLEDGE)
            if company_col is not None and company_col.count() > 0:
                results = company_col.query(
                    query_texts=[f"Company profile and sector knowledge for {stock}"],
                    n_results=5,
                )
                if results and results.get("documents"):
                    for i, doc in enumerate(results["documents"][0]):
                        meta = {}
                        if results.get("metadatas") and results["metadatas"][0]:
                            meta = results["metadatas"][0][i]
                        sector_context.append({"content": doc, "metadata": meta})
        except Exception as e:
            logger.warning("Failed to query company knowledge: %s", e)

    # 3. Compute recent P&L from trade history (last 90 days)
    recent_90d_pnl = _compute_recent_pnl(trade_history, days=90)

    # 4. Performance metrics from learning_journal (SQLite)
    perf_metrics = _get_stock_performance(stock)

    # 5. Build context summary
    context_parts: List[str] = []
    if past_reports:
        context_parts.append(f"Past research available for {stock}.")
    if trade_history:
        context_parts.append(f"{len(trade_history)} past trade(s) found for {stock}.")
    if similar_trades_learnings:
        context_parts.append(
            f"{len(similar_trades_learnings)} relevant learning(s) from past decisions."
        )
    if sector_context:
        context_parts.append(
            f"{len(sector_context)} company knowledge entry/entries found."
        )
    if recent_90d_pnl != 0.0:
        direction = "profit" if recent_90d_pnl > 0 else "loss"
        context_parts.append(
            f"Recent 90-day P&L: ₹{recent_90d_pnl:,.0f} ({direction})."
        )
    if not context_parts:
        context_parts.append(f"No prior memory found for {stock}.")

    return {
        "stock": stock,
        "past_research": past_reports.get("report") if past_reports and isinstance(past_reports, dict) else past_reports,
        "trade_history": trade_history,
        "past_decisions": None,  # Retrieved separately if needed
        "similar_trades_learnings": similar_trades_learnings,
        "sector_context": sector_context,
        "recent_90d_pnl": recent_90d_pnl,
        "performance": perf_metrics,
        "context_summary": " ".join(context_parts),
    }


def _get_stock_performance(stock: str) -> Dict[str, Any]:
    """
    Get performance metrics for a specific stock from the learning_journal table.

    Args:
        stock: NSE symbol

    Returns:
        Dict with total_trades, wins, losses, win_rate, total_pnl, avg_pnl
    """
    from .database import get_connection

    try:
        conn = get_connection()
        try:
            rows = conn.execute(
                """
                SELECT pnl_pct, outcome
                FROM learning_journal
                WHERE symbol = ? AND outcome IS NOT NULL
                """,
                (stock.upper(),),
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
            }

        total = len(rows)
        wins = sum(1 for _, o in rows if o == "completed")
        pnls = [p for p, _ in rows if p is not None]
        total_pnl = sum(pnls)

        return {
            "total_trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": wins / total if total > 0 else 0.0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(pnls) if pnls else 0.0,
        }
    except Exception as e:
        logger.warning("Failed to get stock performance for %s: %s", stock, e)
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
        }


def _compute_recent_pnl(trades: List[Dict[str, Any]], days: int = 90) -> float:
    """
    Compute total P&L from trades in the last N days.

    Args:
        trades: List of trade records
        days: Number of days to look back

    Returns:
        Total P&L as float
    """
    if not trades:
        return 0.0

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    total_pnl = 0.0

    for trade in trades:
        # Check recorded_at or timestamp
        ts_str = trade.get("recorded_at") or trade.get("timestamp")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts < cutoff:
                continue
        except (ValueError, TypeError):
            continue

        pnl = trade.get("pnl_pct") or trade.get("pnl") or 0.0
        total_pnl += float(pnl)

    return total_pnl


# ---------------------------------------------------------------------------
# Vector Store write helpers
# ---------------------------------------------------------------------------

def store_strategy_learning(
    vector_store: VectorStore,
    content: str,
    metadata: Dict[str, Any],
    doc_id: Optional[str] = None,
) -> Optional[str]:
    """
    Store a strategy learning (reflection on a past trade) in the vector store.

    Args:
        vector_store: VectorStore instance
        content: Text content to embed
        metadata: Metadata dict (symbol, date, outcome, etc.)
        doc_id: Optional explicit ID

    Returns:
        Stored document ID
    """
    import uuid

    collection = vector_store.get_collection(COLLECTION_STRATEGY_LEARNINGS)
    if collection is None:
        collection = get_or_create_collection(
            vector_store.client, COLLECTION_STRATEGY_LEARNINGS
        )

    doc_id = doc_id or str(uuid.uuid4())
    collection.add(
        documents=[content],
        metadatas=[metadata],
        ids=[doc_id],
    )
    logger.info("Stored strategy learning %s", doc_id)
    return doc_id


def store_company_knowledge(
    vector_store: VectorStore,
    content: str,
    metadata: Dict[str, Any],
    doc_id: Optional[str] = None,
) -> Optional[str]:
    """
    Store company/sector knowledge in the vector store.

    Args:
        vector_store: VectorStore instance
        content: Text content to embed
        metadata: Metadata dict (symbol, sector, etc.)
        doc_id: Optional explicit ID

    Returns:
        Stored document ID
    """
    import uuid

    collection = vector_store.get_collection(COLLECTION_COMPANY_KNOWLEDGE)
    if collection is None:
        collection = get_or_create_collection(
            vector_store.client, COLLECTION_COMPANY_KNOWLEDGE
        )

    doc_id = doc_id or str(uuid.uuid4())
    collection.add(
        documents=[content],
        metadatas=[metadata],
        ids=[doc_id],
    )
    logger.info("Stored company knowledge %s", doc_id)
    return doc_id


def store_market_knowledge(
    vector_store: VectorStore,
    content: str,
    metadata: Dict[str, Any],
    doc_id: Optional[str] = None,
) -> Optional[str]:
    """
    Store market knowledge (embedded research summaries) in the vector store.

    Args:
        vector_store: VectorStore instance
        content: Text content to embed
        metadata: Metadata dict (symbol, date, type, etc.)
        doc_id: Optional explicit ID

    Returns:
        Stored document ID
    """
    import uuid

    collection = vector_store.get_collection(COLLECTION_MARKET_KNOWLEDGE)
    if collection is None:
        collection = get_or_create_collection(
            vector_store.client, COLLECTION_MARKET_KNOWLEDGE
        )

    doc_id = doc_id or str(uuid.uuid4())
    collection.add(
        documents=[content],
        metadatas=[metadata],
        ids=[doc_id],
    )
    logger.info("Stored market knowledge %s", doc_id)
    return doc_id
