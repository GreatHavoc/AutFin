"""
Memory System — 3-Type Persistent Agent Memory.

Based on HedgeAgents (2025) and OpenClaw architectures. Three memory types
that compound over time, enabling agents to learn from past decisions:

1. Market Memory — what happened (prices, news, events, regime changes)
2. Reflection Memory — what worked/failed (trade outcomes, lessons learned)
3. Experience Memory — patterns that repeat (sector rotations, seasonal)

Each memory type has its own storage and retrieval strategy. All are
accessible via natural language queries and updated after each session.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

from src.autfin.llm.client import get_client, primary_model

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    MARKET = "market"        # What happened (prices, news, events)
    REFLECTION = "reflection"  # What worked/failed (trade outcomes)
    EXPERIENCE = "experience"  # Patterns that repeat (seasonal, sector)


@dataclass
class MemoryEntry:
    """One memory in the system."""
    memory_type: MemoryType
    key: str
    value: str
    symbol: Optional[str]
    confidence: float  # 0.0-1.0, how confident we are in this memory
    source: str  # which agent created this memory
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    tags: list[str] = field(default_factory=list)
    related_symbols: list[str] = field(default_factory=list)


@dataclass
class MemoryQuery:
    """Query for retrieving memories."""
    query: str
    memory_type: Optional[MemoryType] = None
    symbol: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    min_confidence: float = 0.0
    max_age_days: Optional[int] = None
    limit: int = 10


@dataclass
class MemoryStats:
    """Statistics about the memory system."""
    total_memories: int
    by_type: dict[str, int]
    avg_confidence: float
    most_accessed: list[dict]
    recent_additions: int  # last 7 days


class MemorySystem:
    """
    Persistent memory system for the trading agent.

    Stores and retrieves three types of memory:
    1. Market Memory: what happened in the market
    2. Reflection Memory: what worked/failed in our trades
    3. Experience Memory: patterns that repeat

    All memories are stored in the LangGraph Store via namespaces,
    enabling cross-session persistence and natural language retrieval.
    """

    def __init__(self, store=None):
        self.store = store
        self._cache: dict[str, list[MemoryEntry]] = {}

    # -------------------------------------------------------------------
    # Market Memory — what happened
    # -------------------------------------------------------------------

    async def record_market_event(
        self,
        event: str,
        symbols: list[str],
        regime: str = "normal",
        confidence: float = 0.8,
        source: str = "global_analyst",
    ) -> MemoryEntry:
        """Record a market event (news, regime change, macro event)."""
        entry = MemoryEntry(
            memory_type=MemoryType.MARKET,
            key=f"market_{datetime.now().strftime('%Y%m%d_%H%M')}_{symbols[0] if symbols else 'general'}",
            value=event,
            symbol=symbols[0] if symbols else None,
            confidence=confidence,
            source=source,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            tags=[regime, "market_event"],
            related_symbols=symbols,
        )

        await self._store_entry(entry)
        return entry

    async def get_market_context(
        self,
        symbol: str,
        days: int = 7,
    ) -> list[MemoryEntry]:
        """Get recent market events relevant to a symbol."""
        query = MemoryQuery(
            query=f"market events for {symbol}",
            memory_type=MemoryType.MARKET,
            symbol=symbol,
            max_age_days=days,
            limit=10,
        )
        return await self._retrieve_entries(query)

    # -------------------------------------------------------------------
    # Reflection Memory — what worked/failed
    # -------------------------------------------------------------------

    async def record_trade_outcome(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        exit_price: float,
        thesis: str,
        outcome: str,
        pnl_pct: float,
        lessons: list[str],
        confidence: float = 0.9,
        source: str = "trade_review",
    ) -> MemoryEntry:
        """Record a completed trade and its lessons."""
        value = json.dumps({
            "symbol": symbol,
            "action": action,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
            "thesis": thesis,
            "outcome": outcome,
            "lessons": lessons,
        })

        entry = MemoryEntry(
            memory_type=MemoryType.REFLECTION,
            key=f"reflection_{symbol}_{datetime.now().strftime('%Y%m%d')}",
            value=value,
            symbol=symbol,
            confidence=confidence,
            source=source,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            tags=[action.lower(), "trade_outcome", outcome],
            related_symbols=[symbol],
        )

        await self._store_entry(entry)
        return entry

    async def get_trade_reflections(
        self,
        symbol: Optional[str] = None,
        outcome: Optional[str] = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Get past trade reflections, optionally filtered."""
        query = MemoryQuery(
            query=f"trade reflections for {symbol or 'all symbols'}",
            memory_type=MemoryType.REFLECTION,
            symbol=symbol,
            limit=limit,
        )
        return await self._retrieve_entries(query)

    async def get_calibration_data(self) -> dict:
        """
        Get conviction calibration data from past trades.

        Returns accuracy of conviction scores:
        - conviction 0.8-1.0: what was actual win rate?
        - conviction 0.6-0.8: what was actual win rate?
        - conviction 0.4-0.6: what was actual win rate?
        """
        reflections = await self.get_trade_reflections(limit=50)

        tiers = {
            "high": {"range": (0.8, 1.0), "trades": [], "actual_win_rate": 0},
            "medium": {"range": (0.6, 0.8), "trades": [], "actual_win_rate": 0},
            "low": {"range": (0.4, 0.6), "trades": [], "actual_win_rate": 0},
        }

        for entry in reflections:
            try:
                data = json.loads(entry.value)
                # We don't store conviction in the reflection, but we can
                # infer from confidence
                confidence = entry.confidence
                if confidence >= 0.8:
                    tiers["high"]["trades"].append(data)
                elif confidence >= 0.6:
                    tiers["medium"]["trades"].append(data)
                else:
                    tiers["low"]["trades"].append(data)
            except (json.JSONDecodeError, KeyError):
                continue

        # Calculate actual win rates
        for tier_name, tier_data in tiers.items():
            trades = tier_data["trades"]
            if trades:
                wins = sum(1 for t in trades if t.get("pnl_pct", 0) > 0)
                tier_data["actual_win_rate"] = wins / len(trades)
            tier_data["count"] = len(trades)

        return tiers

    # -------------------------------------------------------------------
    # Experience Memory — patterns that repeat
    # -------------------------------------------------------------------

    async def record_pattern(
        self,
        pattern: str,
        symbols: list[str],
        confidence: float = 0.7,
        evidence: str = "",
        source: str = "experience_conference",
    ) -> MemoryEntry:
        """Record a discovered pattern (sector rotation, seasonal, etc.)."""
        entry = MemoryEntry(
            memory_type=MemoryType.EXPERIENCE,
            key=f"pattern_{datetime.now().strftime('%Y%m%d')}_{pattern[:30]}",
            value=json.dumps({
                "pattern": pattern,
                "evidence": evidence,
                "symbols": symbols,
            }),
            symbol=symbols[0] if symbols else None,
            confidence=confidence,
            source=source,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            tags=["pattern", "experience"],
            related_symbols=symbols,
        )

        await self._store_entry(entry)
        return entry

    async def get_patterns(
        self,
        symbol: Optional[str] = None,
        min_confidence: float = 0.5,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Get discovered patterns, optionally filtered by symbol."""
        query = MemoryQuery(
            query=f"patterns for {symbol or 'general'}",
            memory_type=MemoryType.EXPERIENCE,
            symbol=symbol,
            min_confidence=min_confidence,
            limit=limit,
        )
        return await self._retrieve_entries(query)

    # -------------------------------------------------------------------
    # Cross-type queries
    # -------------------------------------------------------------------

    async def get_context_for_decision(
        self,
        symbol: str,
        decision_type: str,
    ) -> dict:
        """
        Get all relevant context for a trading decision.

        Combines market memory, trade reflections, and patterns
        into a single context dict for the decision-maker.
        """
        market = await self.get_market_context(symbol, days=30)
        reflections = await self.get_trade_reflections(symbol=symbol, limit=5)
        patterns = await self.get_patterns(symbol=symbol, min_confidence=0.6)
        calibration = await self.get_calibration_data()

        return {
            "symbol": symbol,
            "decision_type": decision_type,
            "market_events": [
                {"event": e.value, "confidence": e.confidence, "date": e.created_at.isoformat()}
                for e in market
            ],
            "past_trades": [
                {"outcome": e.value, "confidence": e.confidence}
                for e in reflections
            ],
            "relevant_patterns": [
                {"pattern": e.value, "confidence": e.confidence}
                for e in patterns
            ],
            "calibration": calibration,
            "total_memories_consulted": len(market) + len(reflections) + len(patterns),
        }

    # -------------------------------------------------------------------
    # Storage (via LangGraph Store)
    # -------------------------------------------------------------------

    async def _store_entry(self, entry: MemoryEntry):
        """Store a memory entry in the LangGraph Store."""
        if self.store is None:
            # Cache only (no persistent store)
            cache_key = entry.memory_type.value
            if cache_key not in self._cache:
                self._cache[cache_key] = []
            self._cache[cache_key].append(entry)
            return

        try:
            from src.autfin.memory.langgraph_store import (
                NS_TRADE_HISTORY, store_upsert,
            )
            await store_upsert(
                self.store,
                NS_TRADE_HISTORY,
                entry.key,
                entry.value,
                {
                    "memory_type": entry.memory_type.value,
                    "symbol": entry.symbol or "",
                    "confidence": entry.confidence,
                    "source": entry.source,
                    "tags": ",".join(entry.tags),
                },
            )
        except Exception as e:
            logger.error("Failed to store memory: %s", e)

    async def _retrieve_entries(self, query: MemoryQuery) -> list[MemoryEntry]:
        """Retrieve memory entries matching a query."""
        if self.store is None:
            # Use cache
            cache_key = query.memory_type.value if query.memory_type else "all"
            cached = self._cache.get(cache_key, [])
            results = [
                e for e in cached
                if (query.symbol is None or e.symbol == query.symbol)
                and e.confidence >= query.min_confidence
            ]
            return results[:query.limit]

        try:
            from src.autfin.memory.langgraph_store import store_search
            results = await store_search(
                self.store,
                NS_TRADE_HISTORY,
                query.query,
                limit=query.limit,
            )
            # Convert to MemoryEntry objects
            entries = []
            for r in results:
                try:
                    # Try to parse as JSON
                    value = r.get("value", "")
                    data = json.loads(value) if isinstance(value, str) else value
                    if isinstance(data, dict):
                        entry = MemoryEntry(
                            memory_type=MemoryType(data.get("memory_type", "market")),
                            key=r.get("key", ""),
                            value=value,
                            symbol=data.get("symbol"),
                            confidence=float(data.get("confidence", 0.5)),
                            source=data.get("source", "unknown"),
                            created_at=datetime.now(),
                            last_accessed=datetime.now(),
                        )
                        entries.append(entry)
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
            return entries[:query.limit]
        except Exception as e:
            logger.error("Failed to retrieve memories: %s", e)
            return []

    async def get_stats(self) -> MemoryStats:
        """Get statistics about the memory system."""
        total = 0
        by_type = {}
        for cache_key, entries in self._cache.items():
            count = len(entries)
            by_type[cache_key] = count
            total += count

        return MemoryStats(
            total_memories=total,
            by_type=by_type,
            avg_confidence=0.0,  # placeholder
            most_accessed=[],
            recent_additions=0,
        )
