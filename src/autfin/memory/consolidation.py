"""Memory consolidation and TTL management for AutFin.

Manages expiration of old entries, size limits per namespace,
and periodic cleanup to prevent unbounded memory growth.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# TTL configurations (in days)
TTL_CONFIG: Dict[str, Optional[int]] = {
    "trade_history": 365,          # Keep trades for 1 year
    "research_reports": 90,        # Keep research for 90 days
    "strategy_learnings": None,    # Never expire learnings
    "company_knowledge": 180,      # Keep company info for 6 months
    "market_knowledge": 30,        # Keep market data for 30 days
}

# Max size per namespace
MAX_SIZE_CONFIG: Dict[str, int] = {
    "trade_history": 500,          # Max 500 trade records
    "research_reports": 100,       # Max 100 research reports
    "strategy_learnings": 200,     # Max 200 learnings
    "company_knowledge": 200,      # Max 200 company profiles
    "market_knowledge": 100,       # Max 100 market snapshots
}


class MemoryConsolidator:
    """Manages TTL expiration and size limits for stored memories."""

    def __init__(self, store=None):
        self._store = store

    async def consolidate(self) -> Dict[str, int]:
        """Run full consolidation: expire old entries, enforce size limits.

        Returns:
            Stats dict with expired, trimmed, and errors counts
        """
        # from src.autfin.memory.store import get_store  # REMOVED
        store = self._store or get_store()

        stats: Dict[str, int] = {"expired": 0, "trimmed": 0, "errors": 0}

        for namespace, ttl_days in TTL_CONFIG.items():
            try:
                if ttl_days is not None:
                    expired = await self._expire_old(store, namespace, ttl_days)
                    stats["expired"] += expired

                max_size = MAX_SIZE_CONFIG.get(namespace)
                if max_size:
                    trimmed = await self._enforce_size_limit(store, namespace, max_size)
                    stats["trimmed"] += trimmed
            except Exception as e:
                logger.warning("Consolidation failed for %s: %s", namespace, e)
                stats["errors"] += 1

        logger.info("Memory consolidation complete: %s", stats)
        return stats

    async def _expire_old(self, store, namespace: str, ttl_days: int) -> int:
        """Remove entries older than TTL.

        Args:
            store: AutFinStore instance
            namespace: Namespace to clean
            ttl_days: Time-to-live in days

        Returns:
            Number of expired entries removed
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
        expired = 0

        try:
            keys = await store._list_ns_keys(namespace)
            for key in keys:
                try:
                    value = await store.aget(namespace, key)
                    if value and isinstance(value, dict):
                        created_at = value.get("created_at") or value.get("timestamp")
                        if created_at:
                            try:
                                stored_at = datetime.fromisoformat(str(created_at))
                                if stored_at.tzinfo is None:
                                    stored_at = stored_at.replace(tzinfo=timezone.utc)
                                if stored_at < cutoff:
                                    await store.adelete(namespace, key)
                                    expired += 1
                            except (ValueError, TypeError):
                                pass
                except Exception:
                    continue
        except Exception as e:
            logger.debug("expire_old failed for %s: %s", namespace, e)

        return expired

    async def _enforce_size_limit(self, store, namespace: str, max_size: int) -> int:
        """Keep only the most recent max_size entries.

        Args:
            store: AutFinStore instance
            namespace: Namespace to trim
            max_size: Maximum number of entries to keep

        Returns:
            Number of entries trimmed
        """
        trimmed = 0

        try:
            keys = await store._list_ns_keys(namespace)
            if len(keys) <= max_size:
                return 0

            # Collect entries with timestamps for sorting
            entries = []
            for key in keys:
                try:
                    value = await store.aget(namespace, key)
                    if value and isinstance(value, dict):
                        ts = value.get("created_at") or value.get("timestamp") or ""
                        entries.append((key, str(ts)))
                    else:
                        entries.append((key, ""))
                except Exception:
                    entries.append((key, ""))

            # Sort by timestamp ascending (oldest first)
            entries.sort(key=lambda x: x[1])

            # Delete oldest entries exceeding limit
            to_delete = len(entries) - max_size
            for key, _ in entries[:to_delete]:
                try:
                    await store.adelete(namespace, key)
                    trimmed += 1
                except Exception:
                    continue
        except Exception as e:
            logger.debug("enforce_size_limit failed for %s: %s", namespace, e)

        return trimmed


async def consolidate_memory() -> Dict[str, int]:
    """Run memory consolidation (TTL expiry + capacity trimming).

    Returns:
        Stats dict with expired, trimmed, and errors counts
    """
    consolidator = MemoryConsolidator()
    return await consolidator.consolidate()
