"""Tests for memory consolidation and TTL management."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock


class TestTTLConfig:
    def test_ttl_config_has_required_namespaces(self):
        from src.autfin.memory.consolidation import TTL_CONFIG
        assert "trade_history" in TTL_CONFIG
        assert "research_reports" in TTL_CONFIG
        assert "strategy_learnings" in TTL_CONFIG

    def test_trade_history_ttl_is_1_year(self):
        from src.autfin.memory.consolidation import TTL_CONFIG
        assert TTL_CONFIG["trade_history"] == 365

    def test_learnings_never_expire(self):
        from src.autfin.memory.consolidation import TTL_CONFIG
        assert TTL_CONFIG["strategy_learnings"] is None


class TestMaxSizeConfig:
    def test_max_size_limits_set(self):
        from src.autfin.memory.consolidation import MAX_SIZE_CONFIG
        assert MAX_SIZE_CONFIG["trade_history"] == 500
        assert MAX_SIZE_CONFIG["research_reports"] == 100


class TestConsolidatorBasics:
    def test_consolidator_init_no_store(self):
        from src.autfin.memory.consolidation import MemoryConsolidator
        c = MemoryConsolidator()
        assert c._store is None

    def test_consolidator_init_with_store(self):
        from src.autfin.memory.consolidation import MemoryConsolidator
        mock_store = MagicMock()
        c = MemoryConsolidator(store=mock_store)
        assert c._store is mock_store


class TestConsolidate:
    @pytest.mark.asyncio
    async def test_consolidate_returns_stats_dict(self):
        from src.autfin.memory.consolidation import MemoryConsolidator

        mock_store = MagicMock()
        mock_store._list_ns_keys = AsyncMock(return_value=[])
        mock_store.aget = AsyncMock(return_value=None)

        c = MemoryConsolidator(store=mock_store)
        stats = await c.consolidate()

        assert "expired" in stats
        assert "trimmed" in stats
        assert "errors" in stats
        assert isinstance(stats["expired"], int)
        assert isinstance(stats["trimmed"], int)
        assert isinstance(stats["errors"], int)


class TestExpireOld:
    @pytest.mark.asyncio
    async def test_expire_old_returns_count(self):
        from src.autfin.memory.consolidation import MemoryConsolidator

        mock_store = MagicMock()
        mock_store._list_ns_keys = AsyncMock(return_value=["key1", "key2"])
        mock_store.aget = AsyncMock(return_value={"created_at": "2020-01-01T00:00:00+00:00"})
        mock_store.adelete = AsyncMock(return_value=True)

        c = MemoryConsolidator(store=mock_store)
        result = await c._expire_old(mock_store, "trade_history", ttl_days=365)

        assert isinstance(result, int)
        assert result >= 0


class TestEnforceSizeLimit:
    @pytest.mark.asyncio
    async def test_below_limit_no_trim(self):
        from src.autfin.memory.consolidation import MemoryConsolidator

        mock_store = MagicMock()
        # Below limit
        mock_store._list_ns_keys = AsyncMock(return_value=["k1", "k2"])

        c = MemoryConsolidator(store=mock_store)
        result = await c._enforce_size_limit(mock_store, "trade_history", max_size=100)

        assert result == 0


class TestConsolidateMemory:
    @pytest.mark.asyncio
    async def test_function_alias(self):
        """Test the helper function the scheduler will call."""
        from src.autfin.memory.consolidation import consolidate_memory

        # Just ensure it's callable
        assert callable(consolidate_memory)
