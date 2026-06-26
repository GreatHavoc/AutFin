"""
Dashboard tests for Phase 4.4 — Global Markets and Sentiment panels.
Tests verify functions exist, sidebar has new nav items, and main routes correctly.
Streamlit rendering is tested via source inspection (no real Streamlit needed).
"""

import inspect
import pytest
from unittest.mock import MagicMock, patch

from src.autfin.dashboard.app import (
    render_global_markets,
    render_sentiment,
    render_sidebar,
    main,
)


# ---------------------------------------------------------------------------
# Function existence tests
# ---------------------------------------------------------------------------

class TestNewPanelFunctions:
    """Verify new panel functions exist and are callable."""

    def test_render_global_markets_is_callable(self):
        assert callable(render_global_markets)

    def test_render_sentiment_is_callable(self):
        assert callable(render_sentiment)


# ---------------------------------------------------------------------------
# Sidebar navigation tests
# ---------------------------------------------------------------------------

class TestSidebarNav:
    """Verify sidebar includes new navigation items."""

    def test_sidebar_has_global_markets(self):
        source = inspect.getsource(render_sidebar)
        assert "Global Markets" in source

    def test_sidebar_has_sentiment(self):
        source = inspect.getsource(render_sidebar)
        assert "Sentiment" in source

    def test_sidebar_nav_order(self):
        """Global Markets and Sentiment should appear in nav list."""
        source = inspect.getsource(render_sidebar)
        gm_idx = source.index("Global Markets")
        sent_idx = source.index("Sentiment")
        # Global Markets should come before Sentiment
        assert gm_idx < sent_idx


# ---------------------------------------------------------------------------
# Main routing tests
# ---------------------------------------------------------------------------

class TestMainRouting:
    """Verify main() routes to new panels."""

    def test_main_routes_global_markets(self):
        source = inspect.getsource(main)
        assert "render_global_markets" in source

    def test_main_routes_sentiment(self):
        source = inspect.getsource(main)
        assert "render_sentiment" in source

    def test_main_has_all_pages(self):
        """Main should handle all page options."""
        source = inspect.getsource(main)
        for page in ["Overview", "Trades", "Signals", "Performance",
                      "Stock Pitches", "Fan-Out Progress",
                      "Learning Insights", "Global Markets", "Sentiment",
                      "Paper Trading", "System"]:
            assert f'"{page}"' in source or f"'{page}'" in source


# ---------------------------------------------------------------------------
# Supervisor prompt tests
# ---------------------------------------------------------------------------

class TestSupervisorPromptV4:
    """Verify supervisor prompt reflects v4 upgrades."""

    def test_prompt_mentions_multi_source(self):
        from src.autfin.prompts.supervisor import SUPERVISOR_SYSTEM
        assert "Multi-source" in SUPERVISOR_SYSTEM or "multi-source" in SUPERVISOR_SYSTEM

    def test_prompt_mentions_reddit(self):
        from src.autfin.prompts.supervisor import SUPERVISOR_SYSTEM
        assert "Reddit" in SUPERVISOR_SYSTEM

    def test_prompt_mentions_finbert(self):
        from src.autfin.prompts.supervisor import SUPERVISOR_SYSTEM
        assert "FinBERT" in SUPERVISOR_SYSTEM

    def test_prompt_mentions_india_10y(self):
        from src.autfin.prompts.supervisor import SUPERVISOR_SYSTEM
        assert "India 10Y" in SUPERVISOR_SYSTEM

    def test_prompt_mentions_16_indices(self):
        from src.autfin.prompts.supervisor import SUPERVISOR_SYSTEM
        assert "16" in SUPERVISOR_SYSTEM
