"""
Integration tests for Phase 4.4 — Supervisor prompt, build_stock_pitch,
multi-source sentiment wiring, and dashboard panels.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json


# ---------------------------------------------------------------------------
# Supervisor prompt tests
# ---------------------------------------------------------------------------

class TestSupervisorPrompt:
    """Tests for updated supervisor system prompt."""

    def test_prompt_mentions_multi_source_sentiment(self):
        from src.autfin.prompts.supervisor import SUPERVISOR_SYSTEM
        assert "Moneycontrol" in SUPERVISOR_SYSTEM
        assert "Reddit" in SUPERVISOR_SYSTEM
        assert "FinBERT" in SUPERVISOR_SYSTEM

    def test_prompt_mentions_global_macro(self):
        from src.autfin.prompts.supervisor import SUPERVISOR_SYSTEM
        assert "India 10Y" in SUPERVISOR_SYSTEM
        assert "16 global indices" in SUPERVISOR_SYSTEM

    def test_prompt_has_all_agents(self):
        from src.autfin.prompts.supervisor import SUPERVISOR_SYSTEM
        for agent in ["market_data", "global_macro", "fundamentals", "technical",
                       "sentiment", "researcher", "risk_manager", "portfolio_manager",
                       "executor", "human_review"]:
            assert agent in SUPERVISOR_SYSTEM

    def test_quick_decision_prompt(self):
        from src.autfin.prompts.supervisor import SUPERVISOR_QUICK_DECISION
        assert "next" in SUPERVISOR_QUICK_DECISION


# ---------------------------------------------------------------------------
# build_stock_pitch sentiment integration
# ---------------------------------------------------------------------------

class TestBuildStockPitchSentiment:
    """Tests for build_stock_pitch including sentiment context."""

    @pytest.mark.asyncio
    async def test_pitch_prompt_mentions_sentiment(self):
        """Verify the pitch system prompt references sentiment."""
        from src.autfin.tools.supervisor_tools import build_stock_pitch

        mock_client = MagicMock()
        mock_client.generate = AsyncMock(return_value=json.dumps({
            "investment_thesis": "Strong growth",
            "catalysts": ["Jio IPO"],
            "risks": ["Regulatory"],
            "valuation_summary": "Fair value",
            "competitive_moat": "Network effects",
            "price_target": {"base": 2500, "upside": 3000, "downside": 2000, "rationale": "DCF"},
        }))

        with patch("src.autfin.llm.client.get_llm_client", return_value=mock_client), \
             patch("src.autfin.dashboard.data.save_stock_pitch"):
            result = await build_stock_pitch(
                "RELIANCE",
                {"sentiment": {"score": 0.4, "category": "bullish"}, "technicals": {}}
            )

        assert result["status"] == "success"
        # Verify the prompt sent to LLM includes sentiment reference
        call_args = mock_client.generate.call_args
        system_prompt = call_args.kwargs.get("system", "") if call_args.kwargs else ""
        assert "sentiment" in system_prompt.lower()

    @pytest.mark.asyncio
    async def test_pitch_with_sentiment_in_research_data(self):
        """Pitch generation works with sentiment in research_data."""
        from src.autfin.tools.supervisor_tools import build_stock_pitch

        mock_client = MagicMock()
        mock_client.generate = AsyncMock(return_value=json.dumps({
            "investment_thesis": "Test thesis",
            "catalysts": [],
            "risks": [],
            "valuation_summary": "Test",
            "competitive_moat": "Test",
            "price_target": {"base": 100, "upside": 120, "downside": 80, "rationale": "Test"},
        }))

        research_data = {
            "sentiment": {
                "multi_source": {
                    "weighted_score": 0.35,
                    "category": "slightly_positive",
                    "sources": [
                        {"source": "marketaux", "score": 0.4, "confidence": 0.6},
                        {"source": "reddit", "score": 0.25, "confidence": 0.3},
                    ],
                }
            }
        }

        with patch("src.autfin.llm.client.get_llm_client", return_value=mock_client), \
             patch("src.autfin.dashboard.data.save_stock_pitch"):
            result = await build_stock_pitch("RELIANCE", research_data)

        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Multi-source sentiment wiring tests
# ---------------------------------------------------------------------------

class TestMultiSourceSentimentWiring:
    """Tests for multi-source sentiment integration with supervisor tools."""

    def test_analyze_macro_context_includes_india_10y(self):
        """Macro context function should reference India 10Y."""
        import inspect
        from src.autfin.tools.supervisor_tools import analyze_macro_context
        source = inspect.getsource(analyze_macro_context)
        # The function should include India 10Y in the rates response
        assert "india_10y" in source

    def test_analyze_macro_context_has_15_tickers(self):
        """Macro context should fetch 15 global tickers."""
        import inspect
        from src.autfin.tools.supervisor_tools import analyze_macro_context
        source = inspect.getsource(analyze_macro_context)
        assert "Shanghai" in source or "^SSEC" in source
        assert "Hang Seng" in source or "^HSI" in source
        assert "DAX" in source or "^GDAXI" in source

    def test_sentiment_tools_has_multi_source(self):
        """SentimentTools should have get_multi_source_sentiment method."""
        from src.autfin.tools.sentiment_tools import SentimentTools
        tools = SentimentTools(api_token="test")
        assert hasattr(tools, "get_multi_source_sentiment")
        assert hasattr(tools, "get_moneycontrol_sentiment")
        assert hasattr(tools, "get_economic_times_sentiment")
        assert hasattr(tools, "get_reddit_sentiment")

    def test_reddit_tool_has_analyze(self):
        """RedditSentimentTool should have analyze method."""
        from src.autfin.tools.reddit_sentiment import RedditSentimentTool
        tool = RedditSentimentTool()
        assert hasattr(tool, "analyze")
        assert hasattr(tool, "_categorize")
        assert hasattr(tool, "_volume_signal")

    def test_hybrid_analyzer_has_batch(self):
        """HybridSentimentAnalyzer should have analyze_batch method."""
        from src.autfin.tools.sentiment_analyzer import HybridSentimentAnalyzer
        analyzer = HybridSentimentAnalyzer()
        assert hasattr(analyzer, "analyze")
        assert hasattr(analyzer, "analyze_batch")


# ---------------------------------------------------------------------------
# Dashboard panel tests (mocked Streamlit)
# ---------------------------------------------------------------------------

class TestDashboardPanels:
    """Tests for new dashboard panels (Global Markets, Sentiment)."""

    def _mock_streamlit(self):
        """Create a mock streamlit module."""
        mock_st = MagicMock()
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        mock_st.metric.return_value = None
        mock_st.subheader.return_value = None
        mock_st.divider.return_value = None
        mock_st.info.return_value = None
        mock_st.warning.return_value = None
        mock_st.error.return_value = None
        mock_st.dataframe.return_value = None
        mock_st.multiselect.return_value = ["RELIANCE"]
        mock_st.expander.return_value.__enter__ = MagicMock()
        mock_st.expander.return_value.__exit__ = MagicMock()
        mock_st.write.return_value = None
        mock_st.radio.return_value = "Overview"
        mock_st.checkbox.return_value = False
        mock_st.select_slider.return_value = 60
        mock_st.button.return_value = False
        mock_st.caption.return_value = None
        mock_st.success.return_value = None
        mock_st.warning.return_value = None
        mock_st.rerun.return_value = None
        mock_st.set_page_config.return_value = None
        mock_st.title.return_value = None
        mock_st.header.return_value = None
        mock_st.sidebar = MagicMock()
        mock_st.sidebar.header.return_value = None
        mock_st.sidebar.caption.return_value = None
        mock_st.sidebar.divider.return_value = None
        mock_st.sidebar.metric.return_value = None
        mock_st.sidebar.radio.return_value = "Overview"
        mock_st.sidebar.checkbox.return_value = False
        mock_st.sidebar.select_slider.return_value = 60
        mock_st.sidebar.button.return_value = False
        mock_st.sidebar.caption.return_value = None
        return mock_st

    def test_render_global_markets_exists(self):
        """render_global_markets function should exist."""
        from src.autfin.dashboard.app import render_global_markets
        assert callable(render_global_markets)

    def test_render_sentiment_exists(self):
        """render_sentiment function should exist."""
        from src.autfin.dashboard.app import render_sentiment
        assert callable(render_sentiment)

    def test_sidebar_has_new_nav_items(self):
        """Sidebar nav should include Global Markets and Sentiment."""
        import inspect
        from src.autfin.dashboard.app import render_sidebar
        source = inspect.getsource(render_sidebar)
        assert "Global Markets" in source
        assert "Sentiment" in source

    def test_main_routes_to_new_panels(self):
        """main() should route to Global Markets and Sentiment panels."""
        import inspect
        from src.autfin.dashboard.app import main
        source = inspect.getsource(main)
        assert "render_global_markets" in source
        assert "render_sentiment" in source
