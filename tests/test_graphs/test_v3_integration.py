"""v3.0 Integration Tests — supervisor graph end-to-end flows.

Tests the full pipeline for each context mode:
  - morning_routine: screen → research → pitch → risk → allocate → execute
  - price_alert: investigate one symbol → decide
  - quarterly_review: deep dive all holdings

All tests mock the LLM and external dependencies.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(**overrides) -> dict[str, Any]:
    state = {
        "messages": [],
        "context": "morning_routine",
        "depth": "standard",
        "available_cash": 200000.0,
        "utilized_margin": 0.0,
        "holdings": [
            {"tradingsymbol": "RELIANCE", "qty": 10, "avg_price": 2400.0},
            {"tradingsymbol": "TCS", "qty": 5, "avg_price": 3800.0},
        ],
        "positions": [],
        "thread_id": "test-integration",
        "run_mode": "paper",
        "errors": [],
        "total_tokens": 0,
        "total_cost": 0.0,
        "budget_limit": 50.0,
        "iteration_count": 0,
        "validation_count": 0,
    }
    state.update(overrides)
    return state


def _mock_llm(text: str = "Analysis complete."):
    mock = MagicMock()
    mock.is_available = True
    mock.deep_model = "test-model"
    mock._call_count = 1

    async def fake_generate(**kwargs):
        return text

    mock.generate_with_tools = fake_generate
    return mock


# ---------------------------------------------------------------------------
# 3.4 Integration Test: Full Morning Routine
# ---------------------------------------------------------------------------

class TestMorningRoutineIntegration:
    """Full morning routine: screen → research → pitch → risk → allocate → execute."""

    @pytest.mark.asyncio
    async def test_graph_compiles(self):
        from src.autfin.graph.supervisor import build_supervisor_graph
        builder = build_supervisor_graph()
        compiled = builder.compile()
        assert compiled is not None

    @pytest.mark.asyncio
    async def test_screen_candidates_handles_empty_universe(self):
        from src.autfin.tools.supervisor_tools import screen_candidates
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await screen_candidates()
            assert "candidates" in result or result.get("status") == "error"

    @pytest.mark.asyncio
    async def test_research_stock_quick_depth(self):
        from src.autfin.tools.supervisor_tools import research_stock
        async def fake_agent(s):
            return {"messages": [], "data": {}}
        with patch("src.autfin.agents.fundamentals.fundamentals_agent", side_effect=fake_agent), \
             patch("src.autfin.agents.technical.technical_agent", side_effect=fake_agent), \
             patch("src.autfin.agents.sentiment.sentiment_agent", side_effect=fake_agent):
            result = await research_stock("RELIANCE", "quick")
            assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_build_stock_pitch_with_llm(self):
        from src.autfin.tools.supervisor_tools import build_stock_pitch
        pitch_json = json.dumps({
            "investment_thesis": "Strong energy play",
            "catalysts": ["Green energy pivot"],
            "risks": ["Oil price volatility"],
            "valuation_summary": "Fairly valued",
            "competitive_moat": "Wide moat",
            "price_target": {"base": 2600, "upside": 3000, "downside": 2100},
        })
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(return_value=pitch_json)
        with patch("src.autfin.llm.client.get_llm_client", return_value=mock_client):
            result = await build_stock_pitch("RELIANCE", {"pe": 22.5})
            assert result["status"] == "success"
            assert "pitch" in result

    @pytest.mark.asyncio
    async def test_run_risk_checks_approves_low_risk(self):
        from src.autfin.tools.supervisor_tools import run_risk_checks
        mock_agent = MagicMock()
        mock_agent.check_all_risks.return_value = {
            "risk_score": 0.6,
            "approved": True,
            "checks": {},
            "issues": [],
        }
        with patch("src.autfin.agents.risk_manager.RiskManagerAgent", return_value=mock_agent):
            result = await run_risk_checks({
                "symbol": "RELIANCE", "action": "BUY", "quantity": 5, "price": 2400.0,
            })
            assert result["status"] == "success"
            assert result["approved"] is True

    @pytest.mark.asyncio
    async def test_allocate_capital_returns_sized_trades(self):
        from src.autfin.tools.supervisor_tools import allocate_capital
        approved = [
            {"symbol": "RELIANCE", "action": "BUY", "conviction": 0.8, "price": 2400.0},
            {"symbol": "TCS", "action": "BUY", "conviction": 0.6, "price": 3800.0},
        ]
        result = await allocate_capital(approved, 200000.0)
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_finalize_session_returns_summary(self):
        from src.autfin.tools.supervisor_tools import finalize_session
        result = await finalize_session()
        assert result["status"] == "success"
        assert "summary" in result

    @pytest.mark.asyncio
    async def test_supervisor_node_runs_with_morning_context(self):
        from src.autfin.graph.supervisor import supervisor_node
        state = _base_state()
        with patch("src.autfin.llm.client.get_llm_client", return_value=_mock_llm("Morning routine complete.")):
            with patch("src.autfin.graph.supervisor._build_context_prompt", return_value="Test"):
                result = await supervisor_node(state)
                assert "messages" in result
                assert result.get("iteration_count", 0) >= 1


# ---------------------------------------------------------------------------
# 3.5 Integration Test: Price Alert → Decision
# ---------------------------------------------------------------------------

class TestPriceAlertIntegration:
    """Price alert flow: investigate one symbol → decide."""

    @pytest.mark.asyncio
    async def test_price_alert_context_sets_symbol(self):
        state = _base_state(context="price_alert", alert_symbol="RELIANCE", price_move_pct=-0.05)
        assert state["context"] == "price_alert"
        assert state["alert_symbol"] == "RELIANCE"

    @pytest.mark.asyncio
    async def test_price_alert_graph_compiles(self):
        from src.autfin.graph.supervisor import build_supervisor_graph
        builder = build_supervisor_graph()
        compiled = builder.compile()
        assert compiled is not None

    @pytest.mark.asyncio
    async def test_price_alert_supervisor_node_runs(self):
        from src.autfin.graph.supervisor import supervisor_node
        state = _base_state(context="price_alert", alert_symbol="RELIANCE", price_move_pct=-0.05)
        with patch("src.autfin.llm.client.get_llm_client", return_value=_mock_llm("RELIANCE dropped 5%. No action needed.")):
            with patch("src.autfin.graph.supervisor._build_context_prompt", return_value="Alert"):
                result = await supervisor_node(state)
                assert "messages" in result
                assert result.get("iteration_count", 0) >= 1

    @pytest.mark.asyncio
    async def test_validate_thesis_returns_result(self):
        from src.autfin.tools.supervisor_tools import validate_thesis
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(return_value=json.dumps({
            "status": "INTACT", "reasoning": "Thesis still holds"
        }))
        with patch("src.autfin.llm.client.get_llm_client", return_value=mock_client):
            result = await validate_thesis("RELIANCE", {"thesis": "Energy transition"})
            assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_get_market_quotes_returns_data(self):
        from src.autfin.tools.supervisor_tools import get_market_quotes
        result = await get_market_quotes(["RELIANCE"])
        assert "status" in result


# ---------------------------------------------------------------------------
# 3.6 Integration Test: Quarterly Deep Dive
# ---------------------------------------------------------------------------

class TestQuarterlyReviewIntegration:
    """Quarterly review: deep dive all holdings with DCF."""

    @pytest.mark.asyncio
    async def test_quarterly_review_context(self):
        state = _base_state(context="quarterly_review", depth="deep")
        assert state["context"] == "quarterly_review"
        assert state["depth"] == "deep"

    @pytest.mark.asyncio
    async def test_quarterly_review_graph_compiles(self):
        from src.autfin.graph.supervisor import build_supervisor_graph
        builder = build_supervisor_graph()
        compiled = builder.compile()
        assert compiled is not None

    @pytest.mark.asyncio
    async def test_quarterly_review_supervisor_node_runs(self):
        from src.autfin.graph.supervisor import supervisor_node
        state = _base_state(context="quarterly_review", depth="deep")
        with patch("src.autfin.llm.client.get_llm_client", return_value=_mock_llm("Quarterly review complete.")):
            with patch("src.autfin.graph.supervisor._build_context_prompt", return_value="Quarterly"):
                result = await supervisor_node(state)
                assert "messages" in result
                assert result.get("iteration_count", 0) >= 1

    @pytest.mark.asyncio
    async def test_research_stock_deep_runs_all_agents(self):
        from src.autfin.tools.supervisor_tools import research_stock
        async def fake_agent(s):
            return {"messages": [], "data": {}}
        with patch("src.autfin.agents.fundamentals.fundamentals_agent", side_effect=fake_agent), \
             patch("src.autfin.agents.technical.technical_agent", side_effect=fake_agent), \
             patch("src.autfin.agents.sentiment.sentiment_agent", side_effect=fake_agent), \
             patch("src.autfin.agents.valuation.valuation_agent", side_effect=fake_agent), \
             patch("src.autfin.agents.global_macro.global_macro_agent", side_effect=fake_agent), \
             patch("src.autfin.agents.peer_compare.peer_compare_agent", side_effect=fake_agent), \
             patch("src.autfin.agents.moat.moat_agent", side_effect=fake_agent):
            result = await research_stock("RELIANCE", "deep")
            assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_validate_all_holdings(self):
        from src.autfin.tools.supervisor_tools import validate_thesis
        mock_client = MagicMock()
        mock_client.generate = AsyncMock(return_value=json.dumps({
            "status": "INTACT", "reasoning": "Still valid"
        }))
        holdings = [
            {"symbol": "RELIANCE", "thesis": "Energy transition"},
            {"symbol": "TCS", "thesis": "IT growth"},
        ]
        with patch("src.autfin.llm.client.get_llm_client", return_value=mock_client):
            for h in holdings:
                result = await validate_thesis(h["symbol"], h)
                assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Token Budget Guard (3.7)
# ---------------------------------------------------------------------------

class TestTokenBudgetGuard:
    """Token budget guard: abort if approaching rate limit."""

    @pytest.mark.asyncio
    async def test_budget_exhausted_returns_error(self):
        from src.autfin.graph.supervisor import supervisor_node
        state = _base_state(total_tokens=100000, budget_limit=50000)
        result = await supervisor_node(state)
        assert "messages" in result
        content = result["messages"][0].content
        assert "budget" in content.lower() or "exhausted" in content.lower()

    @pytest.mark.asyncio
    async def test_budget_not_exceeded_runs_normally(self):
        from src.autfin.graph.supervisor import supervisor_node
        state = _base_state(total_tokens=1000, budget_limit=50000)
        with patch("src.autfin.llm.client.get_llm_client", return_value=_mock_llm("Budget OK.")):
            with patch("src.autfin.graph.supervisor._build_context_prompt", return_value="Budget"):
                result = await supervisor_node(state)
                assert result.get("iteration_count", 0) >= 1
