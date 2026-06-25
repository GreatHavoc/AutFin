"""
Tests for HITL (Human-in-the-Loop) nodes — trade_gate and human_review.

Covers:
  - _needs_human_approval helper
  - trade_gate_node: paper mode, empty queue, small trades, large trades, interrupt behavior
  - human_review_node: approval and rejection flows
  - _route_after_trade_gate routing
"""
import pytest
from unittest.mock import patch, MagicMock
from typing import Any

from src.autfin.graph.hitl import (
    trade_gate_node,
    human_review_node,
    _route_after_trade_gate,
    _needs_human_approval,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(**overrides) -> dict[str, Any]:
    """Return a minimal valid AgentState-like dict."""
    state = {
        "messages": [],
        "available_cash": 50000.0,
        "utilized_margin": 0.0,
        "holdings": [],
        "positions": [],
        "research_debates": None,
        "risk_assessments": None,
        "trade_queue": None,
        "execution_reports": None,
        "errors": [],
        "thread_id": "test-hitl-001",
        "run_mode": "daily_morning",
    }
    state.update(overrides)
    return state


def _trade(symbol: str, action: str, qty: int, price: float, **extra) -> dict:
    """Build a trade dict."""
    trade = {
        "symbol": symbol,
        "action": action,
        "qty": qty,
        "price": price,
        "order_type": "MARKET",
        "reason": extra.pop("reason", "Test trade"),
    }
    trade.update(extra)
    return trade


# ---------------------------------------------------------------------------
# _needs_human_approval
# ---------------------------------------------------------------------------

class TestNeedsHumanApproval:
    """Tests for the _needs_human_approval helper."""

    def test_empty_queue(self):
        assert _needs_human_approval([], 5000.0) == []

    def test_small_trades_within_limit(self):
        trades = [_trade("RELIANCE", "BUY", 5, 1000.0)]  # ₹5,000
        result = _needs_human_approval(trades, 5000.0)
        assert len(result) == 0  # 5000 == 5000, not >, so no HITL

    def test_small_trades_below_limit(self):
        trades = [_trade("RELIANCE", "BUY", 4, 1000.0)]  # ₹4,000
        result = _needs_human_approval(trades, 5000.0)
        assert len(result) == 0

    def test_large_trade_above_limit(self):
        trades = [_trade("RELIANCE", "BUY", 10, 1000.0)]  # ₹10,000
        result = _needs_human_approval(trades, 5000.0)
        assert len(result) == 1
        assert result[0]["symbol"] == "RELIANCE"
        assert result[0]["trade_value"] == 10000.0

    def test_mixed_trades(self):
        trades = [
            _trade("RELIANCE", "BUY", 3, 1000.0),   # ₹3,000 — small
            _trade("TCS", "BUY", 10, 2000.0),       # ₹20,000 — large
            _trade("INFY", "BUY", 2, 1000.0),       # ₹2,000 — small
        ]
        result = _needs_human_approval(trades, 5000.0)
        assert len(result) == 1
        assert result[0]["symbol"] == "TCS"
        assert result[0]["trade_value"] == 20000.0

    def test_multiple_large_trades(self):
        trades = [
            _trade("RELIANCE", "BUY", 10, 1000.0),  # ₹10,000
            _trade("TCS", "BUY", 10, 2000.0),       # ₹20,000
        ]
        result = _needs_human_approval(trades, 5000.0)
        assert len(result) == 2

    def test_trade_value_calculation(self):
        """Verify trade_value is computed from qty * price."""
        trade = [_trade("INFY", "BUY", 7, 1500.0)]  # ₹10,500
        result = _needs_human_approval(trade, 5000.0)
        assert result[0]["trade_value"] == 10500.0


# ---------------------------------------------------------------------------
# trade_gate_node — Paper mode
# ---------------------------------------------------------------------------

class TestTradeGatePaperMode:
    """trade_gate_node in paper mode skips HITL."""

    @pytest.mark.asyncio
    @patch("src.autfin.graph.hitl.AutFinSettings")
    async def test_paper_mode_skips_hitl(self, mock_settings_cls):
        mock_settings_cls.return_value = MagicMock(
            execution_mode="paper",
            auto_execute_max_amount=5000.0,
        )
        state = _base_state(
            trade_queue=[_trade("RELIANCE", "BUY", 10, 1000.0)]
        )
        result = await trade_gate_node(state)
        assert result["pending_human_approval"] is None
        assert result["human_decision"] is None
        assert any("paper mode" in m.content.lower() for m in result["messages"])


# ---------------------------------------------------------------------------
# trade_gate_node — Empty queue
# ---------------------------------------------------------------------------

class TestTradeGateEmptyQueue:
    """trade_gate_node with no trades."""

    @pytest.mark.asyncio
    @patch("src.autfin.graph.hitl.AutFinSettings")
    async def test_empty_queue(self, mock_settings_cls):
        mock_settings_cls.return_value = MagicMock(
            execution_mode="auto-small",
            auto_execute_max_amount=5000.0,
        )
        state = _base_state(trade_queue=[])
        result = await trade_gate_node(state)
        assert result["pending_human_approval"] is None
        assert any("empty" in m.content.lower() for m in result["messages"])


# ---------------------------------------------------------------------------
# trade_gate_node — Small trades (no HITL)
# ---------------------------------------------------------------------------

class TestTradeGateSmallTrades:
    """trade_gate_node passes through when all trades are small."""

    @pytest.mark.asyncio
    @patch("src.autfin.graph.hitl.AutFinSettings")
    async def test_small_trades_no_hitl(self, mock_settings_cls):
        mock_settings_cls.return_value = MagicMock(
            execution_mode="auto-small",
            auto_execute_max_amount=5000.0,
        )
        state = _base_state(
            trade_queue=[
                _trade("RELIANCE", "BUY", 3, 1000.0),  # ₹3,000
                _trade("INFY", "BUY", 2, 1000.0),       # ₹2,000
            ]
        )
        result = await trade_gate_node(state)
        assert result["pending_human_approval"] is None
        assert result["human_decision"] is None
        assert any("within auto-execute" in m.content.lower() for m in result["messages"])


# ---------------------------------------------------------------------------
# trade_gate_node — Large trades (HITL triggered)
# ---------------------------------------------------------------------------

class TestTradeGateLargeTrades:
    """trade_gate_node pauses for HITL when large trades exist."""

    @pytest.mark.asyncio
    @patch("src.autfin.graph.hitl.AutFinSettings")
    async def test_large_trade_triggers_interrupt(self, mock_settings_cls):
        """Large trade should call interrupt() — we test by verifying the function
        raises the interrupt exception (which is how LangGraph implements it)."""
        mock_settings_cls.return_value = MagicMock(
            execution_mode="auto-small",
            auto_execute_max_amount=5000.0,
        )
        state = _base_state(
            trade_queue=[_trade("RELIANCE", "BUY", 10, 1000.0)]  # ₹10,000
        )
        # interrupt() raises a special exception in standalone context
        # In a compiled graph, LangGraph catches it and pauses
        from langgraph.types import interrupt as real_interrupt
        with patch("src.autfin.graph.hitl.interrupt", side_effect=real_interrupt):
            with pytest.raises(Exception):
                await trade_gate_node(state)

    @pytest.mark.asyncio
    @patch("src.autfin.graph.hitl.AutFinSettings")
    async def test_large_trade_payload_structure(self, mock_settings_cls):
        """Verify the interrupt payload contains correct trade details."""
        mock_settings_cls.return_value = MagicMock(
            execution_mode="auto-small",
            auto_execute_max_amount=5000.0,
        )
        trade = _trade("TCS", "BUY", 10, 2000.0, reason="Strong signal")
        state = _base_state(trade_queue=[trade])

        captured_payload = {}
        def fake_interrupt(payload):
            captured_payload.update(payload)
            raise SystemExit("interrupted")

        with patch("src.autfin.graph.hitl.interrupt", side_effect=fake_interrupt):
            with pytest.raises(SystemExit):
                await trade_gate_node(state)

        assert captured_payload["action"] == "approve_trades"
        assert len(captured_payload["trades"]) == 1
        assert captured_payload["total_value"] == 20000.0
        assert captured_payload["trade_details"][0]["symbol"] == "TCS"
        assert captured_payload["trade_details"][0]["action"] == "BUY"
        assert captured_payload["trade_details"][0]["qty"] == 10
        assert captured_payload["trade_details"][0]["price"] == 2000.0
        assert captured_payload["trade_details"][0]["trade_value"] == 20000.0
        assert captured_payload["trade_details"][0]["reason"] == "Strong signal"
        assert "20,000" in captured_payload["message"]

    @pytest.mark.asyncio
    @patch("src.autfin.graph.hitl.AutFinSettings")
    async def test_resume_with_approval(self, mock_settings_cls):
        """After interrupt, resume with approved=True should keep trade_queue."""
        mock_settings_cls.return_value = MagicMock(
            execution_mode="auto-small",
            auto_execute_max_amount=5000.0,
        )
        state = _base_state(
            trade_queue=[_trade("RELIANCE", "BUY", 10, 1000.0)]  # ₹10,000
        )
        # Simulate resume: interrupt returns the human's decision
        with patch("src.autfin.graph.hitl.interrupt", return_value={"approved": True}):
            result = await trade_gate_node(state)

        assert result["human_decision"] == "approved"
        assert result["trade_queue"]  # queue preserved
        assert any("approved" in m.content.lower() for m in result["messages"])

    @pytest.mark.asyncio
    @patch("src.autfin.graph.hitl.AutFinSettings")
    async def test_resume_with_rejection(self, mock_settings_cls):
        """After interrupt, resume with approved=False should clear trade_queue."""
        mock_settings_cls.return_value = MagicMock(
            execution_mode="auto-small",
            auto_execute_max_amount=5000.0,
        )
        state = _base_state(
            trade_queue=[_trade("RELIANCE", "BUY", 10, 1000.0)]
        )
        with patch("src.autfin.graph.hitl.interrupt", return_value={"approved": False}):
            result = await trade_gate_node(state)

        assert result["human_decision"] == "rejected"
        assert result["trade_queue"] == []
        assert any("rejected" in m.content.lower() for m in result["messages"])

    @pytest.mark.asyncio
    @patch("src.autfin.graph.hitl.AutFinSettings")
    async def test_resume_with_modified_trades(self, mock_settings_cls):
        """Human can modify trades during approval (e.g., reduce qty)."""
        mock_settings_cls.return_value = MagicMock(
            execution_mode="auto-small",
            auto_execute_max_amount=5000.0,
        )
        state = _base_state(
            trade_queue=[_trade("RELIANCE", "BUY", 10, 1000.0)]  # ₹10,000
        )
        modified = [_trade("RELIANCE", "BUY", 5, 1000.0)]  # Reduced to ₹5,000
        with patch("src.autfin.graph.hitl.interrupt", return_value={
            "approved": True,
            "trades": modified,
        }):
            result = await trade_gate_node(state)

        assert result["human_decision"] == "approved"
        assert len(result["trade_queue"]) == 1
        assert result["trade_queue"][0]["qty"] == 5

    @pytest.mark.asyncio
    @patch("src.autfin.graph.hitl.AutFinSettings")
    async def test_resume_with_truthy_non_dict(self, mock_settings_cls):
        """Resume with a truthy non-dict value (e.g., True) should approve."""
        mock_settings_cls.return_value = MagicMock(
            execution_mode="auto-small",
            auto_execute_max_amount=5000.0,
        )
        state = _base_state(
            trade_queue=[_trade("RELIANCE", "BUY", 10, 1000.0)]
        )
        with patch("src.autfin.graph.hitl.interrupt", return_value=True):
            result = await trade_gate_node(state)

        assert result["human_decision"] == "approved"
        assert result["trade_queue"]  # preserved


# ---------------------------------------------------------------------------
# trade_gate_node — Recommendation mode
# ---------------------------------------------------------------------------

class TestTradeGateRecommendationMode:
    """trade_gate_node in recommendation mode — all trades need approval."""

    @pytest.mark.asyncio
    @patch("src.autfin.graph.hitl.AutFinSettings")
    async def test_recommendation_mode_always_gates(self, mock_settings_cls):
        """In recommendation mode, even small trades trigger HITL."""
        mock_settings_cls.return_value = MagicMock(
            execution_mode="recommendation",
            auto_execute_max_amount=5000.0,
        )
        state = _base_state(
            trade_queue=[_trade("RELIANCE", "BUY", 2, 1000.0)]  # ₹2,000 — small
        )
        captured_payload = {}
        def fake_interrupt(payload):
            captured_payload.update(payload)
            raise SystemExit("interrupted")

        with patch("src.autfin.graph.hitl.interrupt", side_effect=fake_interrupt):
            with pytest.raises(SystemExit):
                await trade_gate_node(state)

        # All trades need approval in recommendation mode
        assert captured_payload["action"] == "approve_trades"
        assert len(captured_payload["trades"]) == 1


# ---------------------------------------------------------------------------
# human_review_node
# ---------------------------------------------------------------------------

class TestHumanReviewNode:
    """Tests for the human_review_node."""

    @pytest.mark.asyncio
    @patch("src.autfin.graph.hitl.interrupt")
    async def test_human_review_interrupts(self, mock_interrupt):
        """human_review_node should call interrupt() with review payload."""
        mock_interrupt.return_value = {"approved": True, "feedback": "Looks good"}
        state = _base_state(
            trade_queue=[_trade("RELIANCE", "BUY", 5, 1000.0)],
            risk_assessments={"RELIANCE": {"approved": False, "recommendation": "SELL"}},
        )
        result = await human_review_node(state)

        mock_interrupt.assert_called_once()
        payload = mock_interrupt.call_args[0][0]
        assert payload["action"] == "human_review"
        assert payload["trades_pending"] == 1
        assert payload["risk_vetoes"] == 1
        assert len(payload["trade_details"]) == 1

    @pytest.mark.asyncio
    @patch("src.autfin.graph.hitl.interrupt")
    async def test_human_review_approval(self, mock_interrupt):
        """Approval should keep trade_queue and set human_decision."""
        mock_interrupt.return_value = {"approved": True, "feedback": "Go ahead"}
        state = _base_state(
            trade_queue=[_trade("TCS", "BUY", 5, 3000.0)],
        )
        result = await human_review_node(state)

        assert result["human_decision"] == "approved"
        assert result["trade_queue"]  # preserved
        assert any("approved" in m.content.lower() for m in result["messages"])

    @pytest.mark.asyncio
    @patch("src.autfin.graph.hitl.interrupt")
    async def test_human_review_rejection(self, mock_interrupt):
        """Rejection should clear trade_queue."""
        mock_interrupt.return_value = {"approved": False, "feedback": "Too risky"}
        state = _base_state(
            trade_queue=[_trade("TCS", "BUY", 5, 3000.0)],
        )
        result = await human_review_node(state)

        assert result["human_decision"] == "rejected"
        assert result["trade_queue"] == []
        assert any("rejected" in m.content.lower() for m in result["messages"])

    @pytest.mark.asyncio
    @patch("src.autfin.graph.hitl.interrupt")
    async def test_human_review_truthy_non_dict(self, mock_interrupt):
        """Truthy non-dict value should approve."""
        mock_interrupt.return_value = True
        state = _base_state()
        result = await human_review_node(state)

        assert result["human_decision"] == "approved"

    @pytest.mark.asyncio
    @patch("src.autfin.graph.hitl.interrupt")
    async def test_human_review_falsy_non_dict(self, mock_interrupt):
        """Falsy non-dict value should reject."""
        mock_interrupt.return_value = False
        state = _base_state()
        result = await human_review_node(state)

        assert result["human_decision"] == "rejected"
        assert result["trade_queue"] == []

    @pytest.mark.asyncio
    @patch("src.autfin.graph.hitl.interrupt")
    async def test_human_review_empty_state(self, mock_interrupt):
        """Should work with empty state (no trades, no risk assessments)."""
        mock_interrupt.return_value = {"approved": True}
        state = _base_state()
        result = await human_review_node(state)

        payload = mock_interrupt.call_args[0][0]
        assert payload["trades_pending"] == 0
        assert payload["risk_vetoes"] == 0
        assert result["human_decision"] == "approved"


# ---------------------------------------------------------------------------
# _route_after_trade_gate
# ---------------------------------------------------------------------------

class TestRouteAfterTradeGate:
    """Tests for the trade gate routing function."""

    def test_approved_with_trades(self):
        state = _base_state(
            human_decision="approved",
            trade_queue=[_trade("RELIANCE", "BUY", 5, 1000.0)],
        )
        assert _route_after_trade_gate(state) == "to_executor"

    def test_rejected(self):
        state = _base_state(
            human_decision="rejected",
            trade_queue=[],
        )
        assert _route_after_trade_gate(state) == "skip_to_report"

    def test_no_decision_empty_queue(self):
        state = _base_state(human_decision=None, trade_queue=[])
        assert _route_after_trade_gate(state) == "skip_to_report"

    def test_approved_but_queue_cleared(self):
        """Approved but queue was cleared (shouldn't happen but test edge case)."""
        state = _base_state(human_decision="approved", trade_queue=[])
        assert _route_after_trade_gate(state) == "skip_to_report"


# ---------------------------------------------------------------------------
# AgentState HITL fields
# ---------------------------------------------------------------------------

class TestAgentStateHITLFields:
    """Verify AgentState has the HITL fields."""

    def test_has_pending_human_approval(self):
        from src.autfin.state.agent_state import AgentState
        import typing
        hints = typing.get_type_hints(AgentState)
        assert "pending_human_approval" in hints

    def test_has_human_decision(self):
        from src.autfin.state.agent_state import AgentState
        import typing
        hints = typing.get_type_hints(AgentState)
        assert "human_decision" in hints
