"""Tests for online evaluation of the live morning routine (B20).

Offline evals (`evals/dataset.py`) score cases; production traffic was ungraded,
so a regression in what the supervisor queues stayed invisible until someone ran
the suite by hand. The daemon's routine now carries the deterministic evaluator
through `pydantic_evals.online.evaluate`.
"""
from __future__ import annotations

import pytest
from pydantic_evals.online import CallbackSink

from autfin.types import SupervisorOutput, TradeOrder


def _queue(rr: float | None, conviction: float | None) -> SupervisorOutput:
    return SupervisorOutput(
        action="complete",
        message="done",
        trade_queue=[
            TradeOrder(symbol="RELIANCE", action="buy", quantity=1, risk_reward=rr, conviction=conviction)
        ],
    )


def _score(output) -> float:
    from autfin.evals.online import ONLINE_EVALUATORS

    class _Ctx:
        pass

    # the evaluator only reads ctx.output
    ctx = _Ctx()
    ctx.output = output
    return ONLINE_EVALUATORS[0].evaluate(ctx)


def test_a_sound_queue_scores_full_marks():
    assert _score(_queue(2.5, 0.8)) == 1.0


def test_a_trade_below_the_floors_scores_partial():
    assert _score(_queue(1.2, 0.4)) == 0.3


def test_an_unsized_trade_scores_zero():
    assert _score(_queue(None, 0.9)) == 0.0


def test_an_empty_queue_is_neutral_not_a_failure():
    assert _score(SupervisorOutput(action="complete", message="nothing to do")) == 0.5


def test_an_error_output_scores_zero():
    assert _score(SupervisorOutput(action="error", message="")) == 0.0


def test_the_daemon_routine_carries_the_evaluator():
    """The point of B20: production traffic is graded, not just offline cases."""
    import run

    assert hasattr(run._run_morning, "__wrapped__"), "the decorator must be applied"


@pytest.mark.asyncio
async def test_a_decorated_function_delivers_a_score_to_the_sink():
    from pydantic_evals.online import DEFAULT_CONFIG, evaluate, wait_for_evaluations

    from autfin.evals.online import ONLINE_EVALUATORS

    payloads: list = []
    previous_sink = DEFAULT_CONFIG.default_sink
    # CallbackSink.submit calls the callback with (results, failures, context)
    DEFAULT_CONFIG.default_sink = CallbackSink(
        lambda results, failures, context: payloads.append((results, failures, context))
    )
    try:

        @evaluate(*ONLINE_EVALUATORS, target="probe")
        async def routine() -> SupervisorOutput:
            return _queue(3.0, 0.9)

        await routine()
        await wait_for_evaluations(timeout=10.0)
    finally:
        DEFAULT_CONFIG.default_sink = previous_sink

    assert payloads, "the evaluation never reached the sink"
    values = [r.value for results, _failures, _ctx in payloads for r in results]
    assert 1.0 in values, values
