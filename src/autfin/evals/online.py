"""Online evaluation: grade the daemon's live runs in the background (B20).

`evals/dataset.py` scores cases offline; nothing grades production traffic, so a
regression in what the supervisor actually queues is invisible until someone
runs the suite by hand. `pydantic_evals.online.evaluate` attaches the same
`Evaluator` objects to a live function, runs them on a background thread, and
delivers the score to a sink (here: an OTel event on the call span).

The evaluator is deliberately deterministic and shaped for the real output
types: `TradeQualityEvaluator` in `dataset.py` indexes `ctx.output` as a dict,
but the supervisor returns a `SupervisorOutput` model, so it would raise on
every production call.
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from autfin.types import SupervisorOutput

# The floors the offline dataset uses for trade quality: risk/reward at least
# 2.0 and conviction at least 0.5. Kept identical so an offline case and a live
# run are graded by the same bar.
MIN_RISK_REWARD = 2.0
MIN_CONVICTION = 0.5


@dataclass
class SupervisorOutputEvaluator(Evaluator[SupervisorOutput, SupervisorOutput]):
    """Score one morning routine's output: a decision was reached, and every
    queued trade clears the risk/reward and conviction floors."""

    def evaluate(self, ctx: EvaluatorContext[SupervisorOutput, SupervisorOutput]) -> float:
        output = ctx.output
        if not isinstance(output, SupervisorOutput):
            return 0.0
        if output.action == "error" or not output.message:
            return 0.0
        if not output.trade_queue:
            # a routine that queues nothing is a legitimate outcome (no setups),
            # graded as neutral rather than a failure
            return 0.5

        scores = []
        for order in output.trade_queue:
            if order.risk_reward is None or order.conviction is None:
                scores.append(0.0)  # an unsized order has no business being queued
                continue
            if order.risk_reward < MIN_RISK_REWARD or order.conviction < MIN_CONVICTION:
                scores.append(0.3)
                continue
            scores.append(1.0)
        return sum(scores) / len(scores)


ONLINE_EVALUATORS = [SupervisorOutputEvaluator()]
