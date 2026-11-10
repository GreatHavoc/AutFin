"""Self-learning engine: retrieve → decide → execute → observe → learn.

State lives in the SQLite `learning_journal` table (see ``autfin.memory.database``).
The online learner keeps in-memory running statistics on top of that journal.
"""
from __future__ import annotations

from autfin.learning.ab_testing import ABTestingFramework, Experiment
from autfin.learning.calibrator import (
    CalibrationTracker,
    calibrate_conviction,
    detect_patterns,
    record_learning,
)
from autfin.learning.critique import (
    build_critique_prompt,
    get_decisions_needing_review,
    parse_critique_response,
    run_weekly_critique,
    store_critique,
)
from autfin.learning.feedback_loop import (
    get_learning_summary,
    post_trade_learn,
    pre_trade_context,
)
from autfin.learning.judge import _claim_in_context, _extract_claims
from autfin.learning.online import (
    OnlineLearner,
    get_online_learner,
    reset_online_learner,
)
from autfin.learning.retriever import retrieve_learnings, search_similar_setups

__all__ = [
    "ABTestingFramework",
    "CalibrationTracker",
    "Experiment",
    "OnlineLearner",
    "build_critique_prompt",
    "calibrate_conviction",
    "detect_patterns",
    "get_decisions_needing_review",
    "get_learning_summary",
    "get_online_learner",
    "parse_critique_response",
    "post_trade_learn",
    "pre_trade_context",
    "record_learning",
    "reset_online_learner",
    "retrieve_learnings",
    "run_weekly_critique",
    "search_similar_setups",
    "store_critique",
]
