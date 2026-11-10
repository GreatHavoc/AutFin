"""Self-learning engine for AutFin v3.0.
Closes the feedback loop: retrieve → decide → execute → observe → learn.
"""
from .calibrator import CalibrationTracker, calibrate_conviction, detect_patterns, record_learning
from .feedback_loop import get_learning_summary, post_trade_learn, pre_trade_context
from .retriever import retrieve_learnings, search_similar_setups

__all__ = [
    "retrieve_learnings",
    "search_similar_setups",
    "calibrate_conviction",
    "detect_patterns",
    "record_learning",
    "CalibrationTracker",
    "pre_trade_context",
    "post_trade_learn",
    "get_learning_summary",
]
