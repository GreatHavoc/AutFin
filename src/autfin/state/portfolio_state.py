"""
PortfolioState - DecisionRecord and portfolio types
"""
from typing import TypedDict


class DecisionRecord(TypedDict):
    """Stored in LangGraph Store for cross-thread memory."""
    decision_id: str                # UUID
    timestamp: str                  # ISO 8601
    symbol: str
    action: str                     # "BUY" | "SELL" | "HOLD" | "REDUCE"
    conviction: float               # 0.0 - 1.0
    
    # Input scores
    fundamentals_score: float       # 0-1
    technical_score: float          # 0-1
    sentiment_score: float          # -1 to +1
    macro_score: float              # -1 to +1
    risk_score: float               # 0-1
    
    # Output
    rationale: str                  # LLM-generated
    position_size: float | None     # ₹ amount
    price: float | None
    budget_deployed: float | None
    
    # Outcome (filled later)
    outcome_pnl: float | None
    outcome_date: str | None
    was_correct: bool | None