"""
Researcher Agent - Bull/Bear debate synthesis agent.
Analyzes conflicting research signals and synthesizes balanced recommendations.
"""
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from langgraph.config import RunnableConfig

from src.autfin.state.agent_state import AgentState
from src.autfin.llm.client import get_llm_client

# Configure logging
logger = logging.getLogger(__name__)

# ================== SIGNAL WEIGHT CONFIGURATION ==================

# PRD-aligned proportional weights (must sum to 1.0)
# fundamentals: 0.35, technical: 0.25, sentiment: 0.15, macro: 0.25 (includes risk impact)
SIGNAL_WEIGHTS = {
    "fundamentals": 0.35,
    "technical": 0.25,
    "sentiment": 0.15,
    "macro": 0.25,
}

# Signal category mappings
# Bull signals are positive, Bear signals are negative
SIGNAL_CATEGORIES = {
    "technical": {
        "bullish": ["buy", "strong_buy", "oversold_reversal", "bullish_pattern", "macd_crossover_up"],
        "bearish": ["sell", "strong_sell", "overbought_reversal", "bearish_pattern", "macd_crossover_down"],
        "neutral": ["hold"],
    },
    "fundamentals": {
        "bullish": ["strong_buy", "buy"],
        "bearish": ["strong_sell", "sell"],
        "neutral": ["hold"],
    },
    "sentiment": {
        "bullish": ["strong_buy", "buy"],
        "bearish": ["strong_sell", "sell"],
        "neutral": ["neutral"],
    },
    "macro": {
        "bullish": ["high_positive", "medium_positive"],
        "bearish": ["high_negative", "medium_negative"],
        "neutral": ["neutral"],
    },
}


# ================== SIGNAL PARSING FUNCTIONS ==================

def parse_technical_signal(signal_data: dict) -> Dict[str, Any]:
    """
    Parse technical signal into standardized format.
    
    Args:
        signal_data: Technical agent output
        
    Returns:
        Standardized signal dict with type, direction, strength
    """
    signal = signal_data.get("signal", {})
    signal_type = signal.get("signal", "HOLD").lower()
    
    # Map signal type to standardized direction
    if signal_type in ["strong_buy"]:
        direction = "bullish"
        strength = 2.0
    elif signal_type in ["buy"]:
        direction = "bullish"
        strength = 1.0
    elif signal_type in ["hold"]:
        direction = "neutral"
        strength = 0.0
    elif signal_type in ["sell"]:
        direction = "bearish"
        strength = -1.0
    elif signal_type in ["strong_sell"]:
        direction = "bearish"
        strength = -2.0
    else:
        direction = "neutral"
        strength = 0.0
    
    return {
        "type": "technical",
        "direction": direction,
        "strength": strength,
        "confidence": signal.get("confidence", 0.0),
        "indicators": signal.get("indicator_scores", {}),
        "reasoning": signal.get("reasoning", []),
        "patterns": signal.get("pattern_signals", []),
        "timestamp": signal_data.get("timestamp", datetime.now().isoformat()),
    }


def parse_fundamentals_signal(signal_data: dict) -> Dict[str, Any]:
    """
    Parse fundamentals signal into standardized format.
    
    Args:
        signal_data: Fundamentals agent output
        
    Returns:
        Standardized signal dict with type, direction, strength
    """
    if not signal_data:
        return {
            "type": "fundamentals",
            "direction": "neutral",
            "strength": 0.0,
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat(),
        }
    
    health_score = signal_data.get("health_score", 50.0)
    
    # Map health score to direction
    if health_score > 80:
        direction = "bullish"
        strength = 2.0
    elif health_score > 60:
        direction = "bullish"
        strength = 1.0
    elif health_score > 40:
        direction = "neutral"
        strength = 0.0
    elif health_score > 20:
        direction = "bearish"
        strength = -1.0
    else:
        direction = "bearish"
        strength = -2.0
    
    return {
        "type": "fundamentals",
        "direction": direction,
        "strength": strength,
        "confidence": health_score / 100.0,
        "health_score": health_score,
        "health_breakdown": signal_data.get("health_breakdown", {}),
        "ratios": signal_data.get("ratios", {}),
        "timestamp": signal_data.get("timestamp", datetime.now().isoformat()),
    }


def parse_sentiment_signal(signal_data: dict) -> Dict[str, Any]:
    """
    Parse sentiment signal into standardized format.
    
    Args:
        signal_data: Sentiment agent output
        
    Returns:
        Standardized signal dict with type, direction, strength
    """
    if not signal_data:
        return {
            "type": "sentiment",
            "direction": "neutral",
            "strength": 0.0,
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat(),
        }
    
    # Get aggregate sentiment
    aggregate = signal_data.get("aggregate", {})
    weighted_score = aggregate.get("weighted_score", 0.0)
    
    # Map sentiment score to direction
    if weighted_score > 0.6:
        direction = "bullish"
        strength = 2.0
    elif weighted_score > 0.3:
        direction = "bullish"
        strength = 1.0
    elif weighted_score > -0.3:
        direction = "neutral"
        strength = 0.0
    elif weighted_score > -0.6:
        direction = "bearish"
        strength = -1.0
    else:
        direction = "bearish"
        strength = -2.0
    
    return {
        "type": "sentiment",
        "direction": direction,
        "strength": strength,
        "confidence": aggregate.get("confidence", 0.0),
        "sentiment_score": weighted_score,
        "articles_count": aggregate.get("symbol_count", 0),
        "categories": aggregate.get("sentiment_categories", {}),
        "timestamp": signal_data.get("timestamp", datetime.now().isoformat()),
    }


def parse_macro_signal(signal_data: dict) -> Dict[str, Any]:
    """
    Parse macro signal into standardized format.
    
    Args:
        signal_data: Macro agent output
        
    Returns:
        Standardized signal dict with type, direction, strength
    """
    if not signal_data:
        return {
            "type": "macro",
            "direction": "neutral",
            "strength": 0.0,
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat(),
        }
    
    # Get impact analysis
    impact_analysis = signal_data.get("impact_analysis", {})
    impact_score = impact_analysis.get("impact_score", "neutral")
    
    # Map impact score to direction
    if impact_score in ["high_positive", "medium_positive"]:
        direction = "bullish"
        strength = 2.0 if impact_score == "high_positive" else 1.0
    elif impact_score in ["high_negative", "medium_negative"]:
        direction = "bearish"
        strength = -2.0 if impact_score == "high_negative" else -1.0
    else:
        direction = "neutral"
        strength = 0.0
    
    return {
        "type": "macro",
        "direction": direction,
        "strength": strength,
        "confidence": 0.8,  # Macro signals generally have high confidence
        "impact_score": impact_score,
        "indicators_count": signal_data.get("indicators", {}).get("total_indicators", 0),
        "recommendations": impact_analysis.get("recommendations", []),
        "correlations": impact_analysis.get("correlations", {}),
        "timestamp": signal_data.get("timestamp", datetime.now().isoformat()),
    }


# ================== SIGNAL ANALYSIS ==================

def analyze_signals(research_results: dict) -> dict:
    """
    Analyze all research agent outputs and extract signals.
    
    Parses signals from technical, fundamentals, sentiment, and macro agents.
    Organizes signals by type and direction (bullish/bearish).
    
    Args:
        research_results: Dictionary containing all research agent outputs
        
    Returns:
        Dictionary with categorized signals
    """
    signals = {
        "technical": [],
        "fundamentals": [],
        "sentiment": [],
        "macro": [],
        "summary": {
            "total_signals": 0,
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
        },
    }
    
    # Extract research results from state structure
    technical_reports = research_results.get("technical_reports", {})
    fundamentals_reports = research_results.get("fundamental_reports", {})
    sentiment_reports = research_results.get("sentiment_reports", {})
    macro_signals = research_results.get("macro_signals", {})
    
    # Process technical signals — handles both nested {"reports": {...}} and flat {symbol: {...}}
    _tech_items = {}
    if isinstance(technical_reports, dict):
        if "reports" in technical_reports:
            _tech_items = technical_reports["reports"]
        else:
            # Flat format: {symbol: report_dict, ...}
            _tech_items = {k: v for k, v in technical_reports.items() if isinstance(v, dict)}
    for symbol, report in _tech_items.items():
        if isinstance(report, dict) and "signal" in report:
            signal = parse_technical_signal(report)
            signals["technical"].append(signal)
            signals["summary"]["total_signals"] += 1
            if signal["direction"] == "bullish":
                signals["summary"]["bullish_count"] += 1
            elif signal["direction"] == "bearish":
                signals["summary"]["bearish_count"] += 1
            else:
                signals["summary"]["neutral_count"] += 1
    
    # Process fundamentals signals — handles both nested and flat formats
    _fund_items = {}
    if isinstance(fundamentals_reports, dict):
        if "reports" in fundamentals_reports:
            _fund_items = fundamentals_reports["reports"]
        else:
            _fund_items = {k: v for k, v in fundamentals_reports.items() if isinstance(v, dict)}
    for symbol, report in _fund_items.items():
        if isinstance(report, dict) and "health_score" in report:
                signal = parse_fundamentals_signal(report)
                signals["fundamentals"].append(signal)
                signals["summary"]["total_signals"] += 1
                if signal["direction"] == "bullish":
                    signals["summary"]["bullish_count"] += 1
                elif signal["direction"] == "bearish":
                    signals["summary"]["bearish_count"] += 1
                else:
                    signals["summary"]["neutral_count"] += 1
    
    # Process sentiment signals
    if isinstance(sentiment_reports, dict):
        aggregate = sentiment_reports.get("aggregate", {})
        if aggregate:
            signal = parse_sentiment_signal(sentiment_reports)
            signals["sentiment"].append(signal)
            signals["summary"]["total_signals"] += 1
            if signal["direction"] == "bullish":
                signals["summary"]["bullish_count"] += 1
            elif signal["direction"] == "bearish":
                signals["summary"]["bearish_count"] += 1
            else:
                signals["summary"]["neutral_count"] += 1
    
    # Process macro signals
    if isinstance(macro_signals, dict):
        signal = parse_macro_signal(macro_signals)
        signals["macro"].append(signal)
        signals["summary"]["total_signals"] += 1
        if signal["direction"] == "bullish":
            signals["summary"]["bullish_count"] += 1
        elif signal["direction"] == "bearish":
            signals["summary"]["bearish_count"] += 1
        else:
            signals["summary"]["neutral_count"] += 1
    
    return signals


# ================== SYNTHESIS LOGIC ==================

def synthesize_recommendation(signals: dict) -> dict:
    """
    Synthesize conflicting signals into balanced recommendation.
    
    Uses weighted scoring to combine signals from different agents.
    Fundamentals have highest weight, followed by macro, technical, and sentiment.
    
    Args:
        signals: Dictionary with categorized signals from analyze_signals()
        
    Returns:
        Dictionary with synthesized recommendation
    """
    # Calculate weighted scores for each signal type
    weighted_scores = {}
    
    for signal_type in ["technical", "fundamentals", "sentiment", "macro"]:
        signal_list = signals.get(signal_type, [])
        weight = SIGNAL_WEIGHTS.get(signal_type, 1.0)
        
        if signal_list:
            # Calculate average strength for this signal type
            total_strength = sum(s.get("strength", 0.0) for s in signal_list)
            avg_strength = total_strength / len(signal_list)
            weighted_scores[signal_type] = avg_strength * weight
        else:
            weighted_scores[signal_type] = 0.0
    
    # Calculate total weighted score
    total_weight = sum(SIGNAL_WEIGHTS.values())
    total_score = sum(weighted_scores.values())
    
    # Calculate normalized conviction score (0-1)
    # Based on absolute weighted score and agreement
    if total_weight > 0:
        normalized_score = total_score / total_weight
        # Convert to 0-1 range
        conviction = (normalized_score + 2) / 4  # (-2 to +2) -> (0 to 1)
    else:
        conviction = 0.5  # Default to neutral when no signals
    
    # PRD-aligned decision matrix:
    # > 0.70 → STRONG BUY
    # 0.55-0.70 → BUY
    # 0.40-0.55 → HOLD
    # 0.25-0.40 → SELL/REDUCE
    # < 0.25 → STRONG SELL
    if conviction > 0.70 and total_score > 0:
        recommendation = "strong_buy"
    elif conviction > 0.55 and total_score > 0:
        recommendation = "buy"
    elif conviction > 0.40:
        recommendation = "hold"
    elif conviction > 0.25:
        recommendation = "sell"
    elif total_score < 0:
        recommendation = "strong_sell"
    else:
        recommendation = "hold"
    
    # Calculate agreement level (how consistent are the signals)
    signal_types_with_data = [k for k, v in weighted_scores.items() if v != 0]
    if len(signal_types_with_data) >= 2:
        # Check if signals agree on direction
        bullish_types = sum(1 for t in signal_types_with_data if weighted_scores[t] > 0)
        bearish_types = sum(1 for t in signal_types_with_data if weighted_scores[t] < 0)
        
        if bullish_types >= 2 or bearish_types >= 2:
            agreement = "high"
        elif bullish_types == 1 or bearish_types == 1:
            agreement = "medium"
        else:
            agreement = "low"
    else:
        agreement = "low"
    
    # Generate summary
    summary = {
        "recommendation": recommendation,
        "conviction_score": round(conviction, 2),
        "agreement": agreement,
        "weighted_scores": {k: round(v, 2) for k, v in weighted_scores.items()},
        "signal_count": {
            "total": signals["summary"]["total_signals"],
            "bullish": signals["summary"]["bullish_count"],
            "bearish": signals["summary"]["bearish_count"],
            "neutral": signals["summary"]["neutral_count"],
        },
    }
    
    # Add detailed breakdown by signal type
    summary["signal_breakdown"] = {}
    for signal_type in ["technical", "fundamentals", "sentiment", "macro"]:
        signal_list = signals.get(signal_type, [])
        if signal_list:
            summary["signal_breakdown"][signal_type] = {
                "count": len(signal_list),
                "average_strength": round(sum(s.get("strength", 0.0) for s in signal_list) / len(signal_list), 2),
                "directions": [s.get("direction") for s in signal_list],
            }
    
    return summary


# ================== RESEARCH NOTES GENERATION ==================

def generate_research_notes(recommendation: dict) -> list:
    """
    Generate detailed research notes with reasoning.
    
    Creates human-readable notes explaining the recommendation,
    including key indicators and any conflicting signals.
    
    Args:
        recommendation: Dictionary from synthesize_recommendation()
        
    Returns:
        List of research notes as strings
    """
    notes = []
    
    # Header
    notes.append(f"## Research Synthesis Report")
    notes.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    notes.append("")
    
    # Recommendation
    notes.append(f"### Recommendation: **{recommendation['recommendation'].upper()}**")
    notes.append(f"**Conviction:** {recommendation['conviction_score'] * 100:.0f}%")
    notes.append(f"**Agreement Level:** {recommendation['agreement']}")
    notes.append("")
    
    # Signal Summary
    summary = recommendation.get("signal_count", {})
    notes.append("### Signal Summary")
    notes.append(f"- **Bullish Signals:** {summary.get('bullish_count', summary.get('bullish', 0))}")
    notes.append(f"- **Bearish Signals:** {summary.get('bearish_count', summary.get('bearish', 0))}")
    notes.append(f"- **Neutral Signals:** {summary.get('neutral_count', summary.get('neutral', 0))}")
    notes.append("")
    
    # Weighted Scores
    notes.append("### Weighted Signal Scores")
    weighted_scores = recommendation.get("weighted_scores", {})
    score_descriptions = {
        "technical": "Technical indicators and patterns",
        "fundamentals": "Financial health and valuation",
        "sentiment": "Market sentiment and news",
        "macro": "Macroeconomic impact",
    }
    
    for signal_type, score in weighted_scores.items():
        description = score_descriptions.get(signal_type, signal_type)
        direction = "positive" if score > 0 else "negative" if score < 0 else "neutral"
        notes.append(f"- **{description}:** {score:+.2f} ({direction})")
    
    notes.append("")
    
    # Signal Breakdown
    notes.append("### Detailed Signal Breakdown")
    breakdown = recommendation.get("signal_breakdown", {})
    
    for signal_type, data in breakdown.items():
        count = data.get("count", 0)
        avg_strength = data.get("average_strength", 0)
        directions = data.get("directions", [])
        
        notes.append(f"#### {signal_type.capitalize()}")
        notes.append(f"- Signals analyzed: {count}")
        notes.append(f"- Average strength: {avg_strength:+.2f}")
        notes.append(f"- Direction: {', '.join(directions) if directions else 'None'}")
        notes.append("")
    
    # Key Factors
    notes.append("### Key Factors")
    key_factors = []
    
    if weighted_scores.get("fundamentals", 0) > 0:
        key_factors.append("Strong fundamentals support the recommendation")
    elif weighted_scores.get("fundamentals", 0) < 0:
        key_factors.append("Weak fundamentals are a concern")
    
    if weighted_scores.get("technical", 0) > 0:
        key_factors.append("Technical indicators are bullish")
    elif weighted_scores.get("technical", 0) < 0:
        key_factors.append("Technical indicators are bearish")
    
    if weighted_scores.get("sentiment", 0) > 0:
        key_factors.append("Market sentiment is positive")
    elif weighted_scores.get("sentiment", 0) < 0:
        key_factors.append("Market sentiment is negative")
    
    if weighted_scores.get("macro", 0) > 0:
        key_factors.append("Macro environment is favorable")
    elif weighted_scores.get("macro", 0) < 0:
        key_factors.append("Macro environment is unfavorable")
    
    if key_factors:
        for factor in key_factors:
            notes.append(f"- {factor}")
    else:
        notes.append("- Insufficient data for key factor analysis")
    
    notes.append("")
    
    # Conflicting Signals
    notes.append("### Conflicting Signals")
    conflict_analysis = analyze_conflicts(recommendation)
    
    if conflict_analysis["has_conflicts"]:
        notes.append(f"**Found {conflict_analysis['conflict_count']} areas of conflict:**")
        for conflict in conflict_analysis["conflicts"]:
            notes.append(f"- {conflict}")
    else:
        notes.append("No major conflicts detected between signal types.")
    
    notes.append("")
    
    # Risk Considerations
    notes.append("### Risk Considerations")
    risk_notes = generate_risk_notes(recommendation)
    
    for note in risk_notes:
        notes.append(f"- {note}")
    
    notes.append("")
    notes.append("---")
    notes.append("*This report is synthesized from multiple research agents.*")
    
    return notes


def analyze_conflicts(recommendation: dict) -> dict:
    """
    Analyze conflicts between signal types.
    
    Args:
        recommendation: Dictionary from synthesize_recommendation()
        
    Returns:
        Dictionary with conflict analysis
    """
    weighted_scores = recommendation.get("weighted_scores", {})
    conflicts = []
    
    # Check for conflicts between high-weight signals
    fundamentals_score = weighted_scores.get("fundamentals", 0)
    technical_score = weighted_scores.get("technical", 0)
    
    # Fundamental vs Technical conflict
    if fundamentals_score * technical_score < 0:  # Different signs
        conflicts.append(
            "Fundamentals and technical indicators disagree "
            f"(Fundamentals: {fundamentals_score:+.2f}, Technical: {technical_score:+.2f})"
        )
    
    # Sentiment vs Price action
    sentiment_score = weighted_scores.get("sentiment", 0)
    if sentiment_score * technical_score < 0:
        conflicts.append(
            "Sentiment and technical indicators disagree "
            f"(Sentiment: {sentiment_score:+.2f}, Technical: {technical_score:+.2f})"
        )
    
    # Macro vs others
    macro_score = weighted_scores.get("macro", 0)
    if abs(macro_score) > 0.5:
        if macro_score * fundamentals_score < 0:
            conflicts.append(
                "Macro environment conflicts with fundamentals "
                f"(Macro: {macro_score:+.2f}, Fundamentals: {fundamentals_score:+.2f})"
            )
    
    return {
        "has_conflicts": len(conflicts) > 0,
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
    }


def generate_risk_notes(recommendation: dict) -> list:
    """
    Generate risk considerations based on recommendation.
    
    Args:
        recommendation: Dictionary from synthesize_recommendation()
        
    Returns:
        List of risk notes
    """
    notes = []
    conviction = recommendation.get("conviction_score", 0)
    recommendation_type = recommendation.get("recommendation", "hold")
    
    # Low conviction warnings
    if conviction < 0.5:
        notes.append("Low conviction score - proceed with caution")
    
    # High conviction warnings
    if conviction > 0.9:
        notes.append("High conviction - ensure this aligns with risk tolerance")
    
    # Recommendation-specific risks
    if recommendation_type == "strong_buy":
        notes.append("Consider valuation at current levels")
        notes.append("Monitor for signs of overextension")
    elif recommendation_type == "strong_sell":
        notes.append("Verify this is not temporary distress")
        notes.append("Assess recovery potential")
    elif recommendation_type == "buy":
        notes.append("Entry point may be favorable")
        notes.append("Set stop-loss to manage downside")
    elif recommendation_type == "sell":
        notes.append("Consider tax implications")
        notes.append("Rebalance portfolio appropriately")
    else:  # hold
        notes.append("Maintain position and monitor developments")
        notes.append("Reassess when new data available")
    
    # Signal agreement risks
    agreement = recommendation.get("agreement", "low")
    if agreement == "low":
        notes.append("Low signal agreement - wait for clearer consensus")
    elif agreement == "medium":
        notes.append("Mixed signals - consider partial position")
    
    return notes


# ================== MAIN RESEARCHER AGENT ==================

class ResearcherAgent:
    """
    Bull/Bear debate synthesis agent.
    
    Analyzes conflicting research signals from multiple agents
    and synthesizes a balanced recommendation with conviction scoring.
    """
    
    def __init__(self):
        """Initialize the Researcher Agent."""
        pass
    
    async def analyze_signals(self, research_results: dict) -> dict:
        """
        Parse and categorize signals from all research agents.
        
        Args:
            research_results: Dictionary containing all research agent outputs
            
        Returns:
            Dictionary with categorized signals
        """
        return analyze_signals(research_results)
    
    async def synthesize_recommendation(self, signals: dict) -> dict:
        """
        Synthesize conflicting signals into balanced recommendation.
        
        Args:
            signals: Dictionary with categorized signals
            
        Returns:
            Dictionary with synthesized recommendation
        """
        return synthesize_recommendation(signals)
    
    async def generate_research_notes(self, recommendation: dict) -> list:
        """
        Generate detailed research notes with reasoning.
        
        Args:
            recommendation: Dictionary from synthesize_recommendation()
            
        Returns:
            List of research notes as strings
        """
        return generate_research_notes(recommendation)
    
    async def generate_llm_rationale(
        self,
        signals: dict,
        recommendation: dict,
    ) -> Optional[str]:
        """
        Generate natural language rationale using LLM.

        Falls back to None if LLM is unavailable or fails.

        Args:
            signals: Categorized signals from analyze_signals()
            recommendation: Synthesized recommendation dict

        Returns:
            LLM-generated rationale string, or None
        """
        llm = get_llm_client()
        if not llm.is_available:
            return None

        # Build context summary for the LLM
        weighted = recommendation.get("weighted_scores", {})
        rec = recommendation.get("recommendation", "hold")
        conviction = recommendation.get("conviction_score", 0.5)
        agreement = recommendation.get("agreement", "low")

        bull_count = signals.get("summary", {}).get("bullish_count", 0)
        bear_count = signals.get("summary", {}).get("bearish_count", 0)
        neutral_count = signals.get("summary", {}).get("neutral_count", 0)

        prompt = (
            f"You are a senior equity research analyst at an Indian brokerage.\n"
            f"Synthesize a 2-3 paragraph investment rationale for this stock.\n\n"
            f"## Signal Summary\n"
            f"- Recommendation: {rec.upper()}\n"
            f"- Conviction: {conviction:.0%}\n"
            f"- Signal agreement: {agreement}\n"
            f"- Bullish signals: {bull_count}, Bearish: {bear_count}, Neutral: {neutral_count}\n\n"
            f"## Weighted Scores\n"
        )
        for k, v in weighted.items():
            prompt += f"- {k}: {v:+.2f}\n"

        # Add top bull/bear signals
        all_signals = (
            signals.get("technical", [])
            + signals.get("fundamentals", [])
            + signals.get("sentiment", [])
            + signals.get("macro", [])
        )
        bull_signals = [s for s in all_signals if s.get("direction") == "bullish"][:3]
        bear_signals = [s for s in all_signals if s.get("direction") == "bearish"][:3]

        if bull_signals:
            prompt += "\n## Top Bullish Factors\n"
            for s in bull_signals:
                prompt += f"- [{s.get('type', '?')}] strength={s.get('strength', 0):+.1f}\n"

        if bear_signals:
            prompt += "\n## Top Bearish Factors\n"
            for s in bear_signals:
                prompt += f"- [{s.get('type', '?')}] strength={s.get('strength', 0):+.1f}\n"

        prompt += (
            "\n## Instructions\n"
            "Write a concise investment rationale (2-3 paragraphs). Cover:\n"
            "1. The bull case — what supports this recommendation\n"
            "2. The bear case — key risks and concerns\n"
            "3. Your overall assessment and what to watch\n"
            "Use plain English. Reference specific signal types where relevant.\n"
        )

        system = (
            "You are an equity research analyst specializing in Indian stock markets "
            "(NSE/BSE). Be objective, data-driven, and concise. "
            "Do not use markdown formatting — write in plain text paragraphs."
        )

        return await llm.generate(
            prompt=prompt,
            system_prompt=system,
            max_tokens=512,
        )

    async def _extract_catalysts(self, signals: dict, research_results: dict) -> List[dict]:
        """
        Pillar 4: Extract and classify catalysts (hard/soft) from research context.

        Uses LLM to identify concrete events that could move the stock price in
        6-12 months. Classifies each as hard (dated event like earnings) or soft
        (possible event like expansion).

        Falls back to empty list if LLM is unavailable.
        """
        llm = get_llm_client()
        if not llm.is_available:
            return []

        all_signals = (
            signals.get("technical", [])
            + signals.get("fundamentals", [])
            + signals.get("sentiment", [])
            + signals.get("macro", [])
        )
        signal_names = [
            s.get("name", s.get("type", "unknown"))
            for s in all_signals
            if s.get("direction") in ("bullish", "bearish")
        ][:10]

        fundamental_reports = research_results.get("fundamental_reports", {})
        firm_data = json.dumps(fundamental_reports, default=str)[:1000]

        llm_result = await llm.generate_json(
            prompt=(
                f"You are an equity research analyst. Extract specific catalysts "
                f"that could move this stock in 6-12 months.\n\n"
                f"## Active signals\n{', '.join(signal_names) if signal_names else 'none'}\n\n"
                f"## Fundamental data\n{firm_data}\n\n"
                f"Return a JSON object with a 'catalysts' array. Each catalyst needs:\n"
                f"  - name: short label (e.g. 'Q3 earnings Aug 15', 'new plant approval')\n"
                f"  - type: 'hard' (dated event) or 'soft' (possible event)\n"
                f"  - description: 1-sentence why it matters\n"
                f"  - direction: 'up' or 'down'\n"
                f"  - estimated_impact: number between 0.02 and 0.15 (2-15% price impact)\n"
                f"  - timeframe: string like '3 months', '6-12 months'\n"
                f"  - confidence: number 0-1 how likely\n"
                f"\nPrioritize HARD catalysts (dated events). Include 2-4 catalysts max."
            ),
            system_prompt=(
                "You are an equity research analyst. Return only valid JSON. "
                "Hard catalysts = dated events (earnings, trial results, M&A close). "
                "Soft catalysts = possible events (expansion, market share shifts)."
            ),
        )
        if not llm_result or not isinstance(llm_result, dict):
            return []
        return llm_result.get("catalysts", [])

    async def _self_audit(
        self,
        signals: dict,
        recommendation: dict,
        memory_context: Optional[dict] = None,
    ) -> dict:
        """
        Pillar 2: Self-reflective audit — "What am I missing?"

        After synthesis, the researcher introspects: Did I skip any data source?
        Is the confidence justified? What alternative interpretation could exist?
        Produces a gap report and optionally adjusts the recommendation.

        Falls back to a no-gaps result when LLM unavailable.
        """
        llm = get_llm_client()
        if not llm.is_available:
            return {"gaps_found": False, "gap_notes": [], "action": "none"}

        rec = recommendation.get("recommendation", "hold")
        conviction = recommendation.get("conviction_score", 0.5)
        agreement = recommendation.get("agreement", "low")
        weighted = recommendation.get("weighted_scores", {})

        missing_agents = [
            k for k, v in weighted.items()
            if abs(v) < 0.03
        ]
        available_agents = [
            k for k, v in weighted.items()
            if abs(v) >= 0.03
        ]

        mem_ctx = ""
        if memory_context and isinstance(memory_context, dict):
            past_learnings = memory_context.get("past_learnings") or memory_context.get("similar_trades") or ""
            if past_learnings:
                mem_ctx = f"\n## Past learnings\n{str(past_learnings)[:500]}"

        llm_result = await llm.generate_json(
            prompt=(
                f"You are reviewing an investment research output for completeness.\n\n"
                f"## Current Recommendation\n"
                f"- Rating: {rec}\n"
                f"- Conviction: {conviction:.0%}\n"
                f"- Signal agreement: {agreement}\n"
                f"- Weighted scores: {json.dumps(weighted)}\n"
                f"- Agents providing signals: {available_agents}\n"
                f"- Agents with weak/no signal: {missing_agents}\n"
                f"{mem_ctx}\n\n"
                f"## Self-Audit Questions\n"
                f"1. What data source did I skip that could provide meaningful signal?\n"
                f"2. What alternative interpretation of the data is possible?\n"
                f"3. What could prove my recommendation wrong?\n"
                f"4. Is the conviction level justified?\n\n"
                f"Return JSON: {{'gaps_found': bool, 'gap_notes': [str, ...], "
                f"'action': 'none'|'adjust_down'|'flag_for_deep_dive', 'revised_conviction': 0.0-1.0}}"
            ),
            system_prompt=(
                "You are a skeptical senior analyst reviewing a junior analyst's work. "
                "Be critical but constructive. Return only valid JSON."
            ),
        )
        if not llm_result or not isinstance(llm_result, dict):
            return {"gaps_found": False, "gap_notes": [], "action": "none"}
        return llm_result

    async def _generate_variant_view(
        self,
        signals: dict,
        recommendation: dict,
    ) -> Optional[str]:
        """
        Pillar 2.3: Generate a counter-consensus variant view.

        What is the market pricing incorrectly and WHY? This is the highest-value
        output of professional equity research — the differentiated thesis.

        Falls back to None when LLM unavailable.
        """
        llm = get_llm_client()
        if not llm.is_available:
            return None

        rec = recommendation.get("recommendation", "hold")
        weighted = recommendation.get("weighted_scores", {})

        return await llm.generate(
            prompt=(
                f"You are a fund manager looking for a VARIANT VIEW — a thesis that "
                f"80% of the market is missing or pricing incorrectly.\n\n"
                f"## Current signals\n"
                f"- Recommendation: {rec}\n"
                f"- Weighted scores: {json.dumps(weighted)}\n\n"
                f"Answer these questions in 2-3 tightly-written paragraphs:\n"
                f"1. What is the market consensus on this stock right now?\n"
                f"2. What is the market WRONG about? (Be specific — what data point "
                f"or trend is being mispriced?)\n"
                f"3. What would need to happen for the market to realize it's wrong? "
                f"(The catalyst for repricing)\n\n"
                f"Do not repeat the recommendation. Write the variant view."
            ),
            system_prompt=(
                "You are a fund manager generating variant views. Be specific, "
                "non-consensus, and evidence-based. No markdown. "
                "If the data doesn't support a strong variant view, say so honestly."
            ),
            max_tokens=300,
        )

    async def researcher_agent(self, state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
        """
        Orchestrate research synthesis workflow.

        Runs all research agents, analyzes their signals,
        synthesizes recommendations, and generates research notes.

        Args:
            state: Current agent state

        Returns:
            Updated state with researcher signals and recommendations
        """
        errors = state.get("errors", [])

        try:
            # Step 1: Gather research results from state
            research_results = {
                "technical_reports": state.get("technical_reports", {}),
                "fundamental_reports": state.get("fundamental_reports", {}),
                "sentiment_reports": state.get("sentiment_reports", {}),
                "macro_signals": state.get("macro_signals", {}),
            }

            # Check if we have any research data
            if not any(research_results.values()):
                errors.append("No research data available for synthesis")
                return {
                    **state,
                    "research_debates": None,
                    "researcher_notes": [],
                    "errors": errors,
                }

            # Step 2: Analyze all signals
            signals = await self.analyze_signals(research_results)

            # Step 3: Synthesize recommendation
            recommendation = await self.synthesize_recommendation(signals)

            # Step 4: Generate research notes (template-based)
            research_notes = await self.generate_research_notes(recommendation)

            # Step 5: Generate LLM-powered rationale (optional enhancement)
            llm_rationale = await self.generate_llm_rationale(signals, recommendation)
            if llm_rationale:
                research_notes.append("")
                research_notes.append("### AI-Generated Analysis")
                research_notes.append(llm_rationale)

            # Step 5.5: Self-audit — "What am I missing?" (Pillar 2)
            memory_context = state.get("memory_context", {})
            self_audit = await self._self_audit(signals, recommendation, memory_context)
            if self_audit.get("gaps_found"):
                recommendation["self_audit"] = self_audit
                if self_audit.get("revised_conviction") is not None:
                    old_conv = recommendation.get("conviction_score", 0.5)
                    # Clamp revised conviction to [0.0, 1.0]
                    revised = max(0.0, min(1.0, float(self_audit["revised_conviction"])))
                    recommendation["conviction_score"] = revised
                    recommendation["pre_audit_conviction"] = old_conv
                research_notes.append("")
                research_notes.append("### Self-Audit Findings")
                for gap_note in self_audit.get("gap_notes", []):
                    research_notes.append(f"- {gap_note}")
                if self_audit.get("action") == "flag_for_deep_dive":
                    research_notes.append("- **FLAGGED: Recommendation requires deep-dive reanalysis**")

            # Step 6: Extract catalysts (Pillar 4)
            catalysts = await self._extract_catalysts(signals, research_results)
            if catalysts:
                research_notes.append("")
                research_notes.append("### Catalysts (6-12 Month Outlook)")
                for cat in catalysts:
                    direction = "⬆️" if cat.get("direction") == "up" else "⬇️"
                    tag = "[HARD]" if cat.get("type") == "hard" else "[SOFT]"
                    research_notes.append(
                        f"- {direction} {tag} {cat.get('name', '?')}: "
                        f"{cat.get('description', '')}"
                    )

            # Step 7: Variant view (Pillar 2.3)
            variant_view = await self._generate_variant_view(signals, recommendation)
            if variant_view:
                research_notes.append("")
                research_notes.append("### Variant View (What the Market is Missing)")
                research_notes.append(variant_view)

            # Build research debates structure
            research_debates = {
                "summary": {
                    "recommendation": recommendation["recommendation"],
                    "conviction_score": recommendation["conviction_score"],
                    "agreement": recommendation["agreement"],
                    "timestamp": datetime.now().isoformat(),
                },
                "bull": {
                    "signals": [
                        s
                        for s in signals["technical"]
                        + signals["fundamentals"]
                        + signals["sentiment"]
                        + signals["macro"]
                        if s.get("direction") == "bullish"
                    ],
                    "count": signals["summary"]["bullish_count"],
                },
                "bear": {
                    "signals": [
                        s
                        for s in signals["technical"]
                        + signals["fundamentals"]
                        + signals["sentiment"]
                        + signals["macro"]
                        if s.get("direction") == "bearish"
                    ],
                    "count": signals["summary"]["bearish_count"],
                },
                "neutral": {
                    "count": signals["summary"]["neutral_count"],
                },
            }

            # Include LLM rationale in debates if available
            if llm_rationale:
                research_debates["summary"]["llm_rationale"] = llm_rationale

            # Include catalysts and variant view
            if catalysts:
                research_debates["catalysts"] = catalysts
            if variant_view:
                research_debates["variant_view"] = variant_view
            if self_audit:
                research_debates["self_audit"] = self_audit

            # Update state with research output
            updated_state = {
                "research_debates": research_debates,
                "researcher_notes": research_notes,
                "researcher_recommendation": recommendation,
                "agent": "researcher",
                "errors": errors,
            }

            return updated_state

        except Exception as e:
            errors.append(f"Researcher agent failed: {str(e)}")
            logger.error(f"Researcher agent error: {str(e)}")

            return {
                **state,
                "research_debates": None,
                "researcher_notes": [],
                "researcher_recommendation": None,
                "errors": errors,
            }


# ================== CONVENIENCE FUNCTION ==================

# Global researcher agent instance
_researcher_agent: Optional[ResearcherAgent] = None


def create_researcher_agent() -> ResearcherAgent:
    """
    Create or get global researcher agent instance.
    
    Returns:
        ResearcherAgent instance
    """
    global _researcher_agent
    
    if _researcher_agent is None:
        _researcher_agent = ResearcherAgent()
    
    return _researcher_agent


async def researcher_agent(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """
    Convenience function to run researcher agent.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with researcher signals
    """
    agent = create_researcher_agent()
    return await agent.researcher_agent(state, config)
