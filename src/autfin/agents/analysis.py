"""AutFin v7.0 — Analysis Agent

Performs technical, fundamental, and sentiment analysis on symbols.
Uses combined toolsets for all analysis dimensions.
"""
from __future__ import annotations

from pydantic_ai import RunContext
from pydantic_ai.capabilities import Thinking, WebSearch

from autfin.agents.base import create_agent
from autfin.agents.managed_prompts import analysis_instructions
from autfin.tools import analysis_toolset
from autfin.types import (
    AgentDeps,
    AnalysisReport,
    FundamentalReport,
    MoatAssessment,
    PeerComparison,
    SentimentReport,
    TechnicalReport,
    ValuationReport,
)

analysis_agent = create_agent(
    name="analyze",
    output_type=AnalysisReport,
    model_type="deep",
    instructions=analysis_instructions.default,
    capabilities=[
        Thinking(effort="high"),
        WebSearch(local="duckduckgo"),
    ],
    toolsets=[analysis_toolset],
)


@analysis_agent.tool
async def analyze_all(ctx: RunContext[AgentDeps], symbol: str) -> AnalysisReport:
    """Run all analyses in parallel and synthesize."""
    import asyncio

    # Import tool functions from toolsets
    from autfin.tools.fundamental_toolset import (
        compute_health_score,
        compare_peers_tool,
        estimate_valuation,
        get_financials,
        get_ratios,
    )
    from autfin.tools.sentiment_toolset import get_sentiment_data
    from autfin.tools.technical_toolset import get_atr, get_technical_data

    # Parallel execution of all dimensions
    fundamental_task = get_financials(ctx, symbol)
    ratios_task = get_ratios(ctx, symbol)
    health_task = compute_health_score(ctx, symbol)
    technical_task = get_technical_data(ctx, symbol)
    sentiment_task = get_sentiment_data(ctx, symbol)
    atr_task = get_atr(ctx, symbol)
    valuation_task = estimate_valuation(ctx, symbol)

    fundamental_data, ratios, health, technical_data, sentiment_data, atr, valuation_data = (
        await asyncio.gather(
            fundamental_task,
            ratios_task,
            health_task,
            technical_task,
            sentiment_task,
            atr_task,
            valuation_task,
            return_exceptions=True,
        )
    )

    # Handle exceptions
    for item in [fundamental_data, ratios, health, technical_data, sentiment_data, atr, valuation_data]:
        if isinstance(item, Exception):
            item = None

    # Build reports
    fundamental = None
    if fundamental_data and "error" not in fundamental_data:
        fundamental = FundamentalReport(
            symbol=symbol,
            pe_ratio=fundamental_data.get("pe"),
            pb_ratio=fundamental_data.get("pb"),
            ev_ebitda=fundamental_data.get("ev_ebitda"),
            roe=fundamental_data.get("roe"),
            roa=fundamental_data.get("roa"),
            net_margin=fundamental_data.get("net_margin"),
            operating_margin=fundamental_data.get("operating_margin"),
            debt_equity=fundamental_data.get("debt_equity"),
            revenue_growth=fundamental_data.get("revenue_growth_3y"),
            eps_growth=fundamental_data.get("profit_growth_3y"),
            sector=fundamental_data.get("sector", ""),
            market_cap=fundamental_data.get("market_cap"),
            health_score=health.get("health_score", 50) if health else 50,
            moat_level=fundamental_data.get("moat_level", "none"),
            recommendation=fundamental_data.get("recommendation", "hold"),
            conviction=fundamental_data.get("conviction", 0.5),
            strength=fundamental_data.get("strength", 50),
            risks=fundamental_data.get("risks", []),
            catalysts=fundamental_data.get("catalysts", []),
            summary=fundamental_data.get("summary", ""),
        )

    technical = None
    if technical_data and "error" not in technical_data:
        technical = TechnicalReport(
            symbol=symbol,
            sma_20=technical_data.get("sma_20"),
            sma_50=technical_data.get("sma_50"),
            sma_200=technical_data.get("sma_200"),
            rsi=technical_data.get("rsi"),
            macd=technical_data.get("macd"),
            macd_signal=technical_data.get("macd_signal"),
            macd_histogram=technical_data.get("macd_histogram"),
            bollinger_upper=technical_data.get("bollinger_upper"),
            bollinger_middle=technical_data.get("bollinger_middle"),
            bollinger_lower=technical_data.get("bollinger_lower"),
            atr=atr if atr and not isinstance(atr, Exception) else technical_data.get("atr"),
            volume_trend=technical_data.get("volume_trend", "stable"),
            candlestick_patterns=technical_data.get("candlestick_patterns", []),
            signal=technical_data.get("signal", "hold"),
            strength=technical_data.get("strength", 50),
            conviction=technical_data.get("conviction", 0.5),
            summary=technical_data.get("summary", ""),
        )

    sentiment = None
    if sentiment_data and "error" not in sentiment_data:
        sentiment = SentimentReport(
            symbol=symbol,
            sentiment_score=sentiment_data.get("sentiment_score", 0.0),
            sentiment_label=sentiment_data.get("sentiment_label", "neutral"),
            news_sentiment=sentiment_data.get("news_sentiment", 0.0),
            announcement_sentiment=sentiment_data.get("announcement_sentiment", 0.0),
            catalysts=sentiment_data.get("catalysts", []),
            risks=sentiment_data.get("risks", []),
            summary=sentiment_data.get("summary", ""),
        )

    # Valuation
    valuation = None
    if valuation_data and "error" not in valuation_data:
        fv_range = valuation_data.get("fair_value_range")
        valuation = ValuationReport(
            symbol=symbol,
            dcf_base=valuation_data.get("dcf_base"),
            fair_value_range=tuple(fv_range) if fv_range else None,
            upside_pct=valuation_data.get("upside_pct"),
            key_assumptions=valuation_data.get("key_assumptions", {}),
            summary=valuation_data.get("summary", ""),
        )

    # Peer comparison - derive peers from sector
    peers = None
    if fundamental and fundamental.sector:
        sector_peers = {
            "Energy": ["ONGC", "IOC", "BPCL", "HINDPETRO"],
            "Financial Services": ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK"],
            "Technology": ["TCS", "INFY", "HCLTECH", "WIPRO"],
            "Consumer Goods": ["HUL", "ITC", "NESTLEIND", "BRITANNIA"],
            "Automobiles": ["MARUTI", "M&M", "TATAMOTORS", "BAJAJ-AUTO"],
        }
        peer_list = sector_peers.get(fundamental.sector, [])
        if peer_list:
            try:
                peer_data = await compare_peers_tool(ctx, symbol, peer_list)
                if peer_data and "error" not in peer_data:
                    peers = PeerComparison(
                        symbol=symbol,
                        peers=peer_list,
                        rankings=peer_data.get("peers", {}),
                        summary=f"Compared against {len(peer_list)} peers in {fundamental.sector}",
                    )
            except Exception:
                pass

    # Moat - derive from fundamental signals
    moat = None
    if fundamental:
        moat_score = 0
        moat_sources = []
        if fundamental.roe and fundamental.roe > 15:
            moat_score += 25
            moat_sources.append("High ROE")
        if fundamental.debt_equity and fundamental.debt_equity < 0.5:
            moat_score += 20
            moat_sources.append("Low debt")
        if fundamental.revenue_growth and fundamental.revenue_growth > 10:
            moat_score += 20
            moat_sources.append("Strong growth")
        if fundamental.operating_margin and fundamental.operating_margin > 20:
            moat_score += 20
            moat_sources.append("High margins")
        if fundamental.market_cap and fundamental.market_cap > 1e11:
            moat_score += 15
            moat_sources.append("Large cap")
        rating = "wide" if moat_score >= 70 else "narrow" if moat_score >= 40 else "none"
        moat = MoatAssessment(
            symbol=symbol,
            score=moat_score,
            rating=rating,
            sources=moat_sources,
            summary=f"Moat score {moat_score}/100 ({rating})",
        )

    # Compute overall score (weighted average)
    scores = []
    weights = []

    if fundamental:
        scores.append(fundamental.health_score)
        weights.append(0.30)
    if technical:
        scores.append(technical.strength)
        weights.append(0.20)
    if sentiment:
        scores.append((sentiment.sentiment_score + 1) * 50)  # -1..1 -> 0..100
        weights.append(0.15)
    if valuation and valuation.upside_pct is not None:
        scores.append(min(max(valuation.upside_pct + 50, 0), 100))
        weights.append(0.20)
    if moat:
        scores.append(moat.score)
        weights.append(0.15)

    if scores and weights:
        overall = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    else:
        overall = 50.0

    # Determine recommendation
    if overall >= 80:
        rec = "strong_buy"
    elif overall >= 65:
        rec = "buy"
    elif overall >= 45:
        rec = "hold"
    elif overall >= 30:
        rec = "sell"
    else:
        rec = "strong_sell"

    # Aggregate confidence
    confidences = []
    if fundamental:
        confidences.append(fundamental.conviction)
    if technical:
        confidences.append(technical.conviction)
    if sentiment:
        confidences.append(abs(sentiment.sentiment_score))
    confidence = sum(confidences) / len(confidences) if confidences else 0.5

    # Aggregate risks and catalysts
    all_risks = []
    all_catalysts = []
    for r in [fundamental, technical, sentiment]:
        if r:
            all_risks.extend(r.risks)
            all_catalysts.extend(r.catalysts)

    return AnalysisReport(
        symbol=symbol,
        fundamental=fundamental,
        technical=technical,
        sentiment=sentiment,
        valuation=valuation,
        peer_comparison=peers,
        moat=moat,
        overall_score=overall,
        recommendation=rec,
        confidence=confidence,
        key_risks=list(set(all_risks))[:10],
        catalyst=all_catalysts[0] if all_catalysts else "",
    )


@analysis_agent.tool
async def analyze_distortion(ctx: RunContext[AgentDeps], symbol: str) -> dict:
    """Check for price distortions (gaps, anomalies, manipulation signals)."""
    from autfin.agents.distortion import check_distortion

    return await check_distortion(symbol, ctx.deps.as_of)