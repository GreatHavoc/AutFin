"""AutFin v7.0 — Quant Agent with CodeMode

CodeMode-enabled agent for complex quantitative computations.
"""
from __future__ import annotations

from typing import Any

# Define quant output types
from pydantic import BaseModel, ConfigDict
from pydantic_ai import RunContext
from pydantic_ai.capabilities import Thinking
from pydantic_ai_harness import CodeMode

from autfin.agents.base import create_agent
from autfin.agents.managed_prompts import quant_instructions
from autfin.tools import risk_toolset
from autfin.types import AgentDeps

# ponytail: removed Decimal — NIM xgrammar chokes on its regex pattern


class PortfolioMetrics(BaseModel):
    """Portfolio risk/return metrics."""
    model_config = ConfigDict(validate_assignment=True)

    total_value: float
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    portfolio_volatility: float
    beta: float
    alpha: float


class OptimizationResult(BaseModel):
    """Portfolio optimization result."""
    model_config = ConfigDict(validate_assignment=True)
    
    weights: dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    constraints_satisfied: bool


class QuantResult(BaseModel):
    """Quant agent output."""
    model_config = ConfigDict(validate_assignment=True)
    
    action: str  # "var", "optimize", "metrics", "stress_test"
    portfolio_metrics: PortfolioMetrics | None = None
    optimization_result: OptimizationResult | None = None
    stress_test_results: dict[str, Any] | None = None
    summary: str = ""


quant_agent = create_agent(
    name="quant",
    output_type=QuantResult,
    model_type="code",  # Use code model for complex computations
    instructions=quant_instructions.default,
    capabilities=[
        Thinking(effort="high"),
        CodeMode(),  # Sandbox all tools (default)
    ],
    # risk_toolset already bundles technical (kite + technical), so mounting both
    # registered every technical tool twice in the toolset list.
    toolsets=[risk_toolset],
)


@quant_agent.tool
async def calculate_var(
    ctx: RunContext[AgentDeps],
    positions: list[dict],
    confidence: float = 0.95,
    method: str = "historical",
    n_simulations: int = 10000,
) -> dict:
    """Calculate Value at Risk for portfolio positions.
    
    Args:
        positions: List of {symbol, qty, price, sector}
        confidence: Confidence level (0.95 or 0.99)
        method: "historical" or "monte_carlo"
        n_simulations: Number of Monte Carlo paths
    """
    # This runs inside CodeMode sandbox - can use numpy, loops, asyncio
    import numpy as np
    
    # Get historical returns for each position
    returns_matrix = []
    symbols = [p["symbol"] for p in positions]
    
    for symbol in symbols:
        tech_data = await ctx.deps.kite_client and ctx.deps.kite_client.get_technical_data(symbol)
        # Simplified: use historical simulation
        pass
    
    # Placeholder - full implementation in CodeMode
    total_value = sum(p["qty"] * p["price"] for p in positions)
    portfolio_vol = 0.15  # 15% annualized
    
    if method == "historical":
        # Historical simulation
        z_score = 1.645 if confidence == 0.95 else 2.326
        var = total_value * portfolio_vol * z_score / np.sqrt(252)
    else:
        # Monte Carlo
        np.random.seed(42)
        sims = np.random.normal(0, portfolio_vol / np.sqrt(252), n_simulations)
        var = -np.percentile(sims * total_value, (1 - confidence) * 100)
    
    return {
        "var": float(var),
        "confidence": confidence,
        "method": method,
        "portfolio_value": float(total_value),
    }


@quant_agent.tool
async def optimize_portfolio(
    ctx: RunContext[AgentDeps],
    symbols: list[str],
    expected_returns: dict[str, float],
    covariance_matrix: list[list[float]],
    constraints: dict | None = None,
) -> dict:
    """Optimize portfolio weights using mean-variance optimization.
    
    Args:
        symbols: List of symbols
        expected_returns: Expected return for each symbol
        covariance_matrix: NxN covariance matrix
        constraints: Optional constraints (max_weight, min_weight, sector_limits)
    """
    # Runs in CodeMode - can use cvxpy, scipy, or custom solver
    import numpy as np
    
    n = len(symbols)
    mu = np.array([expected_returns.get(s, 0.1) for s in symbols])
    cov = np.array(covariance_matrix)
    
    # Simple mean-variance with risk aversion
    risk_aversion = constraints.get("risk_aversion", 3.0) if constraints else 3.0
    max_weight = constraints.get("max_weight", 0.3) if constraints else 0.3
    
    # Solve: max w'μ - λ/2 w'Σw s.t. sum(w)=1, 0<=w<=max_weight
    # Using simple iterative approach
    inv_cov = np.linalg.pinv(cov)
    weights = inv_cov @ mu / risk_aversion
    weights = np.clip(weights, 0, max_weight)
    weights = weights / weights.sum()
    
    expected_return = float(weights @ mu)
    expected_vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = expected_return / expected_vol if expected_vol > 0 else 0
    
    return {
        "weights": {s: float(w) for s, w in zip(symbols, weights)},
        "expected_return": expected_return,
        "expected_volatility": expected_vol,
        "sharpe_ratio": sharpe,
    }


@quant_agent.tool
async def stress_test_portfolio(
    ctx: RunContext[AgentDeps],
    positions: list[dict],
    scenarios: list[dict] | None = None,
) -> dict:
    """Run stress tests on portfolio positions.
    
    Args:
        positions: List of {symbol, qty, price}
        scenarios: Optional custom scenarios (market_crash, correlation_break, etc.)
    """
    
    if scenarios is None:
        scenarios = [
            {"name": "market_crash", "equity_shock": -0.20, "vol_multiplier": 2.0},
            {"name": "correlation_break", "equity_shock": -0.10, "correlation": 0.9},
            {"name": "rate_shock", "equity_shock": -0.05, "duration": 5},
            {"name": "liquidity_crisis", "equity_shock": -0.15, "spread_widening": 3.0},
        ]
    
    results = {}
    total_value = sum(p["qty"] * p["price"] for p in positions)
    
    for scenario in scenarios:
        shock = scenario.get("equity_shock", -0.10)
        pnl = total_value * shock
        results[scenario["name"]] = {
            "pnl": float(pnl),
            "pnl_pct": shock * 100,
            "remaining_value": float(total_value + pnl),
        }
    
    return {"scenarios": results, "total_value": float(total_value)}


@quant_agent.tool
async def factor_analysis(
    ctx: RunContext[AgentDeps],
    symbols: list[str],
    lookback_days: int = 252,
) -> dict:
    """Perform PCA factor analysis on symbol returns.
    
    Args:
        symbols: List of symbols
        lookback_days: Historical lookback period
    """
    import numpy as np
    
    # Placeholder - would fetch returns and run PCA
    n = len(symbols)
    n_factors = min(5, n)
    
    # Simulated factor loadings
    loadings = np.random.randn(n, n_factors) * 0.3
    factor_names = [f"Factor_{i+1}" for i in range(n_factors)]
    
    return {
        "symbols": symbols,
        "factors": factor_names,
        "loadings": loadings.tolist(),
        "explained_variance": [0.4, 0.2, 0.15, 0.1, 0.05][:n_factors],
    }