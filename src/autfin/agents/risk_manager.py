"""Risk manager agent for AutFin."""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from langgraph.config import RunnableConfig

from src.autfin.config.settings import AutFinSettings
from src.autfin.state.agent_state import AgentState
from src.autfin.llm.client import get_llm_client

logger = logging.getLogger(__name__)


# Risk limits configuration
RISK_LIMITS = {
    "position_size_limit": 0.10,        # Max 10% of portfolio per position
    "sector_exposure_limit": 0.25,      # Max 25% per sector
    "volatility_exposure_limit": 0.03,  # Max 3% daily drawdown
    "margin_usage_limit": 0.80,         # Max 80% margin usage
    "stock_concentration_limit": 0.15,  # Max 15% per stock
    "daily_loss_limit": 0.02,           # Max 2% daily loss
    "max_open_positions": 15,           # Max 15 positions
    "stop_loss_distance": 0.05,         # Min 5% from entry
    "min_risk_reward_ratio": 2.0,       # Min 1:2 R:R
}

# Sector mapping for Indian stocks
SECTOR_MAP = {
    # IT
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT",
    "TECHM": "IT", "TATACONS": "IT", "LTTS": "IT", "MPHASIS": "IT",
    # Banking
    "HDFCBANK": "Bank", "ICICIBANK": "Bank", "SBIN": "Bank",
    "KOTAKBANK": "Bank", "AXISBANK": "Bank", "INDUSIND": "Bank",
    # Pharma
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "DIVISLAB": "Pharma",
    "CIPLA": "Pharma", "LUPIN": "Pharma", "AJANTPHARM": "Pharma",
    # Auto
    "TATAMOTORS": "Auto", "M&M": "Auto",
    "HEROMOTOCO": "Auto", "MARUTI": "Auto", "EICHERMOT": "Auto",
    # FMCG
    "NESTLEIND": "FMCG", "ULTRACEMCO": "FMCG", "BRITANNIA": "FMCG",
    "ITC": "FMCG", "HINDUNILVR": "FMCG", "COLPAL": "FMCG",
    # Energy
    "RELIANCE": "Energy", "TATAPOWER": "Energy", "NTPC": "Energy",
    "POWERGRID": "Energy", "COALINDIA": "Energy",
    # Metals
    "TATASTEEL": "Metals", "JSWSTEEL": "Metals", "ADANIENT": "Metals",
    "VEDANTA": "Metals",
    # Infrastructure
    "LARSEN": "Infrastructure", "SHREECEM": "Infrastructure",
    "ACC": "Infrastructure", "AMBUJACEM": "Infrastructure",
    # Finance
    "HDFC": "Finance", "BAJFINANCE": "Finance",
    "BAJAJFINSV": "Finance", "SBILIFE": "Finance",
}


class RiskManagerAgent:
    """Manages and monitors portfolio risk with veto power."""

    def __init__(self):
        settings = AutFinSettings()
        self.risk_limits = {
            **RISK_LIMITS,
            "position_size_limit": settings.max_single_position_pct,
            "sector_exposure_limit": settings.max_sector_exposure_pct,
        }

    def _map_symbol_to_sector(self, symbol: str) -> str:
        """Map stock symbol to sector."""
        return SECTOR_MAP.get(symbol, "Other")

    def _get_portfolio_value(self, portfolio: Dict) -> float:
        """Calculate total portfolio value."""
        holdings = portfolio.get("holdings", [])
        cash = portfolio.get("available_cash", 0)
        portfolio_value = sum(
            h.get("qty", 0) * h.get("ltp", h.get("avg_price", 0))
            for h in holdings
        )
        return portfolio_value + cash

    def _calculate_position_value(self, trade: Dict) -> float:
        """Calculate value of a position."""
        qty = trade.get("qty", 0)
        price = trade.get("price", 0)
        return qty * price

    def _calculate_daily_pnl(self, portfolio: Dict, trades: List[Dict]) -> float:
        """Calculate daily PnL from recent trades."""
        today = time.strftime("%Y-%m-%d")
        daily_pnl = 0.0

        for trade in trades:
            trade_date = trade.get("timestamp", "")[:10]
            if trade_date == today:
                entry_price = trade.get("price", 0)
                current_price = trade.get("current_price", entry_price)
                qty = trade.get("qty", 0)
                daily_pnl += (current_price - entry_price) * qty

        return daily_pnl

    def _calculate_atr(self, portfolio: Dict, symbol: str) -> float:
        """Calculate Average True Range for volatility estimation."""
        # Simplified ATR calculation - in production would use OHLC data
        holdings = portfolio.get("holdings", [])
        for h in holdings:
            if h.get("trading_symbol") == symbol:
                avg_price = h.get("avg_price", 0)
                # Assume 1.5% ATR as typical for Indian stocks
                return avg_price * 0.015
        return 0.0

    def check_position_size(self, trade: Dict, portfolio: Dict) -> Tuple[bool, str]:
        """
        Check if position size exceeds limit (max 10% of portfolio).

        Args:
            trade: Trade details (symbol, qty, price)
            portfolio: Current portfolio state

        Returns:
            Tuple of (passes, reason)
        """
        portfolio_value = self._get_portfolio_value(portfolio)
        position_value = self._calculate_position_value(trade)
        position_pct = position_value / portfolio_value if portfolio_value > 0 else 0

        limit = self.risk_limits["position_size_limit"]

        if position_pct > limit:
            return False, (
                f"Position size {position_pct:.1%} exceeds limit {limit:.1%}. "
                f"Position: ₹{position_value:,.0f}, Portfolio: ₹{portfolio_value:,.0f}"
            )
        return True, f"Position size OK: {position_pct:.1%}"

    def check_sector_exposure(self, trade: Dict, portfolio: Dict) -> Tuple[bool, str]:
        """
        Check sector exposure limit (max 25% per sector).

        Args:
            trade: Trade details
            portfolio: Current portfolio state

        Returns:
            Tuple of (passes, reason)
        """
        symbol = trade.get("symbol", "")
        sector = self._map_symbol_to_sector(symbol)
        holdings = portfolio.get("holdings", [])

        # Calculate current sector exposure
        sector_value = 0.0
        total_value = 0.0

        for h in holdings:
            value = h.get("qty", 0) * h.get("ltp", h.get("avg_price", 0))
            total_value += value
            if self._map_symbol_to_sector(h.get("trading_symbol", "")) == sector:
                sector_value += value

        # Add new position
        new_position = self._calculate_position_value(trade)
        new_sector_value = sector_value + new_position
        total_with_new = total_value + new_position

        sector_pct = new_sector_value / total_with_new if total_with_new > 0 else 0
        limit = self.risk_limits["sector_exposure_limit"]

        if sector_pct > limit:
            return False, (
                f"Sector {sector} exposure {sector_pct:.1%} exceeds limit {limit:.1%}. "
                f"Sector value: ₹{new_sector_value:,.0f}"
            )
        return True, f"Sector {sector} exposure OK: {sector_pct:.1%}"

    def check_volatility_exposure(self, trade: Dict, portfolio: Dict) -> Tuple[bool, str]:
        """
        Check volatility exposure (max 3% daily drawdown).

        Args:
            trade: Trade details
            portfolio: Current portfolio state

        Returns:
            Tuple of (passes, reason)
        """
        symbol = trade.get("symbol", "")
        atr = self._calculate_atr(portfolio, symbol)
        entry_price = trade.get("price", 0)

        if entry_price > 0:
            volatility_pct = atr / entry_price
        else:
            volatility_pct = 0.0

        limit = self.risk_limits["volatility_exposure_limit"]

        if volatility_pct > limit:
            return False, (
                f"Volatility {volatility_pct:.1%} exceeds limit {limit:.1%}. "
                f"ATR: ₹{atr:.2f}, Entry: ₹{entry_price:.2f}"
            )
        return True, f"Volatility OK: {volatility_pct:.1%}"

    def check_margin_usage(self, portfolio: Dict) -> Tuple[bool, str]:
        """
        Check margin usage limit (max 80% of available margin).

        Args:
            portfolio: Current portfolio state

        Returns:
            Tuple of (passes, reason)
        """
        available_margin = portfolio.get("available_margin", 0)
        utilized_margin = portfolio.get("utilized_margin", 0)

        if available_margin > 0:
            margin_usage = utilized_margin / available_margin
        else:
            margin_usage = 1.0

        limit = self.risk_limits["margin_usage_limit"]

        if margin_usage > limit:
            return False, (
                f"Margin usage {margin_usage:.1%} exceeds limit {limit:.1%}. "
                f"Utilized: ₹{utilized_margin:,.0f}, Available: ₹{available_margin:,.0f}"
            )
        return True, f"Margin usage OK: {margin_usage:.1%}"

    def check_stock_concentration(self, portfolio: Dict, new_position: float) -> Tuple[bool, str]:
        """
        Check single stock concentration (max 15% per stock).

        Args:
            portfolio: Current portfolio state
            new_position: Value of new position being added

        Returns:
            Tuple of (passes, reason)
        """
        holdings = portfolio.get("holdings", [])
        total_value = self._get_portfolio_value(portfolio)

        # Find largest position
        max_position_value = 0.0
        for h in holdings:
            value = h.get("qty", 0) * h.get("ltp", h.get("avg_price", 0))
            if value > max_position_value:
                max_position_value = value

        # Check if new position would create concentration
        new_total = total_value + new_position
        max_concentration = max(max_position_value, new_position) / new_total if new_total > 0 else 0

        limit = self.risk_limits["stock_concentration_limit"]

        if max_concentration > limit:
            return False, (
                f"Concentration {max_concentration:.1%} exceeds limit {limit:.1%}. "
                f"Max position: ₹{max(max_position_value, new_position):,.0f}"
            )
        return True, f"Concentration OK: {max_concentration:.1%}"

    def check_daily_loss(self, portfolio: Dict, trades: List[Dict]) -> Tuple[bool, str]:
        """
        Check daily loss limit (max 2% of portfolio loss per day).

        Args:
            portfolio: Current portfolio state
            trades: Recent trades

        Returns:
            Tuple of (passes, reason)
        """
        portfolio_value = self._get_portfolio_value(portfolio)
        daily_pnl = self._calculate_daily_pnl(portfolio, trades)
        daily_loss_pct = abs(daily_pnl) / portfolio_value if portfolio_value > 0 else 0

        limit = self.risk_limits["daily_loss_limit"]

        if daily_pnl < 0 and daily_loss_pct > limit:
            return False, (
                f"Daily loss {daily_loss_pct:.1%} exceeds limit {limit:.1%}. "
                f"Loss: ₹{daily_pnl:,.0f}"
            )
        return True, f"Daily PnL OK: ₹{daily_pnl:,.0f} ({daily_loss_pct:.2%})"

    def check_max_positions(self, portfolio: Dict) -> Tuple[bool, str]:
        """
        Check maximum open positions (max 15 positions).

        Args:
            portfolio: Current portfolio state

        Returns:
            Tuple of (passes, reason)
        """
        holdings = portfolio.get("holdings", [])
        positions = portfolio.get("positions", [])
        open_positions = len(holdings) + len(positions)

        limit = self.risk_limits["max_open_positions"]

        if open_positions >= limit:
            return False, (
                f"Open positions {open_positions} at limit {limit}. "
                f"Cannot open more positions."
            )
        return True, f"Open positions OK: {open_positions}/{limit}"

    def check_stop_loss_distance(self, trade: Dict) -> Tuple[bool, str]:
        """
        Check stop loss distance (min 5% from entry).

        Args:
            trade: Trade details including stop_loss_price

        Returns:
            Tuple of (passes, reason)
        """
        entry_price = trade.get("price", 0)
        stop_loss_price = trade.get("stop_loss_price", 0)

        if entry_price <= 0 or stop_loss_price <= 0:
            return False, (
                f"Trade rejected: stop loss is mandatory but not provided. "
                f"Entry: ₹{entry_price:.2f}, Stop: ₹{stop_loss_price:.2f}"
            )

        distance = abs(entry_price - stop_loss_price) / entry_price

        limit = self.risk_limits["stop_loss_distance"]

        if distance < limit:
            return False, (
                f"Stop loss distance {distance:.1%} below minimum {limit:.1%}. "
                f"Entry: ₹{entry_price:.2f}, Stop: ₹{stop_loss_price:.2f}"
            )
        return True, f"Stop loss distance OK: {distance:.1%}"

    def check_risk_reward_ratio(self, trade: Dict) -> Tuple[bool, str]:
        """
        Check risk/reward ratio (min 1:2 R:R).

        Args:
            trade: Trade details including target_price

        Returns:
            Tuple of (passes, reason)
        """
        entry_price = trade.get("price", 0)
        stop_loss_price = trade.get("stop_loss_price", 0)
        target_price = trade.get("target_price", 0)

        if entry_price > 0 and stop_loss_price > 0 and target_price > 0:
            potential_loss = abs(entry_price - stop_loss_price)
            potential_gain = abs(target_price - entry_price)
            risk_reward_ratio = potential_gain / potential_loss if potential_loss > 0 else 0
        else:
            potential_loss = 0.0
            potential_gain = 0.0
            risk_reward_ratio = 0.0

        limit = self.risk_limits["min_risk_reward_ratio"]

        if risk_reward_ratio < limit:
            return False, (
                f"Risk/Reward {risk_reward_ratio:.1f}:{limit:.1f} below minimum 1:{limit:.1f}. "
                f"Loss: ₹{potential_loss:.2f}, Gain: ₹{potential_gain:.2f}"
            )
        return True, f"Risk/Reward OK: {risk_reward_ratio:.1f}:{limit:.1f}"

    def check_correlation(self, trade: Dict, portfolio: Dict) -> Tuple[bool, str]:
        """
        Check if adding this position creates over-concentration.
        
        Flags when >30% of portfolio is in the same sector or >40% in top 3 stocks.
        
        Args:
            trade: Trade details
            portfolio: Current portfolio
            
        Returns:
            Tuple of (passes, reason)
        """
        symbol = trade.get("symbol", "")
        sector = self._map_symbol_to_sector(symbol)
        trade_value = self._calculate_position_value(trade)
        total_value = self._get_portfolio_value(portfolio)
        
        if total_value <= 0:
            return True, "Portfolio empty — correlation check skipped"
        
        holdings = portfolio.get("holdings", [])
        
        # Calculate sector exposure after this trade
        sector_value = trade_value  # New position
        symbol_values = {}
        for h in holdings:
            h_symbol = h.get("tradingsymbol", h.get("symbol", ""))
            h_value = h.get("qty", 0) * h.get("ltp", h.get("avg_price", 0))
            h_sector = self._map_symbol_to_sector(h_symbol)
            if h_sector == sector:
                sector_value += h_value
            if h_symbol not in symbol_values:
                symbol_values[h_symbol] = 0
            symbol_values[h_symbol] += h_value
        
        # Add new position
        if symbol in symbol_values:
            symbol_values[symbol] += trade_value
        else:
            symbol_values[symbol] = trade_value
        
        # Check sector concentration (max 30%)
        sector_pct = (sector_value / total_value) * 100
        if sector_pct > 30:
            return False, (
                f"Sector '{sector}' exposure {sector_pct:.1f}% exceeds 30% limit "
                f"(₹{sector_value:,.0f} of ₹{total_value:,.0f})"
            )
        
        # Check top-3 concentration (max 40%)
        top_3 = sorted(symbol_values.values(), reverse=True)[:3]
        top_3_pct = (sum(top_3) / total_value) * 100
        if top_3_pct > 40:
            return False, (
                f"Top 3 stocks concentration {top_3_pct:.1f}% exceeds 40% limit"
            )
        
        return True, f"Correlation OK: sector {sector_pct:.1f}%, top-3 {top_3_pct:.1f}%"

    def check_liquidity(self, trade: Dict, portfolio: Dict) -> Tuple[bool, str]:
        """
        Check if the stock has sufficient trading liquidity.
        
        Requires avg daily volume to support at least 5x the proposed position
        (so the position can be exited within a day without moving the market).
        
        Args:
            trade: Trade details
            portfolio: Current portfolio
            
        Returns:
            Tuple of (passes, reason)
        """
        symbol = trade.get("symbol", "")
        avg_volume = trade.get("avg_volume", 0)
        avg_daily_volume_value = trade.get("avg_daily_volume_value", 0)
        trade_value = self._calculate_position_value(trade)
        
        # If volume data not provided, skip with warning
        if avg_volume <= 0 and avg_daily_volume_value <= 0:
            return True, f"Liquidity check skipped — no volume data for {symbol}"
        
        # Use value-based check if available
        if avg_daily_volume_value > 0:
            # Position should be < 20% of daily volume value (can exit in 1 day)
            position_vs_volume = (trade_value / avg_daily_volume_value) * 100
            if position_vs_volume > 20:
                return False, (
                    f"Position ₹{trade_value:,.0f} is {position_vs_volume:.1f}% of "
                    f"daily volume (₹{avg_daily_volume_value:,.0f}). Max 20% for liquidity."
                )
            return True, f"Liquidity OK: position is {position_vs_volume:.1f}% of daily volume"
        
        # Quantity-based fallback
        trade_qty = trade.get("qty", 0)
        if trade_qty > 0 and avg_volume > 0:
            days_to_exit = trade_qty / (avg_volume * 0.2)  # Can use 20% of daily volume
            if days_to_exit > 1:
                return False, (
                    f"Would take ~{days_to_exit:.0f} days to exit {symbol} "
                    f"(qty {trade_qty}, avg vol {avg_volume:,.0f}). Max 1 day."
                )
            return True, f"Liquidity OK: ~{days_to_exit:.1f} day(s) to exit"
        
        return True, "Liquidity check passed"

    def check_var(self, trade: Dict, portfolio: Dict) -> Tuple[bool, str]:
        """
        Check Value-at-Risk (simplified parametric VaR).
        
        Estimates 1-day 95% VaR and blocks if it exceeds 3% of portfolio value.
        
        Args:
            trade: Trade details
            portfolio: Current portfolio
            
        Returns:
            Tuple of (passes, reason)
        """
        volatility = trade.get("volatility", 0.0)
        trade_value = self._calculate_position_value(trade)
        total_value = self._get_portfolio_value(portfolio)
        
        if total_value <= 0 or trade_value <= 0:
            return True, "VaR check skipped — no position value"
        
        if volatility <= 0:
            # Estimate volatility from ATR if available
            atr = trade.get("atr", 0)
            price = trade.get("price", 0)
            if atr > 0 and price > 0:
                volatility = atr / price  # Convert ATR to percentage
            else:
                return True, f"VaR check skipped — no volatility data for {trade.get('symbol', '')}"
        
        # Parametric VaR at 95% confidence (z = 1.645)
        # VaR = position_value * volatility * z_score
        z_score = 1.645
        var_amount = trade_value * volatility * z_score
        var_pct = (var_amount / total_value) * 100
        
        if var_pct > 3.0:
            return False, (
                f"1-day 95% VaR ₹{var_amount:,.0f} ({var_pct:.1f}% of portfolio) "
                f"exceeds 3% limit. Volatility: {volatility*100:.1f}%"
            )
        
        return True, f"VaR OK: ₹{var_amount:,.0f} ({var_pct:.1f}% of portfolio)"

    def check_cvar(self, trade: Dict, portfolio: Dict) -> Tuple[bool, str]:
        """
        Check Conditional Value at Risk (Expected Shortfall).
        
        CVaR estimates the expected loss in the worst 5% of scenarios.
        Uses a simplified approach: CVaR ≈ 1.5 * VaR for normal distributions.
        
        Args:
            trade: Trade details
            portfolio: Current portfolio
            
        Returns:
            Tuple of (passes, reason)
        """
        volatility = trade.get("volatility", 0.0)
        trade_value = self._calculate_position_value(trade)
        total_value = self._get_portfolio_value(portfolio)
        
        if total_value <= 0 or trade_value <= 0:
            return True, "CVaR check skipped — no position value"
        
        if volatility <= 0:
            atr = trade.get("atr", 0)
            price = trade.get("price", 0)
            if atr > 0 and price > 0:
                volatility = atr / price
            else:
                return True, f"CVaR check skipped — no volatility data for {trade.get('symbol', '')}"
        
        # Parametric CVaR at 95% confidence
        # CVaR ≈ position_value * volatility * z_score * 1.5 (scaling factor)
        z_score = 1.645
        cvar_amount = trade_value * volatility * z_score * 1.5
        cvar_pct = (cvar_amount / total_value) * 100
        
        if cvar_pct > 5.0:
            return False, (
                f"1-day 95% CVaR ₹{cvar_amount:,.0f} ({cvar_pct:.1f}% of portfolio) "
                f"exceeds 5% limit. Expected shortfall in worst scenarios too high."
            )
        
        return True, f"CVaR OK: ₹{cvar_amount:,.0f} ({cvar_pct:.1f}% of portfolio)"

    def check_news_risk(self, trade: Dict, portfolio: Dict) -> Tuple[bool, str]:
        """
        Check if there is negative news sentiment for the stock.
        
        Uses sentiment_reports from state if available.
        Blocks buys if sentiment is strongly negative (score < -0.3).
        
        Args:
            trade: Trade details (may include 'sentiment_score' key)
            portfolio: Current portfolio (may include 'sentiment_reports')
            
        Returns:
            Tuple of (passes, reason)
        """
        symbol = trade.get("symbol", "")
        action = trade.get("action", "BUY")
        
        # Only applies to BUY orders
        if action != "BUY":
            return True, "News risk check skipped — only applies to BUY orders"
        
        # Check if sentiment score is in the trade dict
        sentiment_score = trade.get("sentiment_score")
        if sentiment_score is None:
            # Try to get from portfolio's sentiment_reports
            sentiment_reports = portfolio.get("sentiment_reports", {})
            if sentiment_reports and isinstance(sentiment_reports, dict):
                stock_sentiment = sentiment_reports.get(symbol, {})
                sentiment_score = stock_sentiment.get("score") if isinstance(stock_sentiment, dict) else None
        
        if sentiment_score is None:
            return True, f"News risk check passed — no sentiment data for {symbol}"
        
        try:
            score = float(sentiment_score)
        except (ValueError, TypeError):
            return True, "News risk check passed — invalid sentiment score"
        
        if score < -0.3:
            return False, (
                f"Strong negative news sentiment for {symbol} (score: {score:.2f}). "
                f"Consider waiting for sentiment to improve before buying."
            )
        elif score < -0.1:
            return True, f"News risk warning: mild negative sentiment for {symbol} (score: {score:.2f})"
        
        return True, f"News risk OK: sentiment for {symbol} is {'positive' if score > 0.1 else 'neutral'} ({score:.2f})"

    def check_drawdown(self, portfolio: Dict) -> Tuple[bool, str]:
        """
        Check portfolio drawdown halt circuit breaker.

        Compares current portfolio value against the peak value recorded
        in portfolio_snapshots. If drawdown exceeds the configured threshold
        (default 10%), blocks all new BUY orders.

        Args:
            portfolio: Current portfolio state (may include 'peak_portfolio_value')

        Returns:
            Tuple of (passes, reason)
        """
        settings = AutFinSettings()
        halt_pct = settings.portfolio_drawdown_halt_pct  # default 0.10

        peak_value = portfolio.get("peak_portfolio_value", 0.0)
        current_value = self._get_portfolio_value(portfolio)

        if peak_value <= 0 or current_value <= 0:
            return True, "Drawdown check skipped — no peak or current value"

        drawdown = (peak_value - current_value) / peak_value

        if drawdown >= halt_pct:
            return False, (
                f"Portfolio drawdown {drawdown:.1%} exceeds halt threshold {halt_pct:.1%}. "
                f"Peak: ₹{peak_value:,.0f}, Current: ₹{current_value:,.0f}. "
                f"HALTING all new buys until recovery."
            )
        return True, f"Drawdown OK: {drawdown:.1%} (halt at {halt_pct:.1f}%)"

    def check_correlation_risk(self, portfolio: Dict) -> Tuple[bool, str]:
        """
        Check if portfolio is over-correlated with the market (Nifty50).

        When average pairwise correlation with Nifty50 > 0.65, the portfolio
        has minimal diversification benefit and is vulnerable to market-wide drawdowns.

        Uses yfinance to fetch 60-day rolling correlation.

        Args:
            portfolio: Current portfolio with holdings

        Returns:
            Tuple of (passes, reason)
        """
        holdings = portfolio.get("holdings", [])
        if len(holdings) < 2:
            return True, "Correlation check skipped — fewer than 2 holdings"

        symbols = []
        for h in holdings:
            sym = h.get("tradingsymbol") or h.get("symbol") or ""
            if sym:
                symbols.append(sym)

        if len(symbols) < 2:
            return True, "Correlation check skipped — fewer than 2 valid symbols"

        try:
            import yfinance as yf
            import numpy as np

            # Fetch 60-day history for all holdings + Nifty50
            tickers = [f"{s}.NS" for s in symbols[:15]] + ["^NSEI"]
            data = yf.download(tickers, period="3mo", progress=False, group_by="ticker")

            if data is None or data.empty:
                return True, "Correlation check skipped — no price data available"

            # Compute daily returns
            returns = {}
            for sym in symbols[:15]:
                try:
                    col = f"{sym}.NS"
                    if col in data.columns:
                        prices = data[col]["Close"].dropna()
                        if len(prices) >= 20:
                            returns[sym] = prices.pct_change().dropna()
                except Exception:
                    continue

            nifty_col = "^NSEI"
            if nifty_col not in data.columns:
                return True, "Correlation check skipped — Nifty50 data unavailable"

            nifty_returns = data[nifty_col]["Close"].pct_change().dropna()

            if not returns or len(nifty_returns) < 20:
                return True, "Correlation check skipped — insufficient data"

            # Compute average correlation with Nifty50
            correlations = []
            for sym, ret in returns.items():
                common_idx = ret.index.intersection(nifty_returns.index)
                if len(common_idx) >= 20:
                    corr = float(np.corrcoef(
                        ret.loc[common_idx].values,
                        nifty_returns.loc[common_idx].values,
                    )[0, 1])
                    correlations.append((sym, corr))

            if not correlations:
                return True, "Correlation check skipped — no valid correlations computed"

            avg_corr = sum(c for _, c in correlations) / len(correlations)
            max_corr_sym = max(correlations, key=lambda x: abs(x[1]))
            min_corr_sym = min(correlations, key=lambda x: abs(x[1]))

            THRESHOLD = 0.65
            if avg_corr > THRESHOLD:
                return False, (
                    f"Portfolio average Nifty50 correlation {avg_corr:.2f} > {THRESHOLD} threshold. "
                    f"Most correlated: {max_corr_sym[0]} ({max_corr_sym[1]:.2f}), "
                    f"Least: {min_corr_sym[0]} ({min_corr_sym[1]:.2f}). "
                    f"Consider adding uncorrelated assets (gold, pharma, IT)."
                )

            return True, (
                f"Correlation OK: avg={avg_corr:.2f} (threshold {THRESHOLD}). "
                f"Range: {min_corr_sym[0]}={min_corr_sym[1]:.2f} to "
                f"{max_corr_sym[0]}={max_corr_sym[1]:.2f}"
            )

        except Exception as e:
            logger.warning("Correlation check failed: %s", e)
            return True, f"Correlation check skipped — computation failed: {e}"

    def calculate_risk_score(self, risk_report: Dict) -> int:
        """
        Calculate overall risk score (0-100).

        Scoring:
        - 0-30: Low risk (pass)
        - 31-60: Medium risk (pass with warnings)
        - 61-90: High risk (veto possible)
        - 91-100: Extreme risk (veto required)

        Args:
            risk_report: Dictionary from check_all_risks

        Returns:
            Risk score (0-100)
        """
        checks = risk_report.get("checks", [])
        flagged_checks = [c for c in checks if not c.get("passes", True)]
        total_checks = len(checks)

        if total_checks == 0:
            return 0

        # Base score from failure rate
        failure_rate = len(flagged_checks) / total_checks
        base_score = int(failure_rate * 50)

        # Add severity points
        severity_points = 0
        for check in flagged_checks:
            severity = check.get("severity", "medium")
            if severity == "high":
                severity_points += 20
            elif severity == "medium":
                severity_points += 10
            else:
                severity_points += 5

        score = min(base_score + severity_points, 100)
        return score

    def check_all_risks(self, trade: Dict, portfolio: Dict, signals: Dict = None) -> Dict:
        """
        Run all 14 risk checks and return comprehensive report.

        Args:
            trade: Trade details
            portfolio: Current portfolio state
            signals: Optional trading signals for R:R calculation

        Returns:
            Risk report with all check results
        """
        if signals:
            # Use signals to populate trade details
            trade = {**trade, **signals}

        checks = []

        limit_pct = int(self.risk_limits["position_size_limit"] * 100)
        sector_pct = int(self.risk_limits["sector_exposure_limit"] * 100)

        # 1. Position size
        passes, reason = self.check_position_size(trade, portfolio)
        checks.append({
            "name": "position_size",
            "description": f"Position size limit (max {limit_pct}% of portfolio)",
            "passes": passes,
            "reason": reason,
            "severity": "high" if not passes else "low",
        })

        # 2. Sector exposure
        passes, reason = self.check_sector_exposure(trade, portfolio)
        checks.append({
            "name": "sector_exposure",
            "description": f"Sector exposure limit (max {sector_pct}% per sector)",
            "passes": passes,
            "reason": reason,
            "severity": "high" if not passes else "low",
        })

        # 3. Volatility exposure
        passes, reason = self.check_volatility_exposure(trade, portfolio)
        checks.append({
            "name": "volatility_exposure",
            "description": "Volatility exposure (max 3% daily drawdown)",
            "passes": passes,
            "reason": reason,
            "severity": "medium" if not passes else "low",
        })

        # 4. Margin usage
        passes, reason = self.check_margin_usage(portfolio)
        checks.append({
            "name": "margin_usage",
            "description": "Margin usage limit (max 80%)",
            "passes": passes,
            "reason": reason,
            "severity": "high" if not passes else "low",
        })

        # 5. Stock concentration
        position_value = self._calculate_position_value(trade)
        passes, reason = self.check_stock_concentration(portfolio, position_value)
        checks.append({
            "name": "stock_concentration",
            "description": "Single stock concentration (max 15%)",
            "passes": passes,
            "reason": reason,
            "severity": "medium" if not passes else "low",
        })

        # 6. Daily loss
        trades = portfolio.get("recent_trades", [])
        passes, reason = self.check_daily_loss(portfolio, trades)
        checks.append({
            "name": "daily_loss",
            "description": "Daily loss limit (max 2% per day)",
            "passes": passes,
            "reason": reason,
            "severity": "high" if not passes else "low",
        })

        # 7. Max positions
        passes, reason = self.check_max_positions(portfolio)
        checks.append({
            "name": "max_positions",
            "description": "Maximum open positions (max 15)",
            "passes": passes,
            "reason": reason,
            "severity": "high" if not passes else "low",
        })

        # 8. Stop loss distance
        passes, reason = self.check_stop_loss_distance(trade)
        checks.append({
            "name": "stop_loss_distance",
            "description": "Stop loss distance (min 5% from entry)",
            "passes": passes,
            "reason": reason,
            "severity": "medium" if not passes else "low",
        })

        # 9. Risk/reward ratio
        passes, reason = self.check_risk_reward_ratio(trade)
        checks.append({
            "name": "risk_reward_ratio",
            "description": "Risk/reward ratio (min 1:2)",
            "passes": passes,
            "reason": reason,
            "severity": "medium" if not passes else "low",
        })

        # 10. Correlation / concentration
        passes, reason = self.check_correlation(trade, portfolio)
        checks.append({
            "name": "correlation",
            "description": "Sector (max 30%) and top-3 (max 40%) concentration",
            "passes": passes,
            "reason": reason,
            "severity": "high" if not passes else "low",
        })

        # 11. Liquidity
        passes, reason = self.check_liquidity(trade, portfolio)
        checks.append({
            "name": "liquidity",
            "description": "Liquidity check (position < 20% daily volume)",
            "passes": passes,
            "reason": reason,
            "severity": "high" if not passes else "low",
        })

        # 12. VaR (Value at Risk)
        passes, reason = self.check_var(trade, portfolio)
        checks.append({
            "name": "var",
            "description": "1-day 95% VaR (max 3% of portfolio)",
            "passes": passes,
            "reason": reason,
            "severity": "high" if not passes else "low",
        })

        # 13. CVaR (Conditional Value at Risk / Expected Shortfall)
        passes, reason = self.check_cvar(trade, portfolio)
        checks.append({
            "name": "cvar",
            "description": "1-day 95% CVaR (max 5% of portfolio)",
            "passes": passes,
            "reason": reason,
            "severity": "high" if not passes else "low",
        })

        # 14. News Risk
        passes, reason = self.check_news_risk(trade, portfolio)
        checks.append({
            "name": "news_risk",
            "description": "Negative news sentiment check",
            "passes": passes,
            "reason": reason,
            "severity": "medium" if not passes else "low",
        })

        # 15. Drawdown circuit breaker
        passes, reason = self.check_drawdown(portfolio)
        checks.append({
            "name": "drawdown_halt",
            "description": f"Portfolio drawdown halt (max {int(self.risk_limits.get('drawdown_halt_pct', 0.10) * 100)}%)",
            "passes": passes,
            "reason": reason,
            "severity": "critical" if not passes else "low",
        })

        risk_score = self.calculate_risk_score({"checks": checks})
        flagged_issues = [c for c in checks if not c.get("passes", True)]

        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "risk_score": risk_score,
            "risk_level": self._get_risk_level(risk_score),
            "checks": checks,
            "flagged_issues": flagged_issues,
            "total_checks": len(checks),
            "passed_checks": len([c for c in checks if c.get("passes", False)]),
        }

    def _get_risk_level(self, risk_score: int) -> str:
        """Get risk level string from score."""
        if risk_score < 30:
            return "LOW"
        elif risk_score < 60:
            return "MEDIUM"
        elif risk_score < 90:
            return "HIGH"
        else:
            return "EXTREME"

    def apply_veto(self, risk_score: int) -> Tuple[bool, str]:
        """
        Apply veto based on risk score.

        Args:
            risk_score: Calculated risk score (0-100)

        Returns:
            Tuple of (should_veto, reason)
        """
        if risk_score < 60:
            return False, "APPROVED - Risk score within acceptable limits"
        elif risk_score < 90:
            return False, "APPROVED_WITH_WARNINGS - Risk score elevated but acceptable"
        else:
            return True, "VETOED - Risk score too high (>= 90)"

    async def risk_manager_agent(self, state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
        """
        Main agent function - orchestrate risk management.

        Args:
            state: AgentState with portfolio and trade info

        Returns:
            Partial state update with risk assessment
        """
        trade = state.get("pending_trade", {})
        portfolio = {
            "holdings": state.get("holdings", []),
            "positions": state.get("positions", []),
            "available_cash": state.get("available_cash", 0),
            "utilized_margin": state.get("utilized_margin", 0),
            "available_margin": state.get("available_margin", 0),
            "recent_trades": state.get("recent_trades", []),
        }
        debates = state.get("research_debates", {})

        # Build per-symbol risk assessments when symbols can be inferred.
        risk_assessments: Dict[str, Any] = {}
        symbols: List[str] = []

        if isinstance(debates, dict):
            if "summary" in debates:
                # Aggregate debate format: assess current holdings.
                symbols = [
                    h.get("tradingsymbol") or h.get("symbol") or h.get("trading_symbol", "")
                    for h in state.get("holdings", [])
                ]
            else:
                symbols = list(debates.keys())

        for symbol in symbols:
            if not symbol:
                continue
            debate = debates.get(symbol, {}) if not (isinstance(debates, dict) and "summary" in debates) else {}
            trade_for_symbol = {**trade, "symbol": symbol}
            if isinstance(debate, dict):
                price = debate.get("price", trade_for_symbol.get("price", 0))
                market_data = state.get("market_snapshot") or {}
                quotes = market_data.get("quotes", {})
                quote = quotes.get(symbol, {})
                ltp = quote.get("ltp", 0) if isinstance(quote, dict) else 0
                trade_for_symbol["price"] = price or ltp or trade_for_symbol.get("price", 0)

            risk_report = self.check_all_risks(trade_for_symbol, portfolio, debate)
            risk_score = risk_report["risk_score"]
            should_veto, veto_reason = self.apply_veto(risk_score)

            # Generate LLM rationale for this symbol (optional enhancement)
            llm_rationale = await self.generate_llm_risk_rationale(
                symbol, risk_report, should_veto
            )

            entry = {
                "approved": not should_veto,
                "risk_score": risk_score,
                "risk_level": risk_report["risk_level"],
                "checks": risk_report["checks"],
                "veto_reason": veto_reason if should_veto else None,
            }
            if llm_rationale:
                entry["llm_rationale"] = llm_rationale

            risk_assessments[symbol] = entry

        # Fallback: keep a top-level summary for backward compatibility.
        if len(symbols) <= 1:
            risk_report = self.check_all_risks(trade, portfolio, debates)
            risk_score = risk_report["risk_score"]
            should_veto, veto_reason = self.apply_veto(risk_score)
            if not risk_assessments:
                risk_assessments["_summary"] = {
                    "approved": not should_veto,
                    "risk_score": risk_score,
                    "risk_level": risk_report["risk_level"],
                    "checks": risk_report["checks"],
                    "veto_reason": veto_reason if should_veto else None,
                }

        return {
            "risk_assessments": risk_assessments,
            "next_agent": None,
        }

    async def generate_llm_risk_rationale(
        self, symbol: str, risk_report: Dict, veto: bool
    ) -> Optional[str]:
        """
        Generate LLM-powered risk rationale for a symbol.

        Falls back to None if LLM unavailable or fails.

        Args:
            symbol: Stock symbol
            risk_report: Risk report dict from check_all_risks
            veto: Whether the trade was vetoed

        Returns:
            LLM rationale string or None
        """
        llm = get_llm_client()
        if not llm.is_available:
            return None

        checks = risk_report.get("checks", [])
        flagged = risk_report.get("flagged_issues", [])
        score = risk_report.get("risk_score", 0)
        level = risk_report.get("risk_level", "UNKNOWN")

        prompt = (
            f"You are a risk manager at an Indian brokerage.\n"
            f"Write a 1-2 paragraph risk assessment for {symbol}.\n\n"
            f"## Risk Summary\n"
            f"- Risk Score: {score}/100 ({level})\n"
            f"- Checks Passed: {risk_report.get('passed_checks', 0)}/{risk_report.get('total_checks', 0)}\n"
            f"- Vetoed: {'Yes' if veto else 'No'}\n\n"
        )

        if flagged:
            prompt += "## Flagged Issues\n"
            for issue in flagged:
                prompt += f"- {issue.get('name', '?')}: {issue.get('reason', 'N/A')} [{issue.get('severity', 'medium')}]\n"
            prompt += "\n"

        # Add passing checks summary
        passing = [c for c in checks if c.get("passes")]
        if passing:
            prompt += "## Passing Checks\n"
            for c in passing[:5]:
                prompt += f"- {c.get('name', '?')}: {c.get('reason', 'N/A')}\n"
            prompt += "\n"

        prompt += (
            "## Instructions\n"
            "1. Summarize the key risk factors\n"
            "2. Explain why the trade was " + ("blocked" if veto else "approved") + "\n"
            "3. Note any concerns or conditions for the recommendation\n"
            "Write in plain text, no markdown."
        )

        system = (
            "You are a risk management analyst at an Indian brokerage. "
            "Be specific about which risk checks failed and why. "
            "Write plain text, no markdown formatting."
        )

        return await llm.generate(
            prompt=prompt,
            system_prompt=system,
            max_tokens=400,
        )

    def generate_risk_report(self, risk_report: Dict) -> str:
        """
        Generate human-readable risk report.

        Args:
            risk_report: Risk report dict from check_all_risks

        Returns:
            Formatted risk report string
        """
        lines = [
            "=" * 60,
            "RISK MANAGEMENT REPORT",
            "=" * 60,
            f"Generated: {risk_report.get('timestamp', 'N/A')}",
            f"Risk Score: {risk_report.get('risk_score', 0)}/100",
            f"Risk Level: {risk_report.get('risk_level', 'UNKNOWN')}",
            f"Passed Checks: {risk_report.get('passed_checks', 0)}/{risk_report.get('total_checks', 0)}",
            "-" * 60,
            "",
        ]

        for check in risk_report.get("checks", []):
            status = "✓ PASS" if check.get("passes") else "✗ FAIL"
            severity = f"[{check.get('severity', 'unknown').upper()}]"
            lines.append(f"{status} {check.get('name', 'unknown')} {severity}")
            lines.append(f"  {check.get('description', 'N/A')}")
            lines.append(f"  {check.get('reason', 'N/A')}")
            lines.append("")

        flagged = risk_report.get("flagged_issues", [])
        if flagged:
            lines.append("-" * 60)
            lines.append("FLAGGED ISSUES:")
            for issue in flagged:
                lines.append(f"  - {issue.get('name')}: {issue.get('reason')}")

        lines.append("=" * 60)

        return "\n".join(lines)


# Standalone functions for convenience
def check_position_size(trade: Dict, portfolio: Dict) -> Tuple[bool, str]:
    """Check position size limit (configured in AutFinSettings)."""
    return _risk_agent.check_position_size(trade, portfolio)


def check_sector_exposure(trade: Dict, portfolio: Dict) -> Tuple[bool, str]:
    """Check sector exposure limit (configured in AutFinSettings)."""
    return _risk_agent.check_sector_exposure(trade, portfolio)


def check_volatility_exposure(trade: Dict, portfolio: Dict) -> Tuple[bool, str]:
    """Check volatility exposure (max 3% daily drawdown)."""
    return _risk_agent.check_volatility_exposure(trade, portfolio)


def check_margin_usage(portfolio: Dict) -> Tuple[bool, str]:
    """Check margin usage limit (max 80%)."""
    return _risk_agent.check_margin_usage(portfolio)


def check_stock_concentration(portfolio: Dict, new_position: float) -> Tuple[bool, str]:
    """Check single stock concentration (max 15%)."""
    return _risk_agent.check_stock_concentration(portfolio, new_position)


def check_daily_loss(portfolio: Dict, trades: List[Dict]) -> Tuple[bool, str]:
    """Check daily loss limit (max 2% per day)."""
    return _risk_agent.check_daily_loss(portfolio, trades)


def check_max_positions(portfolio: Dict) -> Tuple[bool, str]:
    """Check maximum open positions (max 15)."""
    return _risk_agent.check_max_positions(portfolio)


def check_stop_loss_distance(trade: Dict) -> Tuple[bool, str]:
    """Check stop loss distance (min 5% from entry)."""
    return _risk_agent.check_stop_loss_distance(trade)


def check_risk_reward_ratio(trade: Dict) -> Tuple[bool, str]:
    """Check risk/reward ratio (min 1:2)."""
    return _risk_agent.check_risk_reward_ratio(trade)


def check_all_risks(trade: Dict, portfolio: Dict, signals: Dict = None) -> Dict:
    """Run all 9 risk checks and return comprehensive report."""
    return _risk_agent.check_all_risks(trade, portfolio, signals)


def calculate_risk_score(risk_report: Dict) -> int:
    """Calculate overall risk score (0-100)."""
    return _risk_agent.calculate_risk_score(risk_report)


def apply_veto(risk_score: int) -> Tuple[bool, str]:
    """Apply veto based on risk score."""
    return _risk_agent.apply_veto(risk_score)


def check_drawdown(portfolio: Dict) -> Tuple[bool, str]:
    """Check portfolio drawdown circuit breaker."""
    return _risk_agent.check_drawdown(portfolio)


_risk_agent = RiskManagerAgent()


async def risk_manager_agent(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Module-level entry point for LangGraph routing."""
    return await _risk_agent.risk_manager_agent(state, config)
