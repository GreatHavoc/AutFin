"""Kite (Zerodha) integration tools for AutFin.

Uses KiteConnect SDK (Personal tier — free):
- Auth (login, TOTP-based auto-login)
- Portfolio (holdings, positions, margins)
- Orders (place, modify, cancel)
- GTT (Good Till Triggered) orders

NOT available on Personal tier (require Connect ₹500/month):
- Real-time market quotes
- Historical candle data
- WebSocket streaming
"""

import os
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

try:
    from kiteconnect import KiteConnect
    KITECONNECT_AVAILABLE = True
except ImportError:
    KITECONNECT_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import pyotp
    PYOTP_AVAILABLE = True
except ImportError:
    PYOTP_AVAILABLE = False

logger = logging.getLogger(__name__)


class KiteTools:
    """Tools for interacting with Zerodha Kite API."""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """Initialize KiteTools with credentials.
        
        Args:
            api_key: Zerodha API key. If not provided, reads from environment.
            api_secret: Zerodha API secret. If not provided, reads from environment.
        """
        self.api_key = api_key or os.getenv("KITE_API_KEY")
        self.api_secret = api_secret or os.getenv("KITE_API_SECRET")
        self.kite: Optional[KiteConnect] = None
        self.access_token: Optional[str] = None
        self._token_file = os.path.expanduser("~/.autfin/kite_token.pkl")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("KITE_API_KEY and KITE_API_SECRET must be set")
    
    def _init_kite(self) -> KiteConnect:
        """Initialize and return KiteConnect instance."""
        if not KITECONNECT_AVAILABLE:
            raise ImportError("kiteconnect package not installed. Run: pip install kiteconnect")
        
        if self.kite is None:
            self.kite = KiteConnect(api_key=self.api_key)
        return self.kite
    
    def _save_token(self, token: str) -> None:
        """Save access token to file (JSON format, not pickle)."""
        import json
        os.makedirs(os.path.dirname(self._token_file), exist_ok=True)
        json_path = self._token_file.replace(".pkl", ".json")
        with open(json_path, "w") as f:
            json.dump({"token": token, "timestamp": datetime.now().timestamp()}, f)
        # Remove legacy pickle file if present
        if os.path.exists(self._token_file):
            try:
                os.remove(self._token_file)
            except OSError:
                pass

    def _load_token(self) -> Optional[str]:
        """Load access token from file if valid."""
        import json
        json_path = self._token_file.replace(".pkl", ".json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                    token_age = datetime.now().timestamp() - data.get("timestamp", 0)
                    # Token is valid for 24 hours
                    if token_age < 86400:
                        return data.get("token")
            except Exception:
                pass
        return None

    def auto_login(self) -> Dict[str, Any]:
        """Perform programmatic login using TOTP (no browser required).

        Requires KITE_USER_ID, KITE_PASSWORD, and KITE_TOTP_SECRET in .env.
        The TOTP secret is obtained once from Zerodha console:
            2FA settings → external authenticator → copy key.

        Returns:
            Dictionary with status, access_token, user info, or error.
        """
        user_id = os.getenv("KITE_USER_ID")
        password = os.getenv("KITE_PASSWORD")
        totp_secret = os.getenv("KITE_TOTP_SECRET")

        if not all([user_id, password, totp_secret]):
            return {
                "status": "error",
                "message": (
                    "KITE_USER_ID, KITE_PASSWORD, and KITE_TOTP_SECRET must be set "
                    "for auto_login. Set them in .env or environment."
                ),
            }

        if not KITECONNECT_AVAILABLE:
            return {"status": "error", "message": "kiteconnect package not installed"}
        if not REQUESTS_AVAILABLE:
            return {"status": "error", "message": "requests package not installed"}
        if not PYOTP_AVAILABLE:
            return {"status": "error", "message": "pyotp package not installed"}

        try:
            kite = self._init_kite()
            session = requests.Session()

            # Step 1: Get the login page URL (we'll need it after 2FA)
            login_page_url = f"https://kite.trade/connect/login?v=3&api_key={self.api_key}"
            login_page_res = session.get(url=login_page_url)
            redirect_after_login = login_page_res.url  # This is the full redirect URL

            # Step 2: POST login credentials
            login_resp = session.post(
                "https://kite.zerodha.com/api/login",
                data={"user_id": user_id, "password": password},
            )
            login_data = login_resp.json()

            if login_data.get("status") != "success":
                return {
                    "status": "error",
                    "message": f"Login failed: {login_data.get('message', 'Unknown error')}",
                }

            request_id = login_data["data"]["request_id"]

            # Step 3: Submit TOTP
            totp = pyotp.TOTP(totp_secret)
            twofa_resp = session.post(
                "https://kite.zerodha.com/api/twofa",
                data={"user_id": user_id, "request_id": request_id, "twofa_value": totp.now()},
            )
            twofa_data = twofa_resp.json()

            if twofa_data.get("status") != "success":
                return {
                    "status": "error",
                    "message": f"2FA failed: {twofa_data.get('message', 'Unknown error')}",
                }

            # Step 4: Hit the original login URL — now authenticated, redirect chain starts
            # Follow redirects manually to extract the request_token from the final redirect
            current_url = redirect_after_login
            request_token = None
            max_redirects = 5
            
            for _ in range(max_redirects):
                resp = session.get(url=current_url, allow_redirects=False)
                location = resp.headers.get("Location", "")
                if not location:
                    break
                
                # Check if this redirect URL contains the request_token
                if "request_token" in location:
                    from urllib.parse import urlparse, parse_qs
                    parsed = urlparse(location)
                    params = parse_qs(parsed.query)
                    request_token = params.get("request_token", [None])[0]
                    break
                
                current_url = location
            
            if not request_token:
                return {
                    "status": "error",
                    "message": f"Could not extract request_token after {max_redirects} redirects",
                }

            if not request_token:
                return {
                    "status": "error",
                    "message": "Could not extract request_token from redirect URL",
                }

            # Step 5: Exchange request_token for access_token
            return self.generate_session(request_token)

        except Exception as e:
            logger.error("auto_login failed: %s", e)
            return {"status": "error", "message": str(e)}

    def get_login_url(self) -> str:
        """Generate the login URL for OAuth flow.
        
        Returns:
            URL for user to authenticate
        """
        kite = self._init_kite()
        return kite.login_url()
    
    def generate_session(self, request_token: str) -> Dict[str, Any]:
        """Generate a session by exchanging request token for access token.
        
        Args:
            request_token: Token received from OAuth callback
            
        Returns:
            Dictionary with user information and access token
        """
        kite = self._init_kite()
        try:
            data = kite.generate_session(request_token, api_secret=self.api_secret)
            self.access_token = data["access_token"]
            self._save_token(self.access_token)
            return {
                "status": "success",
                "access_token": self.access_token,
                "user_id": data.get("user_id"),
                "user_name": data.get("user_name"),
                "email": data.get("email")
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _ensure_auth(self) -> KiteConnect:
        """Ensure we have a valid authentication session.
        
        Tries cached token first. If invalid, auto-login with TOTP.
        Token is validated with a lightweight profile() API call.
        """
        kite = self._init_kite()
        
        # Try cached token
        if self.access_token is None:
            self.access_token = self._load_token()
        
        if self.access_token:
            kite.set_access_token(self.access_token)
            # Validate: make a lightweight call
            try:
                kite.profile()
                return kite
            except Exception:
                logger.debug("Cached token invalid — re-authenticating")
                self.access_token = None  # Clear invalid token
        
        # No valid token — auto-login
        result = self.auto_login()
        if result.get("status") != "success":
            raise PermissionError(
                f"Auto-login failed: {result.get('message', 'Unknown error')}"
            )
        
        self.access_token = result["access_token"]
        kite.set_access_token(self.access_token)
        return kite
    
    # ================== MARKET DATA ENDPOINTS ==================
    
    async def get_quotes(self, instruments: list) -> Dict[str, Any]:
        """Get real-time quotes for instruments.
        
        Args:
            instruments: List of instrument tokens or 'exchange:tradingsymbol'
                        e.g., ["NSE:RELIANCE", "BSE:INFY", 408065]
                        
        Returns:
            Dictionary with instrument tokens as keys and quote data as values
        """
        try:
            kite = self._ensure_auth()
            result = kite.quote(instruments)
            return {"status": "success", "data": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def get_ohlc(self, instruments: list) -> Dict[str, Any]:
        """Get OHLC (Open, High, Low, Close) data for instruments.
        
        Args:
            instruments: List of instrument tokens or 'exchange:tradingsymbol'
            
        Returns:
            Dictionary with OHLC data for each instrument
        """
        try:
            kite = self._ensure_auth()
            result = kite.ohlc(instruments)
            return {"status": "success", "data": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def get_instruments(self, exchange: Optional[str] = None) -> list:
        """Get list of all instruments or instruments for a specific exchange.
        
        Args:
            exchange: Exchange filter (NSE, BSE, NFO, BFO, CDS, MCX, etc.)
                     If None, returns all instruments.
                     
        Returns:
            List of instrument dictionaries with details
        """
        try:
            kite = self._init_kite()
            instruments = kite.instruments(exchange)
            return {"status": "success", "data": instruments}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def get_historical_data(
        self,
        instrument_token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str = "day"
    ) -> list:
        """Get historical data for an instrument.
        
        Args:
            instrument_token: The instrument token (instrument_id)
            from_date: Start date for historical data
            to_date: End date for historical data
            interval: Data interval - 
                     'day', '30minute', 'hour', 'week', 'month'
                     Note: For NFO instruments:
                     - day: 2 years max
                     - 30minute: 1 year max  
                     - hour: 1 year max
                     - week: 10 years max
                     - month: 10 years max
                     
        Returns:
            List of candle data [timestamp, open, high, low, close, volume]
        """
        try:
            kite = self._ensure_auth()
            data = kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            return {"status": "success", "data": data}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    # ================== PORTFOLIO ENDPOINTS ==================
    
    async def get_positions(self) -> Dict[str, Any]:
        """Get current positions (both day and net).
        
        Returns:
            Dictionary with day and net positions
        """
        try:
            kite = self._ensure_auth()
            result = kite.positions()
            return {"status": "success", "data": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def get_holdings(self) -> list:
        """Get end-of-day holdings.
        
        Returns:
            List of holding dictionaries with quantity, average_price, etc.
        """
        try:
            kite = self._ensure_auth()
            result = kite.holdings()
            return {"status": "success", "data": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def get_margins(self) -> Dict[str, Any]:
        """Get available margins and balances.
        
        Returns:
            Dictionary with various margin components:
            - equity: Equity margins
            - commodity: Commodity margins  
            - currency: Currency margins
        """
        try:
            kite = self._ensure_auth()
            result = kite.margins()
            return {"status": "success", "data": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    # ================== ORDER MANAGEMENT ENDPOINTS ==================
    
    async def place_order(
        self,
        variety: str = "regular",
        exchange: str = "NSE",
        tradingsymbol: str = "",
        transaction_type: str = "BUY",
        quantity: int = 0,
        product: str = "MIS",
        order_type: str = "MARKET",
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        validity: str = "DAY",
        disclosed_quantity: Optional[int] = None,
        tag: str = ""
    ) -> Dict[str, Any]:
        """Place a new order.
        
        Args:
            variety: Order variety - 'regular', 'bo', 'co', 'amo'
            exchange: Exchange - 'NSE', 'BSE', 'NFO', 'BFO', 'CDS', 'MCX'
            tradingsymbol: Trading symbol (e.g., 'RELIANCE', 'INFY')
            transaction_type: 'BUY' or 'SELL'
            quantity: Order quantity
            product: Product type - 'MIS', 'NRML', 'CNC', 'BO', 'CO'
            order_type: Order type - 'MARKET', 'LIMIT', 'SL', 'SL-M'
            price: Limit price (required for LIMIT, SL, SL-M)
            trigger_price: Trigger price for stop loss orders
            validity: Order validity - 'DAY', 'IOC', 'TWO_DAY', 'GTD', 'GTC'
            disclosed_quantity: Quantity to be disclosed in order book
            tag: Custom tag for tracking orders
            
        Returns:
            Dictionary with order_id or error message
        """
        try:
            kite = self._ensure_auth()
            order_id = kite.place_order(
                variety=variety,
                exchange=exchange,
                tradingsymbol=tradingsymbol,
                transaction_type=transaction_type,
                quantity=quantity,
                product=product,
                order_type=order_type,
                price=price,
                trigger_price=trigger_price,
                validity=validity,
                disclosed_quantity=disclosed_quantity,
                tag=tag
            )
            return {"status": "success", "order_id": order_id}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def modify_order(
        self,
        variety: str,
        order_id: str,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        order_type: Optional[str] = None,
        trigger_price: Optional[float] = None,
        validity: Optional[str] = None,
        disclosed_quantity: Optional[int] = None
    ) -> Dict[str, Any]:
        """Modify an existing order.
        
        Args:
            variety: Order variety
            order_id: Order ID to modify
            quantity: New quantity (optional)
            price: New price (optional)
            order_type: New order type (optional)
            trigger_price: New trigger price (optional)
            validity: New validity (optional)
            disclosed_quantity: New disclosed quantity (optional)
            
        Returns:
            Dictionary with order_id or error message
        """
        try:
            kite = self._ensure_auth()
            modified_order_id = kite.modify_order(
                variety=variety,
                order_id=order_id,
                quantity=quantity,
                price=price,
                order_type=order_type,
                trigger_price=trigger_price,
                validity=validity,
                disclosed_quantity=disclosed_quantity
            )
            return {"status": "success", "order_id": modified_order_id}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def cancel_order(self, variety: str, order_id: str) -> Dict[str, Any]:
        """Cancel an existing order.
        
        Args:
            variety: Order variety
            order_id: Order ID to cancel
            
        Returns:
            Dictionary with order_id or error message
        """
        try:
            kite = self._ensure_auth()
            cancelled_order_id = kite.cancel_order(variety=variety, order_id=order_id)
            return {"status": "success", "order_id": cancelled_order_id}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def get_orders(self) -> list:
        """Get all orders in the order book.
        
        Returns:
            List of order dictionaries with status, quantity, price, etc.
        """
        try:
            kite = self._ensure_auth()
            result = kite.orders()
            return {"status": "success", "data": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def get_order_history(self, order_id: str) -> list:
        """Get history of a specific order.
        
        Args:
            order_id: Order ID to get history for
            
        Returns:
            List of order update dictionaries showing status changes
        """
        try:
            kite = self._ensure_auth()
            result = kite.order_history(order_id=order_id)
            return {"status": "success", "data": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    # ================== UTILITY METHODS ==================
    
    async def profile(self) -> Dict[str, Any]:
        """Get user profile details.
        
        Returns:
            Dictionary with user_id, user_name, email, broker, exchanges, etc.
        """
        try:
            kite = self._ensure_auth()
            result = kite.profile()
            return {"status": "success", "data": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def trades(self) -> Dict[str, Any]:
        """Get all executed trades for the day.
        
        Returns:
            Dictionary with status and list of trade objects.
        """
        try:
            kite = self._ensure_auth()
            result = kite.trades()
            return {"status": "success", "data": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def position_convert(
        self,
        tradingsymbol: str,
        exchange: str = "NSE",
        transaction_type: str = "BUY",
        position_type: str = "overnight",
        quantity: int = 1,
        old_product: str = "MIS",
        new_product: str = "CNC",
    ) -> Dict[str, Any]:
        """Convert an open position from one product type to another.
        
        Example: Convert intraday MIS to delivery CNC.
        
        Args:
            tradingsymbol: Trading symbol
            exchange: Exchange (NSE, BSE, NFO, etc.)
            transaction_type: BUY or SELL
            position_type: "day" or "overnight"
            quantity: Quantity to convert
            old_product: Current product type (MIS, CNC, NRML)
            new_product: New product type to convert to
            
        Returns:
            Dictionary with status and message
        """
        try:
            kite = self._ensure_auth()
            result = kite.convert_position(
                tradingsymbol=tradingsymbol,
                exchange=exchange,
                transaction_type=transaction_type,
                position_type=position_type,
                quantity=quantity,
                old_product=old_product,
                new_product=new_product,
            )
            return {"status": "success", "data": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def get_full_tokens(self) -> Dict[str, Any]:
        """Get complete instrument information including tokens.
        
        Returns:
            Dictionary mapping instrument identifiers to tokens
        """
        try:
            kite = self._init_kite()
            instruments = kite.instruments()
            
            # Build mapping dictionaries
            nse_map = {i["tradingsymbol"]: i["instrument_token"] for i in instruments 
                      if i["exchange"] == "NSE"}
            bse_map = {i["tradingsymbol"]: i["instrument_token"] for i in instruments 
                      if i["exchange"] == "BSE"}
            
            return {
                "status": "success",
                "data": {
                    "all_instruments": instruments,
                    "nse_map": nse_map,
                    "bse_map": bse_map
                }
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def get_ltp(self, instruments: list[str]) -> dict[str, Any]:
        """Get Last Traded Price for instruments.

        LTP is a lightweight quote endpoint that supports up to 1000 instruments
        per call and has a higher rate limit than full quotes.

        Args:
            instruments: List of instrument tokens or trading symbols
                         e.g., ["NSE:RELIANCE", "NSE:TCS"] or instrument tokens

        Returns:
            Dictionary with status and data containing ltp values
        """
        try:
            kite = self._ensure_auth()
            ltp_data = kite.ltp(instruments)
            return {"status": "success", "data": ltp_data}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def place_gtt(
        self,
        trigger_type: str,
        tradingsymbol: str,
        exchange: str,
        trigger_values: list[float],
        last_price: float,
        orders: list[dict],
    ) -> dict[str, Any]:
        """Place a GTT (Good Till Triggered) order.

        GTT orders automatically place when a trigger condition is met.
        Useful for stop-loss and target orders that survive overnight.

        Args:
            trigger_type: "single" or "two-leg" (OCO)
            tradingsymbol: Trading symbol, e.g., "RELIANCE"
            exchange: Exchange, e.g., "NSE"
            trigger_values: Trigger price values [trigger_price] or [trigger1, trigger2]
            last_price: Current last traded price
            orders: List of order dicts [{"transaction_type": "SELL", "quantity": 1,
                    "product": "CNC", "order_type": "LIMIT", "price": 2490}]

        Returns:
            Dictionary with status, trigger_id, and message
        """
        try:
            kite = self._ensure_auth()
            result = kite.place_gtt(
                trigger_type=trigger_type,
                tradingsymbol=tradingsymbol,
                exchange=exchange,
                trigger_values=trigger_values,
                last_price=last_price,
                orders=orders,
            )
            return {"status": "success", "trigger_id": result.get("trigger_id"), "data": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def get_gtts(self) -> dict[str, Any]:
        """Get all active GTT orders.

        Returns:
            Dictionary with status and list of active GTT orders
        """
        try:
            kite = self._ensure_auth()
            gtts = kite.get_gtts()
            return {"status": "success", "data": gtts}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def get_gtt(self, trigger_id: int) -> dict[str, Any]:
        """Get a specific GTT order by trigger ID.

        Args:
            trigger_id: The GTT trigger ID

        Returns:
            Dictionary with status and GTT order details
        """
        try:
            kite = self._ensure_auth()
            gtt = kite.get_gtt(trigger_id)
            return {"status": "success", "data": gtt}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def modify_gtt(
        self,
        trigger_id: int,
        trigger_type: str,
        tradingsymbol: str,
        exchange: str,
        trigger_values: list[float],
        last_price: float,
        orders: list[dict],
    ) -> dict[str, Any]:
        """Modify an existing GTT order.

        Args:
            trigger_id: The GTT trigger ID to modify
            trigger_type: "single" or "two-leg"
            tradingsymbol: Trading symbol
            exchange: Exchange
            trigger_values: Updated trigger values
            last_price: Current last traded price
            orders: Updated order list

        Returns:
            Dictionary with status and message
        """
        try:
            kite = self._ensure_auth()
            result = kite.modify_gtt(
                trigger_id=trigger_id,
                trigger_type=trigger_type,
                tradingsymbol=tradingsymbol,
                exchange=exchange,
                trigger_values=trigger_values,
                last_price=last_price,
                orders=orders,
            )
            return {"status": "success", "data": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def delete_gtt(self, trigger_id: int) -> dict[str, Any]:
        """Delete/cancel a GTT order.

        Args:
            trigger_id: The GTT trigger ID to cancel

        Returns:
            Dictionary with status and message
        """
        try:
            kite = self._ensure_auth()
            result = kite.delete_gtt(trigger_id)
            return {"status": "success", "data": result}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def disconnect(self) -> None:
        """Disconnect from Kite API and clear session."""
        self.kite = None
        self.access_token = None


# Convenience function for common use
def create_kite_tools(api_key: Optional[str] = None, api_secret: Optional[str] = None) -> KiteTools:
    """Create a KiteTools instance.
    
    Args:
        api_key: Zerodha API key (optional, reads from KITE_API_KEY env var)
        api_secret: Zerodha API secret (optional, reads from KITE_API_SECRET env var)
        
    Returns:
        KiteTools instance
    """
    return KiteTools(api_key=api_key, api_secret=api_secret)
