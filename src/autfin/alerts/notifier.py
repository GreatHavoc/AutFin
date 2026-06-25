"""
Notification system for AutFin alerts.

Supports multiple backends:
  - Console (always available, Rich console output)
  - Desktop (requires plyer — optional)
  - Pushover (requires PUSHOVER_USER_KEY and PUSHOVER_API_TOKEN)
  - Email (requires SMTP_* settings)

Alert types:
  - Price: significant intraday price moves
  - StopLoss: position hit stop-loss threshold
  - Trade: trade executed or recommended
  - Summary: daily portfolio summary
  - Error: critical system errors (circuit breaker open, etc.)
"""
from __future__ import annotations

import asyncio
import logging
import smtplib
import time
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert types & severity
# ---------------------------------------------------------------------------

class AlertType(str, Enum):
    """Categories of alerts."""
    PRICE = "price"
    STOP_LOSS = "stop_loss"
    TRADE = "trade"
    SUMMARY = "summary"
    ERROR = "error"
    INFO = "info"


class Severity(str, Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Alert data
# ---------------------------------------------------------------------------

@dataclass
class Alert:
    """A single alert to be delivered."""
    alert_type: AlertType
    title: str
    message: str
    severity: Severity = Severity.MEDIUM
    symbol: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def emoji(self) -> str:
        """Emoji prefix by type for console readability."""
        return {
            AlertType.PRICE: "📈",
            AlertType.STOP_LOSS: "🛑",
            AlertType.TRADE: "💹",
            AlertType.SUMMARY: "📊",
            AlertType.ERROR: "🚨",
            AlertType.INFO: "ℹ️",
        }.get(self.alert_type, "🔔")

    @property
    def severity_label(self) -> str:
        return self.severity.value.upper()


# ---------------------------------------------------------------------------
# Notification backends (base + implementations)
# ---------------------------------------------------------------------------

class BaseNotifier:
    """Abstract base for notification backends."""

    name: str = "base"

    async def send(self, alert: Alert) -> bool:
        raise NotImplementedError

    def is_available(self) -> bool:
        return False


class ConsoleNotifier(BaseNotifier):
    """Always-available console notifier using logger."""

    name = "console"

    def __init__(self) -> None:
        self._alert_count = 0

    async def send(self, alert: Alert) -> bool:
        self._alert_count += 1
        prefix = f"{alert.emoji} [{alert.severity_label}] {alert.alert_type.value.upper()}"
        symbol_part = f" ({alert.symbol})" if alert.symbol else ""
        full_msg = f"{prefix}{symbol_part}: {alert.title}\n  {alert.message}"

        if alert.severity in (Severity.HIGH, Severity.CRITICAL):
            logger.warning(full_msg)
        else:
            logger.info(full_msg)
        return True

    def is_available(self) -> bool:
        return True

    @property
    def alert_count(self) -> int:
        return self._alert_count


class DesktopNotifier(BaseNotifier):
    """Desktop notifications via plyer (optional dependency)."""

    name = "desktop"

    def __init__(self) -> None:
        self._plyer_available: bool | None = None

    def is_available(self) -> bool:
        if self._plyer_available is None:
            try:
                import plyer  # noqa: F401
                self._plyer_available = True
            except ImportError:
                self._plyer_available = False
        return self._plyer_available

    async def send(self, alert: Alert) -> bool:
        if not self.is_available():
            return False
        try:
            from plyer import notification
            notification.notify(
                title=f"{alert.emoji} {alert.title}",
                message=alert.message[:256],  # plyer truncates long messages
                timeout=10,
            )
            return True
        except Exception as exc:
            logger.debug("Desktop notification failed: %s", exc)
            return False


class PushoverNotifier(BaseNotifier):
    """Pushover notifications (requires user key + API token)."""

    name = "pushover"

    def __init__(self, user_key: str = "", api_token: str = "") -> None:
        self._user_key = user_key
        self._api_token = api_token

    def is_available(self) -> bool:
        return bool(self._user_key and self._api_token)

    async def send(self, alert: Alert) -> bool:
        if not self.is_available():
            return False
        try:
            import aiohttp
            data = {
                "token": self._api_token,
                "user": self._user_key,
                "title": f"{alert.emoji} {alert.title}",
                "message": alert.message,
                "priority": {
                    Severity.LOW: -2,
                    Severity.MEDIUM: -1,
                    Severity.HIGH: 1,
                    Severity.CRITICAL: 2,
                }.get(alert.severity, 0),
                "timestamp": int(time.time()),
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.pushover.net/1/messages.json",
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        return True
                    logger.warning("Pushover returned %d", resp.status)
                    return False
        except Exception as exc:
            logger.debug("Pushover notification failed: %s", exc)
            return False


class EmailNotifier(BaseNotifier):
    """Email notifications via SMTP."""

    name = "email"

    def __init__(
        self,
        host: str = "",
        port: int = 587,
        username: str = "",
        password: str = "",
        from_addr: str = "",
        to_addr: str = "",
    ) -> None:
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._from = from_addr
        self._to = to_addr

    def is_available(self) -> bool:
        return bool(self._host and self._from and self._to)

    async def send(self, alert: Alert) -> bool:
        if not self.is_available():
            return False
        try:
            msg = MIMEMultipart()
            msg["From"] = self._from
            msg["To"] = self._to
            msg["Subject"] = f"[AutFin {alert.severity_label}] {alert.title}"

            from html import escape as html_escape
            safe_title = html_escape(str(alert.title))
            safe_message = html_escape(str(alert.message))
            safe_symbol = html_escape(str(alert.symbol)) if alert.symbol else ""
            safe_emoji = html_escape(str(alert.emoji))

            html = f"""\
<html>
<body style="font-family: Arial, sans-serif;">
<h2>{safe_emoji} {safe_title}</h2>
<p><strong>Type:</strong> {alert.alert_type.value} | <strong>Severity:</strong> {alert.severity_label}</p>
{'<p><strong>Symbol:</strong> ' + safe_symbol + '</p>' if safe_symbol else ''}
<p>{safe_message}</p>
<hr>
<p style="color: #666; font-size: 12px;">AutFin Alert System | {alert.timestamp}</p>
</body>
</html>
"""
            msg.attach(MIMEText(html, "html"))

            # Run synchronous SMTP in executor to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None, self._send_smtp, msg
            )
            return True
        except Exception as exc:
            logger.debug("Email notification failed: %s", exc)
            return False

    def _send_smtp(self, msg: MIMEMultipart) -> None:
        with smtplib.SMTP(self._host, self._port, timeout=15) as server:
            server.starttls()
            if self._username and self._password:
                server.login(self._username, self._password)
            server.send_message(msg)


# ---------------------------------------------------------------------------
# AlertManager — the main interface
# ---------------------------------------------------------------------------

class AlertManager:
    """
    Central alert dispatcher.

    Usage::

        manager = AlertManager(settings)
        await manager.send(AlertType.TRADE, "Buy executed", "RELIANCE x 5 @ ₹2,850")
        await manager.send_price_alert("TCS", 3.5, "price_up")
    """

    def __init__(self, settings: Any = None) -> None:
        from src.autfin.config.settings import AutFinSettings
        if settings is None:
            settings = AutFinSettings()
        self._settings = settings
        self._backends: list[BaseNotifier] = []
        self._alert_log: list[Alert] = []
        self._init_backends()

    def _init_backends(self) -> None:
        """Initialize enabled backends based on settings."""
        s = self._settings
        if not s.alerts_enabled:
            return

        # Console is always first and always available
        if s.alert_console:
            self._backends.append(ConsoleNotifier())

        # Desktop (optional)
        if s.alert_desktop:
            self._backends.append(DesktopNotifier())

        # Pushover
        if s.alert_pushover_enabled:
            pushover = PushoverNotifier(
                user_key=getattr(s, "pushover_user_key", ""),
                api_token=getattr(s, "pushover_api_token", ""),
            )
            self._backends.append(pushover)

        # Email
        if s.alert_email_enabled:
            email = EmailNotifier(
                host=s.smtp_host,
                port=s.smtp_port,
                username=s.smtp_username,
                password=s.smtp_password,
                from_addr=s.smtp_from,
                to_addr=s.smtp_to,
            )
            self._backends.append(email)

    @property
    def backends(self) -> list[BaseNotifier]:
        return list(self._backends)

    @property
    def alert_log(self) -> list[Alert]:
        return list(self._alert_log)

    async def send(
        self,
        alert_type: AlertType,
        title: str,
        message: str,
        severity: Severity = Severity.MEDIUM,
        symbol: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> int:
        """
        Send an alert through all available backends.

        Returns the number of backends that successfully delivered the alert.
        """
        alert = Alert(
            alert_type=alert_type,
            title=title,
            message=message,
            severity=severity,
            symbol=symbol,
            data=data or {},
        )
        self._alert_log.append(alert)

        sent_count = 0
        for backend in self._backends:
            if backend.is_available():
                try:
                    ok = await backend.send(alert)
                    if ok:
                        sent_count += 1
                except Exception as exc:
                    logger.debug("Backend %s failed: %s", backend.name, exc)

        logger.info(
            "Alert '%s' sent to %d/%d backends",
            title, sent_count, len(self._backends),
        )
        return sent_count

    # ── Convenience methods for common alert types ──

    async def send_price_alert(
        self, symbol: str, price_pct: float, direction: str = "price"
    ) -> int:
        """Send a price movement alert."""
        direction_label = "up" if price_pct > 0 else "down"
        return await self.send(
            alert_type=AlertType.PRICE,
            title=f"{symbol} moved {price_pct:+.2f}%",
            message=f"{symbol} price {direction_label} {abs(price_pct):.2f}% — consider reviewing position.",
            severity=Severity.HIGH if abs(price_pct) > 5 else Severity.MEDIUM,
            symbol=symbol,
            data={"price_pct": price_pct},
        )

    async def send_stop_loss_alert(
        self, symbol: str, current_price: float, stop_price: float, loss_pct: float
    ) -> int:
        """Send a stop-loss alert."""
        return await self.send(
            alert_type=AlertType.STOP_LOSS,
            title=f"{symbol} hit stop-loss",
            message=(
                f"{symbol} at ₹{current_price:,.2f} — stop-loss at ₹{stop_price:,.2f} "
                f"(loss: {loss_pct:.1f}%)"
            ),
            severity=Severity.HIGH,
            symbol=symbol,
            data={
                "current_price": current_price,
                "stop_price": stop_price,
                "loss_pct": loss_pct,
            },
        )

    async def send_trade_alert(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        status: str,
    ) -> int:
        """Send a trade execution alert."""
        emoji_label = "✅" if status == "executed" else "📝"
        return await self.send(
            alert_type=AlertType.TRADE,
            title=f"{emoji_label} {action.upper()} {symbol}",
            message=(
                f"{action.upper()} {quantity} shares of {symbol} @ ₹{price:,.2f}\n"
                f"Status: {status}"
            ),
            severity=Severity.MEDIUM,
            symbol=symbol,
            data={
                "action": action,
                "quantity": quantity,
                "price": price,
                "status": status,
            },
        )

    async def send_daily_summary(self, summary: dict[str, Any]) -> int:
        """Send a daily portfolio summary alert."""
        lines = [
            f"Cash: ₹{summary.get('cash', 0):,.2f}",
            f"Holdings: ₹{summary.get('holdings_value', 0):,.2f}",
            f"Day P&L: ₹{summary.get('day_pnl', 0):,.2f}",
        ]
        if summary.get("trades_today"):
            lines.append(f"Trades today: {summary['trades_today']}")
        if summary.get("alerts_fired"):
            lines.append(f"Alerts fired: {summary['alerts_fired']}")
        return await self.send(
            alert_type=AlertType.SUMMARY,
            title="Daily Portfolio Summary",
            message="\n".join(lines),
            severity=Severity.LOW,
            data=summary,
        )

    async def send_error_alert(self, error_msg: str, context: str = "") -> int:
        """Send a critical error alert."""
        return await self.send(
            alert_type=AlertType.ERROR,
            title="System Error",
            message=f"{error_msg}\nContext: {context}" if context else error_msg,
            severity=Severity.CRITICAL,
            data={"error": error_msg, "context": context},
        )

    async def send_info(self, title: str, message: str) -> int:
        """Send a low-severity informational alert."""
        return await self.send(
            alert_type=AlertType.INFO,
            title=title,
            message=message,
            severity=Severity.LOW,
        )

    def clear_log(self) -> None:
        """Clear the in-memory alert log."""
        self._alert_log.clear()


# ---------------------------------------------------------------------------
# Module-level singleton (lazy init)
# ---------------------------------------------------------------------------

_default_manager: AlertManager | None = None


def get_alert_manager(settings: Any = None) -> AlertManager:
    """Get or create the default AlertManager singleton."""
    global _default_manager
    if _default_manager is None:
        _default_manager = AlertManager(settings)
    return _default_manager


def reset_alert_manager() -> None:
    """Reset the singleton (useful for testing)."""
    global _default_manager
    _default_manager = None
