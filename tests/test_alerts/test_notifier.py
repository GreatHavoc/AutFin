"""
Tests for the AutFin alert notification system.
"""
from __future__ import annotations

import asyncio
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from src.autfin.alerts.notifier import (
    Alert,
    AlertManager,
    AlertType,
    BaseNotifier,
    ConsoleNotifier,
    DesktopNotifier,
    EmailNotifier,
    PushoverNotifier,
    Severity,
    get_alert_manager,
    reset_alert_manager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_alert(**overrides) -> Alert:
    defaults = dict(
        alert_type=AlertType.TRADE,
        title="Test Alert",
        message="This is a test",
        severity=Severity.MEDIUM,
    )
    defaults.update(overrides)
    return Alert(**defaults)


def _make_settings(**overrides):
    """Create a mock AutFinSettings with sensible defaults for tests."""
    s = MagicMock()
    s.alerts_enabled = True
    s.alert_console = True
    s.alert_desktop = False
    s.alert_pushover_enabled = False
    s.alert_email_enabled = False
    s.alert_price_threshold_pct = 2.0
    s.alert_stop_loss_enabled = True
    s.alert_trade_enabled = True
    s.alert_daily_summary = True
    s.alert_error_enabled = True
    s.smtp_host = ""
    s.smtp_port = 587
    s.smtp_username = ""
    s.smtp_password = ""
    s.smtp_from = ""
    s.smtp_to = ""
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------

class TestAlert:
    def test_default_values(self):
        a = Alert(alert_type=AlertType.PRICE, title="T", message="M")
        assert a.severity == Severity.MEDIUM
        assert a.symbol is None
        assert a.data == {}
        assert a.timestamp  # non-empty

    def test_emoji_map(self):
        assert Alert(alert_type=AlertType.PRICE, title="", message="").emoji == "📈"
        assert Alert(alert_type=AlertType.STOP_LOSS, title="", message="").emoji == "🛑"
        assert Alert(alert_type=AlertType.TRADE, title="", message="").emoji == "💹"
        assert Alert(alert_type=AlertType.SUMMARY, title="", message="").emoji == "📊"
        assert Alert(alert_type=AlertType.ERROR, title="", message="").emoji == "🚨"
        assert Alert(alert_type=AlertType.INFO, title="", message="").emoji == "ℹ️"

    def test_severity_label(self):
        assert Alert(alert_type=AlertType.INFO, title="", message="", severity=Severity.LOW).severity_label == "LOW"
        assert Alert(alert_type=AlertType.INFO, title="", message="", severity=Severity.CRITICAL).severity_label == "CRITICAL"


# ---------------------------------------------------------------------------
# ConsoleNotifier
# ---------------------------------------------------------------------------

class TestConsoleNotifier:
    def test_is_available(self):
        assert ConsoleNotifier().is_available() is True

    @pytest.mark.asyncio
    async def test_send_returns_true(self):
        n = ConsoleNotifier()
        result = await n.send(_make_alert())
        assert result is True
        assert n.alert_count == 1

    @pytest.mark.asyncio
    async def test_send_increments_count(self):
        n = ConsoleNotifier()
        await n.send(_make_alert())
        await n.send(_make_alert())
        assert n.alert_count == 2

    @pytest.mark.asyncio
    async def test_send_logs_warning_for_high_severity(self, caplog):
        n = ConsoleNotifier()
        with caplog.at_level(logging.WARNING):
            await n.send(_make_alert(severity=Severity.HIGH))
        assert "HIGH" in caplog.text

    @pytest.mark.asyncio
    async def test_send_logs_info_for_medium_severity(self, caplog):
        n = ConsoleNotifier()
        with caplog.at_level(logging.INFO):
            await n.send(_make_alert(severity=Severity.MEDIUM))
        assert "MEDIUM" in caplog.text

    @pytest.mark.asyncio
    async def test_send_with_symbol(self, caplog):
        n = ConsoleNotifier()
        with caplog.at_level(logging.INFO):
            await n.send(_make_alert(symbol="RELIANCE"))
        assert "RELIANCE" in caplog.text


# ---------------------------------------------------------------------------
# DesktopNotifier
# ---------------------------------------------------------------------------

class TestDesktopNotifier:
    def test_not_available_without_plyer(self):
        with patch.dict("sys.modules", {"plyer": None}):
            n = DesktopNotifier()
            n._plyer_available = None
            assert n.is_available() is False

    @pytest.mark.asyncio
    async def test_send_returns_false_when_unavailable(self):
        n = DesktopNotifier()
        n._plyer_available = False
        assert await n.send(_make_alert()) is False

    @pytest.mark.asyncio
    async def test_send_success_with_plyer(self):
        mock_plyer = MagicMock()
        mock_notification = MagicMock()
        mock_plyer.notification = mock_notification

        with patch.dict("sys.modules", {"plyer": mock_plyer}):
            n = DesktopNotifier()
            n._plyer_available = None
            result = await n.send(_make_alert())
            assert result is True
            mock_notification.notify.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_handles_plyer_exception(self):
        mock_plyer = MagicMock()
        mock_plyer.notification.notify.side_effect = RuntimeError("display error")

        with patch.dict("sys.modules", {"plyer": mock_plyer}):
            n = DesktopNotifier()
            n._plyer_available = None
            result = await n.send(_make_alert())
            assert result is False


# ---------------------------------------------------------------------------
# PushoverNotifier
# ---------------------------------------------------------------------------

class TestPushoverNotifier:
    def test_not_available_without_keys(self):
        assert PushoverNotifier("", "").is_available() is False
        assert PushoverNotifier("user", "").is_available() is False
        assert PushoverNotifier("", "token").is_available() is False

    def test_available_with_keys(self):
        assert PushoverNotifier("user", "token").is_available() is True

    @pytest.mark.asyncio
    async def test_send_returns_false_when_unavailable(self):
        n = PushoverNotifier()
        assert await n.send(_make_alert()) is False

    @pytest.mark.asyncio
    async def test_send_success(self):
        mock_response = MagicMock()
        mock_response.status = 200

        mock_post_cm = AsyncMock()
        mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post.return_value = mock_post_cm

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession.return_value = mock_session_cm
        mock_aiohttp.ClientTimeout.return_value = MagicMock()

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            n = PushoverNotifier("user_key", "api_token")
            result = await n.send(_make_alert())
            assert result is True

    @pytest.mark.asyncio
    async def test_send_handles_http_error(self):
        mock_response = MagicMock()
        mock_response.status = 400

        mock_post_cm = AsyncMock()
        mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_cm.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post.return_value = mock_post_cm

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession.return_value = mock_session_cm
        mock_aiohttp.ClientTimeout.return_value = MagicMock()

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            n = PushoverNotifier("user_key", "api_token")
            result = await n.send(_make_alert())
            assert result is False

    @pytest.mark.asyncio
    async def test_send_handles_network_error(self):
        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__ = AsyncMock(side_effect=ConnectionError("network down"))
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession.return_value = mock_session_cm
        mock_aiohttp.ClientTimeout.return_value = MagicMock()

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            n = PushoverNotifier("user_key", "api_token")
            result = await n.send(_make_alert())
            assert result is False


# ---------------------------------------------------------------------------
# EmailNotifier
# ---------------------------------------------------------------------------

class TestEmailNotifier:
    def test_not_available_without_config(self):
        assert EmailNotifier().is_available() is False
        assert EmailNotifier(host="smtp.gmail.com").is_available() is False
        assert EmailNotifier(host="smtp.gmail.com", from_addr="a@b.com").is_available() is False

    def test_available_with_full_config(self):
        assert EmailNotifier(
            host="smtp.gmail.com", from_addr="a@b.com", to_addr="c@d.com"
        ).is_available() is True

    @pytest.mark.asyncio
    async def test_send_returns_false_when_unavailable(self):
        n = EmailNotifier()
        assert await n.send(_make_alert()) is False

    @pytest.mark.asyncio
    async def test_send_success(self):
        n = EmailNotifier(
            host="smtp.test.com",
            port=587,
            username="user",
            password="pass",
            from_addr="from@test.com",
            to_addr="to@test.com",
        )
        n._send_smtp = MagicMock()
        result = await n.send(_make_alert(symbol="RELIANCE"))
        assert result is True
        n._send_smtp.assert_called_once()
        call_args = n._send_smtp.call_args
        msg = call_args[0][0]
        assert isinstance(msg, MIMEMultipart)
        assert "[AutFin MEDIUM]" in msg["Subject"]
        body_parts = [p.get_payload(decode=True).decode() for p in msg.walk() if p.get_content_type() == "text/html"]
        assert any("RELIANCE" in part for part in body_parts)

    @pytest.mark.asyncio
    async def test_send_handles_smtp_exception(self):
        mock_smtp_class = MagicMock(side_effect=smtplib.SMTPException("connection refused"))

        with patch("src.autfin.alerts.notifier.smtplib.SMTP", mock_smtp_class):
            n = EmailNotifier(
                host="smtp.test.com",
                from_addr="from@test.com",
                to_addr="to@test.com",
            )
            result = await n.send(_make_alert())
            assert result is False


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------

class TestAlertManager:
    def setup_method(self):
        reset_alert_manager()

    def teardown_method(self):
        reset_alert_manager()

    def test_init_with_defaults(self):
        m = AlertManager(_make_settings())
        assert len(m.backends) == 1  # console only
        assert isinstance(m.backends[0], ConsoleNotifier)

    def test_init_disabled(self):
        m = AlertManager(_make_settings(alerts_enabled=False))
        assert len(m.backends) == 0

    def test_init_console_only(self):
        m = AlertManager(_make_settings(alert_console=True, alert_desktop=False))
        assert len(m.backends) == 1

    def test_init_with_desktop(self):
        with patch.dict("sys.modules", {"plyer": MagicMock()}):
            m = AlertManager(_make_settings(alert_desktop=True))
            assert len(m.backends) == 2

    def test_init_with_pushover(self):
        m = AlertManager(_make_settings(
            alert_pushover_enabled=True,
            pushover_user_key="u",
            pushover_api_token="t",
        ))
        assert any(isinstance(b, PushoverNotifier) for b in m.backends)

    def test_init_with_email(self):
        m = AlertManager(_make_settings(
            alert_email_enabled=True,
            smtp_host="smtp.test.com",
            smtp_from="a@b.com",
            smtp_to="c@d.com",
        ))
        assert any(isinstance(b, EmailNotifier) for b in m.backends)

    @pytest.mark.asyncio
    async def test_send_returns_count(self):
        m = AlertManager(_make_settings())
        count = await m.send(AlertType.TRADE, "T", "M")
        assert count == 1

    @pytest.mark.asyncio
    async def test_send_logs_to_alert_log(self):
        m = AlertManager(_make_settings())
        await m.send(AlertType.TRADE, "T", "M")
        assert len(m.alert_log) == 1
        assert m.alert_log[0].title == "T"

    @pytest.mark.asyncio
    async def test_send_multiple_backends(self):
        b1 = MagicMock(spec=BaseNotifier)
        b1.is_available.return_value = True
        b1.send = AsyncMock(return_value=True)
        b1.name = "mock1"

        b2 = MagicMock(spec=BaseNotifier)
        b2.is_available.return_value = True
        b2.send = AsyncMock(return_value=True)
        b2.name = "mock2"

        m = AlertManager(_make_settings())
        m._backends = [b1, b2]
        count = await m.send(AlertType.TRADE, "T", "M")
        assert count == 2

    @pytest.mark.asyncio
    async def test_send_partial_failure(self):
        b1 = MagicMock(spec=BaseNotifier)
        b1.is_available.return_value = True
        b1.send = AsyncMock(return_value=True)
        b1.name = "ok"

        b2 = MagicMock(spec=BaseNotifier)
        b2.is_available.return_value = True
        b2.send = AsyncMock(return_value=False)
        b2.name = "fail"

        m = AlertManager(_make_settings())
        m._backends = [b1, b2]
        count = await m.send(AlertType.TRADE, "T", "M")
        assert count == 1

    @pytest.mark.asyncio
    async def test_send_backend_exception_is_caught(self):
        b1 = MagicMock(spec=BaseNotifier)
        b1.is_available.return_value = True
        b1.send = AsyncMock(side_effect=RuntimeError("boom"))
        b1.name = "crash"

        m = AlertManager(_make_settings())
        m._backends = [b1]
        count = await m.send(AlertType.TRADE, "T", "M")
        assert count == 0

    def test_clear_log(self):
        m = AlertManager(_make_settings())
        m._alert_log.append(_make_alert())
        m.clear_log()
        assert len(m.alert_log) == 0

    @pytest.mark.asyncio
    async def test_send_price_alert(self):
        m = AlertManager(_make_settings())
        count = await m.send_price_alert("TCS", 3.5)
        assert count == 1
        assert m.alert_log[0].alert_type == AlertType.PRICE
        assert m.alert_log[0].symbol == "TCS"

    @pytest.mark.asyncio
    async def test_send_price_alert_high_severity(self):
        m = AlertManager(_make_settings())
        await m.send_price_alert("TCS", 6.0)
        assert m.alert_log[0].severity == Severity.HIGH

    @pytest.mark.asyncio
    async def test_send_stop_loss_alert(self):
        m = AlertManager(_make_settings())
        count = await m.send_stop_loss_alert("RELIANCE", 2700.0, 2800.0, -3.6)
        assert count == 1
        assert m.alert_log[0].alert_type == AlertType.STOP_LOSS
        assert m.alert_log[0].symbol == "RELIANCE"
        assert m.alert_log[0].severity == Severity.HIGH

    @pytest.mark.asyncio
    async def test_send_trade_alert_executed(self):
        m = AlertManager(_make_settings())
        count = await m.send_trade_alert("INFY", "buy", 10, 1500.0, "executed")
        assert count == 1
        assert m.alert_log[0].alert_type == AlertType.TRADE
        assert m.alert_log[0].symbol == "INFY"

    @pytest.mark.asyncio
    async def test_send_trade_alert_recommended(self):
        m = AlertManager(_make_settings())
        await m.send_trade_alert("WIPRO", "sell", 5, 450.0, "recommended")
        assert "RECOMMENDED" in m.alert_log[0].title or "recommended" in m.alert_log[0].message.lower() or m.alert_log[0].data["status"] == "recommended"

    @pytest.mark.asyncio
    async def test_send_daily_summary(self):
        m = AlertManager(_make_settings())
        summary = {"cash": 50000, "holdings_value": 120000, "day_pnl": 1500, "trades_today": 2}
        count = await m.send_daily_summary(summary)
        assert count == 1
        assert m.alert_log[0].alert_type == AlertType.SUMMARY
        assert "50,000" in m.alert_log[0].message

    @pytest.mark.asyncio
    async def test_send_error_alert(self):
        m = AlertManager(_make_settings())
        count = await m.send_error_alert("Circuit breaker opened", "Kite SDK")
        assert count == 1
        assert m.alert_log[0].alert_type == AlertType.ERROR
        assert m.alert_log[0].severity == Severity.CRITICAL

    @pytest.mark.asyncio
    async def test_send_info(self):
        m = AlertManager(_make_settings())
        count = await m.send_info("Hello", "World")
        assert count == 1
        assert m.alert_log[0].alert_type == AlertType.INFO
        assert m.alert_log[0].severity == Severity.LOW


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def setup_method(self):
        reset_alert_manager()

    def teardown_method(self):
        reset_alert_manager()

    def test_get_creates_instance(self):
        m = get_alert_manager(_make_settings())
        assert isinstance(m, AlertManager)

    def test_get_returns_same_instance(self):
        m1 = get_alert_manager(_make_settings())
        m2 = get_alert_manager()
        assert m1 is m2

    def test_reset_clears_singleton(self):
        m1 = get_alert_manager(_make_settings())
        reset_alert_manager()
        m2 = get_alert_manager(_make_settings())
        assert m1 is not m2
