"""
AutFinSettings - Configuration model using Pydantic
"""
from dotenv import load_dotenv
load_dotenv()  # Load .env into os.environ before model reads it

from pydantic import Field
from pydantic_settings import BaseSettings


class AutFinSettings(BaseSettings):
    """All configurable parameters for AutFin."""
    
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}
    
    # ── LLM ──────────────────────────────────────────
    store_mode: str = Field(
        default="persistent",
        description="Store mode: 'in-memory' or 'persistent' (SQLite)"
    )
    deep_model: str = Field(
        default="nvidia_nim/minimaxai/minimax-m3",
        description="Primary — MiniMax M3 via NVIDIA NIM (1M context, 61 tps, free)"
    )
    fast_model: str = Field(
        default="nvidia_nim/meta/llama-3.1-8b-instruct",
        description="Fast model — Llama 3.1 8B via NVIDIA NIM (912ms, 63 tps, free)"
    )
    nvidia_model: str = Field(
        default="nvidia_nim/meta/llama-3.3-70b-instruct",
        description="NVIDIA NIM — Llama 3.3 70B (free)"
    )
    nvidia_deep_model: str = Field(
        default="nvidia_nim/deepseek-ai/deepseek-v4-flash",
        description="NVIDIA NIM deep — DeepSeek V4 Flash 284B MoE (free, 1M context)"
    )
    model_temperature: float = Field(default=0.0)
    
    # ── Zerodha ──────────────────────────────────────
    kite_user_id: str = Field(
        default="",
        description="Zerodha user ID (e.g., ABC123)"
    )
    kite_password: str = Field(
        default="",
        description="Zerodha Kite login password"
    )
    kite_totp_secret: str = Field(
        default="",
        description="TOTP secret for 2FA (from Zerodha console → 2FA settings)"
    )
    
    # ── Capital Management ──────────────────────────
    min_trade_amount: float = Field(default=1000.0)
    cash_buffer_pct: float = Field(default=0.10)
    deployment_pacing: str = Field(default="spread")
    
    # ── Risk Parameters ─────────────────────────────
    max_single_position_pct: float = Field(default=0.30)
    max_sector_exposure_pct: float = Field(default=0.30)
    default_stop_loss_pct: float = Field(default=0.08)
    trailing_stop_loss_pct: float = Field(default=0.05)
    profit_target_pct: float = Field(default=0.15)
    portfolio_drawdown_halt_pct: float = Field(default=0.10)
    max_daily_trades: int = Field(default=5)
    min_market_cap_crore: float = Field(default=500.0)
    min_avg_volume_lakhs: float = Field(default=5.0)
    
    # ── Execution ───────────────────────────────────
    auto_execute_max_amount: float = Field(default=5000.0)
    execution_mode: str = Field(default="recommendation")
    
    # ── Memory ──────────────────────────────────────
    vector_store_type: str = Field(default="chromadb")
    trade_history_retention_days: int = Field(default=3650)
    research_freshness_days: int = Field(default=7)
    checkpoint_durability: str = Field(default="wal")
    
    # ── Scheduling ──────────────────────────────────
    daily_scan_time: str = Field(default="09:00")
    market_open: str = Field(default="09:15")
    market_close: str = Field(default="15:30")
    data_refresh_interval_min: int = Field(default=5)
    
    # ── Error Recovery ───────────────────────────────
    retry_max_attempts: int = Field(
        default=3,
        description="Max retries for transient errors (API timeouts)"
    )
    retry_initial_interval: float = Field(
        default=1.0,
        description="Initial retry interval in seconds (exponential backoff)"
    )
    retry_backoff_factor: float = Field(
        default=2.0,
        description="Multiplier applied to interval after each retry"
    )
    node_timeout_seconds: float = Field(
        default=300.0,
        description="Max time in seconds for a single node to complete (5 min)"
    )
    circuit_breaker_threshold: int = Field(
        default=3,
        description="Consecutive failures before opening circuit for a service"
    )
    circuit_breaker_reset_seconds: float = Field(
        default=300.0,
        description="Seconds to wait before retrying an opened circuit"
    )
    
    # ── Notifications / Alerts ───────────────────────
    alerts_enabled: bool = Field(
        default=True,
        description="Enable/disable all alert notifications"
    )
    alert_console: bool = Field(
        default=True,
        description="Show alerts in console (always available)"
    )
    alert_desktop: bool = Field(
        default=False,
        description="Show desktop notifications (requires plyer)"
    )
    alert_pushover_enabled: bool = Field(
        default=False,
        description="Send alerts via Pushover (requires PUSHOVER_USER_KEY and PUSHOVER_API_TOKEN)"
    )
    alert_email_enabled: bool = Field(
        default=False,
        description="Send alerts via email (requires SMTP_* settings)"
    )
    alert_price_threshold_pct: float = Field(
        default=2.0,
        description="Minimum price move %% to trigger price alerts"
    )
    alert_stop_loss_enabled: bool = Field(
        default=True,
        description="Alert when a position hits stop-loss"
    )
    alert_trade_enabled: bool = Field(
        default=True,
        description="Alert on trade execution (paper or live)"
    )
    alert_daily_summary: bool = Field(
        default=True,
        description="Send daily portfolio summary alert"
    )
    alert_error_enabled: bool = Field(
        default=True,
        description="Alert on critical errors (circuit breaker open, etc.)"
    )
    
    # ── SMTP (for email alerts) ──────────────────────
    smtp_host: str = Field(default="", description="SMTP server host")
    smtp_port: int = Field(default=587, description="SMTP server port")
    smtp_username: str = Field(default="", description="SMTP username")
    smtp_password: str = Field(default="", description="SMTP password")
    smtp_from: str = Field(default="", description="Sender email address")
    smtp_to: str = Field(default="", description="Recipient email address")
    
    # ── API Keys (loaded from .env, not set here) ───
    # These are loaded automatically via model_config


_settings_cache: AutFinSettings | None = None


def get_settings() -> AutFinSettings:
    """Return a cached singleton instance of AutFinSettings."""
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = AutFinSettings()
    return _settings_cache