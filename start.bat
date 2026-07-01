@echo off
REM AutFin v3.0 — Auto-start script for Windows
REM Run this to start AutFin as a daemon. Press Ctrl+C to stop.

cd /d "%~dp0"

echo ========================================
echo   AutFin v3.0 — Starting Daemon
echo ========================================
echo.
echo Scheduled Jobs:
echo   ^> Daily morning scan  — Mon-Fri 9:00 AM IST
echo   ^> Midday guardian     — Mon-Fri 1:00 PM IST
echo   ^> Weekly review       — Sunday 10:00 AM IST
echo   ^> Monthly deep dive   — 1st of month 8:00 AM IST
echo.
echo LLM: Modal vLLM (Qwen 3.6 35B) + NVIDIA NIM fallback
echo Data: yfinance + Marketaux + FRED + Firecrawl
echo Execution: Kite SDK (Zerodha auto-login via TOTP)
echo Tracing: LangSmith (smith.langchain.com)
echo.
echo Press Ctrl+C to stop.
echo ========================================
echo.

.\.venv\Scripts\python.exe -m autfin.cli.main start
