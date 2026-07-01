@echo off
REM AutFin — LangGraph dev server with blocking calls allowed
REM Uses --allow-blocking to suppress Blockbuster errors from stdlib imports
REM (zoneinfo, sysconfig, pathlib all call os.getcwd at import time)

cd /d "%~dp0"

echo ========================================
echo   AutFin — LangGraph Dev Server
echo ========================================
echo.
echo Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
echo.
echo Press Ctrl+C to stop.
echo ========================================
echo.

.\.venv\Scripts\langgraph.exe dev --allow-blocking
