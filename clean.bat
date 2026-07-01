@echo off
REM AutFin - Cache Clean Script
REM Removes all runtime data, cache, and temp files for a fresh start

cd /d "%~dp0"

echo ========================================
echo   AutFin - Cleaning Cache
echo ========================================
echo.

echo [1/6] Removing Python cache...
for /d /r %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
for /r %%f in (*.pyc) do @if exist "%%f" del /q "%%f" 2>nul
for /r %%f in (*.pyo) do @if exist "%%f" del /q "%%f" 2>nul
echo   Done.

echo [2/6] Removing LangGraph API cache...
if exist ".langgraph_api" rd /s /q ".langgraph_api" 2>nul
echo   Done.

echo [3/6] Removing SQLite databases...
if exist "src\data\autfin.db" del /q "src\data\autfin.db" 2>nul
if exist "src\data\autfin.db-shm" del /q "src\data\autfin.db-shm" 2>nul
if exist "src\data\autfin.db-wal" del /q "src\data\autfin.db-wal" 2>nul
if exist "src\data\checkpoints.db" del /q "src\data\checkpoints.db" 2>nul
if exist "src\data\checkpoints.db-shm" del /q "src\data\checkpoints.db-shm" 2>nul
if exist "src\data\checkpoints.db-wal" del /q "src\data\checkpoints.db-wal" 2>nul
echo   Done.

echo [4/6] Removing ChromaDB vector store...
if exist "src\data\chroma" rd /s /q "src\data\chroma" 2>nul
if exist "data\chroma" rd /s /q "data\chroma" 2>nul
echo   Done.

echo [5/6] Removing uv lockfile...
if exist "uv.lock" del /q "uv.lock" 2>nul
echo   Done.

echo [6/6] Removing build artifacts...
if exist "dist" rd /s /q "dist" 2>nul
if exist "build" rd /s /q "build" 2>nul
for /d %%d in (*.egg-info) do @if exist "%%d" rd /s /q "%%d" 2>nul
for /r %%f in (*.log) do @if exist "%%f" del /q "%%f" 2>nul
echo   Done.

echo.
echo ========================================
echo   Clean complete! Fresh state ready.
echo ========================================
echo.
echo   To start: dev.bat (dev) or start.bat (daemon)
echo.
pause
