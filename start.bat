@echo off
title PinPoint

REM ── Configure your model here ──────────────────────────────────────────────
set OLLAMA_MODEL=llama3.3:latest
REM set OLLAMA_BASE_URL=http://localhost:11434/v1
REM ───────────────────────────────────────────────────────────────────────────

where python >nul 2>nul
if errorlevel 1 (
    echo Python not found. Install from https://python.org
    pause
    exit /b 1
)

python -m pip install -q -r requirements.txt
python main.py %*
pause
