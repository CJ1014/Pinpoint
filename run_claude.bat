@echo off
REM ============================================================
REM  Run PinPoint on Claude (most human mode)
REM  1. Paste your API key below (replace PASTE_YOUR_KEY_HERE)
REM  2. Save this file
REM  3. Double-click it to launch PinPoint on Claude
REM ============================================================

cd /d "C:\Users\cjand\OneDrive\Desktop\pinpoint"

set LLM_PROVIDER=claude
set ANTHROPIC_API_KEY=PASTE_YOUR_KEY_HERE

REM Optional: sonnet = fast+human, opus = smartest (pricier)
set ANTHROPIC_MODEL=claude-sonnet-4-5

set PYTHON=C:\Users\cjand\AppData\Local\Python\pythoncore-3.14-64\python.exe

"%PYTHON%" main.py %*
pause
