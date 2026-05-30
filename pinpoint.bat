@echo off
cd /d "C:\Users\cjand\OneDrive\Desktop\pinpoint"

REM CHAT model — gemma2:9b sounds the most human in conversation.
REM Already downloaded. Used for all chat/idle talk.
set OLLAMA_MODEL=gemma2:9b

REM AGENT model — used for autonomous sessions that build things (needs tool/function-call support).
REM gemma2:9b doesn't support tools so agent sessions need a separate model.
REM llama3.2:3b is only 2 GB. Download it once: ollama pull llama3.2:3b
set AGENT_MODEL=llama3.2:3b

set OLLAMA_KEEP_ALIVE=-1
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_NUM_PARALLEL=1
set OLLAMA_MAX_LOADED_MODELS=2
set PYTHON=C:\Users\cjand\AppData\Local\Python\pythoncore-3.14-64\python.exe

"%PYTHON%" main.py %*
pause
