@echo off
cd /d "C:\Users\cjand\OneDrive\Desktop\pinpoint"

REM gemma2:9b — best free local model for sounding human/conversational.
REM (Was qwen2.5-coder:14b, which is a coding model and sounds robotic.)
set OLLAMA_MODEL=gemma2:9b
set OLLAMA_KEEP_ALIVE=-1
set OLLAMA_FLASH_ATTENTION=1
set OLLAMA_NUM_PARALLEL=1
set OLLAMA_MAX_LOADED_MODELS=2
set PYTHON=C:\Users\cjand\AppData\Local\Python\pythoncore-3.14-64\python.exe

"%PYTHON%" main.py %*
pause
