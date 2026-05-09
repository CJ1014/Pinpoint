@echo off
cd /d "C:\Users\cjand\OneDrive\Desktop\pinpoint"

set OLLAMA_MODEL=qwen3-coder:480b-cloud
set PYTHON=C:\Users\cjand\AppData\Local\Python\pythoncore-3.14-64\python.exe

"%PYTHON%" main.py %*
pause
