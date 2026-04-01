@echo off
cd /d "C:\Users\cjand\OneDrive\Desktop\pinpoint"

set OLLAMA_MODEL=qwen2.5-coder:32b-instruct-q3_K_M
set PYTHON=C:\Users\cjand\AppData\Local\Python\pythoncore-3.14-64\python.exe

"%PYTHON%" main.py %*
pause
