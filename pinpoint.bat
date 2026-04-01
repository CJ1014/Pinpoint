@echo off
cd /d "C:\Users\cjand\OneDrive\Desktop\pinpoint"

set OLLAMA_MODEL=qwen2.5-coder:32b-instruct-q3_K_M
set PYTHON=python
if exist "C:\Users\cjand\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Python\Python 3.14\python.exe" (
    set PYTHON=C:\Users\cjand\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Python\Python 3.14\python.exe
)

"%PYTHON%" main.py %*
pause
