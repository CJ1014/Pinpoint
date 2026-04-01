@echo off
cd /d "C:\Users\cjand\OneDrive\Desktop\pinpoint"

set PYTHON=python
if exist "C:\Users\cjand\AppData\Local\Programs\Python\Python314\python.exe" (
    set PYTHON=C:\Users\cjand\AppData\Local\Programs\Python\Python314\python.exe
)
if exist "C:\Users\cjand\AppData\Local\Python\pythoncore-3.14-64\python.exe" (
    set PYTHON=C:\Users\cjand\AppData\Local\Python\pythoncore-3.14-64\python.exe
)

"%PYTHON%" main.py %*
pause
