@echo off
cd /d "C:\Users\cjand\OneDrive\Desktop\pinpoint"
"C:\Users\cjand\AppData\Local\Programs\Python\Python314\python.exe" main.py 2>nul || ^
"C:\Users\cjand\AppData\Local\Python\pythoncore-3.14-64\python.exe" main.py 2>nul || ^
python main.py
pause
