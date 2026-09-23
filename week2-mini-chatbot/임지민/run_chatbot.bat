@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Python environment is missing. See START_HERE.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" ensure_local_ai.py
".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.maxUploadSize 30 --browser.gatherUsageStats false
pause
