@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Python environment is missing. See START_HERE.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" ensure_local_ai.py
".venv\Scripts\python.exe" -m jupyterlab .
pause
