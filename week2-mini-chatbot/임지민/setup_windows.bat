@echo off
cd /d "%~dp0"
py -3.12 -c "import struct; assert struct.calcsize('P') == 8"
if errorlevel 1 (
    echo Install Python 3.12 64-bit with the Python launcher, then retry.
    pause
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements-notebook.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto failed
".venv\Scripts\python.exe" verify_setup.py
if errorlevel 1 goto failed
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_local_ai.ps1"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" verify_setup.py --models
if errorlevel 1 goto failed
echo Setup complete. Double-click run_chatbot.bat.
pause
exit /b 0
:failed
echo Setup failed. Check the error above and README.md, then retry.
pause
exit /b 1
