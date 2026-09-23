@echo off
cd /d "%~dp0"
set "OLLAMA_MODELS=%CD%\.local-ai\models"
set "OLLAMA_NO_CLOUD=1"
set "OLLAMA_HOST=127.0.0.1:11434"
set "OLLAMA_CONTEXT_LENGTH=8192"
set "OLLAMA_MAX_LOADED_MODELS=1"
if not exist ".local-ai\ollama\ollama.exe" (
    echo Local AI is missing. See setup_local_ai.ps1.
    pause
    exit /b 1
)
echo Starting local AI. Keep this window open. Cloud access is disabled.
".local-ai\ollama\ollama.exe" serve
pause
