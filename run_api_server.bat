@echo off
setlocal
set "PROJECT_DIR=%~dp0"
"%PROJECT_DIR%.venv\Scripts\python.exe" -m uvicorn api_server:app --reload --host 127.0.0.1 --port 8000
