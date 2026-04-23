@echo off
setlocal
set "PROJECT_DIR=%~dp0"
"%PROJECT_DIR%.venv\Scripts\python.exe" -m streamlit run "%PROJECT_DIR%streamlit_app.py"
