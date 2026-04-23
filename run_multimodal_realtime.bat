@echo off
setlocal
set "PROJECT_DIR=%~dp0"
set "PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe"
set "ARTIFACTS=D:\emotion_model_artifacts"

if not exist "%PYTHON%" (
  echo Virtualenv Python not found: %PYTHON%
  echo Run: python -m venv .venv
  exit /b 1
)

if not exist "%ARTIFACTS%" (
  echo Artifacts folder not found: %ARTIFACTS%
  exit /b 1
)

"%PYTHON%" "%PROJECT_DIR%check_artifacts.py" --artifacts "%ARTIFACTS%" || exit /b 1
"%PYTHON%" "%PROJECT_DIR%multimodal_realtime.py" --artifacts "%ARTIFACTS%" --mirror --interactive-text
