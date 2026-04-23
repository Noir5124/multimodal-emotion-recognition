$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$Artifacts = "D:\emotion_model_artifacts"

if (-not (Test-Path $Python)) {
    throw "Virtualenv Python not found: $Python. Run: python -m venv .venv"
}

if (-not (Test-Path $Artifacts)) {
    throw "Artifacts folder not found: $Artifacts"
}

& $Python (Join-Path $ProjectDir "check_artifacts.py") --artifacts $Artifacts
& $Python (Join-Path $ProjectDir "realtime_camera.py") --artifacts $Artifacts --mirror
