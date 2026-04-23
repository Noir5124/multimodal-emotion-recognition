# Setup

## 1. Create Virtual Environment

```powershell
cd C:\Users\HP\Desktop\caveman\emotion_recognition_pipeline
python -m venv .venv
```

## 2. Install Dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Or skip activation and call Python directly from `.venv`.

## 3. Verify Artifacts

Notebook-trained models expected here:

```text
D:\emotion_model_artifacts
```

Check:

```powershell
.\.venv\Scripts\python.exe check_artifacts.py --artifacts D:\emotion_model_artifacts
```

## 4. Run Apps

Gradio:

```powershell
.\run_gradio_demo.bat
```

Streamlit:

```powershell
.\run_streamlit_demo.bat
```

FastAPI:

```powershell
.\run_api_server.bat
```

Realtime multimodal webcam:

```powershell
.\run_multimodal_realtime.bat
```

## 5. Rebuild Evaluation Graphs

```powershell
.\.venv\Scripts\python.exe generate_evaluation_graphs.py --artifacts D:\emotion_model_artifacts
```

## 6. Recommended GitHub Repo Policy

Do not commit:

- `.venv/`
- trained model binaries
- local logs
- temp videos
- notebook checkpoints

Commit:

- source code
- docs
- evaluation graphs
- requirements
- notebook export instructions
