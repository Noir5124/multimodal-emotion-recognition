from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse

from demo_runtime import DEFAULT_ARTIFACTS, predict_image_and_text, predict_text_only, process_video_file


app = FastAPI(title="Emotion Recognition API")


@app.get("/health")
def health():
    return {"ok": True, "artifacts": str(DEFAULT_ARTIFACTS)}


@app.post("/predict/text")
def predict_text(text: str = Form(...)):
    return predict_text_only(text, artifacts=DEFAULT_ARTIFACTS)


@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    text: str | None = Form(default=None),
    face_weight: float = Form(default=0.65),
    text_weight: float = Form(default=0.35),
):
    content = await file.read()
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return {"error": "Could not decode image."}
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _, summary = predict_image_and_text(
        image_rgb,
        text=text,
        artifacts=DEFAULT_ARTIFACTS,
        face_weight=face_weight,
        text_weight=text_weight,
    )
    return {"summary": summary}


@app.post("/predict/video")
async def predict_video(
    file: UploadFile = File(...),
    text: str | None = Form(default=None),
    face_weight: float = Form(default=0.65),
    text_weight: float = Form(default=0.35),
    detect_every: float = Form(default=2.0),
):
    temp_dir = Path(tempfile.mkdtemp(prefix="emotion_api_video_"))
    in_path = temp_dir / file.filename
    in_path.write_bytes(await file.read())
    output_path, summary = process_video_file(
        in_path,
        text=text,
        artifacts=DEFAULT_ARTIFACTS,
        face_weight=face_weight,
        text_weight=text_weight,
        detect_every=detect_every,
    )
    return {"summary": summary, "video_path": output_path}
