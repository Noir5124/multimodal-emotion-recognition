from __future__ import annotations

from pathlib import Path
import tempfile

import cv2
import numpy as np
import streamlit as st

from demo_runtime import (
    DEFAULT_ARTIFACTS,
    predict_image_and_text,
    predict_text_only,
    process_video_file,
)


ARTIFACTS = DEFAULT_ARTIFACTS

st.set_page_config(page_title="Emotion Recognition Demo", layout="wide")
st.title("Emotion Recognition Demo")
st.caption(f"Artifacts: {ARTIFACTS}")


def decode_uploaded_image(uploaded) -> np.ndarray:
    data = uploaded.getvalue()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

tab_text, tab_image, tab_video = st.tabs(["Text", "Image / Webcam", "Video"])

with tab_text:
    text = st.text_area("Text", value="I am feeling happy today")
    if st.button("Predict Text"):
        st.json(predict_text_only(text, artifacts=ARTIFACTS))

with tab_image:
    camera_image = st.camera_input("Capture from webcam")
    uploaded_image = st.file_uploader("Or upload image", type=["png", "jpg", "jpeg"])
    image_text = st.text_input("Optional text for fusion")
    face_weight = st.slider("Face weight", 0.0, 1.0, 0.65, 0.05)
    text_weight = st.slider("Text weight", 0.0, 1.0, 0.35, 0.05)
    if st.button("Predict Image"):
        selected = camera_image or uploaded_image
        if selected is None:
            st.warning("Provide an image first.")
        else:
            image_rgb = decode_uploaded_image(selected)
            annotated, summary = predict_image_and_text(
                image_rgb,
                text=image_text,
                artifacts=ARTIFACTS,
                face_weight=face_weight,
                text_weight=text_weight,
            )
            st.image(annotated)
            st.text(summary)

with tab_video:
    video = st.file_uploader("Upload video", type=["mp4", "avi", "mov"], key="video")
    video_text = st.text_input("Optional text for video fusion")
    detect_every = st.slider("Detect every (seconds)", 0.5, 5.0, 2.0, 0.5)
    face_weight_video = st.slider("Face weight ", 0.0, 1.0, 0.65, 0.05)
    text_weight_video = st.slider("Text weight ", 0.0, 1.0, 0.35, 0.05)
    if st.button("Process Video"):
        if video is None:
            st.warning("Upload a video first.")
        else:
            temp_dir = Path(tempfile.mkdtemp(prefix="streamlit_video_"))
            temp_path = temp_dir / video.name
            temp_path.write_bytes(video.read())
            output_path, summary = process_video_file(
                temp_path,
                text=video_text,
                artifacts=ARTIFACTS,
                face_weight=face_weight_video,
                text_weight=text_weight_video,
                detect_every=detect_every,
            )
            st.video(output_path)
            st.text(summary)
