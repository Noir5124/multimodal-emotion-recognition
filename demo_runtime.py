from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from emotion_pipeline import EmotionFusion, FacialEmotionRecognizer, TextEmotionRecognizer


DEFAULT_ARTIFACTS = Path(r"D:\emotion_model_artifacts")


@dataclass
class RuntimeConfig:
    artifacts: Path
    face_weight: float = 0.65
    text_weight: float = 0.35


def draw_label(frame, x: int, y: int, text: str, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y - th - baseline - 8)
    cv2.rectangle(frame, (x, top), (x + tw + 10, top + th + baseline + 8), color, -1)
    cv2.putText(
        frame,
        text,
        (x + 5, top + th + 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def image_to_bgr(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("No image provided.")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def image_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


@lru_cache(maxsize=4)
def load_runtime(artifacts_dir: str) -> tuple[FacialEmotionRecognizer, TextEmotionRecognizer]:
    artifacts = Path(artifacts_dir)
    face_model = FacialEmotionRecognizer(artifacts_dir=artifacts)
    text_model = TextEmotionRecognizer(artifacts_dir=artifacts)
    return face_model, text_model


def predict_text_only(
    text: str,
    artifacts: Path = DEFAULT_ARTIFACTS,
) -> dict[str, float | str]:
    _, text_model = load_runtime(str(artifacts))
    prediction = text_model.predict(text)
    return {
        "label": prediction.label,
        "confidence": round(prediction.confidence, 4),
        "source": prediction.source,
    }


def predict_image_and_text(
    image: np.ndarray,
    text: str | None = None,
    artifacts: Path = DEFAULT_ARTIFACTS,
    face_weight: float = 0.65,
    text_weight: float = 0.35,
) -> tuple[np.ndarray, str]:
    face_model, text_model = load_runtime(str(artifacts))
    fusion = EmotionFusion(face_weight, text_weight)

    frame = image_to_bgr(image)
    predictions = face_model.predict_frame(frame)
    text_prediction = text_model.predict(text) if text and text.strip() else None

    lines: list[str] = []
    if text_prediction is not None:
        lines.append(f"text: {text_prediction.label} ({text_prediction.confidence:.3f})")

    if not predictions:
        lines.insert(0, "face: no face detected")
        return image_to_rgb(frame), "\n".join(lines)

    for idx, ((x, y, w, h), face_prediction) in enumerate(predictions, start=1):
        fused = fusion.combine(face=face_prediction, text=text_prediction)
        shown = fused or face_prediction
        color = (42, 168, 75)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        draw_label(frame, x, y, f"{shown.label} {shown.confidence:.2f}", color)
        lines.append(
            f"face {idx}: {face_prediction.label} ({face_prediction.confidence:.3f})"
        )
        if fused is not None:
            lines.append(f"fused {idx}: {fused.label} ({fused.confidence:.3f})")

    return image_to_rgb(frame), "\n".join(lines)


def process_video_file(
    video_path: str | Path,
    text: str | None = None,
    artifacts: Path = DEFAULT_ARTIFACTS,
    face_weight: float = 0.65,
    text_weight: float = 0.35,
    detect_every: float = 2.0,
) -> tuple[str, str]:
    face_model, text_model = load_runtime(str(artifacts))
    fusion = EmotionFusion(face_weight, text_weight)
    text_prediction = text_model.predict(text) if text and text.strip() else None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir = Path(tempfile.mkdtemp(prefix="emotion_video_"))
    out_path = out_dir / "annotated.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create annotated output video.")

    last_face_prediction = None
    last_face_box = None
    last_detection_time = 0.0
    frames = 0
    detections = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        current_time = frames / fps
        if current_time - last_detection_time >= detect_every or frames == 0:
            predictions = face_model.predict_frame(frame)
            last_detection_time = current_time
            if predictions:
                detections += 1
                last_face_box, last_face_prediction = predictions[0]
            else:
                last_face_box, last_face_prediction = None, None

        if last_face_box is not None and last_face_prediction is not None:
            x, y, w, h = last_face_box
            fused = fusion.combine(face=last_face_prediction, text=text_prediction)
            shown = fused or last_face_prediction
            color = (42, 168, 75)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            draw_label(frame, x, y, f"{shown.label} {shown.confidence:.2f}", color)

        writer.write(frame)
        frames += 1

    cap.release()
    writer.release()

    summary_lines = [
        f"frames: {frames}",
        f"fps: {fps:.2f}",
        f"video detections run: {detections}",
        f"detect interval: {detect_every:.1f}s",
    ]
    if text_prediction is not None:
        summary_lines.append(
            f"text: {text_prediction.label} ({text_prediction.confidence:.3f})"
        )
    if last_face_prediction is not None:
        summary_lines.append(
            f"last face: {last_face_prediction.label} ({last_face_prediction.confidence:.3f})"
        )

    return str(out_path), "\n".join(summary_lines)
