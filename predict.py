from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from emotion_pipeline import EmotionFusion, FacialEmotionRecognizer, TextEmotionRecognizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run text, facial, or fused emotion prediction."
    )
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--image", type=Path, default=None, help="Image path for face emotion.")
    parser.add_argument("--text", type=str, default=None, help="Text for text emotion.")
    parser.add_argument("--face-weight", type=float, default=0.65)
    parser.add_argument("--text-weight", type=float, default=0.35)
    return parser.parse_args()


def format_prediction(prefix: str, prediction) -> str:
    return f"{prefix}: {prediction.label} ({prediction.confidence:.3f})"


def main() -> None:
    args = parse_args()
    face_prediction = None
    text_prediction = None

    if args.image is not None:
        face_model = FacialEmotionRecognizer(artifacts_dir=args.artifacts)
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {args.image}")
        faces = face_model.predict_frame(frame)
        if not faces:
            raise RuntimeError("No face detected in image.")
        face_prediction = faces[0][1]
        print(format_prediction("Face", face_prediction))

    if args.text:
        text_model = TextEmotionRecognizer(artifacts_dir=args.artifacts)
        text_prediction = text_model.predict(args.text)
        print(format_prediction("Text", text_prediction))

    fused = EmotionFusion(args.face_weight, args.text_weight).combine(
        face=face_prediction,
        text=text_prediction,
    )
    if fused is not None:
        print(format_prediction("Fused", fused))


if __name__ == "__main__":
    main()
