from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from emotion_pipeline import EmotionFusion, FacialEmotionRecognizer, TextEmotionRecognizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Realtime webcam facial emotion recognition with optional fixed text fusion. "
            "Use multimodal_realtime.py for live text updates or video files."
        )
    )
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--face-weight", type=float, default=0.65)
    parser.add_argument("--text-weight", type=float, default=0.35)
    parser.add_argument(
        "--detect-every",
        type=float,
        default=0.1,
        help="Run emotion detection every N seconds; overlay reuses latest result between runs.",
    )
    parser.add_argument("--mirror", action="store_true", help="Mirror camera preview.")
    return parser.parse_args()


def draw_label(frame, x: int, y: int, text: str, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y - th - baseline - 8)
    cv2.rectangle(frame, (x, top), (x + tw + 8, top + th + baseline + 8), color, -1)
    cv2.putText(
        frame,
        text,
        (x + 4, top + th + 2),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    face_model = FacialEmotionRecognizer(artifacts_dir=args.artifacts)
    text_prediction = None

    if args.text:
        text_model = TextEmotionRecognizer(artifacts_dir=args.artifacts)
        text_prediction = text_model.predict(args.text)
        print(f"Text emotion: {text_prediction.label} ({text_prediction.confidence:.3f})")

    fusion = EmotionFusion(args.face_weight, args.text_weight)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    detect_interval = max(0.1, float(args.detect_every))
    last_detection_at = 0.0
    latest_predictions = []

    print(f"Camera started. Detecting emotion every {detect_interval:.1f}s. Press q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            now = time.monotonic()
            if now - last_detection_at >= detect_interval:
                latest_predictions = face_model.predict_frame(frame)
                last_detection_at = now

            age = max(0.0, now - last_detection_at)
            for (x, y, w, h), face_prediction in latest_predictions:
                fused = fusion.combine(face=face_prediction, text=text_prediction)
                shown = fused or face_prediction
                label = f"{shown.label} {shown.confidence:.2f} ({age:.1f}s)"
                if text_prediction is not None:
                    label = f"{shown.label} {shown.confidence:.2f} fused ({age:.1f}s)"
                color = (42, 168, 75)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                draw_label(frame, x, y, label, color)

            if text_prediction is not None:
                draw_label(
                    frame,
                    12,
                    36,
                    f"text: {text_prediction.label} {text_prediction.confidence:.2f}",
                    (70, 70, 70),
                )

            cv2.imshow("Emotion recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
