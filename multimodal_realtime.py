from __future__ import annotations

import argparse
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2

from emotion_pipeline import EmotionFusion, FacialEmotionRecognizer, TextEmotionRecognizer
from emotion_pipeline.fusion import CANONICAL_LABELS, EmotionPrediction, project_scores


@dataclass
class TextState:
    text: str | None = None
    prediction: EmotionPrediction | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime multimodal emotion recognition from video/camera + text."
    )
    parser.add_argument("--artifacts", type=Path, default=None)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", type=Path, default=None, help="Optional video file path.")
    parser.add_argument("--output", type=Path, default=None, help="Optional annotated video output.")
    parser.add_argument("--text", type=str, default=None, help="Initial text for fusion.")
    parser.add_argument(
        "--interactive-text",
        action="store_true",
        help="Allow typing new text in the terminal while video runs.",
    )
    parser.add_argument("--face-weight", type=float, default=0.65)
    parser.add_argument("--text-weight", type=float, default=0.35)
    parser.add_argument("--smooth", type=int, default=5, help="Face smoothing window size.")
    parser.add_argument(
        "--detect-every",
        type=float,
        default=0.0,
        help="Run face emotion detection every N seconds. Use 0 for every frame.",
    )
    parser.add_argument("--mirror", action="store_true", help="Mirror camera preview.")
    return parser.parse_args()


def draw_label(frame, x: int, y: int, text: str, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
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


def draw_panel(frame, lines: list[str]) -> None:
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.56
    thickness = 1
    padding = 10
    line_height = 24
    width = 20
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, scale, thickness)
        width = max(width, tw + padding * 2)
    height = padding * 2 + line_height * len(lines)
    cv2.rectangle(frame, (8, 8), (8 + width, 8 + height), (35, 35, 35), -1)
    for idx, line in enumerate(lines):
        y = 8 + padding + 16 + idx * line_height
        cv2.putText(
            frame,
            line,
            (8 + padding, y),
            font,
            scale,
            (245, 245, 245),
            thickness,
            cv2.LINE_AA,
        )


def average_predictions(history: deque[EmotionPrediction]) -> EmotionPrediction | None:
    if not history:
        return None
    scores = {label: 0.0 for label in CANONICAL_LABELS}
    for prediction in history:
        projected = project_scores(prediction.scores)
        for label in scores:
            scores[label] += projected[label]
    averaged = {label: score / len(history) for label, score in scores.items()}
    label = max(averaged, key=averaged.get)
    return EmotionPrediction(
        label=label,
        confidence=float(averaged[label]),
        scores=averaged,
        source="face_smooth",
    )


def update_text_state(
    state: TextState,
    lock: threading.Lock,
    text_model: TextEmotionRecognizer,
    text: str,
) -> None:
    text = text.strip()
    if not text:
        return
    prediction = text_model.predict(text)
    with lock:
        state.text = text
        state.prediction = prediction
    print(f"Text emotion: {prediction.label} ({prediction.confidence:.3f})")


def start_text_thread(
    state: TextState,
    lock: threading.Lock,
    text_model: TextEmotionRecognizer,
) -> threading.Thread:
    def worker() -> None:
        print("Type new text then press Enter. Empty line ignored. Ctrl+C exits terminal.")
        while True:
            try:
                text = input("text> ")
            except EOFError:
                return
            update_text_state(state, lock, text_model, text)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


def open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    source = str(args.video) if args.video is not None else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


def open_writer(
    output: Path | None,
    cap: cv2.VideoCapture,
    first_frame,
) -> cv2.VideoWriter | None:
    if output is None:
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 25
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video: {output}")
    return writer


def main() -> None:
    args = parse_args()
    use_text = args.text is not None or args.interactive_text

    face_model = FacialEmotionRecognizer(artifacts_dir=args.artifacts)
    text_model = TextEmotionRecognizer(artifacts_dir=args.artifacts) if use_text else None
    fusion = EmotionFusion(args.face_weight, args.text_weight)

    text_state = TextState()
    text_lock = threading.Lock()
    if args.text and text_model is not None:
        update_text_state(text_state, text_lock, text_model, args.text)
    if args.interactive_text and text_model is not None:
        start_text_thread(text_state, text_lock, text_model)

    cap = open_capture(args)
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read first frame.")
    if args.mirror and args.video is None:
        frame = cv2.flip(frame, 1)

    writer = open_writer(args.output, cap, frame)
    face_history: deque[EmotionPrediction] = deque(maxlen=max(1, args.smooth))
    face_prediction = None
    face_box = None
    last_detection_at = 0.0
    detect_interval = max(0.0, float(args.detect_every))

    interval_label = "every frame" if detect_interval == 0.0 else f"every {detect_interval:.1f}s"
    print(
        f"Multimodal realtime started. Detecting emotion {interval_label}. "
        "Press q in video window to quit."
    )
    try:
        while ok:
            now = time.monotonic()
            should_detect = detect_interval == 0.0 or now - last_detection_at >= detect_interval
            if should_detect:
                predictions = face_model.predict_frame(frame)
                last_detection_at = now
                if predictions:
                    face_box, raw_face_prediction = predictions[0]
                    face_history.append(raw_face_prediction)
                    face_prediction = average_predictions(face_history)
                else:
                    face_box = None
                    face_prediction = None
                    face_history.clear()

            with text_lock:
                text_prediction = text_state.prediction
                current_text = text_state.text

            fused = fusion.combine(face=face_prediction, text=text_prediction)
            panel_lines = []
            if face_prediction is not None:
                age = max(0.0, now - last_detection_at)
                panel_lines.append(
                    f"face: {face_prediction.label} {face_prediction.confidence:.2f} ({age:.1f}s)"
                )
            else:
                panel_lines.append("face: no face")
            if text_prediction is not None:
                panel_lines.append(
                    f"text: {text_prediction.label} {text_prediction.confidence:.2f}"
                )
            else:
                panel_lines.append("text: none")
            if fused is not None:
                panel_lines.append(f"fused: {fused.label} {fused.confidence:.2f}")
            if current_text:
                shown_text = current_text if len(current_text) <= 46 else current_text[:43] + "..."
                panel_lines.append(f'input: "{shown_text}"')

            if face_box is not None:
                x, y, w, h = face_box
                color = (42, 168, 75)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                if fused is not None:
                    draw_label(frame, x, y, f"{fused.label} {fused.confidence:.2f} fused", color)
                elif face_prediction is not None:
                    draw_label(frame, x, y, f"{face_prediction.label} {face_prediction.confidence:.2f}", color)

            draw_panel(frame, panel_lines)

            if writer is not None:
                writer.write(frame)

            cv2.imshow("Multimodal emotion recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            ok, frame = cap.read()
            if ok and args.mirror and args.video is None:
                frame = cv2.flip(frame, 1)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
