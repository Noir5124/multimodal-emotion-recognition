from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tensorflow import keras

from .artifacts import artifact_path, read_json
from .fusion import EmotionPrediction


DEFAULT_FER_CLASS_NAMES = [
    "angry",
    "fear",
    "happy",
    "sad",
    "surprise",
]


class FacialEmotionRecognizer:
    def __init__(
        self,
        artifacts_dir: str | Path | None = None,
        model_path: str | Path | None = None,
        label_maps_path: str | Path | None = None,
        cascade_path: str | Path | None = None,
        image_size: tuple[int, int] = (48, 48),
        min_face_size: tuple[int, int] = (40, 40),
        detector: str = "mediapipe",
        min_detection_confidence: float = 0.5,
        mediapipe_model_path: str | Path | None = None,
    ):
        self.model_path = artifact_path(
            artifacts_dir, model_path, "best_fer_model.keras"
        )
        self.label_maps_path = artifact_path(
            artifacts_dir, label_maps_path, "label_maps.json"
        )
        self.image_size = image_size
        self.min_face_size = min_face_size
        self.model = keras.models.load_model(self.model_path)
        self.class_names = self._load_class_names()
        self.detector = detector
        self.mp_face_detection = self._load_mediapipe_detector(
            min_detection_confidence,
            mediapipe_model_path,
        )
        self.face_cascade = self._load_cascade(cascade_path)

    def _load_class_names(self) -> list[str]:
        if not self.label_maps_path.exists():
            return DEFAULT_FER_CLASS_NAMES

        label_maps = read_json(self.label_maps_path)
        shared_labels = label_maps.get("shared_labels")
        if isinstance(shared_labels, list) and shared_labels:
            return [str(label) for label in shared_labels]

        class_indices = label_maps.get("fer_class_indices")
        if isinstance(class_indices, dict) and class_indices:
            return [
                label
                for label, _ in sorted(
                    class_indices.items(), key=lambda item: int(item[1])
                )
            ]

        class_names = label_maps.get("fer_class_names")
        if isinstance(class_names, list) and class_names:
            return [str(label) for label in class_names]

        return DEFAULT_FER_CLASS_NAMES

    def _load_cascade(self, cascade_path: str | Path | None) -> cv2.CascadeClassifier:
        if cascade_path is None:
            cascade_path = (
                Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            )
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            raise FileNotFoundError(f"OpenCV face cascade not found: {cascade_path}")
        return cascade

    def _default_mediapipe_model_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "models" / "blaze_face_short_range.tflite"

    def _load_mediapipe_detector(
        self,
        min_detection_confidence: float,
        mediapipe_model_path: str | Path | None,
    ):
        if self.detector.lower() not in {"mediapipe", "mp"}:
            return None
        model_path = (
            Path(mediapipe_model_path).expanduser().resolve()
            if mediapipe_model_path is not None
            else self._default_mediapipe_model_path()
        )
        if not model_path.exists():
            print(f"MediaPipe model not found: {model_path}. Falling back to OpenCV Haar cascade.")
            self.detector = "opencv"
            return None

        try:
            import mediapipe as mp
            import mediapipe.tasks as tasks
        except ImportError as exc:
            print(f"MediaPipe import failed ({exc}). Falling back to OpenCV Haar cascade.")
            self.detector = "opencv"
            return None

        self._mp = mp
        options = tasks.vision.FaceDetectorOptions(
            base_options=tasks.BaseOptions(model_asset_path=str(model_path)),
            min_detection_confidence=min_detection_confidence,
        )
        return tasks.vision.FaceDetector.create_from_options(options)

    def _detect_faces_mediapipe(
        self, frame_bgr: np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        if self.mp_face_detection is None:
            return []

        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self.mp_face_detection.detect(mp_image)
        if not result.detections:
            return []

        height, width = frame_bgr.shape[:2]
        faces = []
        for detection in result.detections:
            box = detection.bounding_box
            x = int(box.origin_x)
            y = int(box.origin_y)
            w = int(box.width)
            h = int(box.height)

            pad_x = int(w * 0.12)
            pad_y = int(h * 0.18)
            x = max(0, x - pad_x)
            y = max(0, y - pad_y)
            w = min(width - x, w + 2 * pad_x)
            h = min(height - y, h + 2 * pad_y)

            if w >= self.min_face_size[0] and h >= self.min_face_size[1]:
                faces.append((x, y, w, h))

        return sorted(faces, key=lambda rect: rect[2] * rect[3], reverse=True)

    def detect_faces(self, frame_bgr: np.ndarray) -> Iterable[tuple[int, int, int, int]]:
        if self.mp_face_detection is not None:
            return self._detect_faces_mediapipe(frame_bgr)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.min_face_size,
        )
        return sorted(faces, key=lambda rect: rect[2] * rect[3], reverse=True)

    def predict_face(self, face_bgr: np.ndarray) -> EmotionPrediction:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.image_size, interpolation=cv2.INTER_AREA)
        x = resized.astype("float32") / 255.0
        x = np.expand_dims(x, axis=(0, -1))
        probs = self.model.predict(x, verbose=0)[0]
        scores = {
            label: float(probs[idx])
            for idx, label in enumerate(self.class_names)
            if idx < len(probs)
        }
        label = max(scores, key=scores.get)
        return EmotionPrediction(
            label=label,
            confidence=scores[label],
            scores=scores,
            source="face",
        )

    def predict_frame(
        self, frame_bgr: np.ndarray
    ) -> list[tuple[tuple[int, int, int, int], EmotionPrediction]]:
        predictions = []
        for x, y, w, h in self.detect_faces(frame_bgr):
            face = frame_bgr[y : y + h, x : x + w]
            predictions.append(((int(x), int(y), int(w), int(h)), self.predict_face(face)))
        return predictions
