from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

from .artifacts import artifact_path, find_artifacts_dir, read_json
from .fusion import EmotionPrediction


DEFAULT_TEXT_CLASS_NAMES = [
    "angry",
    "fear",
    "happy",
    "sad",
    "surprise",
]


class TextEmotionRecognizer:
    def __init__(
        self,
        artifacts_dir: str | Path | None = None,
        model_path: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
        config_path: str | Path | None = None,
        label_maps_path: str | Path | None = None,
        max_len: int = 150,
    ):
        self.model_path = artifact_path(
            artifacts_dir, model_path, "best_text_model.keras"
        )
        self.tokenizer_path = artifact_path(
            artifacts_dir, tokenizer_path, "text_tokenizer.pkl"
        )
        self.config_path = self._resolve_config_path(artifacts_dir, config_path)
        self.label_maps_path = artifact_path(
            artifacts_dir, label_maps_path, "label_maps.json"
        )

        self.model = keras.models.load_model(self.model_path)
        self.tokenizer = self._load_tokenizer()
        self.max_len = self._load_max_len(default=max_len)
        self.class_names = self._load_class_names()

    def _resolve_config_path(
        self,
        artifacts_dir: str | Path | None,
        config_path: str | Path | None,
    ) -> Path:
        if config_path is not None:
            return artifact_path(artifacts_dir, config_path, "inference_config.json")

        base_dir = find_artifacts_dir(artifacts_dir)
        path = base_dir / "inference_config.json"
        if path.exists():
            return path

        raise FileNotFoundError(
            f"Required artifact missing: {base_dir / 'inference_config.json'}\n"
            "Export emotion_model_artifacts.zip from the notebook, then extract it "
            "into the artifacts folder."
        )

    def _load_tokenizer(self):
        with self.tokenizer_path.open("rb") as f:
            return pickle.load(f)

    def _load_max_len(self, default: int) -> int:
        if not self.config_path.exists():
            return default
        config = read_json(self.config_path)
        text_config = config.get("text", config)
        return int(text_config.get("max_len", text_config.get("max_length", default)))

    def _load_class_names(self) -> list[str]:
        if self.config_path.exists():
            config = read_json(self.config_path)
            text_config = config.get("text", config)
            class_order = text_config.get("class_order") or config.get("shared_labels")
            if isinstance(class_order, list) and class_order:
                return [str(label) for label in class_order]

            class_names = text_config.get("text_class_names")
            if isinstance(class_names, list) and class_names:
                return [str(label) for label in class_names]

        if self.label_maps_path.exists():
            label_maps = read_json(self.label_maps_path)
            shared_labels = label_maps.get("shared_labels")
            if isinstance(shared_labels, list) and shared_labels:
                return [str(label) for label in shared_labels]

            inv = label_maps.get("text_label_map_inv")
            if isinstance(inv, dict) and inv:
                return [
                    str(label)
                    for _, label in sorted(inv.items(), key=lambda item: int(item[0]))
                ]
            class_names = label_maps.get("text_class_names")
            if isinstance(class_names, list) and class_names:
                return [str(label) for label in class_names]

        return DEFAULT_TEXT_CLASS_NAMES

    def predict(self, text: str) -> EmotionPrediction:
        cleaned = text.lower().strip()
        sequence = self.tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=self.max_len, padding="post")
        probs = np.asarray(self.model.predict(padded, verbose=0)[0], dtype=float)
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
            source="text",
        )
