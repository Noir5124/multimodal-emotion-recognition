from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


CANONICAL_LABELS = (
    "angry",
    "fear",
    "happy",
    "sad",
    "surprise",
)

LABEL_ALIASES = {
    "anger": "angry",
    "angry": "angry",
    "fear": "fear",
    "happy": "happy",
    "happiness": "happy",
    "joy": "happy",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprise",
}


@dataclass(frozen=True)
class EmotionPrediction:
    label: str
    confidence: float
    scores: dict[str, float]
    source: str


def normalize_label(label: str) -> str:
    key = label.strip().lower()
    return LABEL_ALIASES.get(key, key)


def project_scores(scores: Mapping[str, float]) -> dict[str, float]:
    projected = {label: 0.0 for label in CANONICAL_LABELS}
    for raw_label, score in scores.items():
        label = normalize_label(raw_label)
        if label in projected:
            projected[label] += float(score)
    total = sum(projected.values())
    if total > 0:
        projected = {label: score / total for label, score in projected.items()}
    return projected


class EmotionFusion:
    def __init__(self, face_weight: float = 0.65, text_weight: float = 0.35):
        if face_weight < 0 or text_weight < 0:
            raise ValueError("Fusion weights must be non-negative.")
        self.face_weight = float(face_weight)
        self.text_weight = float(text_weight)

    def combine(
        self,
        face: EmotionPrediction | None = None,
        text: EmotionPrediction | None = None,
    ) -> EmotionPrediction | None:
        weighted_scores = {label: 0.0 for label in CANONICAL_LABELS}
        total_weight = 0.0

        if face is not None:
            face_scores = project_scores(face.scores)
            for label, score in face_scores.items():
                weighted_scores[label] += score * self.face_weight
            total_weight += self.face_weight

        if text is not None:
            text_scores = project_scores(text.scores)
            for label, score in text_scores.items():
                weighted_scores[label] += score * self.text_weight
            total_weight += self.text_weight

        if total_weight <= 0:
            return None

        combined = {
            label: score / total_weight for label, score in weighted_scores.items()
        }
        label = max(combined, key=combined.get)
        return EmotionPrediction(
            label=label,
            confidence=float(combined[label]),
            scores=combined,
            source="fusion",
        )
