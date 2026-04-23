from __future__ import annotations

from collections import Counter, deque

from .fusion import CANONICAL_LABELS, EmotionPrediction, project_scores


class StableEmotionFilter:
    def __init__(
        self,
        window: int = 5,
        min_confidence: float = 0.55,
        min_margin: float = 0.12,
        confirm_count: int = 2,
        uncertain_label: str = "uncertain",
    ):
        self.history: deque[EmotionPrediction] = deque(maxlen=max(1, window))
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.confirm_count = max(1, confirm_count)
        self.uncertain_label = uncertain_label
        self.current_label: str | None = None
        self.pending_label: str | None = None
        self.pending_count = 0

    def clear(self) -> None:
        self.history.clear()
        self.current_label = None
        self.pending_label = None
        self.pending_count = 0

    def update(self, prediction: EmotionPrediction) -> EmotionPrediction:
        self.history.append(prediction)
        averaged = self._average()
        gated = self._apply_confidence_gate(averaged)
        return self._confirm_label(gated)

    def _average(self) -> EmotionPrediction:
        scores = {label: 0.0 for label in CANONICAL_LABELS}
        for prediction in self.history:
            projected = project_scores(prediction.scores)
            for label in scores:
                scores[label] += projected[label]
        averaged = {label: score / len(self.history) for label, score in scores.items()}
        label = max(averaged, key=averaged.get)
        return EmotionPrediction(
            label=label,
            confidence=float(averaged[label]),
            scores=averaged,
            source="stable_average",
        )

    def _apply_confidence_gate(self, prediction: EmotionPrediction) -> EmotionPrediction:
        ranked = sorted(prediction.scores.items(), key=lambda item: item[1], reverse=True)
        top_label, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_score - second_score
        if top_score < self.min_confidence or margin < self.min_margin:
            return EmotionPrediction(
                label=self.uncertain_label,
                confidence=float(top_score),
                scores=prediction.scores,
                source="stable_uncertain",
            )
        return EmotionPrediction(
            label=top_label,
            confidence=float(top_score),
            scores=prediction.scores,
            source=prediction.source,
        )

    def _confirm_label(self, prediction: EmotionPrediction) -> EmotionPrediction:
        label = prediction.label
        if self.current_label is None:
            self.current_label = label
            return prediction

        if label == self.current_label:
            self.pending_label = None
            self.pending_count = 0
            return prediction

        if label == self.pending_label:
            self.pending_count += 1
        else:
            self.pending_label = label
            self.pending_count = 1

        if self.pending_count >= self.confirm_count:
            self.current_label = label
            self.pending_label = None
            self.pending_count = 0
            return prediction

        current_scores = prediction.scores
        current_confidence = current_scores.get(self.current_label, prediction.confidence)
        return EmotionPrediction(
            label=self.current_label,
            confidence=float(current_confidence),
            scores=prediction.scores,
            source="stable_confirming",
        )

    def votes(self) -> dict[str, int]:
        counts = Counter(prediction.label for prediction in self.history)
        return dict(counts)
