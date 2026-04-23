from .fusion import EmotionFusion, EmotionPrediction

__all__ = [
    "EmotionFusion",
    "EmotionPrediction",
    "FacialEmotionRecognizer",
    "TextEmotionRecognizer",
]


def __getattr__(name: str):
    if name == "FacialEmotionRecognizer":
        from .facial import FacialEmotionRecognizer

        return FacialEmotionRecognizer
    if name == "TextEmotionRecognizer":
        from .text import TextEmotionRecognizer

        return TextEmotionRecognizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
