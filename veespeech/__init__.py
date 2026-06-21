"""Пакет распознавания речи.

Экспортирует основной интерфейс и реализации Whisper и FasterWhisper.
Тяжёлые зависимости импортируются лениво, чтобы лёгкие модули (например,
``post_processing``) можно было использовать без ``torch``/``whisper``.
"""

from __future__ import annotations

__all__ = [
    "AudioProcessingError",
    "FasterWhisperRecognizer",
    "ModelLoadError",
    "SpeechRecognitionError",
    "SpeechRecognizer",
    "WhisperRecognizer",
]


def __getattr__(name: str) -> object:
    if name == "AudioProcessingError":
        from .exceptions import AudioProcessingError

        return AudioProcessingError
    if name == "ModelLoadError":
        from .exceptions import ModelLoadError

        return ModelLoadError
    if name == "SpeechRecognitionError":
        from .exceptions import SpeechRecognitionError

        return SpeechRecognitionError
    if name == "SpeechRecognizer":
        from .models.model_base import SpeechRecognizer

        return SpeechRecognizer
    if name == "FasterWhisperRecognizer":
        from .models.faster_whisper import FasterWhisperRecognizer

        return FasterWhisperRecognizer
    if name == "WhisperRecognizer":
        from .models.whisper import WhisperRecognizer

        return WhisperRecognizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
