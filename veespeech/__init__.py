"""Пакет распознавания речи.

Экспортирует основной интерфейс и реализации Whisper и FasterWhisper.
"""

from .exceptions import AudioProcessingError, ModelLoadError, SpeechRecognitionError
from .faster_whisper import FasterWhisperRecognizer
from .model_base import SpeechRecognizer
from .whisper import WhisperRecognizer

__all__ = [
    "AudioProcessingError",
    "FasterWhisperRecognizer",
    "ModelLoadError",
    "SpeechRecognitionError",
    "SpeechRecognizer",
    "WhisperRecognizer",
]
