"""Пакет распознавания речи.

Экспортирует основной интерфейс и реализации Whisper и FasterWhisper.
"""

from .exceptions import AudioProcessingError, ModelLoadError, SpeechRecognitionError
from .models.faster_whisper import FasterWhisperRecognizer
from .models.model_base import SpeechRecognizer
from .models.whisper import WhisperRecognizer

__all__ = [
    "AudioProcessingError",
    "FasterWhisperRecognizer",
    "ModelLoadError",
    "SpeechRecognitionError",
    "SpeechRecognizer",
    "WhisperRecognizer",
]
