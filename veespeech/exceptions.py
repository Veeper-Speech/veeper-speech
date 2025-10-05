class SpeechRecognitionError(Exception):
    """Базовое исключение для ошибок распознавания речи."""

    pass


class ModelLoadError(SpeechRecognitionError):
    """Исключение при ошибке загрузки модели."""

    pass


class AudioProcessingError(SpeechRecognitionError):
    """Исключение при ошибке обработки аудио."""

    pass
