from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class SpeechRecognizer(ABC):
    """Базовый класс-интерфейс моделей распознавания речи."""

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @abstractmethod
    def recognize(self, audio_data: bytes) -> str:
        """Распознаёт речь из аудиоданных.

        Parameters:
            audio_data: Сырые байты аудио.

        Returns:
            str: Распознанный текст.

        Raises:
            AudioProcessingError: если входные аудиоданные пустые.
            ModelLoadError: если не удалось загрузить модель.
            SpeechRecognitionError: при других ошибках распознавания.
        """
        raise NotImplementedError

    @classmethod
    def prepare_model(cls, model_name: str, device: str, weights_directory: str | Path) -> Any:
        """Подготавливает модель для использования.

        Parameters:
            model_name: Имя модели.
            device: Устройство для использования.
            weights_direcotry: Путь к директории с весами модели.
        """
        raise NotImplementedError

    def cleanup(self) -> None:
        """Очищает ресурсы, занятые распознавателем.

        Этот метод должен быть вызван перед удалением экземпляра распознавателя
        для освобождения памяти и других ресурсов (например, выгрузки моделей из GPU).
        """
        pass
