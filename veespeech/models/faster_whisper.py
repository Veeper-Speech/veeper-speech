import logging
from pathlib import Path
from typing import Any, Optional

import torch
from faster_whisper import WhisperModel

from veespeech.exceptions import ModelLoadError
from veespeech.file_utils import temporary_audio_file

from .model_base import SpeechRecognizer

logger = logging.getLogger(__name__)

# Константы
DEFAULT_MODEL_NAME = "tiny"
DEFAULT_BEAM_SIZE = 5
DEFAULT_TEMPERATURE = 0.0


class FasterWhisperRecognizer(SpeechRecognizer):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        language: Optional[str] = None,
        device: Optional[str] = None,
        weights_directory: Optional[str | Path] = None,
    ) -> None:
        """Инициализирует распознаватель FasterWhisper.

        Parameters:
            model_name: Имя модели Whisper (например, "tiny", "base", "small").
            language: Язык распознавания (например, "ru", "en"). По умолчанию
                None, в этом случае будет использован язык из аудиофайла.
            device: Устройство для загрузки модели (например, "cuda", "cpu").
                По умолчанию используется "cuda" если оно доступно, иначе
                "cpu".
            weights_directory: Директория для загрузки весов модели.
                По умолчанию None, в этом случае используется директория
                по умолчанию faster-whisper.
        """
        self._model_name = model_name
        self._language = language
        self._device = self._determine_device(device)
        self._weights_directory = weights_directory

        # Предварительно загружаем модель при инициализации
        try:
            self._model = _get_or_load_model(
                model_name=self._model_name,
                device=self._device,
                weights_directory=self._weights_directory,
            )
            logger.info(
                "FasterWhisperRecognizer инициализирован с моделью '%s' на устройстве '%s'",
                self._model_name,
                self._device,
            )
        except Exception as e:
            raise ModelLoadError(
                f"Не удалось загрузить модель FasterWhisper '{self._model_name}' "
                f"на устройстве '{self._device}': {e}"
            ) from e

    def _determine_device(self, device: Optional[str]) -> str:
        """Определяет устройство для загрузки модели.

        Parameters:
            device: Предпочтительное устройство или None для автоопределения.

        Returns:
            str: Определенное устройство ("cuda" или "cpu").
        """
        if device:
            return device

        # Проверяем доступность CUDA более безопасно
        try:
            cuda_available = hasattr(torch, "cuda") and torch.cuda.is_available() and torch.cuda.device_count() > 0
            return "cuda" if cuda_available else "cpu"
        except Exception:
            # Если есть проблемы с CUDA, принудительно используем CPU
            logger.warning("Ошибка при проверке CUDA, используем CPU")
            return "cpu"

    def _create_transcription_params(self) -> dict[str, Any]:
        """Создает параметры для транскрипции.

        Returns:
            dict: Словарь с параметрами для whisper.transcribe().
        """
        return {
            "beam_size": DEFAULT_BEAM_SIZE,
            "temperature": DEFAULT_TEMPERATURE,
            "condition_on_previous_text": False,
            "language": self._language,
        }

    def _extract_text_from_result(self, segments: list[Any]) -> str:
        """Извлекает текст из результата транскрипции.

        Parameters:
            segments: Список сегментов от faster_whisper.

        Returns:
            str: Извлеченный и очищенный текст.
        """
        if not segments:
            return ""

        # Объединяем текст из всех сегментов
        text_parts = []
        for segment in segments:
            if hasattr(segment, "text") and segment.text:
                text_parts.append(segment.text.strip())

        return " ".join(text_parts).strip()

    def recognize(self, audio_data: bytes) -> str:
        """Распознаёт речь из аудио байтов, используя faster-whisper.

        Parameters:
            audio_data: Сырые байты аудио. Поддерживаются WAV, MP3, OGG/Opus
                и другие форматы, поддерживаемые ffmpeg.

        Returns:
            str: Распознанный текст.

        Raises:
            AudioProcessingError: если входные аудиоданные пустые.
            ModelLoadError: если не удалось загрузить модель.
            SpeechRecognitionError: при других ошибках распознавания.
        """
        logger.debug("Начинаем распознавание аудио размером %d байт", len(audio_data))

        with temporary_audio_file(audio_data) as tmp_path:
            decode_kwargs = self._create_transcription_params()
            logger.debug(
                "Запускаем транскрипцию с параметрами: beam_size=%s, language=%s",
                decode_kwargs.get("beam_size"),
                decode_kwargs.get("language"),
            )

            segments, _ = self._model.transcribe(tmp_path, **decode_kwargs)
            text = self._extract_text_from_result(list(segments))

            logger.debug("Распознавание завершено, длина текста: %d символов", len(text))
            return text

    def cleanup(self) -> None:
        """Очищает ресурсы, занятые распознавателем.

        Удаляет модель из глобального кеша и очищает память.
        """
        # Удаляем модель из глобального кеша, если она там есть
        weights_dir_str = str(self._weights_directory) if self._weights_directory else None
        key = (self._model_name, self._device, weights_dir_str)

        if key in _CACHED_MODELS:
            try:
                # Очищаем кеш CUDA, если модель была на GPU
                if self._device == "cuda":
                    try:
                        torch.cuda.empty_cache()
                        logger.debug("Очищен кеш CUDA для модели '%s'", self._model_name)
                    except Exception as e:
                        logger.warning("Не удалось очистить кеш CUDA: %s", e)

                del _CACHED_MODELS[key]
                logger.debug("Модель '%s' удалена из кеша", self._model_name)
            except Exception as e:
                logger.warning("Ошибка при очистке модели '%s': %s", self._model_name, e)

        # Очищаем ссылку на модель в экземпляре
        self._model = None

    @classmethod
    def prepare_model(cls, model_name: str, device: str, weights_directory: str | Path) -> WhisperModel:
        """Подготавливает модель FasterWhisper для использования.

        Parameters:
            model_name: Имя модели Whisper.
            device: Устройство для загрузки ("cpu" или "cuda").
            weights_directory: Директория с весами модели.

        Returns:
            WhisperModel: Загруженная модель FasterWhisper.
        """
        return _get_or_load_model(model_name=model_name, device=device, weights_directory=weights_directory)


# --- Внутренняя реализация кеширования модели ---
_CACHED_MODELS: dict[tuple[str, str, Optional[str]], WhisperModel] = {}


def _get_or_load_model(model_name: str, device: str, weights_directory: Optional[str | Path] = None) -> WhisperModel:
    """Возвращает кэшированную модель FasterWhisper или загружает новую.

    Parameters:
        model_name: Имя модели Whisper.
        device: Устройство загрузки ("cpu" или "cuda").
        weights_directory: Директория для загрузки весов модели.

    Returns:
        WhisperModel: Экземпляр загруженной модели FasterWhisper.

    Raises:
        ModelLoadError: Если не удалось загрузить модель.
    """
    # Преобразуем Path в строку для ключа кэша
    weights_dir_str = str(weights_directory) if weights_directory else None
    key = (model_name, device, weights_dir_str)
    model = _CACHED_MODELS.get(key)

    if model is None:
        logger.info("Загружаем модель FasterWhisper '%s' на устройство '%s'", model_name, device)
        if weights_directory:
            logger.info("Используем директорию для весов: '%s'", weights_directory)
        try:
            # Формируем параметры для загрузки модели
            load_params = {"device": device}
            if weights_directory:
                load_params["download_root"] = str(weights_directory)

            model = WhisperModel(model_name, **load_params)
            _CACHED_MODELS[key] = model
            logger.debug("Модель '%s' успешно загружена и закеширована", model_name)
        except Exception as e:
            raise ModelLoadError(
                f"Не удалось загрузить модель FasterWhisper '{model_name}' " f"на устройстве '{device}': {e}"
            ) from e
    else:
        logger.debug("Используем закешированную модель '%s' с устройства '%s'", model_name, device)
    return model
