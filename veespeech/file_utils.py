import logging
import os
import tempfile
from contextlib import contextmanager
from typing import Generator

from .exceptions import AudioProcessingError

logger = logging.getLogger(__name__)


@contextmanager
def temporary_audio_file(audio_data: bytes) -> Generator[str, None, None]:
    """Контекстный менеджер для создания временного аудиофайла.

    Parameters:
        audio_data: Сырые байты аудио.

    Yields:
        str: Путь к временному файлу.

    Raises:
        AudioProcessingError: Если аудиоданные пустые.
    """
    if not audio_data:
        raise AudioProcessingError("Пустые аудиоданные переданы в recognize().")

    suffix = guess_audio_suffix(audio_data)
    logger.debug("Определен формат аудио: %s", suffix)

    tmp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp_file.write(audio_data)
        tmp_file.flush()
        yield tmp_file.name
    finally:
        tmp_file.close()
        try:
            os.remove(tmp_file.name)
        except Exception as e:
            logger.warning("Не удалось удалить временный файл %s: %s", tmp_file.name, e)


# Константы для определения формата аудио
ID3_TAG_PREFIX = b"ID3"
RIFF_HEADER = b"RIFF"
WAVE_HEADER = b"WAVE"
OGG_HEADER = b"OggS"
MP3_SYNC_BYTE = 0xFF
MP3_SYNC_MASK = 0xE0


def guess_audio_suffix(audio_bytes: bytes) -> str:
    """Определяет расширение файла по сигнатуре аудиобайтов.

    Parameters:
        audio_bytes: Сырые байты аудиофайла.

    Returns:
        str: Рекомендуемое расширение (".mp3", ".wav" или ".ogg").
    """
    # OGG header: "OggS"
    if len(audio_bytes) >= 4 and audio_bytes[:4] == OGG_HEADER:
        return ".ogg"
    # MP3 может содержать ID3-тэг ("ID3" в начале) или frame sync (0xFFEx)
    if len(audio_bytes) >= 3 and audio_bytes[:3] == ID3_TAG_PREFIX:
        return ".mp3"
    if len(audio_bytes) >= 2:
        b0, b1 = audio_bytes[0], audio_bytes[1]
        if b0 == MP3_SYNC_BYTE and (b1 & MP3_SYNC_MASK) == MP3_SYNC_MASK:
            return ".mp3"
    # WAV RIFF header: "RIFF" .... "WAVE"
    if len(audio_bytes) >= 12 and audio_bytes[:4] == RIFF_HEADER and audio_bytes[8:12] == WAVE_HEADER:
        return ".wav"
    # По умолчанию отдаём WAV, ffmpeg обычно корректно определяет по содержимому
    return ".wav"
