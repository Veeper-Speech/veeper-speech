"""Тесты для утилит работы с файлами."""

from veespeech.file_utils import guess_audio_suffix


def test_guess_audio_suffix_ogg() -> None:
    """Тестирует определение OGG формата."""
    # OGG начинается с "OggS"
    ogg_header = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00"
    assert guess_audio_suffix(ogg_header) == ".ogg"


def test_guess_audio_suffix_wav() -> None:
    """Тестирует определение WAV формата."""
    # WAV начинается с "RIFF" и содержит "WAVE"
    wav_header = b"RIFF\x00\x00\x00\x00WAVE"
    assert guess_audio_suffix(wav_header) == ".wav"


def test_guess_audio_suffix_mp3_with_id3() -> None:
    """Тестирует определение MP3 формата с ID3 тегом."""
    # MP3 с ID3 тегом
    mp3_header = b"ID3\x03\x00\x00\x00\x00\x00\x00"
    assert guess_audio_suffix(mp3_header) == ".mp3"


def test_guess_audio_suffix_mp3_sync() -> None:
    """Тестирует определение MP3 формата через sync byte."""
    # MP3 sync pattern (0xFF, 0xFB)
    mp3_sync = b"\xff\xfb\x00\x00"
    assert guess_audio_suffix(mp3_sync) == ".mp3"


def test_guess_audio_suffix_unknown() -> None:
    """Тестирует fallback для неизвестного формата."""
    # Неизвестный формат должен вернуть .wav по умолчанию
    unknown_data = b"\x00\x00\x00\x00\x00\x00\x00\x00"
    assert guess_audio_suffix(unknown_data) == ".wav"


def test_guess_audio_suffix_empty() -> None:
    """Тестирует обработку пустых данных."""
    # Пустые данные должны вернуть .wav по умолчанию
    empty_data = b""
    assert guess_audio_suffix(empty_data) == ".wav"


def test_guess_audio_suffix_priority() -> None:
    """Тестирует приоритет определения форматов."""
    # OGG должен определяться первым (если есть сигнатура)
    # Создаём данные, которые содержат "OggS" в начале
    ogg_like_data = b"OggS" + b"\xff\xfb\x00\x00"  # OGG + MP3-like
    assert guess_audio_suffix(ogg_like_data) == ".ogg"

    # MP3 ID3 должен определяться перед sync byte
    mp3_id3_data = b"ID3\xff\xfb\x00\x00"
    assert guess_audio_suffix(mp3_id3_data) == ".mp3"
