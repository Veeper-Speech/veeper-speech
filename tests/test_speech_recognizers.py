import warnings
from pathlib import Path
from typing import Type

import pytest
import torch
import torch.nn

# Импортируем типы для тестов prepare_model
from faster_whisper import WhisperModel

from veespeech import AudioProcessingError, FasterWhisperRecognizer, ModelLoadError, WhisperRecognizer
from veespeech.models.model_base import SpeechRecognizer


def _read_audio_bytes(audio_file: str) -> bytes:
    assets_dir = Path(__file__).resolve().parent / "assets"
    path = assets_dir / audio_file
    assert path.exists(), f"Файл не найден: {path}"
    return path.read_bytes()


def _validate_counting_content(text: str) -> None:
    """Проверяет, что распознанный текст содержит слова или числа счета.

    Parameters:
        text: Распознанный текст для проверки.

    Raises:
        AssertionError: Если текст не содержит слов счета.
    """
    text_lower = text.lower()
    counting_words = [
        "раз",
        "два",
        "три",
        "четыре",
        "пять",
        "1",
        "2",
        "3",
        "4",
        "5",
        "one",
        "two",
        "three",
        "four",
        "five",
    ]
    assert any(
        word in text_lower for word in counting_words
    ), f"Распознанный текст должен содержать слова счета. Получен: '{text}'"


@pytest.mark.parametrize(
    "recognizer_class, audio_file",
    [
        (WhisperRecognizer, "count_ru.wav"),
        (WhisperRecognizer, "count_en.wav"),
        (FasterWhisperRecognizer, "count_ru.wav"),
        (FasterWhisperRecognizer, "count_en.wav"),
    ],
)
def test_recognizer_output_validation(recognizer_class: Type[SpeechRecognizer], audio_file: str) -> None:
    """Тестирует, что распознанный текст содержит корректное содержимое (слова счета)."""
    wav_bytes = _read_audio_bytes(audio_file)
    recognizer = recognizer_class()
    text = recognizer.recognize(wav_bytes)

    assert isinstance(text, str), "Результат должен быть строкой."
    assert text.strip() != "", "Распознанный текст пуст."

    # Используем общую функцию для проверки содержимого
    _validate_counting_content(text)

    print(f"{recognizer_class.__name__}: {text}")


@pytest.mark.parametrize(
    "recognizer_class",
    [WhisperRecognizer, FasterWhisperRecognizer],
)
def test_recognizer_raises_error_on_wrong_model_name(recognizer_class: Type[SpeechRecognizer]) -> None:
    """Тестирует обработку ошибок при неправильном имени модели."""
    with pytest.raises(ModelLoadError):
        recognizer_class(model_name="wrong")


@pytest.mark.parametrize(
    "recognizer_class",
    [WhisperRecognizer, FasterWhisperRecognizer],
)
def test_recognizer_raises_error_on_empty_audio(recognizer_class: Type[SpeechRecognizer]) -> None:
    """Тестирует обработку ошибок при пустых аудиоданных."""
    with pytest.raises(AudioProcessingError):
        recognizer_class().recognize(b"")


@pytest.mark.parametrize(
    "recognizer_class, language, audio_file",
    [
        (WhisperRecognizer, "ru", "count_ru.wav"),
        (WhisperRecognizer, "en", "count_en.wav"),
        (WhisperRecognizer, None, "count_ru.wav"),
        (FasterWhisperRecognizer, "ru", "count_ru.wav"),
        (FasterWhisperRecognizer, "en", "count_en.wav"),
        (FasterWhisperRecognizer, None, "count_ru.wav"),
    ],
)
def test_languages(recognizer_class: Type[SpeechRecognizer], language: str | None, audio_file: str) -> None:
    """Тестирует распознавание с разными языками."""
    recognizer = recognizer_class(language=language)
    text = recognizer.recognize(_read_audio_bytes(audio_file))
    assert isinstance(text, str), "Результат должен быть строкой."
    assert text.strip() != "", "Распознанный текст пуст."


@pytest.mark.parametrize(
    "recognizer_class, device",
    [
        (WhisperRecognizer, "cpu"),
        (WhisperRecognizer, "cuda"),
        (WhisperRecognizer, None),
        (FasterWhisperRecognizer, "cpu"),
        (FasterWhisperRecognizer, "cuda"),
        (FasterWhisperRecognizer, None),
    ],
)
def test_devices(recognizer_class: Type[SpeechRecognizer], device: str | None) -> None:
    """Тестирует работу на разных устройствах."""
    # Пропускаем тест CUDA, если CUDA недоступна
    if device == "cuda":
        cuda_available = hasattr(torch, "cuda") and torch.cuda.is_available() and torch.cuda.device_count() > 0
        if not cuda_available:
            pytest.skip("CUDA недоступна, пропускаем тест")

    recognizer = recognizer_class(device=device)
    if device == "cpu":
        # При наличии CUDA некоторые бекенды могут издавать предупреждения,
        # когда мы намеренно запускаем инференс на CPU. Глушим их локально.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            text = recognizer.recognize(_read_audio_bytes("count_ru.wav"))
    else:
        text = recognizer.recognize(_read_audio_bytes("count_ru.wav"))
    assert isinstance(text, str), "Результат должен быть строкой."
    assert text.strip() != "", "Распознанный текст пуст."


@pytest.mark.parametrize(
    "recognizer_class, expected_type",
    [
        (WhisperRecognizer, torch.nn.Module),
        (FasterWhisperRecognizer, WhisperModel),
    ],
)
def test_prepare_model_return_type(recognizer_class: Type[SpeechRecognizer], expected_type: Type) -> None:
    """Тестирует, что prepare_model возвращает правильный тип модели."""
    model = recognizer_class.prepare_model(model_name="tiny", device="cpu", weights_directory=Path("/tmp"))
    assert isinstance(model, expected_type), f"prepare_model должен возвращать {expected_type.__name__}"


@pytest.mark.parametrize(
    "recognizer_class",
    [WhisperRecognizer, FasterWhisperRecognizer],
)
def test_prepare_model_caching(recognizer_class: Type[SpeechRecognizer]) -> None:
    """Тестирует кеширование моделей в prepare_model."""
    # Первый вызов
    model1 = recognizer_class.prepare_model(model_name="tiny", device="cpu", weights_directory=Path("/tmp"))

    # Второй вызов с теми же параметрами
    model2 = recognizer_class.prepare_model(model_name="tiny", device="cpu", weights_directory=Path("/tmp"))

    # Должна вернуться та же модель (кеширование работает)
    assert model1 is model2, "prepare_model должен возвращать кэшированную модель"


@pytest.mark.parametrize(
    "recognizer_class",
    [WhisperRecognizer, FasterWhisperRecognizer],
)
def test_prepare_model_error_handling(recognizer_class: Type[SpeechRecognizer]) -> None:
    """Тестирует обработку ошибок в prepare_model при неправильном имени модели."""
    with pytest.raises(ModelLoadError):
        recognizer_class.prepare_model(model_name="wrong_model_name", device="cpu", weights_directory=Path("/tmp"))


@pytest.mark.parametrize(
    "recognizer_class",
    [WhisperRecognizer, FasterWhisperRecognizer],
)
def test_weights_directory_is_used(recognizer_class: Type[SpeechRecognizer], tmp_path: Path) -> None:
    """Тестирует, что веса модели загружаются в указанную директорию.

    Каждый тест получает уникальную временную директорию tmp_path от pytest,
    которая автоматически очищается после завершения теста.
    """
    # Создаем уникальную директорию для весов модели внутри tmp_path
    weights_dir = tmp_path / "test_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Получаем список всех файлов до загрузки модели (должно быть пусто)
    files_before = set(weights_dir.rglob("*"))
    assert len(files_before) == 0, f"Директория {weights_dir} должна быть пустой до загрузки модели"

    # Создаем распознаватель с указанием директории для весов
    recognizer = recognizer_class(model_name="tiny", weights_directory=weights_dir)

    # Получаем список всех файлов после загрузки модели
    files_after = set(weights_dir.rglob("*"))

    # Вычисляем разницу - новые файлы, которые появились
    new_files = files_after - files_before

    # Проверяем, что появились новые файлы
    assert len(new_files) > 0, (
        f"В директории {weights_dir} должны появиться новые файлы весов модели. "
        f"До: {len(files_before)} файлов, После: {len(files_after)} файлов"
    )

    # Проверяем, что среди новых файлов есть обычные файлы (не только директории)
    regular_files = [f for f in new_files if f.is_file()]
    assert len(regular_files) > 0, (
        f"Среди новых файлов должны быть обычные файлы (весовые коэффициенты). "
        f"Найдено {len(regular_files)} обычных файлов из {len(new_files)} новых элементов"
    )

    text = recognizer.recognize(_read_audio_bytes("count_ru.wav"))
    assert isinstance(text, str), "Результат должен быть строкой."
    assert text.strip() != "", "Распознанный текст не должен быть пустым."
