# Veeper Speech Recognition (veespeech)

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#)
[![Coverage](https://img.shields.io/badge/coverage-80.41%25-brightgreen)](#)
[![Bandit](https://img.shields.io/badge/Bandit-passing-brightgreen)](#)
[![Safety](https://img.shields.io/badge/Safety-passing-brightgreen)](#)
[![PyPI](https://img.shields.io/pypi/v/veeper-speech?label=PyPI&logo=pypi)](https://pypi.org/project/veeper-speech/)

Небольшая библиотека для распознавания речи Veeper. Встраивается в клиентскую и серверную часть приложения и предоставляет единый интерфейс `SpeechRecognizer` с реализациями на базе OpenAI Whisper и Faster-Whisper.

## Установка

```bash
pip install veeper-speech
```

## Пример использования

```python
from pathlib import Path
from veespeech import WhisperRecognizer, FasterWhisperRecognizer

wav_bytes = Path("path/to/audio.wav").read_bytes()

# Вариант 1: OpenAI Whisper
whisper = WhisperRecognizer(model_name="tiny", language="ru", device="cpu")
text = whisper.recognize(wav_bytes)
print(text)

# Вариант 2: Faster-Whisper
fw = FasterWhisperRecognizer(model_name="tiny", language="ru", device="cpu")
text = fw.recognize(wav_bytes)
print(text)
```

Опционально можно указать директорию для весов через параметр `weights_directory` у распознавателей.

## Поддерживаемые бэкенды

- Whisper (`WhisperRecognizer`)
- Faster-Whisper (`FasterWhisperRecognizer`)

## Запуск тестов

```bash
tox
```

## Лицензия

MIT. См. файл `LICENSE`.
