import sys
from pathlib import Path

# Позволяет запускать тесты без установки пакета в окружение.
# Добавляем корень подпроекта `speech_recognition` в sys.path,
# чтобы можно было импортировать модуль `veespeech`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
