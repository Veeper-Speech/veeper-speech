from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import requests

from .exceptions import PostProcessingError
from .prompts import build_text_enhancement_messages

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_CHAT_COMPLETIONS_URL = OPENROUTER_URL
DEFAULT_TEXT_ENHANCEMENT_MODEL = "deepseek/deepseek-v4-flash"
_TEXT_ENHANCEMENT_TEMPERATURE = 0.2
_TEXT_ENHANCEMENT_REASONING = {"effort": "none", "exclude": True}


@dataclass(slots=True)
class OpenRouterClient:
    """Thin client for the OpenRouter chat-completions endpoint.

    Parameters:
        session: An optional ``requests.Session``-like object. If omitted,
            the top-level ``requests`` module is used.
        logger: An optional logger for debug output. No sensitive data is
            logged.
    """

    session: Any | None = None
    logger: logging.Logger | None = None

    def __post_init__(self) -> None:
        if self.logger is None:
            self.logger = logging.getLogger(__name__)

    def enhance_text(
        self,
        *,
        text: str,
        api_key: str,
        model: str = DEFAULT_TEXT_ENHANCEMENT_MODEL,
        timeout: float = 30.0,
    ) -> str:
        """Send ``text`` to OpenRouter and return the enhanced text.

        Raises:
            PostProcessingError: When the request or response is invalid.
        """
        payload = {
            "model": model,
            "messages": build_text_enhancement_messages(text),
            "temperature": _TEXT_ENHANCEMENT_TEMPERATURE,
            "reasoning": _TEXT_ENHANCEMENT_REASONING.copy(),
        }
        response = self._post(api_key=api_key, payload=payload, timeout=timeout)
        return _extract_content(response)

    def _post(self, *, api_key: str, payload: dict[str, Any], timeout: float) -> Any:
        session = self.session if self.session is not None else requests
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = session.post(
                OPENROUTER_CHAT_COMPLETIONS_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001
            raise PostProcessingError(f"OpenRouter request failed ({exc.__class__.__name__})") from None

        raise_for_status = getattr(response, "raise_for_status", None)
        if callable(raise_for_status):
            try:
                raise_for_status()
            except Exception as exc:  # noqa: BLE001
                raise PostProcessingError(f"OpenRouter request failed ({exc.__class__.__name__})") from None

        _validate_status_code(response)
        return response


def _validate_status_code(response: Any) -> None:
    status_code = getattr(response, "status_code", None)
    if status_code is None:
        return

    try:
        status = int(status_code)
    except (TypeError, ValueError):
        raise PostProcessingError("OpenRouter returned an invalid HTTP response") from None

    if status >= 400:
        raise PostProcessingError(f"OpenRouter request failed with HTTP {status}") from None


def _extract_content(response: Any) -> str:
    try:
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        raise PostProcessingError(f"OpenRouter returned invalid JSON ({exc.__class__.__name__})") from None

    if not isinstance(payload, dict):
        raise PostProcessingError("OpenRouter returned malformed JSON")

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise PostProcessingError("OpenRouter response missing choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise PostProcessingError("OpenRouter response missing message content")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise PostProcessingError("OpenRouter response missing message content")

    content = message.get("content")
    if not isinstance(content, str):
        raise PostProcessingError("OpenRouter response missing message content")

    content = content.strip()
    if not content:
        raise PostProcessingError("OpenRouter returned empty content")
    return content


__all__ = [
    "DEFAULT_TEXT_ENHANCEMENT_MODEL",
    "OPENROUTER_CHAT_COMPLETIONS_URL",
    "OPENROUTER_URL",
    "OpenRouterClient",
]
