from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .openrouter import DEFAULT_TEXT_ENHANCEMENT_MODEL, OpenRouterClient

MIN_TEXT_ENHANCEMENT_LENGTH = 100
DEFAULT_TEXT_ENHANCEMENT_TIMEOUT_SECONDS = 30.0


@dataclass(slots=True)
class TextEnhancementSettings:
    """Configuration for the text enhancement pass.

    Parameters:
        enabled: Whether enhancement is enabled at all.
        api_key: OpenRouter API key. Blank values are treated as missing.
        model: Model identifier passed to OpenRouter.
        min_length: Minimum stripped text length that triggers enhancement.
        timeout: HTTP timeout in seconds.
    """

    enabled: bool = True
    api_key: str | None = None
    model: str = DEFAULT_TEXT_ENHANCEMENT_MODEL
    min_length: int = MIN_TEXT_ENHANCEMENT_LENGTH
    timeout: float = DEFAULT_TEXT_ENHANCEMENT_TIMEOUT_SECONDS

    @property
    def openrouter_api_key(self) -> str | None:
        return self.api_key

    @property
    def text_enhancement_enabled(self) -> bool:
        return self.enabled

    @property
    def text_enhancement_model(self) -> str:
        return self.model


class TextEnhancementService:
    """Orchestrates gating and fallback for the text enhancement step."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        client_cls: type[Any] = OpenRouterClient,
        logger: logging.Logger | None = None,
    ) -> None:
        self._client = client
        self._client_cls = client_cls
        self._logger = logger or logging.getLogger(__name__)

    def enhance(self, text: str, settings: TextEnhancementSettings | Any) -> str:
        """Return enhanced ``text`` when appropriate, otherwise ``text``.

        The original object is returned unchanged whenever enhancement is
        skipped or fails, so callers can rely on identity checks for the
        fallback path.
        """
        original_text = text
        stripped_text = text.strip()
        min_length = _normalize_min_length(_get_setting(settings, "min_length", default=MIN_TEXT_ENHANCEMENT_LENGTH))
        if len(stripped_text) <= min_length:
            return original_text

        enabled = _get_setting(settings, "enabled", "text_enhancement_enabled", default=True)
        if not bool(enabled):
            return original_text

        api_key = _normalize_api_key(_get_setting(settings, "api_key", "openrouter_api_key", default=None))
        if not api_key:
            return original_text

        model = _normalize_model(
            _get_setting(settings, "model", "text_enhancement_model", default=DEFAULT_TEXT_ENHANCEMENT_MODEL)
        )
        timeout = _get_setting(settings, "timeout", default=DEFAULT_TEXT_ENHANCEMENT_TIMEOUT_SECONDS)

        try:
            client = self._client if self._client is not None else self._client_cls()
            enhanced = client.enhance_text(
                text=text,
                api_key=api_key,
                model=model,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Text enhancement failed (%s) for text length %d",
                exc.__class__.__name__,
                len(stripped_text),
            )
            return original_text

        if not isinstance(enhanced, str):
            self._logger.warning("Text enhancement returned invalid content for text length %d", len(stripped_text))
            return original_text

        enhanced = enhanced.strip()
        if not enhanced:
            self._logger.warning("Text enhancement returned invalid content for text length %d", len(stripped_text))
            return original_text

        return enhanced


def _get_setting(settings: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(settings, name):
            return getattr(settings, name)
    return default


def _normalize_api_key(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _normalize_model(value: Any) -> str:
    if not isinstance(value, str):
        return DEFAULT_TEXT_ENHANCEMENT_MODEL
    stripped = value.strip()
    return stripped or DEFAULT_TEXT_ENHANCEMENT_MODEL


def _normalize_min_length(value: Any) -> int:
    if isinstance(value, int) and value >= 0:
        return value
    return MIN_TEXT_ENHANCEMENT_LENGTH


__all__ = [
    "DEFAULT_TEXT_ENHANCEMENT_TIMEOUT_SECONDS",
    "MIN_TEXT_ENHANCEMENT_LENGTH",
    "TextEnhancementService",
    "TextEnhancementSettings",
]
