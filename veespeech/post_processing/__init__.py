"""Post-processing utilities for speech recognition output.

Exports the public API used to enhance recognized text via OpenRouter.
"""

from __future__ import annotations

from .exceptions import PostProcessingError
from .openrouter import (
    DEFAULT_TEXT_ENHANCEMENT_MODEL,
    OPENROUTER_CHAT_COMPLETIONS_URL,
    OPENROUTER_URL,
    OpenRouterClient,
)
from .prompts import SYSTEM_PROMPT, build_text_enhancement_messages
from .service import (
    DEFAULT_TEXT_ENHANCEMENT_TIMEOUT_SECONDS,
    MIN_TEXT_ENHANCEMENT_LENGTH,
    TextEnhancementService,
    TextEnhancementSettings,
)

__all__ = [
    "DEFAULT_TEXT_ENHANCEMENT_MODEL",
    "DEFAULT_TEXT_ENHANCEMENT_TIMEOUT_SECONDS",
    "MIN_TEXT_ENHANCEMENT_LENGTH",
    "OPENROUTER_CHAT_COMPLETIONS_URL",
    "OPENROUTER_URL",
    "OpenRouterClient",
    "PostProcessingError",
    "SYSTEM_PROMPT",
    "TextEnhancementService",
    "TextEnhancementSettings",
    "build_text_enhancement_messages",
]
