"""Real OpenRouter integration smoke test for post-processing.

Self-skipping unless explicitly enabled via environment variables so it never
runs in the ordinary PR gate.
"""

from __future__ import annotations

import logging
import os

import pytest

RUN_FLAG = "VEEPER_RUN_OPENROUTER_INTEGRATION"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"


def _should_skip_openrouter_integration() -> bool:
    flag = os.getenv(RUN_FLAG)
    key = os.getenv("OPENROUTER_API_KEY", "")
    return flag != "1" or not key.strip()


@pytest.mark.skipif(
    _should_skip_openrouter_integration(),
    reason=f"Set {RUN_FLAG}=1 and OPENROUTER_API_KEY to run this test",
)
def test_openrouter_post_processing_preserves_sentinels(caplog) -> None:
    from veespeech.post_processing.openrouter import OpenRouterClient
    from veespeech.post_processing.prompts import build_text_enhancement_messages

    api_key = os.environ["OPENROUTER_API_KEY"].strip()
    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
    timeout = 60.0

    # Safe synthetic text with explicit sentinels to verify preservation.
    text = (
        "This is a synthetic integration test sentence. "
        "It mentions the name Alice, the number 42, and the year 2026. "
    ) * 6

    client = OpenRouterClient()

    with caplog.at_level(logging.DEBUG, logger="veespeech.post_processing.openrouter"):
        result = client.enhance_text(
            text=text,
            api_key=api_key,
            model=model,
            timeout=timeout,
        )

    assert isinstance(result, str)
    assert result.strip()
    assert "Alice" in result
    assert "42" in result
    assert "2026" in result

    # The raw prompt must be built from the stripped input but sentinels are still present.
    messages = build_text_enhancement_messages(text)
    user_content = messages[-1]["content"]
    assert "Alice" in user_content
    assert "42" in user_content
    assert "2026" in user_content

    assert api_key not in caplog.text
    assert user_content not in caplog.text
