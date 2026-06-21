from __future__ import annotations

SYSTEM_PROMPT = (
    "Improve the readability of the user's text only. "
    "Remove filler words, repeated words, and awkward repetition. "
    "Fix punctuation, sentence breaks, and paragraph breaks when useful. "
    "Preserve the original language, meaning, facts, numbers, and names. "
    "Introduce no new facts, details, or opinions. "
    "Return only the final text."
)


def build_text_enhancement_messages(text: str) -> list[dict[str, str]]:
    """Build the system/user message pair for text enhancement.

    Parameters:
        text: The raw recognized text. It is stripped before being placed in
            the user message.

    Returns:
        A list of message dicts suitable for an OpenAI-compatible chat API.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text.strip()},
    ]


__all__ = ["SYSTEM_PROMPT", "build_text_enhancement_messages"]
