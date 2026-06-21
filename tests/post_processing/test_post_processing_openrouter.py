"""Unit tests for the OpenRouter post-processing client."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pytest

from veespeech.post_processing.exceptions import PostProcessingError
from veespeech.post_processing.openrouter import OPENROUTER_CHAT_COMPLETIONS_URL, OpenRouterClient
from veespeech.post_processing.prompts import SYSTEM_PROMPT, build_text_enhancement_messages

FAKE_KEY = "sk-test-not-real"


class FakeResponse:
    """Minimal ``requests.Response`` stand-in."""

    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@dataclass
class FakeHttpRecorder:
    """Records calls and returns a canned response."""

    response: FakeResponse
    calls: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(("POST", url, kwargs))
        return self.response


@dataclass
class FailingHttpRecorder:
    """Records calls and raises a canned exception."""

    exception: Exception
    calls: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(("POST", url, kwargs))
        raise self.exception


def _assert_no_leaks(caplog_text: str, secret: str, full_text: str) -> None:
    """Ensure no API keys or raw transcripts appear in captured logs."""
    assert secret not in caplog_text
    assert full_text not in caplog_text


class TestPromptContract:
    """PP-01 prompt contract."""

    def test_system_prompt_requires_cleanup_and_preservation(self) -> None:
        required = [
            "filler",
            "repetition",
            "punctuation",
            "paragraph",
            "language",
            "meaning",
            "facts",
            "names",
            "numbers",
            "no new facts",
            "Return only the final text",
        ]
        lowered = SYSTEM_PROMPT.lower()
        for fragment in required:
            assert fragment.lower() in lowered, f"Missing prompt requirement: {fragment}"

    def test_build_messages_use_stripped_input(self) -> None:
        raw = "  \n\n  filler and repeated words here  \n\n  "
        messages = build_text_enhancement_messages(raw)

        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert messages[1] == {"role": "user", "content": raw.strip()}


class TestMockedOpenRouterHappyPath:
    """PP-02 mocked OpenRouter happy path."""

    def test_client_posts_expected_request_and_extracts_content(self, caplog) -> None:
        text = f"  {'x' * 101}  "
        full_text = text.strip()
        model = "deepseek/deepseek-v4-flash"
        timeout = 12.5

        response = FakeResponse({"choices": [{"message": {"content": "  polished result  "}}]})
        session = FakeHttpRecorder(response)
        client = OpenRouterClient(session=session)

        with caplog.at_level(logging.DEBUG, logger="veespeech.post_processing.openrouter"):
            result = client.enhance_text(
                text=text,
                api_key=FAKE_KEY,
                model=model,
                timeout=timeout,
            )

        assert result == "polished result"

        assert len(session.calls) == 1
        method, url, kwargs = session.calls[0]
        assert method == "POST"
        assert url == OPENROUTER_CHAT_COMPLETIONS_URL
        assert kwargs["timeout"] == timeout

        headers = kwargs["headers"]
        assert headers["Authorization"] == f"Bearer {FAKE_KEY}"
        assert headers["Content-Type"] == "application/json"

        payload = kwargs["json"]
        assert payload["model"] == model
        assert payload["temperature"] == 0.2
        assert payload["reasoning"] == {"effort": "none", "exclude": True}

        messages = payload["messages"]
        assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert messages[-1]["content"] == full_text

        _assert_no_leaks(caplog.text, FAKE_KEY, full_text)


class TestFailureAndMalformedResponses:
    """PP-03 failures/malformed responses."""

    @pytest.mark.parametrize(
        "recorder_factory,description",
        [
            (lambda: FakeHttpRecorder(FakeResponse({"error": "boom"}, status_code=500)), "HTTP error"),
            (
                lambda: FailingHttpRecorder(RuntimeError(f"network timeout for {FAKE_KEY} and {('x' * 101).strip()}")),
                "request error",
            ),
            (
                lambda: FakeHttpRecorder(FakeResponse(ValueError(f"bad json containing {FAKE_KEY}"))),
                "invalid JSON",
            ),
            (lambda: FakeHttpRecorder(FakeResponse({})), "missing content"),
            (lambda: FakeHttpRecorder(FakeResponse({"choices": []})), "empty choices"),
            (lambda: FakeHttpRecorder(FakeResponse({"choices": [{}]})), "missing message"),
            (lambda: FakeHttpRecorder(FakeResponse({"choices": [{"message": {}}]})), "missing content"),
            (lambda: FakeHttpRecorder(FakeResponse({"choices": [{"message": {"content": ""}}]})), "empty content"),
            (
                lambda: FakeHttpRecorder(FakeResponse({"choices": [{"message": {"content": "   "}}]})),
                "whitespace content",
            ),
            (
                lambda: FakeHttpRecorder(FakeResponse({"choices": [{"message": {"content": 123}}]})),
                "non-string content",
            ),
        ],
        ids=[
            "http",
            "request",
            "json",
            "missing",
            "empty_choices",
            "missing_message",
            "missing_content",
            "empty_content",
            "whitespace_content",
            "nonstring_content",
        ],
    )
    def test_client_raises_post_processing_error_and_does_not_leak_data(
        self, recorder_factory, description, caplog
    ) -> None:
        text = f"  {'y' * 101}  "
        full_text = text.strip()
        model = "deepseek/deepseek-v4-flash"

        client = OpenRouterClient(session=recorder_factory())

        with pytest.raises(PostProcessingError):
            with caplog.at_level(logging.DEBUG, logger="veespeech.post_processing.openrouter"):
                client.enhance_text(
                    text=text,
                    api_key=FAKE_KEY,
                    model=model,
                    timeout=10.0,
                )

        _assert_no_leaks(caplog.text, FAKE_KEY, full_text)
