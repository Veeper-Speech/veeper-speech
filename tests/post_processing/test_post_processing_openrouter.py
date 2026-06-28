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
            "visual context",
            "application",
            "IDE",
            "document editor",
            "chat",
            "communication style",
            "layout",
            "indentation",
            "line breaks",
            "formatting",
            "resolve ambiguity",
            "preserve structure",
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

    def test_data_url_screenshot_produces_multimodal_content(self) -> None:
        raw = "  text with screenshot  "
        data_url = "data:image/png;base64,abc123"
        messages = build_text_enhancement_messages(raw, screenshot=data_url)

        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
        user_message = messages[1]
        assert user_message["role"] == "user"
        content = user_message["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": raw.strip()}
        assert content[1] == {"type": "image_url", "image_url": {"url": data_url}}

    def test_text_only_backward_compatibility(self) -> None:
        raw = "  plain text  "
        messages = build_text_enhancement_messages(raw)

        assert messages[1] == {"role": "user", "content": raw.strip()}
        assert isinstance(messages[1]["content"], str)


class TestMockedOpenRouterHappyPath:
    """PP-02 mocked OpenRouter happy path."""

    def test_client_no_longer_exposes_enhance_text(self) -> None:
        assert not hasattr(OpenRouterClient, "enhance_text")

    def test_client_posts_expected_request_and_extracts_content(self, caplog) -> None:
        text = f"  {'x' * 101}  "
        full_text = text.strip()
        model = "qwen/qwen3.6-flash"
        timeout = 12.5
        temperature = 0.2
        reasoning = {"effort": "none", "exclude": True}

        response = FakeResponse({"choices": [{"message": {"content": "  polished result  "}}]})
        session = FakeHttpRecorder(response)
        client = OpenRouterClient(session=session)

        with caplog.at_level(logging.DEBUG, logger="veespeech.post_processing.openrouter"):
            result = client.call(
                prompt=SYSTEM_PROMPT,
                text=full_text,
                api_key=FAKE_KEY,
                model=model,
                timeout=timeout,
                temperature=temperature,
                reasoning=reasoning,
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
        assert payload["temperature"] == temperature
        assert payload["reasoning"] == reasoning

        messages = payload["messages"]
        assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert messages[-1]["content"] == full_text

        _assert_no_leaks(caplog.text, FAKE_KEY, full_text)

    def test_client_posts_multimodal_payload_when_image_passed(self, caplog) -> None:
        text = f"  {'x' * 101}  "
        full_text = text.strip()
        image = "data:image/png;base64,abc123"
        model = "qwen/qwen3.6-flash"
        timeout = 12.5
        temperature = 0.2
        reasoning = {"effort": "none", "exclude": True}

        response = FakeResponse({"choices": [{"message": {"content": "  polished result  "}}]})
        session = FakeHttpRecorder(response)
        client = OpenRouterClient(session=session)

        with caplog.at_level(logging.DEBUG, logger="veespeech.post_processing.openrouter"):
            result = client.call(
                prompt=SYSTEM_PROMPT,
                text=full_text,
                api_key=FAKE_KEY,
                model=model,
                timeout=timeout,
                image=image,
                temperature=temperature,
                reasoning=reasoning,
            )

        assert result == "polished result"

        assert len(session.calls) == 1
        payload = session.calls[0][2]["json"]
        messages = payload["messages"]
        assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
        assert messages[-1]["role"] == "user"
        content = messages[-1]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": full_text}
        assert content[1] == {"type": "image_url", "image_url": {"url": image}}

        _assert_no_leaks(caplog.text, FAKE_KEY, full_text)

    def test_reasoning_is_copied_not_mutated(self) -> None:
        reasoning = {"effort": "none", "exclude": True}
        response = FakeResponse({"choices": [{"message": {"content": "ok"}}]})
        session = FakeHttpRecorder(response)
        client = OpenRouterClient(session=session)

        client.call(
            prompt=SYSTEM_PROMPT,
            text="some text",
            api_key=FAKE_KEY,
            model="m",
            reasoning=reasoning,
        )

        payload = session.calls[0][2]["json"]
        assert payload["reasoning"] == reasoning
        assert payload["reasoning"] is not reasoning

    def test_temperature_and_reasoning_are_omitted_when_none(self) -> None:
        response = FakeResponse({"choices": [{"message": {"content": "ok"}}]})
        session = FakeHttpRecorder(response)
        client = OpenRouterClient(session=session)

        client.call(
            prompt=SYSTEM_PROMPT,
            text="some text",
            api_key=FAKE_KEY,
            model="m",
        )

        payload = session.calls[0][2]["json"]
        assert "temperature" not in payload
        assert "reasoning" not in payload


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
        model = "qwen/qwen3.6-flash"

        client = OpenRouterClient(session=recorder_factory())

        with pytest.raises(PostProcessingError):
            with caplog.at_level(logging.DEBUG, logger="veespeech.post_processing.openrouter"):
                client.call(
                    prompt=SYSTEM_PROMPT,
                    text=full_text,
                    api_key=FAKE_KEY,
                    model=model,
                    timeout=10.0,
                    temperature=0.2,
                    reasoning={"effort": "none", "exclude": True},
                )

        _assert_no_leaks(caplog.text, FAKE_KEY, full_text)
