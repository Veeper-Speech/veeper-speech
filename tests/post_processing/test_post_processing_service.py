"""Unit tests for the post-processing service orchestration and gating."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pytest

from veespeech.post_processing.exceptions import PostProcessingError
from veespeech.post_processing.prompts import SYSTEM_PROMPT
from veespeech.post_processing.service import (
    MIN_TEXT_ENHANCEMENT_LENGTH,
    TextEnhancementService,
    TextEnhancementSettings,
)

FAKE_KEY = "sk-test-not-real"


@dataclass
class RecordedCall:
    prompt: str
    text: str
    api_key: str
    model: str
    timeout: float
    image: str | None
    temperature: float | None
    reasoning: dict[str, Any] | None


@dataclass
class RecordingClient:
    """Fake OpenRouter client that records arguments and returns a canned result."""

    outcome: str | None = None
    exception: Exception | None = None
    calls: list[RecordedCall] = field(default_factory=list)

    def call(
        self,
        *,
        prompt: str,
        text: str,
        api_key: str,
        model: str,
        timeout: float,
        image: str | None = None,
        temperature: float | None = None,
        reasoning: dict[str, Any] | None = None,
    ) -> str | None:
        self.calls.append(
            RecordedCall(
                prompt=prompt,
                text=text,
                api_key=api_key,
                model=model,
                timeout=timeout,
                image=image,
                temperature=temperature,
                reasoning=reasoning,
            )
        )
        if self.exception is not None:
            raise self.exception
        return self.outcome


class TestServiceGating:
    """PP-04 service gating."""

    @pytest.mark.parametrize("length", [0, 1, 50, 100])
    def test_short_text_returns_original_and_does_not_call_client(self, length) -> None:
        client = RecordingClient(outcome="enhanced")
        service = TextEnhancementService(client=client)
        original = f"  {'x' * length}  "
        settings = TextEnhancementSettings(
            enabled=True,
            api_key=FAKE_KEY,
            model="qwen/qwen3.6-flash",
            timeout=10.0,
        )

        result = service.enhance(original, settings)

        assert result is original
        assert client.calls == []

    def test_disabled_service_returns_original_and_does_not_call_client(self) -> None:
        client = RecordingClient(outcome="enhanced")
        service = TextEnhancementService(client=client)
        original = f"  {'x' * 101}  "
        settings = TextEnhancementSettings(
            enabled=False,
            api_key=FAKE_KEY,
            model="qwen/qwen3.6-flash",
            timeout=10.0,
        )

        result = service.enhance(original, settings)

        assert result is original
        assert client.calls == []

    @pytest.mark.parametrize("api_key", [None, "", "   "])
    def test_missing_or_blank_key_returns_original_and_does_not_call_client(self, api_key) -> None:
        client = RecordingClient(outcome="enhanced")
        service = TextEnhancementService(client=client)
        original = f"  {'x' * 101}  "
        settings = TextEnhancementSettings(
            enabled=True,
            api_key=api_key,
            model="qwen/qwen3.6-flash",
            timeout=10.0,
        )

        result = service.enhance(original, settings)

        assert result is original
        assert client.calls == []

    def test_long_text_can_trigger_client_call(self) -> None:
        client = RecordingClient(outcome="enhanced")
        service = TextEnhancementService(client=client)
        original = f"  {'x' * 101}  "
        settings = TextEnhancementSettings(
            enabled=True,
            api_key=FAKE_KEY,
            model="qwen/qwen3.6-flash",
            timeout=10.0,
        )

        result = service.enhance(original, settings)

        assert result == "enhanced"
        assert len(client.calls) == 1

    def test_min_length_constant_matches_matrix(self) -> None:
        assert MIN_TEXT_ENHANCEMENT_LENGTH == 100


class TestCustomMinLengthThresholds:
    """PP-04b custom min_length is honored."""

    def test_lower_threshold_allows_shorter_text(self) -> None:
        client = RecordingClient(outcome="enhanced")
        service = TextEnhancementService(client=client)
        original = f"  {'x' * 60}  "
        settings = TextEnhancementSettings(
            enabled=True,
            api_key=FAKE_KEY,
            model="qwen/qwen3.6-flash",
            min_length=50,
            timeout=10.0,
        )

        result = service.enhance(original, settings)

        assert result == "enhanced"
        assert len(client.calls) == 1

    def test_lower_threshold_boundary_returns_original(self) -> None:
        client = RecordingClient(outcome="enhanced")
        service = TextEnhancementService(client=client)
        original = f"  {'x' * 50}  "
        settings = TextEnhancementSettings(
            enabled=True,
            api_key=FAKE_KEY,
            model="qwen/qwen3.6-flash",
            min_length=50,
            timeout=10.0,
        )

        result = service.enhance(original, settings)

        assert result is original
        assert client.calls == []

    def test_higher_threshold_blocks_text_below_it(self) -> None:
        client = RecordingClient(outcome="enhanced")
        service = TextEnhancementService(client=client)
        original = f"  {'x' * 150}  "
        settings = TextEnhancementSettings(
            enabled=True,
            api_key=FAKE_KEY,
            model="qwen/qwen3.6-flash",
            min_length=200,
            timeout=10.0,
        )

        result = service.enhance(original, settings)

        assert result is original
        assert client.calls == []

    def test_higher_threshold_allows_text_above_it(self) -> None:
        client = RecordingClient(outcome="enhanced")
        service = TextEnhancementService(client=client)
        original = f"  {'x' * 201}  "
        settings = TextEnhancementSettings(
            enabled=True,
            api_key=FAKE_KEY,
            model="qwen/qwen3.6-flash",
            min_length=200,
            timeout=10.0,
        )

        result = service.enhance(original, settings)

        assert result == "enhanced"
        assert len(client.calls) == 1


class TestServiceOrchestrationAndFallback:
    """PP-05 service orchestration/fallback."""

    def test_service_passes_stripped_text_prompt_temperature_and_reasoning(self) -> None:
        client = RecordingClient(outcome="enhanced")
        service = TextEnhancementService(client=client)
        original = f"  {'x' * 101}  "
        settings = TextEnhancementSettings(
            enabled=True,
            api_key=f"  {FAKE_KEY}  ",
            model="custom/model",
            timeout=12.5,
        )

        result = service.enhance(original, settings)

        assert result == "enhanced"
        assert len(client.calls) == 1
        call = client.calls[0]
        assert call.text == original.strip()
        assert call.prompt == SYSTEM_PROMPT
        assert call.api_key == FAKE_KEY
        assert call.model == "custom/model"
        assert call.timeout == 12.5
        assert call.temperature == 0.2
        assert call.reasoning == {"effort": "none", "exclude": True}

    def test_service_returns_original_on_client_exception(self, caplog) -> None:
        client = RecordingClient(exception=PostProcessingError("openrouter failed"))
        service = TextEnhancementService(client=client)
        original = f"  {'x' * 101}  "
        settings = TextEnhancementSettings(
            enabled=True,
            api_key=FAKE_KEY,
            model="qwen/qwen3.6-flash",
            timeout=10.0,
        )

        with caplog.at_level(logging.WARNING, logger="veespeech.post_processing.service"):
            result = service.enhance(original, settings)

        assert result is original
        _assert_no_leaks(caplog.text, FAKE_KEY, original.strip())

    @pytest.mark.parametrize("outcome", [None, "", "   ", 123])
    def test_service_returns_original_on_invalid_client_output(self, outcome) -> None:
        client = RecordingClient(outcome=outcome)  # type: ignore[arg-type]
        service = TextEnhancementService(client=client)
        original = f"  {'x' * 101}  "
        settings = TextEnhancementSettings(
            enabled=True,
            api_key=FAKE_KEY,
            model="qwen/qwen3.6-flash",
            timeout=10.0,
        )

        result = service.enhance(original, settings)

        assert result is original


class TestServiceScreenshotHandling:
    """PP-05b screenshot forwarding and normalization."""

    def test_text_only_call_passes_none_image(self) -> None:
        client = RecordingClient(outcome="enhanced")
        service = TextEnhancementService(client=client)
        original = f"  {'x' * 101}  "
        settings = TextEnhancementSettings(enabled=True, api_key=FAKE_KEY)

        result = service.enhance(original, settings)

        assert result == "enhanced"
        assert len(client.calls) == 1
        assert client.calls[0].image is None

    def test_screenshot_forwarded_to_client_as_image(self) -> None:
        client = RecordingClient(outcome="enhanced")
        service = TextEnhancementService(client=client)
        original = f"  {'x' * 101}  "
        data_url = "data:image/png;base64,abc123"
        settings = TextEnhancementSettings(enabled=True, api_key=FAKE_KEY)

        result = service.enhance(original, settings, screenshot=data_url)

        assert result == "enhanced"
        assert len(client.calls) == 1
        assert client.calls[0].image == data_url

    def test_raw_base64_screenshot_normalized(self) -> None:
        client = RecordingClient(outcome="enhanced")
        service = TextEnhancementService(client=client)
        original = f"  {'x' * 101}  "
        settings = TextEnhancementSettings(enabled=True, api_key=FAKE_KEY)

        service.enhance(original, settings, screenshot="  abc123  ")

        assert client.calls[0].image == "data:image/png;base64,abc123"

    def test_bytes_screenshot_normalized(self) -> None:
        client = RecordingClient(outcome="enhanced")
        service = TextEnhancementService(client=client)
        original = f"  {'x' * 101}  "
        settings = TextEnhancementSettings(enabled=True, api_key=FAKE_KEY)

        service.enhance(original, settings, screenshot=b"\x00\x01\x02")

        assert client.calls[0].image == "data:image/png;base64,AAEC"

    @pytest.mark.parametrize("screenshot", [None, "", "   ", 123, []])
    def test_blank_or_unsupported_screenshot_treated_as_absent(self, screenshot) -> None:
        client = RecordingClient(outcome="enhanced")
        service = TextEnhancementService(client=client)
        original = f"  {'x' * 101}  "
        settings = TextEnhancementSettings(enabled=True, api_key=FAKE_KEY)

        service.enhance(original, settings, screenshot=screenshot)

        assert client.calls[0].image is None

    def test_screenshot_payload_not_logged_on_client_exception(self, caplog) -> None:
        client = RecordingClient(exception=PostProcessingError("openrouter failed"))
        service = TextEnhancementService(client=client)
        original = f"  {'x' * 101}  "
        settings = TextEnhancementSettings(enabled=True, api_key=FAKE_KEY)
        screenshot = "data:image/png;base64,SECRETSCREENSHOT"

        with caplog.at_level(logging.WARNING, logger="veespeech.post_processing.service"):
            result = service.enhance(original, settings, screenshot=screenshot)

        assert result is original
        assert "SECRETSCREENSHOT" not in caplog.text


def _assert_no_leaks(caplog_text: str, secret: str, full_text: str) -> None:
    assert secret not in caplog_text
    assert full_text not in caplog_text
