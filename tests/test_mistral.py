import json
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from typing import cast

import httpx
import pytest

from llm_abstraction import AIRetryAfterMissingError, MistralBackend, parse_retry_after_seconds


def test_parse_retry_after_seconds_delta() -> None:
    assert parse_retry_after_seconds("4") == 4
    assert parse_retry_after_seconds("0.5") == 0.5


def test_parse_retry_after_seconds_http_date() -> None:
    retry_at = datetime.now(timezone.utc) + timedelta(seconds=10)
    parsed = parse_retry_after_seconds(format_datetime(retry_at, usegmt=True))

    assert parsed is not None
    assert 0 < parsed <= 10


@pytest.mark.asyncio
async def test_mistral_backend_retries_after_header(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.path == "/v1/chat/completions"
        if calls == 1:
            return httpx.Response(429, headers={"retry-after": "0.1"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "done"}}],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                },
            },
        )

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("llm_abstraction.mistral.asyncio.sleep", fake_sleep)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        backend = MistralBackend(api_key="test", client=client, max_attempts=2)
        response = await backend.chat([{"role": "user", "content": "hello"}])

    assert response.content == "done"
    assert response.usage.total_tokens == 5
    assert calls == 2
    assert sleeps == [0.1]


@pytest.mark.asyncio
async def test_mistral_backend_requires_retry_after_for_rate_limit() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(429, headers={"x-ratelimit-remaining-tokens-month": "0"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        backend = MistralBackend(api_key="test", client=client, max_attempts=2)
        with pytest.raises(AIRetryAfterMissingError) as exc_info:
            await backend.complete([{"role": "user", "content": "hello"}])

    assert exc_info.value.response_headers["x-ratelimit-remaining-tokens-month"] == "0"


@pytest.mark.asyncio
async def test_mistral_backend_sends_response_format() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json_from_request(request)
        assert body["response_format"] == {"type": "json_object"}
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok": true}'}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        backend = MistralBackend(api_key="test", client=client)
        response = await backend.complete(
            [{"role": "user", "content": "hello"}],
            response_format={"type": "json_object"},
        )

    assert response == '{"ok": true}'


@pytest.mark.asyncio
async def test_mistral_backend_extracts_embeddings() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json_from_request(request)
        assert request.url.path == "/v1/embeddings"
        assert body["inputs"] == ["one", "two"]
        return httpx.Response(
            200,
            json={
                "data": [
                    {"embedding": [1, "2.5"]},
                    {"embedding": [3.5, 4]},
                ],
                "usage": {"total_tokens": 6},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        backend = MistralBackend(api_key="test", client=client)
        response = await backend.embed(["one", "two"])

    assert response.embeddings == [[1.0, 2.5], [3.5, 4.0]]
    assert response.usage.total_tokens == 6


def json_from_request(request: httpx.Request) -> dict[str, object]:
    body = json.loads(request.content.decode("utf-8"))
    assert isinstance(body, dict)
    return cast(dict[str, object], body)
