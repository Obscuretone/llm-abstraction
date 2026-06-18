from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

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
        del request
        nonlocal calls
        calls += 1
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
        return httpx.Response(429)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        backend = MistralBackend(api_key="test", client=client, max_attempts=2)
        with pytest.raises(AIRetryAfterMissingError):
            await backend.complete([{"role": "user", "content": "hello"}])
