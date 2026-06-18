from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import cast

import httpx

from .base import ChatMessage, ChatResponse, EmbeddingResponse, TokenUsage
from .exceptions import AIBackendError, AIRateLimitError, AITransientError
from .retry_after import parse_retry_after_seconds


class MistralBackend:
    DEFAULT_API_URL = "https://api.mistral.ai/v1"
    RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "mistral-large-latest",
        api_url: str = DEFAULT_API_URL,
        timeout_seconds: float = 60,
        max_attempts: int = 6,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required.")
        self.api_key = api_key
        self.model = model
        self.api_url = api_url
        self.timeout_seconds = max(float(timeout_seconds), 1)
        self.max_attempts = max(int(max_attempts), 1)
        self._client = client

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = 1200,
        response_format: Mapping[str, object] | None = None,
    ) -> ChatResponse:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = dict(response_format)

        body = await self._post_json("chat/completions", payload)
        return ChatResponse(
            content=self._extract_content(body),
            usage=self._extract_usage(body),
            raw=body,
        )

    async def embed(
        self,
        inputs: Sequence[str],
        *,
        model: str = "mistral-embed",
    ) -> EmbeddingResponse:
        payload: dict[str, object] = {
            "model": model,
            "inputs": list(inputs),
        }
        body = await self._post_json("embeddings", payload)
        return EmbeddingResponse(
            embeddings=self._extract_embeddings(body),
            usage=self._extract_usage(body),
            raw=body,
        )

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = 1200,
        response_format: Mapping[str, object] | None = None,
    ) -> str:
        response = await self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        return response.content

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def _post_json(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = await self._post(path, payload)
                if response.status_code in self.RETRYABLE_STATUS_CODES:
                    raise self._retryable_error(response)
                response.raise_for_status()
                return cast(dict[str, object], response.json())
            except AITransientError as exc:
                if attempt >= self.max_attempts:
                    raise
                await asyncio.sleep(exc.resolve_delay_seconds())
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                raise AITransientError(f"Mistral transport failed: {exc}") from exc

        raise AIBackendError("Mistral request failed after retries.")

    async def _post(self, path: str, payload: dict[str, object]) -> httpx.Response:
        url = f"{self.api_url.rstrip('/')}/{path.lstrip('/')}"
        if self._client is not None:
            return await self._client.post(
                url,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout_seconds,
            )

        async with httpx.AsyncClient() as client:
            return await client.post(
                url,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout_seconds,
            )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _retryable_error(self, response: httpx.Response) -> AITransientError:
        retry_after_seconds = parse_retry_after_seconds(response.headers.get("retry-after"))
        message = f"Mistral retryable HTTP {response.status_code}"
        if response.status_code == 429:
            return AIRateLimitError(
                message,
                retry_after_seconds=retry_after_seconds,
                status_code=response.status_code,
                response_headers=response.headers,
            )
        return AITransientError(
            message,
            retry_after_seconds=retry_after_seconds,
            status_code=response.status_code,
            response_headers=response.headers,
        )

    def _extract_content(self, body: Mapping[str, object]) -> str:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise AIBackendError("Mistral response did not contain message content.")
        choices = cast(list[object], choices)

        raw_first_choice = choices[0]
        if not isinstance(raw_first_choice, Mapping):
            raise AIBackendError("Mistral response did not contain message content.")
        first_choice = cast(Mapping[str, object], raw_first_choice)

        message = first_choice.get("message")
        if not isinstance(message, Mapping):
            raise AIBackendError("Mistral response did not contain message content.")
        message = cast(Mapping[str, object], message)

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise AIBackendError("Mistral response content was empty.")
        return content.strip()

    def _extract_usage(self, body: Mapping[str, object]) -> TokenUsage:
        raw_usage = body.get("usage")
        if not isinstance(raw_usage, Mapping):
            return TokenUsage()
        raw_usage = cast(Mapping[str, object], raw_usage)
        return TokenUsage(
            input_tokens=self._int_or_zero(raw_usage.get("prompt_tokens")),
            output_tokens=self._int_or_zero(raw_usage.get("completion_tokens")),
            total_tokens=self._int_or_zero(raw_usage.get("total_tokens")),
        )

    def _extract_embeddings(self, body: Mapping[str, object]) -> list[list[float]]:
        data = body.get("data")
        if not isinstance(data, list):
            raise AIBackendError("Mistral response did not contain embeddings.")
        data = cast(list[object], data)

        embeddings: list[list[float]] = []
        for item in data:
            if not isinstance(item, Mapping):
                raise AIBackendError("Mistral response did not contain embeddings.")
            item = cast(Mapping[str, object], item)
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                raise AIBackendError("Mistral response did not contain embeddings.")
            embedding = cast(list[object], embedding)
            embeddings.append([self._float(value) for value in embedding])

        return embeddings

    def _int_or_zero(self, value: object) -> int:
        if value is None:
            return 0
        if isinstance(value, int):
            return value
        if isinstance(value, float | str):
            try:
                return int(value)
            except ValueError:
                return 0
        return 0

    def _float(self, value: object) -> float:
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError as exc:
                raise AIBackendError("Mistral embedding contained a non-numeric value.") from exc
        raise AIBackendError("Mistral embedding contained a non-numeric value.")
