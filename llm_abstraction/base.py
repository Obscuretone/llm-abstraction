from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

ChatMessage = Mapping[str, str]


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class ChatResponse:
    content: str
    usage: TokenUsage = TokenUsage()
    raw: object | None = None


@dataclass(frozen=True)
class EmbeddingResponse:
    embeddings: list[list[float]]
    usage: TokenUsage = TokenUsage()
    raw: object | None = None


class AIBackend(Protocol):
    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = 1200,
        response_format: Mapping[str, object] | None = None,
    ) -> ChatResponse: ...

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = 1200,
        response_format: Mapping[str, object] | None = None,
    ) -> str: ...
