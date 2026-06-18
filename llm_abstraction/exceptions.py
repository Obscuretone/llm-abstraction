from __future__ import annotations


class AIBackendError(Exception):
    pass


class AITransientError(AIBackendError):
    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds
        self.status_code = status_code

    def resolve_delay_seconds(self) -> float:
        if self.retry_after_seconds is None:
            raise AIRetryAfterMissingError(
                "Retryable AI backend error did not include Retry-After.",
                status_code=self.status_code,
            )
        return max(float(self.retry_after_seconds), 0.001)


class AIRateLimitError(AITransientError):
    pass


class AIRetryAfterMissingError(AIBackendError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
