from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


def parse_retry_after_seconds(retry_after: str | None) -> float | None:
    if not isinstance(retry_after, str) or not retry_after.strip():
        return None

    value = retry_after.strip()
    try:
        return max(float(value), 0.001)
    except ValueError:
        pass

    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None

    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    else:
        retry_at = retry_at.astimezone(timezone.utc)

    delay_seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()
    if delay_seconds <= 0:
        return None
    return max(delay_seconds, 0.001)
