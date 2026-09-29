"""Retry decorator implementing the exponential-backoff-with-jitter policy
described in resilience_patterns.pdf: base delay 1s, doubling per attempt,
capped at 30s, up to 5 attempts, retrying only on transient status codes.
"""
import random
import time
from functools import wraps

TRANSIENT_STATUS_CODES = {429, 502, 503, 504}


class TransientError(Exception):
    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"transient error: {status_code}")


def compute_backoff_delay(attempt: int, base_delay: float = 1.0, max_delay: float = 30.0) -> float:
    """Exponential backoff with +/-20% jitter, capped at max_delay."""
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter_factor = random.uniform(0.8, 1.2)
    return delay * jitter_factor


def with_retries(max_attempts: int = 5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except TransientError as exc:
                    if exc.status_code not in TRANSIENT_STATUS_CODES:
                        raise
                    last_error = exc
                    if attempt < max_attempts - 1:
                        time.sleep(compute_backoff_delay(attempt))
            raise last_error
        return wrapper
    return decorator


class CircuitBreaker:
    """Opens after `failure_threshold` consecutive failures and short-
    circuits calls for `reset_seconds` before allowing another attempt."""

    def __init__(self, failure_threshold: int = 5, reset_seconds: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_seconds = reset_seconds
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self.reset_seconds:
            self._opened_at = None
            self._consecutive_failures = 0
            return False
        return True

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._opened_at = time.monotonic()
