"""Token bucket rate limiter, matching the policy in rate_limiting.pdf:
100-token capacity per API key, refilling at 10 tokens/second.
"""
import time


class TokenBucket:
    def __init__(self, capacity: float = 100.0, refill_rate_per_second: float = 10.0):
        self.capacity = capacity
        self.refill_rate_per_second = refill_rate_per_second
        self._tokens = capacity
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate_per_second)
        self._last_refill = now

    def try_consume(self, tokens: float = 1.0) -> bool:
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    def retry_after_seconds(self, tokens: float = 1.0) -> float:
        """How long the caller should wait before the bucket has `tokens` available."""
        self._refill()
        deficit = tokens - self._tokens
        if deficit <= 0:
            return 0.0
        return deficit / self.refill_rate_per_second


class RateLimiter:
    """Per-key token buckets, e.g. one per API key."""

    def __init__(self, capacity: float = 100.0, refill_rate_per_second: float = 10.0):
        self.capacity = capacity
        self.refill_rate_per_second = refill_rate_per_second
        self._buckets: dict[str, TokenBucket] = {}

    def _bucket_for(self, key: str) -> TokenBucket:
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(self.capacity, self.refill_rate_per_second)
        return self._buckets[key]

    def allow(self, key: str) -> bool:
        return self._bucket_for(key).try_consume()
