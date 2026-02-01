"""Rate limiting for API-bound operations."""

import asyncio
import time


class RateLimiter:
    """Token bucket rate limiter for async operations.

    Prevents API quota exhaustion and thundering herd by allowing
    initial bursts then enforcing steady-state rate limits.

    Attributes:
        rate: Requests per second allowed.
        tokens: Current available tokens.
        last_update: Last token refill timestamp.
    """

    def __init__(self, requests_per_second: float):
        """Initialize rate limiter with specified request rate.

        Args:
            requests_per_second: Maximum sustained request rate.
        """
        self.rate = requests_per_second
        self.capacity = requests_per_second
        self.tokens = requests_per_second  # Start full for initial bursts
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens, blocking if insufficient tokens available.

        Args:
            tokens: Number of tokens to acquire.
        """
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update

            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= tokens:
                self.tokens -= tokens
            else:
                wait_time = (tokens - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
                self.last_update += wait_time
