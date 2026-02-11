"""Tests for RateLimiter token bucket implementation.

This module tests the RateLimiter class which enforces rate limiting
using a token bucket algorithm.
"""

import asyncio
import time

import pytest

from ragmark.index.embedders import SentenceTransformerEmbedder
from ragmark.index.rate_limiter import RateLimiter


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.performance
@pytest.mark.asyncio
class TestEmbedderRateLimiting:
    """Tests for SentenceTransformerEmbedder rate limiting integration."""

    async def test_embedder_with_rate_limit_should_throttle_requests(self) -> None:
        """Verifies embedder respects rate_limit parameter.

        Given:
            SentenceTransformerEmbedder with rate_limit=2.0 and batch_size=2.
        When:
            Embedding 10 texts via embed_async().
        Then:
            Operation takes approximately 1.5 seconds (5 batches, 2/s after burst).
        """
        embedder = SentenceTransformerEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=2,
            rate_limit=2.0,
        )

        # Warmup: load model (not timed)
        await embedder.embed_async(["warmup"])

        texts = [f"Test text {i}" for i in range(10)]

        start = time.time()
        await embedder.embed_async(texts)
        duration = time.time() - start

        # 5 batches at 2/s = ~1.5s rate limiting (2 immediate + 3 at 2/s)
        assert 1.0 < duration < 6.0, f"Rate limit not enforced: {duration:.2f}s"

    async def test_embedder_without_rate_limit_should_run_unrestricted(self) -> None:
        """Verifies embedder without rate_limit runs at full speed.

        Given:
            SentenceTransformerEmbedder with rate_limit=None.
        When:
            Embedding 10 texts (after model warmup).
        Then:
            Completes quickly without artificial delays.
        """
        embedder = SentenceTransformerEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            rate_limit=None,
        )

        texts = [f"Quick test {i}" for i in range(10)]

        # Warmup: load model (not timed)
        await embedder.embed_async(["warmup"])

        start = time.time()
        await embedder.embed_async(texts)
        duration = time.time() - start

        # Should complete in <2s without rate limiting (model already loaded)
        assert duration < 2.0, f"Unrestricted took too long: {duration:.2f}s"

    async def test_embedder_rate_limit_should_work_with_batching(self) -> None:
        """Verifies rate limiting applies per batch, not per text.

        Given:
            SentenceTransformerEmbedder with rate_limit=2.0, batch_size=5.
        When:
            Embedding 15 texts (processed as 3 batches).
        Then:
            Duration reflects 3 batch acquisitions at 2/s (~0.5 second).
        """
        embedder = SentenceTransformerEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=5,
            rate_limit=2.0,
        )

        # Warmup: load model (not timed)
        await embedder.embed_async(["warmup"])

        texts = [f"Batch test {i}" for i in range(15)]

        start = time.time()
        await embedder.embed_async(texts)
        duration = time.time() - start

        # 3 batches at 2/s = ~0.5s rate limiting
        assert 0.3 < duration < 6.0, f"Batch rate limiting: {duration:.2f}s"


@pytest.mark.unit
@pytest.mark.performance
@pytest.mark.asyncio
class TestRateLimiterTokenBucket:
    """Tests for RateLimiter token bucket implementation."""

    async def test_rate_limiter_should_enforce_10_requests_per_second(self) -> None:
        """Verifies rate limiter enforces rate limit after initial burst.

        Given:
            RateLimiter configured for 10 requests/second.
        When:
            Exhausting initial tokens, then acquiring 20 more.
        Then:
            Second batch takes approximately 2 seconds (20 tokens ÷ 10/s).
        """
        limiter = RateLimiter(requests_per_second=10.0)

        # Exhaust initial tokens
        await limiter.acquire(tokens=100)

        start = time.time()
        for _ in range(20):
            await limiter.acquire(tokens=1)
        duration = time.time() - start

        # 20 requests at 10/s = ~2 seconds
        assert 1.8 < duration < 2.2, f"Duration {duration:.2f}s, expected ~2s"

    async def test_rate_limiter_should_allow_burst_within_initial_tokens(self) -> None:
        """Verifies initial token bucket allows immediate bursts.

        Given:
            RateLimiter with rate=5.0 (initial tokens=5).
        When:
            Acquiring 5 tokens immediately.
        Then:
            Completes in under 100ms (no waiting).
        """
        limiter = RateLimiter(requests_per_second=5.0)

        start = time.time()
        await limiter.acquire(tokens=5)
        duration = time.time() - start

        assert duration < 0.1, f"Burst blocked: {duration:.3f}s"

    async def test_rate_limiter_should_refill_tokens_over_time(self) -> None:
        """Verifies token bucket refills at correct rate.

        Given:
            RateLimiter at 20 tokens/second.
        When:
            Consuming all tokens, waiting 0.5s, then acquiring 10 more.
        Then:
            Second acquire completes quickly (refilled ~10 tokens).
        """
        limiter = RateLimiter(requests_per_second=20.0)

        # Consume initial tokens
        await limiter.acquire(tokens=20)

        # Wait for refill
        await asyncio.sleep(0.5)

        # Acquire 10 tokens (should have ~10 refilled)
        start = time.time()
        await limiter.acquire(tokens=10)
        duration = time.time() - start

        # Should complete quickly (refilled tokens available)
        assert duration < 0.2, f"Refill failed: waited {duration:.3f}s"

    async def test_rate_limiter_should_handle_concurrent_acquires(self) -> None:
        """Verifies rate limiter is thread-safe under concurrent load.

        Given:
            RateLimiter with 100 tokens/second, initial tokens exhausted.
        When:
            10 concurrent tasks each acquiring 10 tokens.
        Then:
            All tasks complete and total duration respects rate limit.
        """
        limiter = RateLimiter(requests_per_second=100.0)

        # Exhaust initial tokens
        await limiter.acquire(tokens=100)

        async def acquire_batch():
            await limiter.acquire(tokens=10)

        start = time.time()
        await asyncio.gather(*[acquire_batch() for _ in range(10)])
        duration = time.time() - start

        # 100 tokens at 100/s = ~1 second
        assert 0.8 < duration < 1.5, f"Concurrent acquire: {duration:.2f}s"
