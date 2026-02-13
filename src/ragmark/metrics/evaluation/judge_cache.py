"""Thread-safe cache for LLM judge evaluation results.

Provides both in-memory and optional disk-backed persistence to avoid
redundant LLM calls when the same (answer, context) pair is evaluated
multiple times across benchmark runs.
"""

import asyncio
import hashlib
import shelve
from pathlib import Path
from typing import Any

from ragmark.logger import get_logger

logger = get_logger(__name__)


class JudgeCache:
    """Thread-safe cache for LLM judge results.

    Uses an in-memory dict for fast lookups with optional shelve-backed
    disk persistence. All operations are guarded by asyncio.Lock to
    prevent concurrent corruption.

    Attributes:
        _memory: In-memory hot cache.
        _disk_path: Optional path for shelve-backed persistence.
        _lock: Async lock for concurrent access safety.
    """

    def __init__(self, disk_path: Path | None = None) -> None:
        """Initialize the cache.

        Args:
            disk_path: Optional filesystem path for persistent cache.
                When provided, cache entries are written to a shelve
                database at this location.
        """
        self._memory: dict[str, float] = {}
        self._disk_path = disk_path
        self._lock = asyncio.Lock()
        self._shelf: shelve.Shelf[Any] | None = None

        if disk_path is not None:
            disk_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug("JudgeCache initialized: disk_path=%s", disk_path)
        else:
            logger.debug("JudgeCache initialized: memory-only mode")

    @staticmethod
    def make_key(*parts: str) -> str:
        """Compute a deterministic SHA-256 cache key from input parts.

        Args:
            *parts: String components to hash (e.g., answer, context).

        Returns:
            Hexadecimal SHA-256 digest.
        """
        joined = "|".join(parts)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    async def get(self, key: str) -> float | None:
        """Retrieve a cached score.

        Checks the in-memory cache first, falling back to disk if available.

        Args:
            key: Cache key produced by make_key().

        Returns:
            Cached score, or None on cache miss.
        """
        async with self._lock:
            if key in self._memory:
                logger.debug("JudgeCache hit (memory): key=%s", key[:12])
                return self._memory[key]

            if self._disk_path is not None:
                loop = asyncio.get_running_loop()
                value = await loop.run_in_executor(None, self._disk_get, key)
                if value is not None:
                    self._memory[key] = value
                    logger.debug("JudgeCache hit (disk): key=%s", key[:12])
                    return value

            logger.debug("JudgeCache miss: key=%s", key[:12])
            return None

    async def put(self, key: str, value: float) -> None:
        """Store a score in the cache.

        Writes to both in-memory dict and disk shelf (if configured).

        Args:
            key: Cache key produced by make_key().
            value: Score to cache.
        """
        async with self._lock:
            self._memory[key] = value

            if self._disk_path is not None:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._disk_put, key, value)

            logger.debug("JudgeCache stored: key=%s, value=%.4f", key[:12], value)

    async def close(self) -> None:
        """Flush and close the disk shelf if open."""
        async with self._lock:
            if self._shelf is not None:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._shelf.close)
                self._shelf = None
                logger.debug("JudgeCache disk shelf closed")

    def _open_shelf(self) -> shelve.Shelf[Any]:
        """Lazily open the shelve database."""
        if self._shelf is None:
            self._shelf = shelve.open(str(self._disk_path))
        return self._shelf

    def _disk_get(self, key: str) -> float | None:
        """Synchronous disk read (called via run_in_executor)."""
        try:
            shelf = self._open_shelf()
            return shelf.get(key)
        except Exception as exc:
            logger.warning(
                "JudgeCache disk read failed: key=%s, reason=%s", key[:12], exc
            )
            logger.debug("Disk read error details: %s", exc, exc_info=True)
            return None

    def _disk_put(self, key: str, value: float) -> None:
        """Synchronous disk write (called via run_in_executor)."""
        try:
            shelf = self._open_shelf()
            shelf[key] = value
            shelf.sync()
        except Exception as exc:
            logger.warning(
                "JudgeCache disk write failed: key=%s, reason=%s", key[:12], exc
            )
            logger.debug("Disk write error details: %s", exc, exc_info=True)
