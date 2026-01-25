"""Abstract base class for LLM drivers.

This module defines the unified interface for LLM backends, supporting
both synchronous generation and streaming outputs.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from ragmark.schemas.generation import GenerationResult


class BaseLLMDriver(ABC):
    """Abstract base class for LLM generation backends.

    LLM drivers provide a consistent interface for text generation across
    different model formats (GGUF, HuggingFace, API-based) and inference
    backends (llama.cpp, vLLM, OpenAI-compatible APIs).

    All generation operations are async to support efficient concurrent
    processing and non-blocking inference.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        """Generate text completion for a prompt.

        Args:
            prompt: Input prompt string.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0 = deterministic).
            stop: Optional list of stop sequences.

        Returns:
            Generation result with text, usage stats, and finish reason.

        Raises:
            GenerationError: If generation fails.
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Generate text with streaming output.

        This method yields tokens as they are generated, enabling
        real-time streaming to users.

        Args:
            prompt: Input prompt string.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            stop: Optional list of stop sequences.

        Yields:
            Generated text chunks as they become available.

        Raises:
            GenerationError: If generation fails.
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string.

        Args:
            text: Text to tokenize.

        Returns:
            Number of tokens in the text.
        """
        pass

    @property
    @abstractmethod
    def context_window(self) -> int:
        """Get the maximum context window size.

        Returns:
            Maximum number of tokens that can fit in the context.
        """
        pass

    async def __aenter__(self) -> "BaseLLMDriver":
        """Async context manager entry.

        Returns:
            Self for use in async with statements.
        """
        return self

    @abstractmethod
    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> None:
        """Async context manager exit.

        Implementations should clean up model resources here.
        """
        pass
