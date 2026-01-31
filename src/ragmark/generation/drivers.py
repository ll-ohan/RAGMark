"""Abstract base class for LLM drivers.

This module defines the unified interface for LLM backends, supporting
both synchronous generation and streaming outputs.
"""

import asyncio
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ragmark.exceptions import GenerationError
from ragmark.logger import get_logger
from ragmark.schemas.generation import GenerationResult, TokenUsage

logger = get_logger(__name__)


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
            prompt: Input text to complete.
            max_tokens: Generation length limit.
            temperature: Sampling randomness (0.0 = deterministic).
            stop: Sequences that halt generation when encountered.

        Returns:
            Completion text, token usage statistics, and finish reason.

        Raises:
            GenerationError: If generation fails.
        """
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Generate text with streaming output.

        Yields tokens incrementally as generated, enabling real-time
        streaming to end users.

        Args:
            prompt: Input text to complete.
            max_tokens: Generation length limit.
            temperature: Sampling randomness.
            stop: Sequences that halt generation when encountered.

        Yields:
            Text chunks as they become available.

        Raises:
            GenerationError: If generation fails.
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string.

        Args:
            text: Input text to tokenize.

        Returns:
            Token count using model's tokenizer.
        """
        pass

    @property
    @abstractmethod
    def context_window(self) -> int:
        """Get the maximum context window size.

        Returns:
            Model's context window capacity in tokens.
        """
        pass

    async def __aenter__(self) -> "BaseLLMDriver":
        """Enter the async context manager.

        Returns:
            Driver instance for use in async with statements.
        """
        return self

    @abstractmethod
    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> None:
        """Exit the async context manager.

        Implementations must clean up model resources (memory, threads, GPU).
        """
        pass


class LlamaCppDriver(BaseLLMDriver):
    """LLM driver using llama-cpp-python for local GGUF models.

    This driver provides async access to local LLM models via llama.cpp.
    The model is loaded lazily on first use to avoid blocking during import.
    Thread safety is ensured via single-worker ThreadPoolExecutor since
    llama.cpp is NOT thread-safe.

    Attributes:
        model_path: Path to GGUF model file.
        n_ctx: Context window size in tokens.
        n_gpu_layers: Number of layers offloaded to GPU (0 = CPU only).
    """

    def __init__(
        self,
        model_path: str | Path,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        n_threads: int | None = None,
        verbose: bool = False,
    ):
        """Initialize driver with model configuration.

        Model loading is deferred until __aenter__ to avoid blocking
        the event loop during initialization.

        Args:
            model_path: Location of GGUF model file.
            n_ctx: Context window capacity.
            n_gpu_layers: Layers offloaded to GPU (0 = CPU only).
            n_threads: CPU thread count (defaults to available cores).
            verbose: Enable llama.cpp debug logging.
        """
        self.model_path = Path(model_path)
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._n_threads = n_threads or os.cpu_count()
        self._verbose = verbose
        self._model: Any = None
        self._executor: ThreadPoolExecutor | None = None

        logger.debug(
            "LlamaCppDriver configured: model=%s, n_ctx=%d, n_gpu_layers=%d, n_threads=%d",
            self.model_path,
            n_ctx,
            n_gpu_layers,
            self._n_threads,
        )

    async def __aenter__(self) -> "LlamaCppDriver":
        """Load model on context manager entry.

        Model loading executes in dedicated thread pool to prevent
        blocking the async event loop during GGUF deserialization.

        Returns:
            Driver instance ready for generation.

        Raises:
            GenerationError: If model loading fails or llama-cpp-python not installed.
        """
        logger.info("Model loading started: path=%s", self.model_path)

        loop = asyncio.get_running_loop()
        self._executor = ThreadPoolExecutor(max_workers=1)

        try:
            self._model = await loop.run_in_executor(self._executor, self._load_model)
            logger.info(
                "Model loading completed: path=%s, n_ctx=%d",
                self.model_path,
                self._n_ctx,
            )
        except Exception as e:
            logger.error("Model loading failed: path=%s", self.model_path)
            logger.debug("Loading failure details: %s", e, exc_info=True)
            if self._executor:
                self._executor.shutdown(wait=True)
            raise GenerationError(f"Failed to load model from {self.model_path}") from e

        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> None:
        """Cleanup model resources and thread pool."""
        logger.debug("Resource cleanup started: model=%s", self.model_path)

        if self._executor:
            self._executor.shutdown(wait=True)
            logger.debug("Thread pool executor shutdown complete")

    def _load_model(self) -> Any:
        """Load llama.cpp model in thread pool worker.

        Returns:
            Initialized Llama model instance.

        Raises:
            GenerationError: If llama-cpp-python not installed or model file missing.
        """
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise GenerationError(
                "llama-cpp-python not installed. "
                "Install with: pip install llama-cpp-python"
            ) from e

        if not self.model_path.exists():
            raise GenerationError(f"Model file not found: {self.model_path}")

        logger.debug(
            "Initializing Llama model: n_ctx=%d, n_gpu_layers=%d, n_threads=%d",
            self._n_ctx,
            self._n_gpu_layers,
            self._n_threads,
        )

        return Llama(
            model_path=str(self.model_path),
            n_ctx=self._n_ctx,
            n_gpu_layers=self._n_gpu_layers,
            n_threads=self._n_threads,
            verbose=self._verbose,
        )

    async def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> GenerationResult:
        """Generate text completion for a prompt.

        Args:
            prompt: Input text to complete.
            max_tokens: Generation length limit.
            temperature: Sampling randomness (0.0 = deterministic).
            stop: Sequences that halt generation when encountered.

        Returns:
            Completion text, token usage statistics, and finish reason.

        Raises:
            GenerationError: If generation fails or model not loaded.
        """
        if self._model is None:
            raise GenerationError("Model not loaded. Use async with context manager.")

        logger.debug(
            "Generation started: prompt_chars=%d, max_tokens=%d, temperature=%.2f",
            len(prompt),
            max_tokens,
            temperature,
        )

        loop = asyncio.get_running_loop()

        try:
            output = await loop.run_in_executor(
                self._executor,
                lambda: self._model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop or [],
                    echo=False,
                ),
            )
        except Exception as e:
            logger.error("Generation failed: prompt_chars=%d", len(prompt))
            logger.debug("Generation failure details: %s", e, exc_info=True)
            raise GenerationError("Text generation failed") from e

        logger.debug(
            "Generation completed: output_chars=%d, completion_tokens=%d, finish_reason=%s",
            len(output["choices"][0]["text"]),
            output["usage"]["completion_tokens"],
            output["choices"][0]["finish_reason"],
        )

        return GenerationResult(
            text=output["choices"][0]["text"],
            usage=TokenUsage(
                prompt_tokens=output["usage"]["prompt_tokens"],
                completion_tokens=output["usage"]["completion_tokens"],
                total_tokens=output["usage"]["total_tokens"],
            ),
            finish_reason=output["choices"][0]["finish_reason"],
        )

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Generate text with streaming output.

        Args:
            prompt: Input text to complete.
            max_tokens: Generation length limit.
            temperature: Sampling randomness.
            stop: Sequences that halt generation when encountered.

        Yields:
            Text chunks as they become available.

        Raises:
            GenerationError: If generation fails or model not loaded.
        """
        if self._model is None:
            raise GenerationError("Model not loaded. Use async with context manager.")

        logger.debug(
            "Streaming generation started: prompt_chars=%d, max_tokens=%d",
            len(prompt),
            max_tokens,
        )

        loop = asyncio.get_running_loop()

        try:
            stream = await loop.run_in_executor(
                self._executor,
                lambda: self._model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop or [],
                    stream=True,
                    echo=False,
                ),
            )

            for chunk in stream:
                if chunk["choices"][0]["text"]:
                    yield chunk["choices"][0]["text"]

        except Exception as e:
            logger.error("Streaming generation failed: prompt_chars=%d", len(prompt))
            logger.debug("Streaming failure details: %s", e, exc_info=True)
            raise GenerationError("Streaming generation failed") from e

        logger.debug("Streaming generation completed")

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string.

        Args:
            text: Input text to tokenize.

        Returns:
            Token count using model's tokenizer.

        Raises:
            GenerationError: If model not loaded.
        """
        if self._model is None:
            raise GenerationError("Model not loaded. Use async with context manager.")

        tokens = self._model.tokenize(text.encode("utf-8"))
        token_count = len(tokens)

        logger.debug("Token counting: text_chars=%d, tokens=%d", len(text), token_count)

        return token_count

    @property
    def context_window(self) -> int:
        """Get the maximum context window size.

        Returns:
            Model's context window capacity in tokens.
        """
        return self._n_ctx
