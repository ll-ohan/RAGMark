"""Abstract base class for embedding models.

This module defines the interface for computing dense and sparse
embeddings from text, supporting both synchronous and batch processing.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ragmark.config.profile import EmbedderConfig
from ragmark.exceptions import EmbeddingError, UnsupportedBackendError
from ragmark.logger import get_logger

logger = get_logger(__name__)


class BaseEmbedder(ABC):
    """Abstract base class for embedding models.

    Embedders transform text into vector representations suitable for
    similarity search. This interface supports both dense (continuous)
    and sparse (token-based) embeddings.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: EmbedderConfig) -> "BaseEmbedder":
        """Instantiate embedder from configuration.

        Args:
            config: Configuration parameters.

        Returns:
            Configured embedder instance.
        """
        pass

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Compute dense embeddings for a batch of texts.

        Args:
            texts: Input texts to process.

        Returns:
            Dense embedding vectors.

        Raises:
            EmbeddingError: If embedding computation fails.
        """
        pass

    @abstractmethod
    def embed_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        """Compute sparse embeddings for a batch of texts.

        This is optional and only needed for hybrid retrieval strategies.

        Args:
            texts: Input texts to process.

        Returns:
            Sparse embedding dictionaries (token_id -> weight).
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Get the dimensionality of dense embeddings.

        Returns:
            Vector dimensionality.
        """
        pass

    @property
    def supports_sparse(self) -> bool:
        """Check if this embedder supports sparse embeddings.

        Returns:
            Whether sparse embeddings are supported.
        """
        return False


class SentenceTransformerEmbedder(BaseEmbedder):
    """Sentence transformer embedder using HuggingFace models.

    This embedder uses the sentence-transformers library to generate
    dense embeddings from text. Models are lazy-loaded on first use
    to avoid startup overhead.

    Attributes:
        model_name: HuggingFace model identifier.
        device: Target hardware for inference.
        batch_size: Number of texts to process at once.
    """

    def __init__(
        self,
        model_name: str | Path,
        device: str = "cpu",
        batch_size: int = 32,
        rate_limit: float | None = None,
    ):
        """Initialize the sentence transformer embedder.

        Args:
            model_name: HuggingFace model identifier.
            device: Target hardware for inference.
            batch_size: Number of texts to process at once.
            rate_limit: Maximum embedding batches per second for rate limiting.
        """
        self.model_name: str | Path = model_name
        self.device = device
        self.batch_size = batch_size
        self.rate_limit = rate_limit
        self._model: Any | None = None
        self._dim: int | None = None

        if rate_limit is not None:
            from ragmark.index.rate_limiter import RateLimiter

            self._rate_limiter: RateLimiter | None = RateLimiter(rate_limit)
        else:
            self._rate_limiter = None

    @classmethod
    def from_config(cls, config: EmbedderConfig) -> "SentenceTransformerEmbedder":
        """Create SentenceTransformerEmbedder from configuration.

        Args:
            config: Configuration parameters.

        Returns:
            Configured embedder instance.
        """
        return cls(
            model_name=config.model_name,
            device=config.device,
            batch_size=config.batch_size,
            rate_limit=config.rate_limit,
        )

    def _load_model(self) -> None:
        if self._model is not None:
            logger.debug("Model already loaded: %s", self.model_name)
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            logger.error("sentence-transformers not installed")
            logger.debug("Import error details: %s", e, exc_info=True)
            raise UnsupportedBackendError("sentence-transformers", "embed") from e

        try:
            logger.info("Loading embedding model: %s", self.model_name)

            self._model = SentenceTransformer(str(self.model_name), device=self.device)
            self._dim = self._model.get_sentence_embedding_dimension()

            logger.info(
                "Model loaded: name=%s, dimension=%d", self.model_name, self._dim
            )
        except Exception as e:
            logger.error("Failed to load model: name=%s", self.model_name)
            logger.debug("Model loading error: %s", e, exc_info=True)
            raise EmbeddingError(
                f"Failed to load model '{self.model_name}': {e}"
            ) from e

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Compute dense embeddings for a list of texts.

        Args:
            texts: Input texts to process.

        Returns:
            Dense embedding vectors.

        Raises:
            EmbeddingError: If embedding computation fails.
        """
        if not texts:
            logger.debug("No texts to embed, returning empty list")
            return []

        logger.debug(
            "Computing embeddings: count=%d, batch_size=%d", len(texts), self.batch_size
        )

        self._load_model()

        if self._model is None:
            logger.error("Model not loaded: %s", self.model_name)
            raise EmbeddingError(f"Failed to load model '{self.model_name}'")

        try:
            from typing import cast

            embeddings = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )

            logger.debug("Embeddings computed successfully: count=%d", len(embeddings))
            # Required for Pydantic serialization
            return cast(list[list[float]], embeddings.tolist())

        except Exception as e:
            logger.error("Embedding computation failed")
            logger.debug("Embedding error details: %s", e, exc_info=True)
            raise EmbeddingError(f"Embedding computation failed: {e}") from e

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings asynchronously with optional rate limiting.

        Delegates embedding computation to thread pool to prevent event loop
        blocking, with optional rate limiting for API quota management. Rate
        limiting is applied per internal batch (ceil(len(texts) / batch_size)).

        Args:
            texts: Input texts to embed.

        Returns:
            Dense embedding vectors for each text.

        Raises:
            EmbeddingError: If embedding computation fails.
        """
        import asyncio

        if self._rate_limiter is not None:
            if not texts:
                return []
            batch_count = (len(texts) + self.batch_size - 1) // self.batch_size
            await self._rate_limiter.acquire(tokens=batch_count)

        # Delegate to thread pool (CPU-bound operation)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed, texts)

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimensionality.

        Returns:
            Vector dimensionality.
        """
        if self._dim is None:
            self._load_model()
        return self._dim if self._dim is not None else 0

    def embed_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        """Compute sparse embeddings for a batch of texts.

        Args:
            texts: Input texts to process.

        Raises:
            NotImplementedError: Always, as this backend is dense-only.
        """
        raise NotImplementedError(
            "Sparse embeddings are not supported by SentenceTransformerEmbedder."
        )

    @property
    def supports_sparse(self) -> bool:
        """Check if sparse embeddings are supported.

        Returns:
            Always False for this backend.
        """
        return False


class EmbedderFactory:
    """Factory for creating embedder instances."""

    @staticmethod
    def create(config: EmbedderConfig) -> BaseEmbedder:
        """Create an embedder from configuration.

        Args:
            config: Configuration parameters.

        Returns:
            Initialized embedder.

        Raises:
            UnsupportedBackendError: If model is not available.
        """
        return SentenceTransformerEmbedder.from_config(config)
