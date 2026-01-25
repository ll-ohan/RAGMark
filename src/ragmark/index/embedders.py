"""Abstract base class for embedding models.

This module defines the interface for computing dense and sparse
embeddings from text, supporting both synchronous and batch processing.
"""

from abc import ABC, abstractmethod

from ragmark.config.profile import EmbedderConfig


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
            config: EmbedderConfig instance.

        Returns:
            Configured embedder instance.
        """
        pass

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Compute dense embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of dense embedding vectors (one per input text).

        Raises:
            EmbeddingError: If embedding computation fails.
        """
        pass

    def embed_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        """Compute sparse embeddings for a batch of texts.

        This is optional and only needed for hybrid retrieval strategies.
        Default implementation returns empty sparse vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of sparse embedding dictionaries (token_id -> weight).

        Raises:
            NotImplementedError: If sparse embeddings are not supported.
        """
        return [{} for _ in texts]

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Get the dimensionality of dense embeddings.

        Returns:
            Embedding vector dimensionality.
        """
        pass

    @property
    def supports_sparse(self) -> bool:
        """Check if this embedder supports sparse embeddings.

        Returns:
            True if embed_sparse is implemented.
        """
        return False


class EmbedderFactory:
    """Factory for creating embedder instances."""

    @staticmethod
    def create(config: EmbedderConfig) -> BaseEmbedder:
        """Create an embedder from configuration.

        Args:
            config: EmbedderConfig instance.

        Returns:
            Configured embedder instance.

        Raises:
            UnsupportedBackendError: If model is not available.
        """
        from ragmark.exceptions import UnsupportedBackendError

        # Try to import sentence-transformers
        try:
            from ragmark.index.backends import SentenceTransformerEmbedder

            return SentenceTransformerEmbedder.from_config(config)
        except ImportError as e:
            raise UnsupportedBackendError(
                "sentence-transformers",
                "embeddings",
            ) from e
