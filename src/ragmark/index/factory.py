"""Factory for creating vector index instances.

This module provides factory functions for instantiating the appropriate
vector index backend based on configuration.
"""

import logging

from ragmark.config.profile import EmbedderConfig, IndexConfig
from ragmark.exceptions import UnsupportedBackendError
from ragmark.index.base import VectorIndex
from ragmark.index.embedders import BaseEmbedder, EmbedderFactory

logger = logging.getLogger(__name__)

_INDEX_REGISTRY: dict[str, type[VectorIndex]] = {}


def register_index_backend(name: str, cls: type[VectorIndex]) -> None:
    """Register a custom index backend.

    Enables adding custom implementations without modifying the core library.

    Args:
        name: Unique backend identifier.
        cls: Implementation class.
    """
    _INDEX_REGISTRY[name] = cls
    logger.debug(f"Registered index backend: {name}")


class IndexFactory:
    """Factory for creating vector index instances."""

    @staticmethod
    def create(
        config: IndexConfig,
        embedder_config: EmbedderConfig | None = None,
        embedder: BaseEmbedder | None = None,
        validate_connection: bool = False,
    ) -> VectorIndex:
        """Create a vector index instance from configuration.

        Args:
            config: Configuration parameters.
            embedder_config: Configuration for internal embedder creation.
            embedder: Pre-existing embedder instance. Overrides embedder_config.
            validate_connection: Test connection before returning.

        Returns:
            Configured index instance.

        Raises:
            UnsupportedBackendError: If backend is unknown or missing dependencies.
        """
        backend = config.backend

        if embedder is None and embedder_config is not None:
            embedder = EmbedderFactory.create(embedder_config)
            logger.debug(f"Created embedder: {embedder_config.model_name}")

        if backend in _INDEX_REGISTRY:
            return _INDEX_REGISTRY[backend].from_config(config, embedder=embedder)

        elif backend == "memory":
            from ragmark.index.backends import MemoryIndex

            return MemoryIndex.from_config(config, embedder=embedder)

        elif backend == "qdrant":
            try:
                from ragmark.index.backends import QdrantIndex

                return QdrantIndex.from_config(config, embedder=embedder)
            except ImportError as e:
                raise UnsupportedBackendError("qdrant", "qdrant") from e

        elif backend == "milvus":
            try:
                from ragmark.index.backends import MilvusIndex

                return MilvusIndex.from_config(config, embedder=embedder)
            except ImportError as e:
                raise UnsupportedBackendError("milvus", "milvus") from e

        elif backend == "lancedb":
            try:
                from ragmark.index.backends import LanceDBIndex

                return LanceDBIndex.from_config(config, embedder=embedder)
            except ImportError as e:
                raise UnsupportedBackendError("lancedb", "lancedb") from e

        else:
            raise UnsupportedBackendError(backend)
