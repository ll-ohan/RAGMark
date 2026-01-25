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

# Registry for index backends
_INDEX_REGISTRY: dict[str, type[VectorIndex]] = {}


def register_index_backend(name: str, cls: type[VectorIndex]) -> None:
    """Register a custom index backend.

    This allows users to add custom index implementations without
    modifying the core library.

    Args:
        name: Backend identifier (e.g., 'custom_db').
        cls: VectorIndex implementation class.
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
            config: Index configuration.
            embedder_config: Optional embedder configuration. If provided,
                an embedder will be created and attached to the index.
            embedder: Optional pre-configured embedder instance. Takes
                precedence over embedder_config if both are provided.
            validate_connection: If True, test the connection before returning.

        Returns:
            Configured VectorIndex instance.

        Raises:
            UnsupportedBackendError: If backend is not installed or registered.
            IndexError: If connection validation fails.
        """
        backend = config.backend

        # Create embedder if configuration is provided
        if embedder is None and embedder_config is not None:
            embedder = EmbedderFactory.create(embedder_config)
            logger.debug(f"Created embedder: {embedder_config.model_name}")
        index_class: type[VectorIndex]
        # Check registry first
        if backend in _INDEX_REGISTRY:
            index_class = _INDEX_REGISTRY[backend]

        # Native backends - Import from backends module
        elif backend == "memory":
            from ragmark.index.backends import MemoryIndex

            index_class = MemoryIndex

        elif backend == "qdrant":
            try:
                from ragmark.index.backends import QdrantIndex
            except ImportError as e:
                raise UnsupportedBackendError("qdrant", "qdrant") from e
            index_class = QdrantIndex

        elif backend == "milvus":
            try:
                from ragmark.index.backends import MilvusIndex
            except ImportError as e:
                raise UnsupportedBackendError("milvus", "milvus") from e

            index_class = MilvusIndex

        elif backend == "lancedb":
            try:
                from ragmark.index.backends import LanceDBIndex
            except ImportError as e:
                raise UnsupportedBackendError("lancedb", "lancedb") from e

            index_class = LanceDBIndex
        else:
            raise UnsupportedBackendError(backend)

        return index_class.from_config(config, embedder=embedder)
