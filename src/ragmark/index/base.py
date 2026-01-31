"""Abstract base class for vector index backends.

This module defines the unified interface that all vector database
implementations must provide, ensuring consistent async operations
across different backends.
"""

from abc import ABC, abstractmethod
from typing import Any

from ragmark.config.profile import IndexConfig
from ragmark.index.embedders import BaseEmbedder
from ragmark.schemas.documents import KnowledgeNode
from ragmark.schemas.retrieval import SearchResult


class VectorIndex(ABC):
    """Abstract base class for vector index backends.

    Defines the contract for vector database implementations, providing a
    consistent async interface for CRUD operations and search.

    Attributes:
        config: Index configuration.
        embedder: Optional embedder instance.
    """

    config: IndexConfig | None
    embedder: BaseEmbedder | None

    @classmethod
    @abstractmethod
    def from_config(
        cls, config: IndexConfig, embedder: BaseEmbedder | None = None
    ) -> "VectorIndex":
        """Instantiate index from configuration.

        Args:
            config: Index configuration.
            embedder: Embedder instance.

        Returns:
            Instance of specific VectorIndex subclass.
        """
        pass

    @abstractmethod
    async def add(self, nodes: list[KnowledgeNode]) -> None:
        """Add knowledge nodes to index.

        Nodes with missing embeddings (dense_vector=None) must trigger an error
        unless the index has an associated embedder configured.

        Args:
            nodes: Nodes to index.

        Raises:
            IndexError: If insertion fails.
            ValueError: If nodes lack required embeddings.
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: list[float] | dict[str, list[int] | list[float]],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors using dense embeddings.

        Args:
            query_vector: Dense query embedding or sparse vector.
            top_k: Number of results to return.
            filters: Metadata filters (backend-specific syntax).

        Returns:
            Search results ordered by descending similarity.

        Raises:
            IndexError: If search fails.
        """
        pass

    @abstractmethod
    async def search_hybrid(
        self,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
        top_k: int,
        alpha: float,
    ) -> list[SearchResult]:
        """Perform hybrid dense and sparse retrieval.

        Alpha parameter controls the fusion weight:
        - alpha=0.0: sparse only
        - alpha=1.0: dense only
        - alpha=0.5: equal weight

        Args:
            dense_vector: Dense query embedding.
            sparse_vector: Sparse query embedding (token_id -> weight).
            top_k: Number of results to return.
            alpha: Fusion weight for dense vs sparse (0.0 to 1.0).

        Returns:
            Search results with fused scores.

        Raises:
            IndexError: If search fails.
            ValueError: If alpha is out of range.
        """
        pass

    @abstractmethod
    async def delete(self, node_ids: list[str]) -> int:
        """Delete nodes by ID.

        Args:
            node_ids: Node IDs to delete.

        Returns:
            Count of deleted nodes.

        Raises:
            IndexError: If deletion fails.
        """
        pass

    @abstractmethod
    async def count(self) -> int:
        """Get total number of indexed nodes.

        Returns:
            Node count.

        Raises:
            IndexError: If count operation fails.
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Delete all nodes from index.

        This operation is typically faster than deleting nodes individually.

        Raises:
            IndexError: If clear operation fails.
        """
        pass

    @abstractmethod
    async def exists(self, node_id: str) -> bool:
        """Check if node exists in index.

        Args:
            node_id: Node ID to check.

        Returns:
            True if node exists.

        Raises:
            IndexError: If existence check fails.
        """
        pass

    async def __aenter__(self) -> "VectorIndex":
        """Enter async context manager.

        Returns:
            Instance for context.
        """
        return self

    @abstractmethod
    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> None:
        """Exit async context manager."""
        pass
