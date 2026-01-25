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

    This class defines the contract for all vector database implementations,
    providing a consistent async interface for CRUD operations and search.

    All I/O operations are async to support efficient concurrent processing
    and non-blocking database interactions.
    """

    @classmethod
    @abstractmethod
    def from_config(
        cls, config: IndexConfig, embedder: BaseEmbedder | None = None
    ) -> "VectorIndex":
        """Instantiate the index from a configuration object.

        Args:
            config: The index configuration.
            embedder: Optional embedder instance.

        Returns:
            An instance of the specific VectorIndex subclass.
        """
        pass

    @abstractmethod
    async def add(self, nodes: list[KnowledgeNode]) -> None:
        """Add knowledge nodes to the index.

        Nodes with missing embeddings (dense_vector=None) should trigger
        an error unless the index has an associated embedder configured.

        Args:
            nodes: Knowledge nodes to index.

        Raises:
            IndexError: If insertion fails.
            ValueError: If nodes lack required embeddings.
        """
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: list[float] | dict[int, float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors using dense embeddings.

        Args:
            query_vector: Dense query embedding vector or sparse vector.
            top_k: Number of results to return.
            filters: Optional metadata filters (backend-specific syntax).

        Returns:
            List of search results ordered by descending similarity.

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
        """Search using hybrid dense + sparse retrieval.

        The alpha parameter controls the fusion weight:
        - alpha=0.0: sparse only
        - alpha=1.0: dense only
        - alpha=0.5: equal weight

        Args:
            dense_vector: Dense query embedding.
            sparse_vector: Sparse query embedding (token_id -> weight).
            top_k: Number of results to return.
            alpha: Fusion weight for dense vs sparse (0.0 to 1.0).

        Returns:
            List of search results with fused scores.

        Raises:
            IndexError: If search fails.
            ValueError: If alpha is out of range.
        """
        pass

    @abstractmethod
    async def delete(self, node_ids: list[str]) -> int:
        """Delete nodes from the index by ID.

        Args:
            node_ids: List of node IDs to delete.

        Returns:
            Number of nodes actually deleted.

        Raises:
            IndexError: If deletion fails.
        """
        pass

    @abstractmethod
    async def count(self) -> int:
        """Get the total number of indexed nodes.

        Returns:
            Total count of nodes in the index.

        Raises:
            IndexError: If count operation fails.
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Delete all nodes from the index.

        This operation is typically faster than deleting nodes individually.

        Raises:
            IndexError: If clear operation fails.
        """
        pass

    @abstractmethod
    async def exists(self, node_id: str) -> bool:
        """Check if a node exists in the index.

        Args:
            node_id: Node ID to check.

        Returns:
            True if the node exists in the index.

        Raises:
            IndexError: If existence check fails.
        """
        pass

    async def __aenter__(self) -> "VectorIndex":
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

        Implementations should clean up connections and resources here.
        """
        pass
