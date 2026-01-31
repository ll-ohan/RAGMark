"""Abstract base classes for retrieval and reranking.

This module defines the contracts for retrieval strategies and
reranking/refinement operations.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ragmark.schemas.retrieval import RetrievedNode, TraceContext

if TYPE_CHECKING:
    from ragmark.config.profile import RerankerConfig, RetrievalConfig
    from ragmark.index.base import VectorIndex


class BaseRetriever(ABC):
    """Defines the contract for retrieval strategies.

    Retrievers are responsible for finding relevant knowledge nodes
    given a user query. They encapsulate the embedding, search, and
    optional reranking logic.

    All retrieval operations are async to support efficient I/O with
    vector databases and embedding models.
    """

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        config: "RetrievalConfig",
        index: "VectorIndex",
        refiner: "BaseRefiner | None" = None,
    ) -> "BaseRetriever":
        """Instantiate a retriever from configuration.

        Args:
            config: The retrieval configuration.
            index: The vector index to search.
            refiner: The reranking strategy to apply, if any.

        Returns:
            The configured retriever instance.
        """
        pass

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 10) -> TraceContext:
        """Retrieve relevant knowledge nodes for a query.

        Args:
            query: The search query.
            top_k: The maximum number of nodes to retrieve.

        Returns:
            The trace containing nodes and metadata.

        Raises:
            RetrievalError: If the underlying search operation fails.
        """
        pass

    async def retrieve_batch(
        self, queries: list[str], top_k: int = 10
    ) -> list[TraceContext]:
        """Retrieve results for multiple queries concurrently.

        The default implementation uses asyncio.gather for concurrent execution.

        Args:
            queries: The search queries.
            top_k: The maximum number of nodes to retrieve per query.

        Returns:
            The traces corresponding to the input queries.

        Raises:
            RetrievalError: If any retrieval operation fails.
        """
        tasks = [self.retrieve(query, top_k) for query in queries]
        return await asyncio.gather(*tasks)


class BaseRefiner(ABC):
    """Defines the contract for reranking and refinement strategies.

    Refiners take an initial set of retrieved candidates and reorder
    them to improve precision, typically using more expensive models
    like cross-encoders.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: "RerankerConfig") -> "BaseRefiner":
        """Instantiate a refiner from configuration.

        Args:
            config: The reranker configuration.

        Returns:
            The configured refiner instance.
        """
        pass

    @abstractmethod
    def refine(
        self,
        query: str,
        candidates: list[RetrievedNode],
        top_k: int,
    ) -> list[RetrievedNode]:
        """Rerank retrieved candidates.

        Args:
            query: The search query.
            candidates: The nodes to reorder.
            top_k: The maximum number of nodes to return.

        Returns:
            The reordered nodes.

        Raises:
            RetrievalError: If the reranking operation fails.
        """
        pass
