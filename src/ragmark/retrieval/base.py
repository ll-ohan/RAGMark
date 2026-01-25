"""Abstract base classes for retrieval and reranking.

This module defines the contracts for retrieval strategies and
reranking/refinement operations.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ragmark.schemas.retrieval import RetrievedNode, TraceContext

if TYPE_CHECKING:
    from ragmark.config.profile import RerankerConfig, RetrievalConfig
    from ragmark.index.base import VectorIndex


class BaseRetriever(ABC):
    """Abstract base class for retrieval strategies.

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
        """Instantiate retriever from configuration.

        Args:
            config: RetrievalConfig instance.
            index: Vector index to search.
            refiner: Optional refiner for reranking.

        Returns:
            Configured retriever instance.
        """
        pass

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 10) -> TraceContext:
        """Retrieve relevant knowledge nodes for a query.

        Args:
            query: User query string.
            top_k: Number of results to return.

        Returns:
            Complete trace context with retrieved nodes and metadata.

        Raises:
            RetrievalError: If retrieval fails.
        """
        pass

    async def retrieve_batch(
        self, queries: list[str], top_k: int = 10
    ) -> list[TraceContext]:
        """Retrieve results for multiple queries concurrently.

        Default implementation uses asyncio.gather for concurrent execution.

        Args:
            queries: List of query strings.
            top_k: Number of results per query.

        Returns:
            List of trace contexts (one per query).

        Raises:
            RetrievalError: If any retrieval fails.
        """
        import asyncio

        tasks = [self.retrieve(query, top_k) for query in queries]
        return await asyncio.gather(*tasks)


class BaseRefiner(ABC):
    """Abstract base class for reranking/refinement strategies.

    Refiners take an initial set of retrieved candidates and reorder
    them to improve precision, typically using more expensive models
    like cross-encoders.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: "RerankerConfig") -> "BaseRefiner":
        """Instantiate refiner from configuration.

        Args:
            config: RerankerConfig instance.

        Returns:
            Configured refiner instance.
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
            query: Original user query.
            candidates: Initial retrieval results to rerank.
            top_k: Number of results to keep after reranking.

        Returns:
            Reranked list of retrieved nodes (truncated to top_k).

        Raises:
            RetrievalError: If reranking fails.
        """
        pass
