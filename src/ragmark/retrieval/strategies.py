"""Retrieval strategy implementations.

This module contains concrete implementations of retrieval strategies
(Dense, Sparse, Hybrid, Refined) for vector-based document retrieval.
"""

import asyncio
import time

from ragmark.config.profile import RetrievalConfig
from ragmark.exceptions import RetrievalError
from ragmark.index.base import VectorIndex
from ragmark.index.embedders import BaseEmbedder
from ragmark.logger import get_logger
from ragmark.retrieval.base import BaseRefiner, BaseRetriever
from ragmark.schemas.documents import KnowledgeNode
from ragmark.schemas.retrieval import RetrievedNode, SearchResult, TraceContext

logger = get_logger(__name__)


class DenseRetriever(BaseRetriever):
    """Dense-only retrieval strategy using embeddings.

    This retriever performs semantic search using dense embeddings.
    It embeds the query and searches the vector index for similar nodes.

    Attributes:
        index: The vector index to search.
        embedder: The embedder used for query encoding.
        top_k: The default number of results to return.
    """

    def __init__(self, index: VectorIndex, embedder: BaseEmbedder, top_k: int = 10):
        """Initialize the dense retriever strategy.

        Args:
            index: The vector index to search.
            embedder: The embedder used for query encoding.
            top_k: The default number of results to return.
        """
        self.index = index
        self.embedder = embedder
        self.top_k = top_k

    @classmethod
    def from_config(
        cls,
        config: RetrievalConfig,
        index: VectorIndex,
        refiner: BaseRefiner | None = None,
    ) -> "DenseRetriever":
        """Instantiate the dense retriever from configuration.

        Args:
            config: The retrieval configuration.
            index: The vector index to search.
            refiner: Ignored for dense retrieval.

        Returns:
            The configured retriever instance.

        Raises:
            ValueError: If the index does not have an embedder configured.
        """
        if not hasattr(index, "embedder") or index.embedder is None:
            raise ValueError("DenseRetriever requires an index with an embedder")

        return cls(index=index, embedder=index.embedder, top_k=config.top_k)

    async def retrieve(self, query: str, top_k: int = 10) -> TraceContext:
        """Retrieve relevant nodes for a query.

        Args:
            query: The search query.
            top_k: The maximum number of nodes to return.

        Returns:
            The trace containing nodes and metadata.

        Raises:
            RetrievalError: If the retrieval operation fails.
        """
        logger.debug(
            "Dense retrieval started: query_len=%d, top_k=%d", len(query), top_k
        )

        try:
            # Run in executor to prevent blocking the event loop
            embed_start = time.perf_counter()
            logger.debug("Embedding query")
            loop = asyncio.get_running_loop()
            query_vectors = await loop.run_in_executor(
                None, self.embedder.embed, [query]
            )
            query_vector = query_vectors[0]
            embed_time_ms = (time.perf_counter() - embed_start) * 1000

            logger.debug("Query embedded: time=%.2fms", embed_time_ms)

            search_start = time.perf_counter()
            logger.debug("Searching index: top_k=%d", top_k)
            results = await self.index.search(query_vector, top_k=top_k)
            search_time_ms = (time.perf_counter() - search_start) * 1000

            logger.debug(
                "Search completed: results=%d, time=%.2fms",
                len(results),
                search_time_ms,
            )

            retrieved_nodes: list[RetrievedNode] = []
            for rank, result in enumerate(results, start=1):
                node = self._build_node_from_result(result)

                retrieved_nodes.append(
                    RetrievedNode(node=node, score=result.score, rank=rank)
                )

            total_time_ms = embed_time_ms + search_time_ms
            logger.info(
                "Dense retrieval completed: results=%d, total_time=%.2fms",
                len(retrieved_nodes),
                total_time_ms,
            )

            return TraceContext(
                query=query,
                retrieved_nodes=retrieved_nodes,
                retrieval_metadata={
                    "strategy": "dense",
                    "embedding_time_ms": embed_time_ms,
                    "search_time_ms": search_time_ms,
                    "total_time_ms": total_time_ms,
                    "top_k": top_k,
                },
                reranked=False,
            )

        except Exception as e:
            logger.error("Dense retrieval failed")
            logger.debug("Retrieval error details: %s", e, exc_info=True)
            raise RetrievalError(f"Dense retrieval failed: {e}") from e

    def _build_node_from_result(self, result: SearchResult) -> KnowledgeNode:
        if result.node is not None:
            return result.node

        raise RetrievalError(
            f"Failed to retrieve complete payload for node_id={result.node_id}. "
            "The index must be configured to return full node data."
        )


class SparseRetriever(BaseRetriever):
    """Sparse-only retrieval strategy (to be implemented).

    Attributes:
        index: The vector index to search.
        top_k: The default number of results to return.
    """

    def __init__(self, index: VectorIndex, top_k: int = 10):
        """Initialize the sparse retriever.

        Args:
            index: The vector index to search.
            top_k: The default number of results to return.
        """
        self.index = index
        self.top_k = top_k

    @classmethod
    def from_config(
        cls,
        config: RetrievalConfig,
        index: VectorIndex,
        refiner: BaseRefiner | None = None,
    ) -> "SparseRetriever":
        """Instantiate the sparse retriever from configuration.

        Args:
            config: The retrieval configuration.
            index: The vector index to search.
            refiner: Ignored for sparse retrieval.

        Returns:
            Configured retriever instance.
        """
        return cls(index=index, top_k=config.top_k)

    async def retrieve(self, query: str, top_k: int = 10) -> TraceContext:
        raise NotImplementedError("SparseRetriever will be implemented in Phase 2")


class HybridRetriever(BaseRetriever):
    """Hybrid retrieval strategy with RRF fusion (to be implemented).

    Attributes:
        index: The vector index to search.
        top_k: The default number of results to return.
        alpha: The weight for dense scores (0.0 to 1.0).
    """

    def __init__(self, index: VectorIndex, top_k: int = 10, alpha: float = 0.5):
        """Initialize the hybrid retriever.

        Args:
            index: The vector index to search.
            top_k: The default number of results to return.
            alpha: The weight for dense scores (0.0 to 1.0).
        """
        self.index = index
        self.top_k = top_k
        self.alpha = alpha

    @classmethod
    def from_config(
        cls,
        config: RetrievalConfig,
        index: VectorIndex,
        refiner: BaseRefiner | None = None,
    ) -> "HybridRetriever":
        """Instantiate the hybrid retriever from configuration.

        Args:
            config: The retrieval configuration.
            index: The vector index to search.
            refiner: Ignored for hybrid retrieval.

        Returns:
            Configured retriever instance.

        Raises:
            ValueError: If alpha is not provided in config.
        """
        if config.alpha is None:
            raise ValueError("alpha is required for hybrid retrieval")
        return cls(index=index, top_k=config.top_k, alpha=config.alpha)

    async def retrieve(self, query: str, top_k: int = 10) -> TraceContext:
        raise NotImplementedError("HybridRetriever will be implemented in Phase 2")


class RefinedRetriever(BaseRetriever):
    """Retriever with reranking (to be implemented).

    Attributes:
        base_retriever: The underlying retrieval strategy.
        refiner: The reranking strategy to apply.
    """

    def __init__(self, base_retriever: BaseRetriever, refiner: BaseRefiner):
        """Initialize the refined retriever.

        Args:
            base_retriever: The underlying retrieval strategy.
            refiner: The reranking strategy to apply.
        """
        self.base_retriever = base_retriever
        self.refiner = refiner

    @classmethod
    def from_config(
        cls,
        config: RetrievalConfig,
        index: VectorIndex,
        refiner: BaseRefiner | None = None,
    ) -> "RefinedRetriever":
        """Instantiate the refined retriever from configuration.

        Args:
            config: The retrieval configuration.
            index: The vector index to search.
            refiner: The refiner strategy for reranking (required).

        Returns:
            Configured retriever instance.

        Raises:
            ValueError: If refiner is not provided or mode is unknown.
        """
        if refiner is None:
            raise ValueError("RefinedRetriever requires a refiner")

        base_retriever: BaseRetriever
        if config.mode == "dense":
            base_retriever = DenseRetriever.from_config(config, index)
        elif config.mode == "sparse":
            base_retriever = SparseRetriever.from_config(config, index)
        elif config.mode == "hybrid":
            base_retriever = HybridRetriever.from_config(config, index)
        else:
            raise ValueError(f"Unknown retrieval mode: {config.mode}")

        return cls(base_retriever=base_retriever, refiner=refiner)

    async def retrieve(self, query: str, top_k: int = 10) -> TraceContext:
        raise NotImplementedError("RefinedRetriever will be implemented in Phase 2")
