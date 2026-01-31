"""Integration tests for retrieval strategies using Fakes."""

from collections.abc import Callable
from typing import Any

import pytest

from ragmark.config.profile import RetrievalConfig
from ragmark.exceptions import RetrievalError
from ragmark.retrieval.strategies import (
    DenseRetriever,
    HybridRetriever,
    RefinedRetriever,
    SparseRetriever,
)
from ragmark.schemas.documents import KnowledgeNode, NodePosition
from ragmark.schemas.retrieval import SearchResult


@pytest.mark.unit
class TestRetrievalStrategies:
    """Test suite for retrieval logic flow and error handling."""

    @pytest.mark.asyncio
    async def test_dense_retriever_should_return_ranked_nodes_successfully(
        self, fake_vector_index_factory: Callable[..., Any]
    ) -> None:
        """Verify the complete data flow: Query -> Embed -> Search -> Trace.

        1. Given a populated index with a known node
        2. When performing a retrieve operation
        3. Then the returned TraceContext contains the correct node and score
        4. And the index received the correct parameters
        """
        expected_node = KnowledgeNode(
            node_id="test_node",
            content="Test Content",
            source_id="src1",
            position=NodePosition(start_char=0, end_char=10, page=1, section="Test"),
            metadata={},
            dense_vector=[0.1] * 3,
            sparse_vector=None,
        )
        fake_results = [
            SearchResult(node_id="test_node", score=0.99, node=expected_node)
        ]
        fake_index = fake_vector_index_factory(search_results=fake_results, dimension=3)

        retriever = DenseRetriever(
            index=fake_index, embedder=fake_index.embedder, top_k=5
        )

        query_top_k = 1
        trace = await retriever.retrieve("my query", top_k=query_top_k)

        assert trace.query == "my query"
        assert trace.retrieval_metadata["strategy"] == "dense"
        assert "embedding_time_ms" in trace.retrieval_metadata
        assert "search_time_ms" in trace.retrieval_metadata
        assert "total_time_ms" in trace.retrieval_metadata
        assert len(trace.retrieved_nodes) == 1

        retrieved = trace.retrieved_nodes[0]
        assert retrieved.node.content == "Test Content"
        assert retrieved.score == 0.99
        assert retrieved.rank == 1

        assert len(fake_index.call_history) == 1
        last_call = fake_index.call_history[0]
        assert last_call["top_k"] == query_top_k

    @pytest.mark.asyncio
    async def test_dense_retriever_should_wrap_internal_errors_preserving_cause(
        self, fake_vector_index_factory: Callable[..., Any]
    ) -> None:
        """Ensure backend errors are wrapped in RetrievalError with trace preservation.

        1. Given an index configured to fail with a ValueError
        2. When retrieve is called
        3. Then a RetrievalError is raised and __cause__ is the original ValueError
        """
        fake_index = fake_vector_index_factory()
        fake_index._should_fail = True
        fake_index._fail_exception = ValueError("DB Connection Failed")

        retriever = DenseRetriever(index=fake_index, embedder=fake_index.embedder)

        with pytest.raises(RetrievalError) as exc_info:
            await retriever.retrieve("query")

        assert exc_info.value.__cause__ is not None
        assert "DB Connection Failed" in str(exc_info.value.__cause__)

    @pytest.mark.asyncio
    async def test_dense_retriever_should_raise_error_when_backend_returns_empty_payload(
        self, fake_vector_index_factory: Callable[..., Any]
    ) -> None:
        """Ensure specific error when search succeeds but node payload is missing.

        1. Given an index returning a SearchResult without a Node object (data inconsistency)
        2. When retrieve is called
        3. Then a RetrievalError is raised specifically for missing payload
        """
        corrupted_result = SearchResult(node_id="phantom_node", score=0.88, node=None)
        fake_index = fake_vector_index_factory(
            search_results=[corrupted_result], dimension=3
        )

        retriever = DenseRetriever(index=fake_index, embedder=fake_index.embedder)

        with pytest.raises(RetrievalError, match="Failed to retrieve complete payload"):
            await retriever.retrieve("query")

    @pytest.mark.asyncio
    async def test_dense_retriever_from_config_should_create_instance_correctly(
        self,
        fake_vector_index_factory: Callable[..., Any],
        retrieval_config_factory: Callable[..., RetrievalConfig],
    ) -> None:
        """Verify successful instantiation from configuration.

        1. Given a valid config and an index with an embedder
        2. When creating a DenseRetriever from config
        3. Then the instance is created with correct parameters
        """
        fake_index = fake_vector_index_factory()
        config = retrieval_config_factory(top_k=42)

        retriever = DenseRetriever.from_config(config, fake_index)

        assert isinstance(retriever, DenseRetriever)
        assert retriever.top_k == 42
        assert retriever.embedder is fake_index.embedder

    @pytest.mark.asyncio
    async def test_dense_retriever_from_config_should_raise_value_error_missing_embedder(
        self,
        fake_vector_index_factory: Callable[..., Any],
        retrieval_config_factory: Callable[..., RetrievalConfig],
    ) -> None:
        """Ensure from_config validates the presence of an embedder.

        1. Given a config and an index without an embedder
        2. When creating a DenseRetriever from config
        3. Then a ValueError is raised
        """
        fake_index = fake_vector_index_factory(use_default_embedder=False)
        config = retrieval_config_factory()

        with pytest.raises(ValueError, match="requires an index with an embedder"):
            DenseRetriever.from_config(config, fake_index)

    @pytest.mark.asyncio
    async def test_sparse_retriever_should_raise_not_implemented_error(
        self, fake_vector_index_factory: Callable[..., Any]
    ) -> None:
        """Ensure SparseRetriever is not yet implemented.

        1. Given a SparseRetriever initialized with a valid index
        2. When retrieve is called
        3. Then NotImplementedError is raised
        """
        fake_index = fake_vector_index_factory()
        sparse = SparseRetriever(index=fake_index)

        with pytest.raises(NotImplementedError, match="Phase 2"):
            await sparse.retrieve("query")

    @pytest.mark.asyncio
    async def test_hybrid_retriever_should_raise_not_implemented_error(
        self, fake_vector_index_factory: Callable[..., Any]
    ) -> None:
        """Ensure HybridRetriever is not yet implemented.

        1. Given a HybridRetriever initialized with a valid index
        2. When retrieve is called
        3. Then NotImplementedError is raised
        """
        fake_index = fake_vector_index_factory()
        hybrid = HybridRetriever(index=fake_index, alpha=0.5)

        with pytest.raises(NotImplementedError, match="Phase 2"):
            await hybrid.retrieve("query")

    @pytest.mark.asyncio
    async def test_refined_retriever_should_raise_not_implemented_error(
        self, fake_vector_index_factory: Callable[..., Any]
    ) -> None:
        """Ensure RefinedRetriever is not yet implemented.

        1. Given a RefinedRetriever initialized with base retriever and refiner
        2. When retrieve is called
        3. Then NotImplementedError is raised
        """
        fake_index = fake_vector_index_factory()
        base_dense = DenseRetriever(index=fake_index, embedder=fake_index.embedder)
        mock_refiner = fake_index
        refined = RefinedRetriever(base_retriever=base_dense, refiner=mock_refiner)

        with pytest.raises(NotImplementedError, match="Phase 2"):
            await refined.retrieve("query")

    @pytest.mark.asyncio
    async def test_dense_retriever_should_fail_gracefully_on_dimension_mismatch(
        self,
        fake_vector_index_factory: Callable[..., Any],
        fake_embedder_factory: Callable[..., Any],
    ) -> None:
        """Ensure dimension mismatch between Embedder and Index raises RetrievalError.

        1. Given an index (dim=10) and an embedder (dim=384)
        2. When retrieve is called
        3. Then a RetrievalError is raised causing the ValueError
        """
        mismatched_embedder = fake_embedder_factory(dimension=384)
        fake_index = fake_vector_index_factory(
            dimension=10, embedder=mismatched_embedder
        )

        retriever = DenseRetriever(index=fake_index, embedder=mismatched_embedder)

        with pytest.raises(RetrievalError) as exc_info:
            await retriever.retrieve("query")

        assert exc_info.value.__cause__ is not None
        assert "dimension mismatch" in str(exc_info.value.__cause__).lower()

    @pytest.mark.asyncio
    async def test_refined_retriever_from_config_should_initialize_correct_base_strategy(
        self,
        fake_vector_index_factory: Callable[..., Any],
        retrieval_config_factory: Callable[..., RetrievalConfig],
    ) -> None:
        """Verify RefinedRetriever factory correctly builds the internal base retriever.

        1. Given a config requesting 'dense' mode and a refiner
        2. When creating RefinedRetriever from config
        3. Then the base_retriever attribute is an instance of DenseRetriever
        """
        fake_index = fake_vector_index_factory()
        mock_refiner = fake_index

        config = retrieval_config_factory(mode="dense")

        retriever = RefinedRetriever.from_config(
            config, fake_index, refiner=mock_refiner
        )

        assert isinstance(retriever.base_retriever, DenseRetriever)
        assert retriever.refiner is mock_refiner
