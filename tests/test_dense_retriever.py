"""Unit tests for DenseRetriever."""

import asyncio
from collections.abc import Callable
from typing import Any

import pytest
from numpy.testing import assert_allclose

from ragmark.config.profile import RetrievalConfig
from ragmark.exceptions import RetrievalError
from ragmark.retrieval.strategies import DenseRetriever
from ragmark.schemas.documents import KnowledgeNode
from ragmark.schemas.retrieval import SearchResult


@pytest.mark.unit
class TestDenseRetriever:
    """Test suite for DenseRetriever logic using Fakes."""

    def test_from_config_should_raise_value_error_when_index_has_no_embedder(
        self, fake_vector_index_factory: Callable[..., Any]
    ) -> None:
        """Verifies validation logic in factory method.

        Given: An index instance without an attached embedder.
        When: DenseRetriever.from_config is called.
        Then: ValueError is raised to prevent invalid initialization.
        """

        fake_index = fake_vector_index_factory(dimension=3, use_default_embedder=False)
        config = RetrievalConfig(alpha=None, reranker=None, top_k=5)

        with pytest.raises(ValueError, match="requires an index with an embedder"):
            DenseRetriever.from_config(config=config, index=fake_index)

    @pytest.mark.asyncio
    async def test_retrieve_should_return_correct_node_and_score_when_query_is_valid(
        self,
        node_factory: Callable[..., KnowledgeNode],
        fake_embedder_factory: Callable[..., Any],
        fake_vector_index_factory: Callable[..., Any],
    ) -> None:
        """Verifies basic retrieval flow: Embedding -> Search -> Result Construction.

        Given: A populated index and a valid query.
        When: retrieve() is called.
        Then: The specific node is returned with the expected metadata.
        """

        dimension = 3
        fake_embedder = fake_embedder_factory(dimension=dimension)
        fake_index = fake_vector_index_factory(
            dimension=dimension, embedder=fake_embedder
        )

        target_node = node_factory(content="Expected Result")
        fake_index.storage = [
            SearchResult(node_id=target_node.node_id, score=0.95, node=target_node)
        ]

        retriever = DenseRetriever(index=fake_index, embedder=fake_embedder, top_k=10)

        trace = await retriever.retrieve(query="test query", top_k=1)

        assert trace.query == "test query"
        assert len(trace.retrieved_nodes) == 1
        assert trace.retrieved_nodes[0].node.content == "Expected Result"
        assert_allclose(trace.retrieved_nodes[0].score, 0.95, atol=1e-6)
        assert trace.retrieval_metadata["strategy"] == "dense"
        assert trace.retrieval_metadata["total_time_ms"] >= 0

    @pytest.mark.asyncio
    async def test_retrieve_should_wrap_embedding_error_with_retrieval_error_preserving_cause(
        self,
        fake_embedder_factory: Callable[..., Any],
        fake_vector_index_factory: Callable[..., Any],
    ) -> None:
        """Verifies that embedding failures are caught and re-raised properly.

        Given: An embedder that raises a RuntimeError.
        When: retrieve() is called.
        Then: A RetrievalError is raised, and __cause__ points to the original RuntimeError.
        """

        original_error = RuntimeError("HuggingFace API Down")
        fake_embedder = fake_embedder_factory(side_effect=original_error)
        fake_index = fake_vector_index_factory()

        retriever = DenseRetriever(index=fake_index, embedder=fake_embedder)

        with pytest.raises(RetrievalError, match="Dense retrieval failed") as exc_info:
            await retriever.retrieve(query="test")

        assert exc_info.value.__cause__ is original_error

    @pytest.mark.asyncio
    async def test_retrieve_should_wrap_search_error_with_retrieval_error_preserving_cause(
        self,
        fake_embedder_factory: Callable[..., Any],
        fake_vector_index_factory: Callable[..., Any],
    ) -> None:
        """Verifies that index search failures are caught and re-raised properly.

        Given: An index that raises a ConnectionError during search.
        When: retrieve() is called.
        Then: A RetrievalError is raised, and __cause__ points to the original error.
        """

        fake_embedder = fake_embedder_factory(dimension=3)
        fake_index = fake_vector_index_factory(dimension=3)

        original_error = ConnectionError("DB Timeout")

        async def broken_search(*_: Any, **__: Any) -> None:
            raise original_error

        fake_index.search = broken_search

        retriever = DenseRetriever(index=fake_index, embedder=fake_embedder)

        with pytest.raises(RetrievalError, match="Dense retrieval failed") as exc_info:
            await retriever.retrieve(query="test")

        assert exc_info.value.__cause__ is original_error

    @pytest.mark.asyncio
    async def test_retrieve_should_raise_value_error_on_dimension_mismatch(
        self,
        fake_embedder_factory: Callable[..., Any],
        fake_vector_index_factory: Callable[..., Any],
    ) -> None:
        """Verifies that vector dimension mismatches are detected by the index logic.

        Given: Index dim=3, Embedder output dim=2.
        When: retrieve() is called.
        Then: A RetrievalError is raised, caused by the Index's ValueError.
        """

        fake_index = fake_vector_index_factory(dimension=3)
        fake_embedder = fake_embedder_factory(dimension=2)

        retriever = DenseRetriever(index=fake_index, embedder=fake_embedder)

        with pytest.raises(RetrievalError) as exc_info:
            await retriever.retrieve(query="test")

        assert isinstance(exc_info.value.__cause__, ValueError)
        assert "expected 3, got 2" in str(exc_info.value.__cause__)

    @pytest.mark.asyncio
    async def test_retrieve_should_timeout_gracefully_when_embedding_hangs(
        self,
        fake_embedder_factory: Callable[..., Any],
        fake_vector_index_factory: Callable[..., Any],
    ) -> None:
        """Verifies async timeout handling.

        Given: An embedder that sleeps longer than the timeout.
        When: retrieve() is wrapped in asyncio.wait_for.
        Then: asyncio.TimeoutError is raised.
        """

        fake_embedder = fake_embedder_factory(dimension=3, delay=0.2)
        fake_index = fake_vector_index_factory()

        retriever = DenseRetriever(index=fake_index, embedder=fake_embedder)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(retriever.retrieve(query="test"), timeout=0.05)

    @pytest.mark.asyncio
    async def test_retrieve_should_raise_error_on_missing_node_payload(
        self,
        fake_embedder_factory: Callable[..., Any],
        fake_vector_index_factory: Callable[..., Any],
    ) -> None:
        """Verifies data integrity checks on results.

        Given: Index returns a SearchResult with node=None (e.g. payload disabled).
        When: retrieve() is called.
        Then: RetrievalError is raised regarding missing payload.
        """

        fake_embedder = fake_embedder_factory(dimension=3)
        fake_index = fake_vector_index_factory(dimension=3, embedder=fake_embedder)

        fake_index.storage = [SearchResult(node_id="n1", score=0.9, node=None)]

        retriever = DenseRetriever(index=fake_index, embedder=fake_embedder)

        with pytest.raises(RetrievalError, match="complete payload"):
            await retriever.retrieve(query="test")

    @pytest.mark.asyncio
    @pytest.mark.rag_edge_case
    async def test_retrieve_should_handle_unicode_complex_queries_correctly(
        self,
        node_factory: Callable[..., KnowledgeNode],
        fake_embedder_factory: Callable[..., Any],
        fake_vector_index_factory: Callable[..., Any],
    ) -> None:
        """Verifies robustness against complex unicode characters.

        Given: A query containing Emojis and CJK characters.
        When: retrieve() is called.
        Then: The system handles it without encoding errors and returns results.
        """

        complex_query = "Hello 👨‍👩‍👧‍👦 検索"
        fake_embedder = fake_embedder_factory(dimension=3)
        fake_index = fake_vector_index_factory(dimension=3, embedder=fake_embedder)

        target_node = node_factory(content="Unicode Supported")
        fake_index.storage = [SearchResult(node_id="n1", score=1.0, node=target_node)]

        retriever = DenseRetriever(index=fake_index, embedder=fake_embedder)

        trace = await retriever.retrieve(query=complex_query)

        assert trace.query == complex_query
        assert trace.retrieved_nodes[0].node.content == "Unicode Supported"
