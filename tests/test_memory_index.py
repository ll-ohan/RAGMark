"""Unit and Integration tests for MemoryIndex adhering to RAGMark Quality Standards."""

import asyncio
import pickle
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ragmark.exceptions import IndexError as RagIndexError
from ragmark.index.backends import MemoryIndex
from ragmark.schemas.documents import KnowledgeNode, NodePosition
from ragmark.schemas.retrieval import SearchCursor, SearchResult


@pytest.mark.unit
@pytest.mark.rag_edge_case
@pytest.mark.asyncio
class TestMemoryIndexVectors:
    """Test suite for Vector operations and Mathematical Edge Cases."""

    async def test_search_should_handle_zero_vector_stability(
        self, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Verify handling of zero vectors to ensure numerical stability.

        Given:
            An index containing a node with a zero-vector [0, 0, 0].
        When:
            Searching with a zero query vector.
        Then:
            The node is retrieved with a score of 0.0, and no NaNs are produced.
        """
        index: MemoryIndex = memory_index_factory(embedding_dim=3)
        node_zero = KnowledgeNode(
            node_id="zero_node",
            content="Zero Vector",
            source_id="z",
            position=NodePosition(start_char=0, end_char=1, page=1, section="Test"),
            metadata={},
            dense_vector=[0.0, 0.0, 0.0],
            sparse_vector=None,
        )
        await index.add([node_zero])

        results = await index.search(query_vector=[0.0, 0.0, 0.0], top_k=1)

        assert len(results) == 1
        assert not np.isnan(
            results[0].score
        ), "Score should not be NaN even for zero vectors"
        assert_allclose(results[0].score, 0.0, atol=1e-7)

    async def test_search_should_rank_correctly_with_dimension_one(
        self, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Verify scalar operations when embedding dimension is 1.

        Given:
            An index with dim=1 containing positive and negative scalar vectors.
        When:
            Searching with a positive query scalar.
        Then:
            The positive vector is ranked first with a perfect score.
        """
        index: MemoryIndex = memory_index_factory(embedding_dim=1)
        nodes = [
            KnowledgeNode(
                node_id="pos",
                content="Pos",
                source_id="p",
                position=NodePosition(start_char=0, end_char=1, page=1, section="Test"),
                metadata={},
                dense_vector=[1.0],
                sparse_vector=None,
            ),
            KnowledgeNode(
                node_id="neg",
                content="Neg",
                source_id="n",
                position=NodePosition(start_char=0, end_char=1, page=1, section="Test"),
                metadata={},
                dense_vector=[-1.0],
                sparse_vector=None,
            ),
        ]
        await index.add(nodes)

        results: list[SearchResult] = await index.search(query_vector=[0.5], top_k=2)

        assert results[0].node is not None
        assert results[0].node.content == "Pos"
        assert_allclose(results[0].score, 1.0, atol=1e-6)

    async def test_add_should_raise_index_error_on_vector_dimension_mismatch(
        self, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Verify input validation for added vector dimensions.

        Given:
            A MemoryIndex configured for dimension 3.
        When:
            Attempting to add a node with a vector of dimension 2.
        Then:
            A RagIndexError is raised indicating the mismatch.
        """
        index: MemoryIndex = memory_index_factory(embedding_dim=3)
        bad_node = KnowledgeNode(
            node_id="bad_dim",
            content="Bad Dim",
            source_id="s",
            position=NodePosition(start_char=0, end_char=1, page=1, section="Test"),
            metadata={},
            dense_vector=[1.0, 1.0],
            sparse_vector=None,
        )

        with pytest.raises(RagIndexError, match="Vector dimension mismatch"):
            await index.add([bad_node])

    async def test_add_should_raise_index_error_when_missing_vector_and_embedder(
        self, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Verify error when adding nodes without vectors and no configured embedder.

        Given:
            A MemoryIndex with NO embedder configured.
        When:
            Adding a node that has `dense_vector=None`.
        Then:
            A RagIndexError is raised.
        """
        index: MemoryIndex = memory_index_factory(embedding_dim=3, embedder=None)
        no_vec_node = KnowledgeNode(
            node_id="no_vec",
            content="Content",
            source_id="s",
            position=NodePosition(start_char=0, end_char=1, page=1, section="Test"),
            metadata={},
            dense_vector=None,
            sparse_vector=None,
        )

        with pytest.raises(RagIndexError, match="Nodes must have dense_vector"):
            await index.add([no_vec_node])

    async def test_add_should_generate_embeddings_when_using_configured_embedder(
        self,
        memory_index_factory: Callable[..., Any],
        fake_embedder_factory: Callable[..., Any],
        node_factory: Callable[..., Any],
    ) -> None:
        """Verify auto-embedding functionality using a FakeEmbedder.

        Given:
            A MemoryIndex configured with a FakeEmbedder (dim=3, returns [0.1, ...]).
            A node WITHOUT a dense vector.
        When:
            Adding the node to the index.
        Then:
            The embedding is generated automatically.
            Searching for the expected vector retrieves the node with a perfect score.
        """
        index: MemoryIndex = memory_index_factory(
            embedding_dim=3,
            distance_metric="cosine",
            embedder=fake_embedder_factory(dimension=3),
        )
        node_without_vector = node_factory(
            content="To be embedded", dense_vector=None, embedding_dim=3
        )

        await index.add([node_without_vector])

        assert await index.count() == 1

        results = await index.search(query_vector=[0.1, 0.1, 0.1], top_k=1)

        assert len(results) == 1
        assert_allclose(results[0].score, 1.0, atol=1e-6)

    async def test_add_should_compute_embeddings_for_multiple_nodes_without_vectors(
        self,
        memory_index_factory: Callable[..., Any],
        fake_embedder_factory: Callable[..., Any],
    ) -> None:
        """Verify batch embedding computation for multiple nodes.

        Given:
            A MemoryIndex with an embedder configured.
            Multiple nodes without dense vectors.
        When:
            Adding all nodes at once.
        Then:
            Embeddings are computed in batch via executor.
            All nodes are successfully indexed.
        """
        index: MemoryIndex = memory_index_factory(
            embedding_dim=4,
            embedder=fake_embedder_factory(dimension=4),
        )

        nodes_without_vectors = [
            KnowledgeNode(
                node_id=f"node_{i}",
                content=f"Content {i}",
                source_id="test",
                position=NodePosition(
                    start_char=0, end_char=10, page=1, section="Test"
                ),
                metadata={},
                dense_vector=None,
                sparse_vector=None,
            )
            for i in range(5)
        ]

        await index.add(nodes_without_vectors)

        assert await index.count() == 5

        results = await index.search(query_vector=[0.1, 0.1, 0.1, 0.1], top_k=5)
        assert len(results) == 5

    async def test_search_should_raise_index_error_when_query_is_dict(
        self, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Verify that dense search rejects sparse dictionary queries.

        Given:
            A MemoryIndex.
        When:
            Calling search() with a dictionary (sparse vector format).
        Then:
            A RagIndexError is raised.
        """
        index: MemoryIndex = memory_index_factory(embedding_dim=3)

        with pytest.raises(RagIndexError, match="only supports dense vectors"):
            await index.search(query_vector={"indices": [1], "values": [0.5]}, top_k=1)

    async def test_search_should_raise_index_error_on_query_dimension_mismatch(
        self, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Verify input validation for query vector dimensions.

        Given:
            A MemoryIndex of dim 3.
        When:
            Searching with a vector of dim 2.
        Then:
            A RagIndexError is raised.
        """
        index: MemoryIndex = memory_index_factory(embedding_dim=3)

        with pytest.raises(RagIndexError, match="Query vector dimension mismatch"):
            await index.search(query_vector=[1.0, 2.0], top_k=1)

    async def test_internal_compute_similarities_should_support_integer_slicing(
        self, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Verify internal helper supports integer count for vector slicing.

        This tests a specific internal utility path identified in backends.py:
        `if isinstance(vectors, int): vectors = self._vectors[:vectors]`

        Given:
            An index with 2 vectors.
        When:
            Calling `_compute_similarities` directly with an integer count of 1.
        Then:
            It computes similarity only against the first vector (slicing logic).
        """
        index: MemoryIndex = memory_index_factory(embedding_dim=2)
        nodes = [
            KnowledgeNode(
                node_id="A",
                content="A",
                source_id="s",
                position=NodePosition(start_char=0, end_char=1, page=1, section="Test"),
                metadata={},
                dense_vector=[1.0, 0.0],
                sparse_vector=None,
            ),
            KnowledgeNode(
                node_id="B",
                content="B",
                source_id="s",
                position=NodePosition(start_char=0, end_char=1, page=1, section="Test"),
                metadata={},
                dense_vector=[0.0, 1.0],
                sparse_vector=None,
            ),
        ]
        await index.add(nodes)

        query = np.array([0.0, 1.0], dtype=np.float32)

        similarities = index._compute_similarities(query_vector=query, vectors=1)

        assert len(similarities) == 1
        assert_allclose(similarities[0], 0.0, atol=1e-6)


@pytest.mark.unit
@pytest.mark.asyncio
class TestMemoryIndexMetrics:
    """Test suite for specific distance metrics (Euclidean, Dot Product, Cosine)."""

    async def test_search_should_use_negative_l2_for_euclidean_metric(
        self, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Verify Euclidean distance behavior (negative L2).

        Given:
            An index configured with 'euclidean' metric.
            Node A at [0,0,0], Node B at [0,0,2].
        When:
            Searching with query [0,0,0].
        Then:
            Node A has score 0.0 (dist 0).
            Node B has score -2.0 (dist 2).
            Order is A then B.
        """
        index: MemoryIndex = memory_index_factory(
            embedding_dim=3, distance_metric="euclidean"
        )
        nodes = [
            KnowledgeNode(
                node_id="A",
                content="A",
                source_id="s",
                position=NodePosition(start_char=0, end_char=1, page=1, section="Test"),
                metadata={},
                dense_vector=[0.0, 0.0, 0.0],
                sparse_vector=None,
            ),
            KnowledgeNode(
                node_id="B",
                content="B",
                source_id="s",
                position=NodePosition(start_char=0, end_char=1, page=1, section="Test"),
                metadata={},
                dense_vector=[0.0, 0.0, 2.0],
                sparse_vector=None,
            ),
        ]
        await index.add(nodes)

        results = await index.search(query_vector=[0.0, 0.0, 0.0], top_k=2)

        assert results[0].node_id == "A"
        assert_allclose(results[0].score, 0.0, atol=1e-6)

        assert results[1].node_id == "B"
        assert_allclose(results[1].score, -2.0, atol=1e-6)

    async def test_search_should_return_raw_dot_product_when_configured(
        self, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Verify Dot Product metric (unnormalized).

        Given:
            An index with 'dot' metric.
        When:
            Calculating dot product of aligned vectors [2,0] and [3,0].
        Then:
            Score is exactly 6.0 (2*3), not normalized to 1.0.
        """
        index: MemoryIndex = memory_index_factory(
            embedding_dim=2, distance_metric="dot"
        )
        node = KnowledgeNode(
            node_id="A",
            content="A",
            source_id="s",
            position=NodePosition(start_char=0, end_char=1, page=1, section="Test"),
            metadata={},
            dense_vector=[2.0, 0.0],
            sparse_vector=None,
        )
        await index.add([node])

        results = await index.search(query_vector=[3.0, 0.0], top_k=1)

        assert_allclose(results[0].score, 6.0, atol=1e-6)

    @pytest.mark.rag_edge_case
    async def test_search_should_return_perfect_score_for_identical_vectors(
        self, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Verify Cosine similarity returns 1.0 for identical vectors.

        Given:
            An index with a node having vector V=[1.0, 0.0].
        When:
            Searching with the same query vector V=[1.0, 0.0].
        Then:
            The score is 1.0.
        """
        index: MemoryIndex = memory_index_factory(
            embedding_dim=2, distance_metric="cosine"
        )
        node = KnowledgeNode(
            node_id="A",
            content="A",
            source_id="s",
            position=NodePosition(start_char=0, end_char=1, page=1, section="Test"),
            metadata={},
            dense_vector=[1.0, 0.0],
            sparse_vector=None,
        )
        await index.add([node])

        results = await index.search(query_vector=[1.0, 0.0], top_k=1)

        assert_allclose(results[0].score, 1.0, atol=1e-6)

    @pytest.mark.rag_edge_case
    async def test_search_should_return_zero_score_for_orthogonal_vectors(
        self, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Verify Cosine similarity returns 0.0 for orthogonal vectors.

        Given:
            An index with a node having vector V=[1.0, 0.0].
        When:
            Searching with an orthogonal vector Q=[0.0, 1.0].
        Then:
            The score is 0.0 (indicating no similarity).
        """
        index: MemoryIndex = memory_index_factory(
            embedding_dim=2, distance_metric="cosine"
        )
        node = KnowledgeNode(
            node_id="A",
            content="A",
            source_id="s",
            position=NodePosition(start_char=0, end_char=1, page=1, section="Test"),
            metadata={},
            dense_vector=[1.0, 0.0],
            sparse_vector=None,
        )
        await index.add([node])

        results = await index.search(query_vector=[0.0, 1.0], top_k=1)

        assert_allclose(results[0].score, 0.0, atol=1e-6)


@pytest.mark.unit
@pytest.mark.asyncio
class TestMemoryIndexHybrid:
    """Test suite for Hybrid Retrieval and Score Fusion Logic."""

    async def test_search_hybrid_should_apply_min_max_norm_and_alpha_fusion(
        self, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Verify Min-Max Normalization and Alpha Weighting logic.

        Given:
            Two nodes with contrasting strengths:
            - Node A: Best Dense (1.0), Worst Sparse (10.0)
            - Node B: Worst Dense (0.0), Best Sparse (100.0)
            (Note: Vectors set up to simulate scores relatively)
        When:
            Searching with alpha=0.6 (Dense favored).
        Then:
            Scores are normalized Min-Max per component, then fused.
            Dense Norm: A=1.0, B=0.0
            Sparse Norm: A=0.0, B=1.0
            Score A = 0.6*1 + 0.4*0 = 0.6
            Score B = 0.6*0 + 0.4*1 = 0.4
        """
        index: MemoryIndex = memory_index_factory(embedding_dim=2)
        nodes = [
            KnowledgeNode(
                node_id="A",
                content="A",
                source_id="s",
                position=NodePosition(start_char=0, end_char=1, page=1, section="Test"),
                metadata={},
                dense_vector=[1.0, 0.0],
                sparse_vector={1: 1.0},
            ),
            KnowledgeNode(
                node_id="B",
                content="B",
                source_id="s",
                position=NodePosition(start_char=0, end_char=1, page=1, section="Test"),
                metadata={},
                dense_vector=[0.0, 1.0],
                sparse_vector={1: 10.0},
            ),
        ]
        await index.add(nodes)

        results = await index.search_hybrid(
            dense_vector=[1.0, 0.0], sparse_vector={1: 1.0}, top_k=2, alpha=0.6
        )

        assert results[0].node_id == "A"
        assert results[1].node_id == "B"

        assert_allclose(results[0].score, 0.6, atol=1e-6)
        assert_allclose(results[1].score, 0.4, atol=1e-6)

    @pytest.mark.rag_edge_case
    async def test_search_hybrid_should_handle_zero_variance_normalization(
        self, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Verify normalization behavior when all scores are identical (Zero Variance).

        Given:
            Two identical nodes.
        When:
            Performing hybrid search.
        Then:
            Min-Max normalization detects zero variance (max-min < epsilon)
            and returns 1.0 for all components to avoid division by zero.
            Final score should be 1.0.
        """
        index: MemoryIndex = memory_index_factory(embedding_dim=2)
        nodes = [
            KnowledgeNode(
                node_id="A",
                content="A",
                source_id="s",
                position=NodePosition(start_char=0, end_char=1, page=1, section="Test"),
                metadata={},
                dense_vector=[1.0, 0.0],
                sparse_vector={1: 1.0},
            ),
            KnowledgeNode(
                node_id="B",
                content="B",
                source_id="s",
                position=NodePosition(start_char=0, end_char=1, page=1, section="Test"),
                metadata={},
                dense_vector=[1.0, 0.0],
                sparse_vector={1: 1.0},
            ),
        ]
        await index.add(nodes)

        results = await index.search_hybrid(
            dense_vector=[1.0, 0.0], sparse_vector={1: 1.0}, top_k=2, alpha=0.5
        )

        assert_allclose(results[0].score, 1.0, atol=1e-6)
        assert_allclose(results[1].score, 1.0, atol=1e-6)

    async def test_search_hybrid_should_raise_index_error_on_dense_dimension_mismatch(
        self, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Verify input validation for dense vector dimension in hybrid search.

        Given:
            A MemoryIndex of dim 3.
        When:
            Calling search_hybrid with dense vector of dim 2.
        Then:
            A RagIndexError is raised.
        """
        index: MemoryIndex = memory_index_factory(embedding_dim=3)

        with pytest.raises(RagIndexError, match="Dense vector dimension mismatch"):
            await index.search_hybrid(
                dense_vector=[1.0, 2.0], sparse_vector={1: 1.0}, top_k=1, alpha=0.5
            )

    async def test_search_hybrid_should_return_empty_list_when_index_is_empty(
        self, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Verify early exit when searching hybrid on an empty index.

        Given:
            An empty MemoryIndex.
        When:
            Calling search_hybrid().
        Then:
            An empty list is returned immediately.
        """
        index: MemoryIndex = memory_index_factory(embedding_dim=2)

        results = await index.search_hybrid(
            dense_vector=[1.0, 0.0], sparse_vector={1: 1.0}, top_k=5, alpha=0.5
        )

        assert results == []


@pytest.mark.unit
@pytest.mark.asyncio
class TestMemoryIndexCRUD:
    """Test suite for CRUD operations, Filtering, and Index Management."""

    async def test_add_should_do_nothing_when_nodes_list_is_empty(
        self, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Verify that adding an empty list of nodes performs no operation.

        Given:
            A MemoryIndex.
        When:
            Calling add() with an empty list.
        Then:
            The state remains unchanged (count is 0) and no error is raised.
        """
        index: MemoryIndex = memory_index_factory(embedding_dim=3)

        await index.add([])

        assert await index.count() == 0

    async def test_search_should_return_empty_list_when_index_is_empty(
        self, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Verify early exit when searching an empty index.

        Given:
            An empty MemoryIndex.
        When:
            Calling search().
        Then:
            An empty list is returned immediately.
        """
        index: MemoryIndex = memory_index_factory(embedding_dim=3)

        results = await index.search(query_vector=[1.0, 0.0, 0.0], top_k=5)

        assert results == []

    async def test_search_should_respect_metadata_filters(
        self, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Verify that search respects metadata filters."""
        index: MemoryIndex = memory_index_factory(embedding_dim=2)
        nodes = [
            KnowledgeNode(
                node_id="A",
                content="A",
                source_id="s",
                position=NodePosition(start_char=0, end_char=1, page=1, section="Test"),
                metadata={"category": "tech"},
                dense_vector=[1.0, 0.0],
                sparse_vector=None,
            ),
            KnowledgeNode(
                node_id="B",
                content="B",
                source_id="s",
                position=NodePosition(start_char=0, end_char=1, page=1, section="Test"),
                metadata={"category": "finance"},
                dense_vector=[0.9, 0.0],
                sparse_vector=None,
            ),
        ]
        await index.add(nodes)

        results = await index.search(
            query_vector=[1.0, 0.0], top_k=2, filters={"category": "tech"}
        )

        assert len(results) == 1
        assert results[0].node is not None
        assert results[0].node.metadata["category"] == "tech"
        assert results[0].node_id == "A"

    async def test_search_should_return_empty_list_when_filter_matches_nothing(
        self, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Verify that filters returning no matches result in an empty list.

        Given:
            An index with nodes tagged 'tech'.
        When:
            Searching with filter 'food'.
        Then:
            Result list is empty.
        """
        index: MemoryIndex = memory_index_factory(embedding_dim=2)
        node = KnowledgeNode(
            node_id="A",
            content="A",
            source_id="s",
            position=NodePosition(start_char=0, end_char=1, page=1, section="Test"),
            metadata={"category": "tech"},
            dense_vector=[1.0, 0.0],
            sparse_vector=None,
        )
        await index.add([node])

        results = await index.search(
            query_vector=[1.0, 0.0], top_k=10, filters={"category": "food"}
        )

        assert len(results) == 0

    async def test_delete_should_maintain_vector_integrity_for_remaining_nodes(
        self, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Verify that deleting a node properly shifts internal vector arrays."""
        index: MemoryIndex = memory_index_factory(embedding_dim=2)
        nodes = [
            KnowledgeNode(
                node_id="A",
                content="A",
                source_id="s",
                position=NodePosition(start_char=0, end_char=1, page=1, section="Test"),
                metadata={},
                dense_vector=[1.0, 0.0],
                sparse_vector=None,
            ),
            KnowledgeNode(
                node_id="B",
                content="B",
                source_id="s",
                position=NodePosition(start_char=0, end_char=1, page=1, section="Test"),
                metadata={},
                dense_vector=[0.0, 1.0],
                sparse_vector=None,
            ),
            KnowledgeNode(
                node_id="C",
                content="C",
                source_id="s",
                position=NodePosition(start_char=0, end_char=1, page=1, section="Test"),
                metadata={},
                dense_vector=[1.0, 0.0],
                sparse_vector=None,
            ),
        ]
        await index.add(nodes)

        await index.delete(["B"])

        assert await index.count() == 2
        res_a = await index.search([1.0, 0.0], top_k=2)
        ids = sorted([r.node_id for r in res_a])
        assert ids == ["A", "C"]
        assert not await index.exists("B")

    async def test_delete_should_maintain_hybrid_index_consistency_after_node_removal(
        self, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Verify that sparse inverted index is updated when dense vectors shift.

        Detailed scenario to catch 'shift bugs':
        1. Add Node A (token 100), Node B (token 101), Node C (token 102).
           Indices: A=0, B=1, C=2.
        2. Delete Node A.
           New Expected Indices: B=0, C=1.
        3. If inverted index is NOT updated:
           - Token 101 (B) still points to doc_idx 1 (which is now C's slot!).
           - Token 102 (C) still points to doc_idx 2 (Out of Bounds).

        Given:
            Three nodes with distinct sparse features.
        When:
            The first node is deleted.
        Then:
            Hybrid search for the remaining nodes' tokens retrieves them correctly.
        """
        index: MemoryIndex = memory_index_factory(embedding_dim=2)
        nodes = [
            KnowledgeNode(
                node_id="A",
                content="A",
                source_id="s",
                position=NodePosition(start_char=0, end_char=1, page=1, section="Test"),
                metadata={},
                dense_vector=[1.0, 0.0],
                sparse_vector={100: 1.0},
            ),
            KnowledgeNode(
                node_id="B",
                content="B",
                source_id="s",
                position=NodePosition(start_char=0, end_char=1, page=1, section="Test"),
                metadata={},
                dense_vector=[0.0, 1.0],
                sparse_vector={101: 1.0},
            ),
            KnowledgeNode(
                node_id="C",
                content="C",
                source_id="s",
                position=NodePosition(start_char=0, end_char=1, page=1, section="Test"),
                metadata={},
                dense_vector=[1.0, 0.0],
                sparse_vector={102: 1.0},
            ),
        ]
        await index.add(nodes)

        await index.delete(["A"])
        assert await index.count() == 2

        res_b = await index.search_hybrid(
            dense_vector=[0.0, 0.0], sparse_vector={101: 1.0}, top_k=1, alpha=0.0
        )
        assert len(res_b) == 1
        assert (
            res_b[0].node_id == "B"
        ), "Inverted index pointer for B was not updated after shift."

        res_c = await index.search_hybrid(
            dense_vector=[0.0, 0.0], sparse_vector={102: 1.0}, top_k=1, alpha=0.0
        )
        assert len(res_c) == 1
        assert (
            res_c[0].node_id == "C"
        ), "Inverted index pointer for C was not updated after shift."

    async def test_delete_should_return_zero_and_not_raise_error_for_non_existent_id(
        self, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Verify that deleting a non-existent node fails silently and returns 0.

        Given:
            An empty or populated index.
        When:
            Attempting to delete a node ID that does not exist.
        Then:
            The method returns 0 (deleted count) and raises no exception.
        """
        index: MemoryIndex = memory_index_factory(embedding_dim=2)

        deleted_count = await index.delete(["non_existent_id"])

        assert deleted_count == 0

    async def test_clear_should_remove_all_nodes_and_reset_internal_storage(
        self, memory_index_factory: Callable[..., Any], node_factory: Callable[..., Any]
    ) -> None:
        """Verify that clear() empties the index completely.

        Given:
            An index populated with nodes.
        When:
            The clear() method is called.
        Then:
            The node count becomes 0, and internal vector storage is reset.
        """
        index: MemoryIndex = memory_index_factory(embedding_dim=3)
        await index.add([node_factory(dense_vector=[0.1, 0.2, 0.3])])

        assert await index.count() == 1

        await index.clear()

        assert await index.count() == 0
        assert len(index._node_ids) == 0
        assert index._vectors.shape[0] == 0


@pytest.mark.concurrency
@pytest.mark.asyncio
class TestMemoryIndexConcurrency:
    """Stress tests for thread safety."""

    async def test_concurrent_read_write_delete_should_not_corrupt_index(
        self, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Simulate race conditions: Add/Delete/Search simultaneously.

        Given:
            An index with 100 base nodes.
            Tasks adding 500 new nodes (50 * 10).
            Tasks deleting up to 50 base nodes (50 * 1).
        When:
            These tasks run concurrently.
        Then:
            The internal state (vectors vs metadata) remains consistent.
            The final count reflects the minimum possible nodes retained
            (100 base - 50 deleted + 500 added = 550 minimum).
        """
        index: MemoryIndex = memory_index_factory(embedding_dim=3)

        base_nodes = [
            KnowledgeNode(
                node_id=f"base_{i}",
                content="Base",
                source_id="init",
                position=NodePosition(start_char=0, end_char=1, page=1, section="Test"),
                metadata={},
                dense_vector=[1.0, 0.0, 0.0],
                sparse_vector=None,
            )
            for i in range(100)
        ]
        await index.add(base_nodes)

        async def adder_task() -> None:
            for i in range(50):
                nodes = [
                    KnowledgeNode(
                        node_id=f"new_{i}_{j}",
                        content="New",
                        source_id="add",
                        position=NodePosition(
                            start_char=0, end_char=1, page=1, section="Test"
                        ),
                        metadata={},
                        dense_vector=[0.0, 1.0, 0.0],
                        sparse_vector=None,
                    )
                    for j in range(10)
                ]
                await index.add(nodes)
                await asyncio.sleep(0.001)

        async def deleter_task() -> None:
            for _ in range(50):
                target_id = f"base_{random.randint(0, 99)}"
                await index.delete([target_id])
                await asyncio.sleep(0.001)

        async def searcher_task() -> None:
            for _ in range(50):
                await index.search(query_vector=[1.0, 0.0, 0.0], top_k=5)
                await asyncio.sleep(0.001)

        await asyncio.gather(adder_task(), deleter_task(), searcher_task())

        assert len(index._node_ids) == len(
            index._vectors
        ), "Metadata and Vector arrays desynchronized"
        assert len(index._node_ids) == len(
            index._metadata
        ), "Node IDs and Metadata count mismatch"

        max_idx = len(index._node_ids) - 1
        for _, entries in index._inverted_index.items():
            for doc_idx, _ in entries:
                assert (
                    doc_idx <= max_idx
                ), f"Inverted index points to {doc_idx}, but max index is {max_idx}"

        final_count = await index.count()
        assert final_count >= 550, f"Expected at least 550 nodes, found {final_count}"
        assert final_count <= 600, f"Expected at most 600 nodes, found {final_count}"


@pytest.mark.integration
class TestMemoryIndexPersistence:
    """Test suite for FileSystem I/O and Persistence."""

    @pytest.mark.asyncio
    async def test_save_and_load_round_trip_should_preserve_unicode_and_emojis(
        self, tmp_path: Path, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Verify persistence with complex Unicode characters (Emojis, CJK).

        Given:
            An index with nodes containing Unicode content and metadata.
        When:
            The index is saved and reloaded.
        Then:
            The content is perfectly preserved without encoding errors.
        """
        save_dir = tmp_path / "index_dump"
        original_index: MemoryIndex = memory_index_factory(embedding_dim=3)

        complex_content = "RAG with Brain 🧠 and Kanji 検索"
        node = KnowledgeNode(
            node_id="u_node",
            content=complex_content,
            source_id="src",
            position=NodePosition(start_char=0, end_char=10, page=1, section="Test"),
            metadata={"tag": "🚀", "lang": "日本語"},
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector=None,
        )
        await original_index.add([node])

        await original_index.save(save_dir)
        loaded_index = await MemoryIndex.load(save_dir)

        results = await loaded_index.search(query_vector=[0.1, 0.2, 0.3], top_k=1)

        assert results[0].node is not None
        assert results[0].node.content == complex_content
        assert results[0].node.metadata["tag"] == "🚀"
        assert results[0].node.metadata["lang"] == "日本語"

    @pytest.mark.asyncio
    async def test_save_and_load_should_persist_sparse_inverted_index(
        self, tmp_path: Path, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Verify persistence of the sparse inverted index via JSON.

        This tests the JSON loading block where 'inverted_index' is present:
        `if "inverted_index" in metadata_dict: ...`

        Given:
            An index with nodes containing sparse vectors.
        When:
            The index is saved and reloaded.
        Then:
            The inverted index is correctly reconstructed.
        """
        save_dir = tmp_path / "sparse_dump"
        original_index: MemoryIndex = memory_index_factory(embedding_dim=2)

        node = KnowledgeNode(
            node_id="sparse_node",
            content="Sparse content",
            source_id="src",
            position=NodePosition(start_char=0, end_char=5, page=1, section="Test"),
            metadata={},
            dense_vector=[1.0, 0.0],
            sparse_vector={99: 0.8},
        )
        await original_index.add([node])

        await original_index.save(save_dir)
        loaded_index = await MemoryIndex.load(save_dir)

        assert 99 in loaded_index._inverted_index
        assert len(loaded_index._inverted_index[99]) == 1
        doc_idx, weight = loaded_index._inverted_index[99][0]
        assert_allclose(weight, 0.8, atol=1e-6)

    @pytest.mark.asyncio
    async def test_load_should_raise_file_not_found_error_for_invalid_path(
        self, tmp_path: Path
    ) -> None:
        """Verify error handling when loading from a non-existent path.

        Given:
            A path to a directory that does not exist.
        When:
            Attempting to load a MemoryIndex from this path.
        Then:
            A FileNotFoundError is raised (as per backends.py implementation).
        """
        bad_path = tmp_path / "non_existent_index_dir"

        with pytest.raises(FileNotFoundError):
            await MemoryIndex.load(bad_path)

    @pytest.mark.rag_edge_case
    @pytest.mark.asyncio
    async def test_load_should_fallback_to_pickle_for_legacy_indexes(
        self, tmp_path: Path, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Verify backward compatibility loading for legacy .pkl indexes.

        This tests the specific fallback block:
        `elif pkl_path.exists(): ... pickle.load(f)`

        Given:
            A directory containing only a legacy `metadata.pkl` and `vectors.npy`.
        When:
            Calling MemoryIndex.load().
        Then:
            The index is loaded successfully using the pickle fallback logic.
        """
        save_dir = tmp_path / "legacy_dump"
        save_dir.mkdir()

        metadata_dict: dict[str, Any] = {
            "embedding_dim": 2,
            "collection_name": "legacy_col",
            "distance_metric": "cosine",
            "node_ids": ["legacy_node"],
            "metadata": [{"key": "old"}],
            "content": ["Legacy Content"],
            "inverted_index": {55: [(0, 0.5)]},
            "nodes": [
                KnowledgeNode(
                    node_id="legacy_node",
                    content="Legacy Content",
                    source_id="old_src",
                    position=NodePosition(
                        start_char=0, end_char=1, page=1, section="Old"
                    ),
                    metadata={"key": "old"},
                    dense_vector=[1.0, 0.0],
                    sparse_vector=None,
                )
            ],
        }

        with open(save_dir / "metadata.pkl", "wb") as f:
            pickle.dump(metadata_dict, f)

        vectors = np.array([[1.0, 0.0]], dtype=np.float32)
        np.save(save_dir / "vectors.npy", vectors)

        loaded_index = await MemoryIndex.load(save_dir)

        assert loaded_index.collection_name == "legacy_col"
        assert loaded_index._node_ids == ["legacy_node"]
        assert 55 in loaded_index._inverted_index


@pytest.mark.unit
@pytest.mark.asyncio
class TestAsyncSaveLoadNonBlocking:
    """Tests for async save/load non-blocking behavior."""

    async def test_async_save_should_not_block_concurrent_searches(
        self, tmp_path: Path
    ) -> None:
        """Verifies save() doesn't block event loop during concurrent operations.

        Given:
            An index with 500 nodes.
        When:
            Saving to disk while running 5 concurrent searches.
        Then:
            Both operations complete concurrently.
        """
        index = MemoryIndex(embedding_dim=128)
        nodes = [
            KnowledgeNode(
                node_id=f"node_{i}",
                content=f"Content {i}",
                source_id="test_source",
                position=NodePosition(start_char=i * 20, end_char=(i + 1) * 20),
                dense_vector=[float(i % 128)] * 128,
                metadata={},
            )
            for i in range(500)
        ]
        await index.add(nodes)

        async def run_searches():
            for _ in range(5):
                await index.search([0.5] * 128, top_k=10)
                await asyncio.sleep(0.01)

        save_path = tmp_path / "test_index"
        start = time.perf_counter()

        save_task = asyncio.create_task(index.save(save_path))
        search_task = asyncio.create_task(run_searches())

        await asyncio.gather(save_task, search_task)
        total_duration = time.perf_counter() - start

        assert total_duration < 5.0, f"Async save took too long: {total_duration:.2f}s"

    async def test_async_load_should_restore_all_data_correctly(
        self, tmp_path: Path
    ) -> None:
        """Verifies async load() restores index state completely.

        Given:
            An index with 50 nodes saved to disk.
        When:
            Loading from disk asynchronously.
        Then:
            All nodes, vectors, and metadata are restored correctly.
        """
        index = MemoryIndex(
            embedding_dim=64,
            collection_name="test_collection",
            distance_metric="cosine",
        )
        original_nodes = [
            KnowledgeNode(
                node_id=f"test_{i}",
                content=f"Test content {i}",
                source_id="test_source",
                position=NodePosition(start_char=i * 15, end_char=(i + 1) * 15),
                dense_vector=[float((i + j) % 64) for j in range(64)],
                metadata={"index": i, "batch": i // 10},
            )
            for i in range(50)
        ]
        await index.add(original_nodes)

        save_path = tmp_path / "test_load"
        await index.save(save_path)

        loaded_index = await MemoryIndex.load(save_path)

        assert await loaded_index.count() == 50
        assert loaded_index.collection_name == "test_collection"
        assert loaded_index.embedding_dim == 64

        for i in [0, 24, 49]:
            search_result = await loaded_index.search(
                [float((i + j) % 64) for j in range(64)], top_k=1
            )
            assert len(search_result) == 1
            assert search_result[0].node_id == f"test_{i}"


@pytest.mark.unit
@pytest.mark.benchmark
class TestMemoryIndexDeletePerformance:
    """Benchmark tests for MemoryIndex.delete() O(1) optimization."""

    @pytest.mark.asyncio
    async def test_delete_1000_from_10k_should_complete_under_10ms(self) -> None:
        """Verifies delete() achieves O(1) complexity via dict mapping.

        Given:
            An index containing 10,000 nodes with embeddings.
        When:
            Deleting 1000 nodes by ID.
        Then:
            Operation completes in under 10ms (50x faster than O(n²)).
        """
        index = MemoryIndex(embedding_dim=128)
        nodes = [
            KnowledgeNode(
                node_id=f"node_{i}",
                content=f"Test content {i}",
                source_id="test_source",
                position=NodePosition(start_char=i * 100, end_char=(i + 1) * 100),
                dense_vector=[0.1] * 128,
                metadata={"index": i},
            )
            for i in range(10000)
        ]
        await index.add(nodes)

        to_delete = [f"node_{i}" for i in range(1000)]
        start = time.perf_counter()
        deleted = await index.delete(to_delete)
        duration_ms = (time.perf_counter() - start) * 1000

        assert deleted == 1000
        assert await index.count() == 9000
        assert duration_ms < 10, f"Delete took {duration_ms:.2f}ms, expected <10ms"

    @pytest.mark.asyncio
    async def test_delete_maintains_node_id_to_idx_mapping_consistency(self) -> None:
        """Verifies mapping invariant after deletions.

        Given:
            An index with 100 nodes.
        When:
            Deleting 20 nodes from various positions.
        Then:
            Mapping _node_id_to_idx[node_ids[i]] == i holds for all nodes.
        """
        index = MemoryIndex(embedding_dim=64)
        nodes = [
            KnowledgeNode(
                node_id=f"test_{i}",
                content=f"Content {i}",
                source_id="test_source",
                position=NodePosition(start_char=i * 50, end_char=(i + 1) * 50),
                dense_vector=[float(i)] * 64,
                metadata={},
            )
            for i in range(100)
        ]
        await index.add(nodes)

        to_delete = [f"test_{i}" for i in [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]]
        await index.delete(to_delete)

        for idx, node_id in enumerate(index._node_ids):
            assert index._node_id_to_idx[node_id] == idx
            assert index._nodes[idx].node_id == node_id


@pytest.mark.unit
@pytest.mark.asyncio
class TestMemoryIndexLRUEviction:
    """Tests for LRU eviction policy and bounded memory management."""

    async def test_eviction_should_enforce_max_nodes_limit(self) -> None:
        """Verifies max_nodes limit triggers automatic eviction.

        Given:
            An index with max_nodes=100 and lru eviction policy.
        When:
            Adding 150 nodes sequentially.
        Then:
            Index size plateaus at 100 nodes via automatic eviction.
        """
        index = MemoryIndex(embedding_dim=64, max_nodes=100, eviction_policy="lru")

        for batch_start in range(0, 150, 10):
            nodes = [
                KnowledgeNode(
                    node_id=f"node_{i}",
                    content=f"Content {i}",
                    source_id="test_source",
                    position=NodePosition(start_char=i * 20, end_char=(i + 1) * 20),
                    dense_vector=[float(i % 64)] * 64,
                    metadata={"batch": batch_start},
                )
                for i in range(batch_start, batch_start + 10)
            ]
            await index.add(nodes)

        count = await index.count()
        assert count == 100, f"Expected 100 nodes, got {count}"

    async def test_lru_eviction_should_remove_least_recently_accessed_nodes(
        self,
    ) -> None:
        """Verifies LRU policy evicts nodes not accessed by search.

        Given:
            Index with max_nodes=50, containing 50 nodes.
        When:
            Searching for subset of nodes, then adding 10 more.
        Then:
            Never-searched nodes are evicted first (LRU behavior).
        """
        index = MemoryIndex(embedding_dim=32, max_nodes=50, eviction_policy="lru")
        nodes = [
            KnowledgeNode(
                node_id=f"node_{i}",
                content=f"Content {i}",
                source_id="test_source",
                position=NodePosition(start_char=i * 30, end_char=(i + 1) * 30),
                dense_vector=[float(i)] * 32,
                metadata={},
            )
            for i in range(50)
        ]
        await index.add(nodes)

        await index.search([45.0] * 32, top_k=10)

        new_nodes = [
            KnowledgeNode(
                node_id=f"new_{i}",
                content=f"New content {i}",
                source_id="test_source",
                position=NodePosition(start_char=(i + 50) * 30, end_char=(i + 51) * 30),
                dense_vector=[float(i + 50)] * 32,
                metadata={},
            )
            for i in range(10)
        ]
        await index.add(new_nodes)

        remaining_ids = set(index._node_ids)
        recently_accessed = {f"node_{i}" for i in range(40, 50)}

        retained = recently_accessed & remaining_ids
        assert len(retained) > 0, "LRU should retain recently accessed nodes"

    async def test_fifo_eviction_should_remove_oldest_inserted_nodes(self) -> None:
        """Verifies FIFO policy evicts by insertion order.

        Given:
            Index with max_nodes=30, eviction_policy=fifo.
        When:
            Adding 40 nodes sequentially.
        Then:
            First 10 nodes are evicted (FIFO order).
        """
        index = MemoryIndex(embedding_dim=16, max_nodes=30, eviction_policy="fifo")

        all_nodes = [
            KnowledgeNode(
                node_id=f"seq_{i}",
                content=f"Sequential {i}",
                source_id="test_source",
                position=NodePosition(start_char=i * 15, end_char=(i + 1) * 15),
                dense_vector=[float(i % 16)] * 16,
                metadata={"order": i},
            )
            for i in range(40)
        ]
        await index.add(all_nodes)

        remaining_ids = set(index._node_ids)
        first_ten = {f"seq_{i}" for i in range(10)}
        last_thirty = {f"seq_{i}" for i in range(10, 40)}

        assert len(first_ten & remaining_ids) == 0, "FIFO should evict first 10"
        assert len(last_thirty & remaining_ids) == 30, "FIFO should retain last 30"


@pytest.mark.unit
@pytest.mark.asyncio
class TestSearchPagination:
    """Tests for cursor-based search pagination."""

    async def test_search_paginated_should_return_correct_page_size(self) -> None:
        """Verifies search_paginated() returns requested page size.

        Given:
            Index with 100 nodes and cursor with page_size=10.
        When:
            Calling search_paginated().
        Then:
            Returns exactly 10 results in first page.
        """
        index = MemoryIndex(embedding_dim=32)
        nodes = [
            KnowledgeNode(
                node_id=f"node_{i}",
                content=f"Content {i}",
                source_id="test_source",
                position=NodePosition(start_char=0, end_char=100),
                dense_vector=[float(i)] * 32,
                metadata={},
            )
            for i in range(100)
        ]
        await index.add(nodes)

        cursor = SearchCursor(offset=0, page_size=10)

        page, next_cursor = await index.search_paginated(
            query_vector=[50.0] * 32, cursor=cursor
        )

        assert len(page) == 10
        assert next_cursor.offset == 10
        assert next_cursor.page_size == 10

    async def test_search_paginated_should_support_multiple_pages(self) -> None:
        """Verifies pagination through entire result set.

        Given:
            Index with 50 nodes and page_size=15.
        When:
            Iterating through all pages.
        Then:
            Retrieves all 50 results across 4 pages (15+15+15+5).
        """
        index = MemoryIndex(embedding_dim=16)
        nodes = [
            KnowledgeNode(
                node_id=f"test_{i}",
                content=f"Test {i}",
                source_id="test_source",
                position=NodePosition(start_char=0, end_char=100),
                dense_vector=[float(i % 16)] * 16,
                metadata={},
            )
            for i in range(50)
        ]
        await index.add(nodes)

        cursor = SearchCursor(offset=0, page_size=15)
        all_results = []

        while True:
            page, cursor = await index.search_paginated(
                query_vector=[8.0] * 16, cursor=cursor
            )
            all_results.extend(page)

            if len(page) < cursor.page_size:
                break

        assert len(all_results) == 50
        result_ids = [r.node_id for r in all_results]
        assert len(set(result_ids)) == len(result_ids)

    async def test_search_paginated_should_handle_offset_beyond_results(self) -> None:
        """Verifies pagination handles offset past end of results.

        Given:
            Index with 20 nodes.
        When:
            Requesting page at offset=100.
        Then:
            Returns empty page.
        """
        index = MemoryIndex(embedding_dim=8)
        nodes = [
            KnowledgeNode(
                node_id=f"n_{i}",
                content=f"C {i}",
                source_id="test_source",
                position=NodePosition(start_char=0, end_char=100),
                dense_vector=[1.0] * 8,
                metadata={},
            )
            for i in range(20)
        ]
        await index.add(nodes)

        cursor = SearchCursor(offset=100, page_size=10)

        page, next_cursor = await index.search_paginated(
            query_vector=[1.0] * 8, cursor=cursor
        )

        assert len(page) == 0
        assert next_cursor.offset == 100

    async def test_search_paginated_should_update_total_results(self) -> None:
        """Verifies cursor tracks total_results count.

        Given:
            Index with 75 nodes.
        When:
            First search_paginated() call.
        Then:
            next_cursor.total_results equals 75.
        """
        index = MemoryIndex(embedding_dim=64)
        nodes = [
            KnowledgeNode(
                node_id=f"doc_{i}",
                content=f"Document {i}",
                source_id="test_source",
                position=NodePosition(start_char=0, end_char=100),
                dense_vector=[float(i % 64)] * 64,
                metadata={},
            )
            for i in range(75)
        ]
        await index.add(nodes)

        cursor = SearchCursor(offset=0, page_size=20)

        _, next_cursor = await index.search_paginated(
            query_vector=[32.0] * 64, cursor=cursor
        )

        assert next_cursor.total_results == 75

    async def test_search_paginated_should_respect_filters(self) -> None:
        """Verifies pagination works with metadata filters.

        Given:
            Index with 100 nodes, 50 matching filter.
        When:
            Paginating with filters applied.
        Then:
            Pages contain only filtered results.
        """
        index = MemoryIndex(embedding_dim=32)
        nodes = [
            KnowledgeNode(
                node_id=f"item_{i}",
                content=f"Item {i}",
                source_id="test_source",
                position=NodePosition(start_char=0, end_char=100),
                dense_vector=[float(i)] * 32,
                metadata={"category": "A" if i % 2 == 0 else "B"},
            )
            for i in range(100)
        ]
        await index.add(nodes)

        cursor = SearchCursor(offset=0, page_size=10)

        page, next_cursor = await index.search_paginated(
            query_vector=[25.0] * 32, cursor=cursor, filters={"category": "A"}
        )

        assert len(page) <= 10
        for result in page:
            if result.node is not None:
                assert result.node.metadata.get("category") == "A"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.rag_edge_case
class TestMemoryIndexEdgeCases:
    """Additional edge cases for MemoryIndex coverage."""

    async def test_compute_similarities_should_raise_for_unknown_distance_metric(
        self, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Verify that an unknown distance metric raises ValueError.

        Given:
            A MemoryIndex with an invalid distance metric.
        When:
            Calling _compute_similarities() with this metric.
        Then:
            A ValueError is raised indicating unknown metric.
        """
        index: MemoryIndex = memory_index_factory(embedding_dim=3)
        # Manually set an invalid metric to trigger the ValueError
        index.distance_metric = "invalid_metric"  # type: ignore

        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vectors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        with pytest.raises(ValueError, match="Unknown distance metric"):
            index._compute_similarities(query, vectors)

    async def test_evict_nodes_should_return_early_for_zero_or_negative_count(
        self, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Verify that _evict_nodes handles zero/negative count gracefully.

        Given:
            An index with max_nodes limit and some nodes.
        When:
            Calling _evict_nodes() with count=0.
        Then:
            No eviction occurs and method returns early.
        """
        index: MemoryIndex = memory_index_factory(
            embedding_dim=2, distance_metric="cosine"
        )
        index.max_nodes = 10
        index.eviction_policy = "lru"

        nodes = [
            KnowledgeNode(
                node_id="test",
                content="Test",
                source_id="s",
                position=NodePosition(start_char=0, end_char=1, page=1, section="Test"),
                metadata={},
                dense_vector=[1.0, 0.0],
                sparse_vector=None,
            )
        ]
        await index.add(nodes)

        initial_count = await index.count()

        # Call _evict_nodes with count=0 (should return early)
        await index._evict_nodes(0)

        # Verify count unchanged
        assert await index.count() == initial_count

        # Also test negative count
        await index._evict_nodes(-5)
        assert await index.count() == initial_count

    async def test_index_can_be_used_as_async_context_manager(
        self, memory_index_factory: Callable[..., Any]
    ) -> None:
        """Verify that MemoryIndex can be used with async context manager.

        Given:
            A MemoryIndex instance.
        When:
            Using it with 'async with' statement.
        Then:
            __aexit__ is called without errors.
        """
        index: MemoryIndex = memory_index_factory(embedding_dim=3)

        # Use index as context manager
        async with index:
            await index.add(
                [
                    KnowledgeNode(
                        node_id="ctx",
                        content="Context manager test",
                        source_id="s",
                        position=NodePosition(
                            start_char=0, end_char=1, page=1, section="Test"
                        ),
                        metadata={},
                        dense_vector=[1.0, 0.0, 0.0],
                        sparse_vector=None,
                    )
                ]
            )
            assert await index.count() == 1

        # After context, index should still be usable
        assert await index.count() == 1

    async def test_from_config_should_create_index_with_correct_parameters(
        self,
        index_config_factory: Callable[..., Any],
        fake_embedder_factory: Callable[..., Any],
    ) -> None:
        """Verify from_config creates MemoryIndex with config parameters.

        Given:
            An IndexConfig with specific settings.
        When:
            Creating MemoryIndex via from_config().
        Then:
            Index is initialized with correct dimension, collection, and metric.
        """
        embedder = fake_embedder_factory(dimension=128)
        config = index_config_factory(
            backend="memory",
            collection_name="test_collection",
            embedding_dim=128,
            connection={"distance_metric": "euclidean"},
        )

        index = MemoryIndex.from_config(config, embedder)

        assert index.embedding_dim == 128
        assert index.collection_name == "test_collection"
        assert index.distance_metric == "euclidean"
        assert index.embedder is embedder

    async def test_from_config_should_use_defaults_when_connection_is_none(
        self, index_config_factory: Callable[..., Any]
    ) -> None:
        """Verify from_config uses defaults when connection dict is None.

        Given:
            An IndexConfig with connection=None.
        When:
            Creating MemoryIndex via from_config().
        Then:
            Default distance_metric (cosine) is used.
        """
        config = index_config_factory(
            backend="memory",
            collection_name="default_test",
            embedding_dim=64,
            connection=None,
        )

        index = MemoryIndex.from_config(config, embedder=None)

        assert index.embedding_dim == 64
        assert index.collection_name == "default_test"
        assert index.distance_metric == "cosine"


@pytest.mark.integration
@pytest.mark.asyncio
class TestSearchCursorEdgeCases:
    """Edge case tests for SearchCursor."""

    async def test_cursor_with_page_size_larger_than_results(self) -> None:
        """Verifies large page_size returns all available results.

        Given:
            Index with 30 nodes and page_size=100.
        When:
            First page request.
        Then:
            Returns all 30 results in single page.
        """
        index = MemoryIndex(embedding_dim=16)
        nodes = [
            KnowledgeNode(
                node_id=f"x_{i}",
                content="content",
                source_id="test_source",
                position=NodePosition(start_char=0, end_char=100),
                dense_vector=[1.0] * 16,
                metadata={},
            )
            for i in range(30)
        ]
        await index.add(nodes)

        cursor = SearchCursor(offset=0, page_size=100)

        page, _ = await index.search_paginated(query_vector=[1.0] * 16, cursor=cursor)

        assert len(page) == 30

    async def test_cursor_with_minimum_page_size(self) -> None:
        """Verifies page_size=1 returns single result per page.

        Given:
            Index with 10 nodes and page_size=1.
        When:
            Requesting first page.
        Then:
            Returns exactly 1 result.
        """
        index = MemoryIndex(embedding_dim=8)
        nodes = [
            KnowledgeNode(
                node_id=f"single_{i}",
                content="test",
                source_id="test_source",
                position=NodePosition(start_char=0, end_char=100),
                dense_vector=[0.5] * 8,
                metadata={},
            )
            for i in range(10)
        ]
        await index.add(nodes)

        cursor = SearchCursor(offset=0, page_size=1)

        page, next_cursor = await index.search_paginated(
            query_vector=[0.5] * 8, cursor=cursor
        )

        assert len(page) == 1
        assert next_cursor.offset == 1

    async def test_cursor_should_validate_page_size_bounds(self) -> None:
        """Verifies SearchCursor enforces page_size constraints.

        Given:
            Pydantic model with page_size constraints (ge=1, le=1000).
        When:
            Creating cursor with invalid page_size values.
        Then:
            Raises validation error.
        """
        with pytest.raises(ValueError):
            SearchCursor(offset=0, page_size=0)

        with pytest.raises(ValueError):
            SearchCursor(offset=0, page_size=1001)

    async def test_cursor_should_validate_offset_non_negative(self) -> None:
        """Verifies offset cannot be negative.

        Given:
            SearchCursor with offset validation (ge=0).
        When:
            Creating cursor with offset=-1.
        Then:
            Raises validation error.
        """
        with pytest.raises(ValueError):
            SearchCursor(offset=-1, page_size=10)


@pytest.mark.unit
@pytest.mark.asyncio
class TestProductionResilience:
    """Tests for production-ready error handling and edge cases."""

    async def test_index_should_handle_empty_search_gracefully(self) -> None:
        """Verifies search on empty index returns empty results.

        Given:
            Empty MemoryIndex.
        When:
            Performing search.
        Then:
            Returns empty list without errors.
        """
        index = MemoryIndex(embedding_dim=64)

        results = await index.search(query_vector=[0.5] * 64, top_k=10)

        assert results == []

    async def test_pagination_on_empty_index_should_return_empty_page(self) -> None:
        """Verifies pagination handles empty index gracefully.

        Given:
            Empty index and valid cursor.
        When:
            Calling search_paginated().
        Then:
            Returns empty page with total_results=0.
        """
        index = MemoryIndex(embedding_dim=32)
        cursor = SearchCursor(offset=0, page_size=10)

        page, next_cursor = await index.search_paginated(
            query_vector=[1.0] * 32, cursor=cursor
        )

        assert len(page) == 0
        assert next_cursor.total_results == 0

    async def test_eviction_should_not_break_pagination(self) -> None:
        """Verifies pagination remains consistent after LRU eviction.

        Given:
            Index with max_nodes=50, LRU eviction, containing 60 nodes.
        When:
            Paginating through results.
        Then:
            Returns consistent results without crashes.
        """
        index = MemoryIndex(embedding_dim=16, max_nodes=50, eviction_policy="lru")

        nodes = [
            KnowledgeNode(
                node_id=f"evict_test_{i}",
                content=f"Content {i}",
                source_id="test_source",
                position=NodePosition(start_char=0, end_char=100),
                dense_vector=[float(i % 16)] * 16,
                metadata={},
            )
            for i in range(60)
        ]
        await index.add(nodes)

        cursor = SearchCursor(offset=0, page_size=10)

        pages = []
        for _ in range(5):
            page, cursor = await index.search_paginated(
                query_vector=[8.0] * 16, cursor=cursor
            )
            pages.append(page)
            if len(page) == 0:
                break

        total_results = sum(len(p) for p in pages)
        assert total_results <= 50
        assert total_results > 0
