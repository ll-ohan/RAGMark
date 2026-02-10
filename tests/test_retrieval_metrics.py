"""Tests for retrieval evaluation metrics.

This module tests the standard information retrieval metrics used to evaluate
RAG system retrieval quality.
"""

from typing import Any, cast

import pytest

from ragmark.metrics import MetricRegistry, compute_retrieval_metrics
from ragmark.metrics.base import EvaluationMetric
from ragmark.schemas.documents import KnowledgeNode, NodePosition
from ragmark.schemas.evaluation import CaseResult
from ragmark.schemas.retrieval import RetrievedNode, TraceContext


# Test wrappers using MetricRegistry - for backward compatibility with tests
def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Test wrapper for recall@k using MetricRegistry."""
    metric = cast(
        EvaluationMetric[Any, float], MetricRegistry.create(f"recall@{k}", k=k)
    )
    return metric.compute(retrieved_ids=retrieved_ids, relevant_ids=relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Test wrapper for precision@k using MetricRegistry."""
    metric = cast(
        EvaluationMetric[Any, float], MetricRegistry.create(f"precision@{k}", k=k)
    )
    return metric.compute(retrieved_ids=retrieved_ids, relevant_ids=relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Test wrapper for MRR using MetricRegistry."""
    metric = cast(EvaluationMetric[Any, float], MetricRegistry.create("mrr"))
    return metric.compute(retrieved_ids=retrieved_ids, relevant_ids=relevant_ids)


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Test wrapper for NDCG@k using MetricRegistry."""
    metric = cast(EvaluationMetric[Any, float], MetricRegistry.create(f"ndcg@{k}", k=k))
    return metric.compute(retrieved_ids=retrieved_ids, relevant_ids=relevant_ids)


def map_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Test wrapper for MAP@k using MetricRegistry."""
    metric = cast(EvaluationMetric[Any, float], MetricRegistry.create(f"map@{k}", k=k))
    return metric.compute(retrieved_ids=retrieved_ids, relevant_ids=relevant_ids)


@pytest.mark.unit
class TestRecallAtK:
    """Test suite for recall@k metric calculation."""

    def test_recall_perfect_match(self):
        """Verify recall@k returns 1.0 when all relevant items are retrieved.

        Given:
            Retrieved IDs contain all relevant items in top-k.
        When:
            Computing recall@5.
        Then:
            Score is 1.0 (perfect recall).
        """
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = ["a", "b", "c"]

        score = recall_at_k(retrieved, relevant, k=5)

        assert score == 1.0

    def test_recall_partial_match(self):
        """Verify recall@k correctly calculates partial matches.

        Given:
            2 of 4 relevant items in top-5 results.
        When:
            Computing recall@5.
        Then:
            Score is 0.5 (50% of relevant items found).
        """
        retrieved = ["a", "b", "x", "y", "z"]
        relevant = ["a", "b", "c", "d"]

        score = recall_at_k(retrieved, relevant, k=5)

        assert score == 0.5

    def test_recall_no_matches(self):
        """Verify recall@k returns 0.0 when no relevant items retrieved.

        Given:
            No overlap between retrieved and relevant items.
        When:
            Computing recall@5.
        Then:
            Score is 0.0 (no recall).
        """
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b", "c"]

        score = recall_at_k(retrieved, relevant, k=5)

        assert score == 0.0

    def test_recall_empty_retrieved(self):
        """Verify recall@k handles empty retrieved list.

        Given:
            No items retrieved.
        When:
            Computing recall@5.
        Then:
            Score is 0.0.
        """
        retrieved: list[str] = []
        relevant = ["a", "b", "c"]

        score = recall_at_k(retrieved, relevant, k=5)

        assert score == 0.0

    def test_recall_empty_relevant(self):
        """Verify recall@k handles empty relevant list.

        Given:
            No relevant items defined.
        When:
            Computing recall@5.
        Then:
            Score is 0.0 (undefined case, return 0).
        """
        retrieved: list[str] = ["a", "b", "c"]
        relevant: list[str] = []

        score = recall_at_k(retrieved, relevant, k=5)

        assert score == 0.0

    def test_recall_respects_k_limit(self):
        """Verify recall@k only considers top-k results.

        Given:
            Relevant item at position 6.
        When:
            Computing recall@5.
        Then:
            Score is 0.0 (item beyond top-5).
        """
        retrieved = ["x", "x", "x", "x", "x", "a"]
        relevant = ["a"]

        score = recall_at_k(retrieved, relevant, k=5)

        assert score == 0.0


@pytest.mark.unit
class TestPrecisionAtK:
    """Test suite for precision@k metric calculation."""

    def test_precision_perfect_match(self):
        """Verify precision@k returns 1.0 when all retrieved are relevant.

        Given:
            All top-3 results are relevant.
        When:
            Computing precision@3.
        Then:
            Score is 1.0 (perfect precision).
        """
        retrieved = ["a", "b", "c"]
        relevant = ["a", "b", "c", "d", "e"]

        score = precision_at_k(retrieved, relevant, k=3)

        assert score == 1.0

    def test_precision_partial_match(self):
        """Verify precision@k correctly calculates partial matches.

        Given:
            2 of 5 retrieved items are relevant.
        When:
            Computing precision@5.
        Then:
            Score is 0.4 (40% precision).
        """
        retrieved = ["a", "x", "b", "y", "z"]
        relevant = ["a", "b"]

        score = precision_at_k(retrieved, relevant, k=5)

        assert score == 0.4

    def test_precision_no_matches(self):
        """Verify precision@k returns 0.0 when no relevant items retrieved.

        Given:
            No relevant items in top-k.
        When:
            Computing precision@3.
        Then:
            Score is 0.0.
        """
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b", "c"]

        score = precision_at_k(retrieved, relevant, k=3)

        assert score == 0.0

    def test_precision_empty_retrieved(self):
        """Verify precision@k handles empty retrieved list.

        Given:
            No items retrieved.
        When:
            Computing precision@5.
        Then:
            Score is 0.0.
        """
        retrieved: list[str] = []
        relevant = ["a", "b", "c"]

        score = precision_at_k(retrieved, relevant, k=5)

        assert score == 0.0


@pytest.mark.unit
class TestMRR:
    """Test suite for Mean Reciprocal Rank metric."""

    def test_mrr_first_position(self):
        """Verify MRR returns 1.0 when first item is relevant.

        Given:
            First retrieved item is relevant.
        When:
            Computing MRR.
        Then:
            Score is 1.0 (1/rank = 1/1).
        """
        retrieved = ["a", "x", "y"]
        relevant = ["a"]

        score = mrr(retrieved, relevant)

        assert score == 1.0

    def test_mrr_second_position(self):
        """Verify MRR correctly calculates for second position.

        Given:
            First relevant item at rank 2.
        When:
            Computing MRR.
        Then:
            Score is 0.5 (1/rank = 1/2).
        """
        retrieved = ["x", "a", "y"]
        relevant = ["a"]

        score = mrr(retrieved, relevant)

        assert score == 0.5

    def test_mrr_fifth_position(self):
        """Verify MRR correctly calculates for later positions.

        Given:
            First relevant item at rank 5.
        When:
            Computing MRR.
        Then:
            Score is 0.2 (1/rank = 1/5).
        """
        retrieved = ["w", "x", "y", "z", "a"]
        relevant = ["a", "b"]

        score = mrr(retrieved, relevant)

        assert score == 0.2

    def test_mrr_no_relevant_found(self):
        """Verify MRR returns 0.0 when no relevant items found.

        Given:
            No relevant items in retrieved list.
        When:
            Computing MRR.
        Then:
            Score is 0.0.
        """
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b"]

        score = mrr(retrieved, relevant)

        assert score == 0.0

    def test_mrr_empty_retrieved(self):
        """Verify MRR handles empty retrieved list.

        Given:
            No items retrieved.
        When:
            Computing MRR.
        Then:
            Score is 0.0.
        """
        retrieved: list[str] = []
        relevant: list[str] = ["a"]

        score = mrr(retrieved, relevant)

        assert score == 0.0


@pytest.mark.unit
class TestNDCGAtK:
    """Test suite for NDCG@k metric calculation."""

    def test_ndcg_perfect_ranking(self):
        """Verify NDCG@k returns 1.0 for perfect ranking.

        Given:
            All relevant items ranked first.
        When:
            Computing NDCG@5.
        Then:
            Score is 1.0 (DCG equals IDCG).
        """
        retrieved = ["a", "b", "c", "x", "y"]
        relevant = ["a", "b", "c"]

        score = ndcg_at_k(retrieved, relevant, k=5)

        assert score == 1.0

    def test_ndcg_imperfect_ranking(self):
        """Verify NDCG@k penalizes poor ranking.

        Given:
            Relevant items ranked lower (positions 2, 4, 5).
        When:
            Computing NDCG@5.
        Then:
            Score is less than 1.0 (DCG < IDCG due to ranking).
        """
        retrieved = ["x", "a", "y", "b", "c"]
        relevant = ["a", "b", "c"]

        score = ndcg_at_k(retrieved, relevant, k=5)

        assert 0.0 < score < 1.0

    def test_ndcg_no_relevant_found(self):
        """Verify NDCG@k returns 0.0 when no relevant items found.

        Given:
            No relevant items in top-k.
        When:
            Computing NDCG@5.
        Then:
            Score is 0.0 (DCG = 0).
        """
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b", "c"]

        score = ndcg_at_k(retrieved, relevant, k=5)

        assert score == 0.0

    def test_ndcg_empty_retrieved(self):
        """Verify NDCG@k handles empty retrieved list.

        Given:
            No items retrieved.
        When:
            Computing NDCG@5.
        Then:
            Score is 0.0.
        """
        retrieved: list[str] = []
        relevant: list[str] = ["a"]

        score = ndcg_at_k(retrieved, relevant, k=5)

        assert score == 0.0


@pytest.mark.unit
class TestMAPAtK:
    """Test suite for MAP@k metric calculation."""

    def test_map_perfect_match(self):
        """Verify MAP@k returns 1.0 for perfect early retrieval.

        Given:
            All relevant items at top positions.
        When:
            Computing MAP@5.
        Then:
            Score is 1.0 (all relevant items ranked first).
        """
        retrieved = ["a", "b", "c", "x", "y"]
        relevant = ["a", "b", "c"]

        score = map_at_k(retrieved, relevant, k=5)

        assert score == 1.0

    def test_map_partial_match(self):
        """Verify MAP@k rewards early relevant items.

        Given:
            2 of 3 relevant items found, one early, one late.
        When:
            Computing MAP@5.
        Then:
            Score reflects early retrieval bonus.
        """
        retrieved = ["a", "x", "y", "z", "b"]
        relevant = ["a", "b", "c"]

        score = map_at_k(retrieved, relevant, k=5)

        assert 0.0 < score < 1.0

    def test_map_no_matches(self):
        """Verify MAP@k returns 0.0 when no relevant items found.

        Given:
            No relevant items in top-k.
        When:
            Computing MAP@5.
        Then:
            Score is 0.0.
        """
        retrieved = ["x", "y", "z"]
        relevant = ["a", "b", "c"]

        score = map_at_k(retrieved, relevant, k=5)

        assert score == 0.0

    def test_map_empty_retrieved(self):
        """Verify MAP@k handles empty retrieved list.

        Given:
            No items retrieved.
        When:
            Computing MAP@5.
        Then:
            Score is 0.0.
        """
        retrieved: list[str] = []
        relevant: list[str] = ["a"]

        score = map_at_k(retrieved, relevant, k=5)

        assert score == 0.0


@pytest.mark.unit
class TestComputeRetrievalMetrics:
    """Test suite for aggregated metrics computation."""

    @pytest.fixture
    def sample_nodes(self) -> list[KnowledgeNode]:
        """Create sample KnowledgeNode fixtures for testing."""
        nodes: list[KnowledgeNode] = []
        for i in range(5):
            node = KnowledgeNode(
                node_id=f"node_{i}",
                content=f"Content {i}",
                source_id="source_1",
                position=NodePosition(
                    start_char=0, end_char=100, page=1, section="section"
                ),
                dense_vector=[0.1 * i, 0.2 * i],
                sparse_vector={0: 0.3 * i, 1: 0.4 * i},
            )
            nodes.append(node)
        return nodes

    @pytest.fixture
    def sample_case_results(
        self, sample_nodes: list[KnowledgeNode]
    ) -> list[CaseResult]:
        """Create sample CaseResult fixtures with realistic retrieval traces."""
        case_result_1 = CaseResult(
            case_id="case_1",
            predicted_answer="Sample answer for case 1",
            generation_result=None,
            trace=TraceContext(
                query="test query 1",
                retrieved_nodes=[
                    RetrievedNode(node=sample_nodes[0], score=0.9, rank=1),
                    RetrievedNode(node=sample_nodes[1], score=0.8, rank=2),
                    RetrievedNode(node=sample_nodes[2], score=0.7, rank=3),
                ],
                reranked=False,
            ),
        )

        case_result_2 = CaseResult(
            case_id="case_2",
            predicted_answer="Sample answer for case 2",
            generation_result=None,
            trace=TraceContext(
                query="test query 2",
                retrieved_nodes=[
                    RetrievedNode(node=sample_nodes[1], score=0.95, rank=1),
                    RetrievedNode(node=sample_nodes[3], score=0.85, rank=2),
                ],
                reranked=False,
            ),
        )

        return [case_result_1, case_result_2]

    def test_compute_metrics_single_case(self, sample_case_results: list[CaseResult]):
        """Verify metrics computation for single case.

        Given:
            One case with 3 retrieved nodes, 2 relevant (node_0, node_1).
        When:
            Computing retrieval metrics with k=5.
        Then:
            All standard metrics are computed and recall is 1.0 (both relevant found).
        """
        ground_truth = {
            "case_1": ["node_0", "node_1"],
        }

        results = [sample_case_results[0]]

        aggregated = compute_retrieval_metrics(results, ground_truth, ks=[5])

        assert "recall@5" in aggregated
        assert "precision@5" in aggregated
        assert "mrr" in aggregated
        assert "ndcg@5" in aggregated
        assert "map@5" in aggregated

        assert results[0].case_metrics["recall@5"] == 1.0
        assert results[0].case_metrics["precision@5"] == 2.0 / 3.0

    def test_compute_metrics_multiple_cases(
        self, sample_case_results: list[CaseResult]
    ):
        """Verify metrics aggregation across multiple cases.

        Given:
            Two cases with different retrieval results and ground truth.
        When:
            Computing retrieval metrics with k=5.
        Then:
            Metrics are averaged across both cases and populate case_metrics.
        """
        ground_truth = {
            "case_1": ["node_0", "node_1"],
            "case_2": ["node_1", "node_3"],
        }

        aggregated = compute_retrieval_metrics(
            sample_case_results, ground_truth, ks=[5]
        )

        assert 0.0 < aggregated["recall@5"] <= 1.0
        assert "recall@5" in sample_case_results[0].case_metrics
        assert "precision@5" in sample_case_results[0].case_metrics
        assert "recall@5" in sample_case_results[1].case_metrics
        assert "precision@5" in sample_case_results[1].case_metrics

    def test_compute_metrics_missing_ground_truth(
        self, sample_case_results: list[CaseResult]
    ):
        """Verify handling of cases without ground truth.

        Given:
            Ground truth defined only for case_1, not case_2.
        When:
            Computing retrieval metrics.
        Then:
            Only case_1 metrics are populated; case_2 is skipped gracefully.
        """
        ground_truth = {
            "case_1": ["node_0", "node_1"],
        }

        _ = compute_retrieval_metrics(sample_case_results, ground_truth, ks=[5])

        assert "recall@5" in sample_case_results[0].case_metrics
        assert len(sample_case_results[1].case_metrics) == 0

    def test_compute_metrics_empty_results(self):
        """Verify handling of empty results list.

        Given:
            No case results provided.
        When:
            Computing retrieval metrics.
        Then:
            Returns empty metrics dict.
        """
        ground_truth: dict[str, list[str]] = {}
        results: list[CaseResult] = []

        aggregated = compute_retrieval_metrics(results, ground_truth)

        assert aggregated == {}

    def test_compute_metrics_custom_k_values(
        self, sample_case_results: list[CaseResult]
    ):
        """Verify metrics computed for custom k values.

        Given:
            Custom k values [3, 7, 10] instead of default.
        When:
            Computing retrieval metrics.
        Then:
            All metrics are generated for each specified k value.
        """
        ground_truth = {
            "case_1": ["node_0", "node_1"],
            "case_2": ["node_1", "node_3"],
        }

        aggregated = compute_retrieval_metrics(
            sample_case_results, ground_truth, ks=[3, 7, 10]
        )

        assert "recall@3" in aggregated
        assert "recall@7" in aggregated
        assert "recall@10" in aggregated
        assert "precision@3" in aggregated
        assert "precision@7" in aggregated
        assert "precision@10" in aggregated
        assert "ndcg@3" in aggregated
        assert "map@3" in aggregated


@pytest.mark.unit
class TestMetricsNumericalStability:
    """Test suite for numerical stability and edge cases per TEST_POLICY.md Section 4."""

    def test_recall_no_nan_with_edge_inputs(self):
        """Verify recall@k never produces NaN values.

        Given:
            Various edge case inputs (empty lists, no overlap).
        When:
            Computing recall@k.
        Then:
            Result is always a valid float, never NaN or infinity.
        """
        test_cases: list[tuple[list[str], list[str], int]] = [
            ([], [], 5),
            (["a"], [], 5),
            ([], ["a"], 5),
            (["a", "b"], ["c", "d"], 5),
        ]

        for retrieved, relevant, k in test_cases:
            score = recall_at_k(retrieved, relevant, k=k)
            assert isinstance(score, float)
            assert not (score != score)
            assert score >= 0.0
            assert score <= 1.0

    def test_precision_no_nan_with_edge_inputs(self):
        """Verify precision@k never produces NaN values.

        Given:
            Various edge case inputs including empty lists.
        When:
            Computing precision@k.
        Then:
            Result is always a valid float between 0.0 and 1.0.
        """
        test_cases: list[tuple[list[str], list[str], int]] = [
            ([], [], 5),
            (["a"], [], 5),
            ([], ["a"], 5),
            (["a", "b"], ["c", "d"], 5),
        ]

        for retrieved, relevant, k in test_cases:
            score = precision_at_k(retrieved, relevant, k=k)
            assert isinstance(score, float)
            assert not (score != score)
            assert score >= 0.0
            assert score <= 1.0

    def test_ndcg_no_nan_with_edge_inputs(self):
        """Verify NDCG@k never produces NaN values.

        Given:
            Edge case inputs that could cause division by zero.
        When:
            Computing NDCG@k.
        Then:
            Result is always a valid float, handles division by zero gracefully.
        """
        test_cases: list[tuple[list[str], list[str], int]] = [
            ([], [], 5),
            (["a"], [], 5),
            ([], ["a"], 5),
            (["a", "b"], ["c", "d"], 5),
        ]

        for retrieved, relevant, k in test_cases:
            score = ndcg_at_k(retrieved, relevant, k=k)
            assert isinstance(score, float)
            assert not (score != score)
            assert score >= 0.0
            assert score <= 1.0

    def test_metrics_with_duplicate_retrieved_items(self):
        """Verify metrics handle duplicate items in retrieved list.

        Given:
            Retrieved list containing duplicate node IDs.
        When:
            Computing all metrics.
        Then:
            Metrics are calculated correctly treating duplicates as separate positions.
        """
        retrieved = ["a", "a", "b", "c", "c"]
        relevant = ["a", "b"]

        recall = recall_at_k(retrieved, relevant, k=5)
        precision = precision_at_k(retrieved, relevant, k=5)
        mrr_score = mrr(retrieved, relevant)

        assert recall == 1.0
        assert precision == 0.6
        assert mrr_score == 1.0

    def test_all_metrics_with_identical_lists(self):
        """Verify perfect scores when retrieved equals relevant.

        Given:
            Retrieved and relevant lists are identical.
        When:
            Computing all metrics.
        Then:
            Recall, precision, MRR, NDCG, and MAP all return 1.0.
        """
        items = ["a", "b", "c", "d", "e"]

        assert recall_at_k(items, items, k=5) == 1.0
        assert precision_at_k(items, items, k=5) == 1.0
        assert mrr(items, items) == 1.0
        assert ndcg_at_k(items, items, k=5) == 1.0
        assert map_at_k(items, items, k=5) == 1.0
