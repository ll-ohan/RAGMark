"""Unit and integration tests for evaluation metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from ragmark.metrics.base import EvaluationMetric, MetricValidationError
from ragmark.metrics.evaluation.generation import AnswerRelevancy, Faithfulness
from ragmark.metrics.evaluation.retrieval import (
    RecallAtK,
    _metric_kwargs,  # type: ignore[attr-defined]
    _resolve_metric_k_values,  # type: ignore[attr-defined]
    compute_retrieval_batch,
)
from ragmark.schemas.documents import KnowledgeNode, NodePosition
from ragmark.schemas.evaluation import CaseResult
from ragmark.schemas.retrieval import RetrievedNode, TraceContext


class OrderMetric(EvaluationMetric[Any, int]):
    """Minimal metric to verify call ordering."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return "order_metric"

    @property
    def description(self) -> str:
        return "Verifies validate_inputs precedes compute"

    def validate_inputs(self, **inputs: Any) -> None:
        self.calls.append("validate")

    def compute(self, **inputs: Any) -> int:
        self.calls.append("compute")
        return 42


if TYPE_CHECKING:

    def approx(expected: float) -> object:
        ...
else:

    def approx(expected: float) -> object:
        return pytest.approx(expected)


def _make_node(node_id: str) -> KnowledgeNode:
    return KnowledgeNode(
        node_id=node_id,
        content=f"Content for {node_id}",
        source_id="src",
        position=NodePosition(start_char=0, end_char=10, page=1, section="s"),
        dense_vector=[0.1, 0.2],
        sparse_vector=None,
    )


@pytest.mark.unit
def test_evaluation_metric_call_should_validate_before_compute() -> None:
    """Verifies __call__ invokes validate_inputs before compute.

    Given:
        A metric that records call order.
    When:
        Invoking the metric as a callable.
    Then:
        Validation executes before computation.
        The returned value matches compute output.
    """
    metric = OrderMetric()

    result = metric(value=1)

    assert result == 42
    assert metric.calls == ["validate", "compute"]


@pytest.mark.unit
def test_retrieval_metrics_should_raise_when_relevant_ids_missing() -> None:
    """Validates retrieval metric validation rejects empty ground truth.

    Given:
        An empty list of relevant_ids.
    When:
        Validating metric inputs.
    Then:
        MetricValidationError is raised with an explicit message.
    """
    metric = RecallAtK(k=3)

    with pytest.raises(MetricValidationError, match="relevant_ids cannot be empty"):
        metric.validate_inputs(relevant_ids=[])


@pytest.mark.unit
def test_faithfulness_and_relevancy_should_raise_not_implemented() -> None:
    """Validates generation metrics report not implemented.

    Given:
        Placeholder faithfulness and answer relevancy metrics.
    When:
        Calling compute.
    Then:
        NotImplementedError is raised with explicit guidance.
    """
    faithfulness = Faithfulness()
    relevancy = AnswerRelevancy()

    with pytest.raises(NotImplementedError, match="not yet implemented"):
        faithfulness.compute(answer="A", context=["B"])

    with pytest.raises(NotImplementedError, match="not yet implemented"):
        relevancy.compute(question="Q", answer="A")


@pytest.mark.unit
def test_resolve_metric_k_values_should_prioritize_explicit_ks() -> None:
    """Validates explicit ks list overrides parameters.

    Given:
        Explicit ks list and conflicting metric_parameters.
    When:
        Resolving k values.
    Then:
        Explicit ks are returned as-is.
    """
    result = _resolve_metric_k_values(
        "recall",
        ks=[3, 7],
        metric_parameters={"recall": {"ks": [5]}},
    )

    assert result == [3, 7]


@pytest.mark.unit
def test_metric_kwargs_should_exclude_k_and_ks() -> None:
    """Validates metric kwargs exclude k/ks parameters.

    Given:
        Metric parameters including k, ks, and extra options.
    When:
        Extracting kwargs for metric construction.
    Then:
        Only non-k parameters are returned.
    """
    params: dict[str, dict[str, Any]] = {"recall": {"k": 5, "ks": [1, 2], "alpha": 0.3}}

    kwargs = _metric_kwargs(params, "recall")

    assert "k" not in kwargs
    assert "ks" not in kwargs
    assert kwargs["alpha"] == 0.3


@pytest.mark.integration
def test_compute_retrieval_batch_should_skip_invalid_cases() -> None:
    """Validates compute_retrieval_batch skips invalid cases.

    Given:
        A case with no retrieved nodes and valid ground truth.
    When:
        Computing retrieval metrics.
    Then:
        The case is skipped and aggregated metrics are empty.
    """
    trace = TraceContext(query="query", retrieved_nodes=[], reranked=False)
    case = CaseResult(
        case_id="case-1",
        predicted_answer=None,
        trace=trace,
        generation_result=None,
    )

    metrics = compute_retrieval_batch([case], ground_truth={"case-1": ["n1"]})

    assert metrics == {}
    assert case.case_metrics == {}


@pytest.mark.integration
def test_compute_retrieval_batch_should_aggregate_single_case() -> None:
    """Validates compute_retrieval_batch aggregates metrics correctly.

    Given:
        One valid case with retrieved nodes and ground truth.
    When:
        Computing retrieval metrics with k=2.
    Then:
        Aggregated metrics equal per-case metrics and values are correct.
    """
    node1 = _make_node("n1")
    node2 = _make_node("n2")
    node3 = _make_node("n3")

    retrieved = [
        RetrievedNode(node=node1, score=0.9, rank=1),
        RetrievedNode(node=node2, score=0.8, rank=2),
        RetrievedNode(node=node3, score=0.7, rank=3),
    ]
    trace = TraceContext(query="q", retrieved_nodes=retrieved, reranked=False)
    case = CaseResult(
        case_id="case-2",
        predicted_answer="",
        trace=trace,
        generation_result=None,
    )

    metrics = compute_retrieval_batch(
        [case],
        ground_truth={"case-2": ["n2", "n3"]},
        ks=None,
        metric_parameters={
            "recall": {"k": 2},
            "precision": {"ks": [2]},
            "ndcg": {"ks": [2]},
            "map": {"k": 2},
        },
    )

    expected_recall: float = 0.5
    expected_precision: float = 0.5
    expected_mrr: float = 0.5
    expected_map: float = 0.25
    expected_ndcg: float = 0.6309297535714575 / 1.6309297535714575

    assert metrics["recall@2"] == approx(expected_recall)
    assert metrics["precision@2"] == approx(expected_precision)
    assert metrics["mrr"] == approx(expected_mrr)
    assert metrics["map@2"] == approx(expected_map)
    assert metrics["ndcg@2"] == approx(expected_ndcg)

    case_metrics = case.case_metrics
    assert case_metrics["recall@2"] == approx(expected_recall)
    assert case_metrics["precision@2"] == approx(expected_precision)
    assert case_metrics["mrr"] == approx(expected_mrr)
    assert case_metrics["map@2"] == approx(expected_map)
    assert case_metrics["ndcg@2"] == approx(expected_ndcg)
