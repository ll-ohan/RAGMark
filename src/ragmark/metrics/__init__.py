"""Unified metrics framework for RAG systems.

This package provides abstractions for both runtime monitoring metrics
and evaluation metrics, with a central registry for discovery and instantiation.
"""

from typing import Any

from ragmark.metrics.base import (
    BaseMetric,
    EvaluationMetric,
    MonitoringMetric,
)

# Import concrete metric implementations to trigger auto-registration
from ragmark.metrics.evaluation.retrieval import compute_retrieval_batch
from ragmark.metrics.registry import MetricRegistry
from ragmark.schemas.evaluation import CaseResult


def compute_retrieval_metrics(
    results: list[CaseResult],
    ground_truth: dict[str, list[str]],
    ks: list[int] | None = None,
    metric_parameters: dict[str, dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Compute aggregated retrieval metrics across all cases.

    Delegates to retrieval evaluation helpers for metric computation.

    Args:
        results: Case results with retrieval traces to evaluate.
        ground_truth: Mapping from case_id to relevant node IDs.
        ks: K values for @k metrics (recall@k, precision@k, etc.).
        metric_parameters: Optional metric parameters keyed by base name.

    Returns:
        Aggregated metrics with mean values across all valid cases.
    """
    return compute_retrieval_batch(results, ground_truth, ks, metric_parameters)


__all__ = [
    "BaseMetric",
    "EvaluationMetric",
    "MonitoringMetric",
    "MetricRegistry",
    "compute_retrieval_metrics",
]
