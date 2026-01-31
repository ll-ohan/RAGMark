"""Evaluation metrics for RAG systems.

This package provides metrics for evaluating both retrieval and generation
quality in RAG pipelines.
"""

from ragmark.evaluation.retrieval_metrics import (
    compute_retrieval_metrics,
    map_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "ndcg_at_k",
    "map_at_k",
    "compute_retrieval_metrics",
]
