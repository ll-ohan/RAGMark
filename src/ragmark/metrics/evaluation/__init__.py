"""Evaluation metrics for RAG systems.

Provides metrics for measuring retrieval quality, generation quality, etc.
"""

# Import and register metrics so they're available
from ragmark.metrics.evaluation.retrieval import (  # noqa: F401
    MAP,
    MRR,
    NDCG,
    PrecisionAtK,
    RecallAtK,
)

__all__ = [
    "RecallAtK",
    "PrecisionAtK",
    "MRR",
    "NDCG",
    "MAP",
]
