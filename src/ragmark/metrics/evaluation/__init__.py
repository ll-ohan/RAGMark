"""Evaluation metrics for RAG systems.

Provides metrics for measuring retrieval quality, generation quality, etc.
"""

# Import and register metrics so they're available
from ragmark.metrics.evaluation.generation import (  # noqa: F401
    AnswerRelevancy,
    Correctness,
    Faithfulness,
    JudgePromptConfig,
    compute_generation_metrics,
)
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
    "Faithfulness",
    "AnswerRelevancy",
    "Correctness",
    "JudgePromptConfig",
    "compute_generation_metrics",
]
