"""RAGMark - A comprehensive benchmarking framework for RAG systems.

This package provides tools for ingesting documents, fragmenting text,
indexing vectors, retrieving context, generating responses, and evaluating
RAG pipeline performance.
"""

__version__ = "0.1.0"

from ragmark.exceptions import (
    ConfigError,
    EvaluationError,
    FragmentationError,
    GenerationError,
    IndexError,
    IngestionError,
    RAGMarkError,
    RetrievalError,
)

__all__ = [
    "__version__",
    "RAGMarkError",
    "ConfigError",
    "IngestionError",
    "FragmentationError",
    "IndexError",
    "RetrievalError",
    "GenerationError",
    "EvaluationError",
]
