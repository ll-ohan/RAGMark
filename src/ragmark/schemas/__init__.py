"""Pydantic models for RAGMark data structures.

This package contains all data models used throughout the framework,
organized by functional domain.
"""

from ragmark.schemas.documents import (
    KnowledgeNode,
    NodePosition,
    SourceDoc,
    VectorPayload,
)
from ragmark.schemas.evaluation import (
    AuditReport,
    CaseResult,
    SystemInfo,
    TrialCase,
)
from ragmark.schemas.generation import (
    GenerationResult,
    TokenUsage,
)
from ragmark.schemas.retrieval import (
    RetrievedNode,
    SearchResult,
    TraceContext,
)

__all__ = [
    # Documents
    "SourceDoc",
    "NodePosition",
    "KnowledgeNode",
    "VectorPayload",
    # Retrieval
    "SearchResult",
    "RetrievedNode",
    "TraceContext",
    # Generation
    "TokenUsage",
    "GenerationResult",
    # Evaluation
    "TrialCase",
    "CaseResult",
    "SystemInfo",
    "AuditReport",
]
