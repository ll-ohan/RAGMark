"""Synthetic question-answer pair schemas.

This module defines data models for synthetically generated question-answer
pairs used for evaluation and testing of RAG systems.
"""

from pydantic import BaseModel, ConfigDict, Field


class SyntheticQA(BaseModel):
    """Synthetic question-answer pair generated from knowledge node content.

    Attributes:
        question: Generated question.
        answer: Expected answer based on chunk content.
        confidence: Quality score between 0 and 1.
    """

    model_config = ConfigDict(strict=True)

    question: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Generated question text",
    )
    answer: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Expected answer text",
    )
    confidence: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Quality score (0-1, optional)",
    )


class ChunkQAPairs(BaseModel):
    """QA pairs for a single chunk with chunk identifier.

    Attributes:
        chunk_id: Identifier for the chunk (1-indexed).
        qa_pairs: List of question-answer pairs for this chunk.
    """

    model_config = ConfigDict(strict=True)

    chunk_id: int = Field(
        ...,
        ge=1,
        description="1-indexed chunk identifier",
    )
    qa_pairs: list[SyntheticQA] = Field(
        ...,
        min_length=0,
        max_length=10,
        description="Generated QA pairs for this chunk",
    )


class BatchQAOutput(BaseModel):
    """Structured output for batch QA generation.

    This schema enforces the expected structure from LLM outputs,
    making parsing robust and eliminating fragile regex patterns.

    Attributes:
        chunks: List of chunks with their associated QA pairs.
    """

    model_config = ConfigDict(strict=True)

    chunks: list[ChunkQAPairs] = Field(
        ...,
        min_length=1,
        description="QA pairs organized by chunk",
    )
