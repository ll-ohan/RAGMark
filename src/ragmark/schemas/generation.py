"""Generation data models.

This module defines data structures for LLM generation results
and token usage tracking.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ragmark.schemas.retrieval import TraceContext


class TokenUsage(BaseModel):
    """Token usage statistics for an LLM generation.

    Attributes:
        prompt_tokens: Number of tokens in the input prompt.
        completion_tokens: Number of tokens in the generated completion.
        total_tokens: Total tokens used (prompt + completion).
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    prompt_tokens: int = Field(..., ge=0, description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., ge=0, description="Number of generated tokens")
    total_tokens: int = Field(
        ..., ge=0, description="Total tokens (prompt + completion)"
    )


class GenerationResult(BaseModel):
    """Result of an LLM generation operation.

    Captures the generated text along with metadata about token usage and the
    reason generation stopped.

    Attributes:
        text: Generated text completion.
        usage: Token usage statistics.
        finish_reason: Reason generation stopped.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    text: str = Field(..., description="Generated text completion")
    usage: TokenUsage = Field(..., description="Token usage statistics")
    finish_reason: Literal["stop", "length", "error"] = Field(
        ...,
        description="Reason generation stopped",
    )


class AnswerResult(BaseModel):
    """Result of RAG answer generation.

    Complete result from RAG pipeline including the generated answer,
    retrieval trace, generation metadata, and performance metrics.

    Attributes:
        answer: Generated answer text.
        trace: Retrieval trace with retrieved nodes.
        generation_result: LLM generation metadata.
        total_time_ms: End-to-end latency in milliseconds.
        sources: Optional source references.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    answer: str = Field(..., description="Generated answer text")
    trace: TraceContext = Field(..., description="Retrieval trace with retrieved nodes")
    generation_result: GenerationResult = Field(
        ..., description="LLM generation metadata"
    )
    total_time_ms: float = Field(
        ..., ge=0.0, description="End-to-end latency in milliseconds"
    )
    sources: list[str] | None = Field(None, description="Optional source references")
