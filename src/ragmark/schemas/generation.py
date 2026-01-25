"""Generation data models.

This module defines data structures for LLM generation results
and token usage tracking.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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

    This model captures the generated text along with metadata about
    token usage and the reason generation stopped.

    Attributes:
        text: The generated text completion.
        usage: Token usage statistics.
        finish_reason: Why generation stopped (stop, length, or error).
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    text: str = Field(..., description="Generated text completion")
    usage: TokenUsage = Field(..., description="Token usage statistics")
    finish_reason: Literal["stop", "length", "error"] = Field(
        ...,
        description="Reason generation stopped",
    )
