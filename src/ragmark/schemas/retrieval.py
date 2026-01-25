"""Retrieval data models.

This module defines data structures for representing search results,
retrieved nodes, and retrieval trace contexts.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ragmark.schemas.documents import KnowledgeNode


class SearchResult(BaseModel):
    """A single search result from a vector index.

    This minimal structure represents a match from the index,
    optionally including the full node data.

    Attributes:
        node_id: Unique identifier of the matched node.
        score: Similarity or relevance score.
        node: Optional full KnowledgeNode (if include_vectors=True).
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    node_id: str = Field(..., description="Unique identifier of the matched node")
    score: float = Field(..., description="Similarity or relevance score")
    node: KnowledgeNode | None = Field(
        None,
        description="Full node data (optional, depends on include_vectors flag)",
    )


class RetrievedNode(BaseModel):
    """A retrieved node with its ranking information.

    This extends SearchResult with rank information for tracking
    position in the retrieval results.

    Attributes:
        node: The full KnowledgeNode that was retrieved.
        score: Similarity or relevance score (may be reranked).
        rank: Position in the final ranked list (1-indexed).
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    node: KnowledgeNode = Field(..., description="The retrieved knowledge node")
    score: float = Field(
        ..., description="Similarity or relevance score (post-reranking)"
    )
    rank: int = Field(..., ge=1, description="Position in ranked results (1-indexed)")


class TraceContext(BaseModel):
    """Complete trace of a retrieval operation.

    This model captures all information about a retrieval operation,
    including the query, results, metadata about the retrieval process,
    and whether reranking was applied.

    Attributes:
        query: The original user query string.
        retrieved_nodes: List of retrieved and ranked nodes.
        retrieval_metadata: Timing, strategy, and other process metadata.
        reranked: Whether reranking was applied to the results.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    query: str = Field(..., description="The original user query")
    retrieved_nodes: list[RetrievedNode] = Field(
        default_factory=list[RetrievedNode],
        description="Retrieved and ranked nodes",
    )
    retrieval_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the retrieval process (timing, strategy, etc.)",
    )
    reranked: bool = Field(False, description="Whether reranking was applied")
