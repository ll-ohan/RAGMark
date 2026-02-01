"""Retrieval data models.

Defines data structures for representing search results, retrieved nodes,
and retrieval trace contexts.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ragmark.schemas.documents import KnowledgeNode


class SearchResult(BaseModel):
    """Represents a single search result from a vector index.

    This minimal structure represents a match from the index,
    optionally including the full node data.

    Attributes:
        node_id: Unique identifier of the matched node.
        score: Similarity or relevance score.
        node: Full node data if requested.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    node_id: str = Field(..., description="Unique identifier of the matched node")
    score: float = Field(..., description="Similarity or relevance score")
    node: KnowledgeNode | None = Field(
        None,
        description="Full node data (optional, depends on include_vectors flag)",
    )


class RetrievedNode(BaseModel):
    """Represents a retrieved node with ranking information.

    Wraps a node with rank information for tracking position in
    retrieval results.

    Attributes:
        node: The retrieved node.
        score: Similarity or relevance score.
        rank: Position in the final ranked list (1-indexed).
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    node: KnowledgeNode = Field(..., description="The retrieved knowledge node")
    score: float = Field(
        ..., description="Similarity or relevance score (post-reranking)"
    )
    rank: int = Field(..., ge=1, description="Position in ranked results (1-indexed)")


class TraceContext(BaseModel):
    """Captures the complete trace of a retrieval operation.

    Captures all information about a retrieval operation, including the query,
    results, metadata about the retrieval process, and whether reranking
    was applied.

    Attributes:
        query: The original user query.
        retrieved_nodes: Retrieved and ranked nodes.
        retrieval_metadata: Metadata about the retrieval process.
        reranked: Indicates if reranking was applied.
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


class SearchCursor(BaseModel):
    """Cursor for paginating search results.

    Enables cursor-based pagination for large result sets, tracking
    current position and page size for iterative retrieval.

    Attributes:
        offset: Current position in result set (0-indexed).
        page_size: Number of results per page.
        total_results: Total number of results available.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    offset: int = Field(default=0, ge=0, description="Current offset in results")
    page_size: int = Field(default=10, ge=1, le=1000, description="Results per page")
    total_results: int | None = Field(
        default=None, ge=0, description="Total results available"
    )
