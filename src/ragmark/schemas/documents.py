"""Document and knowledge node data models.

This module defines the core data structures for representing source
documents and their fragmented knowledge nodes throughout the ingestion
and indexing pipeline.
"""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class SourceDoc(BaseModel):
    """Represents an ingested source document.

    This model captures the raw text content extracted from a source file,
    along with metadata that helps track provenance and support filtering.

    Attributes:
        content: Extracted text content from the document.
        metadata: Arbitrary metadata dictionary (title, author, etc.).
        source_id: Unique identifier for this source document.
        mime_type: MIME type of the source file.
        page_count: Number of pages if applicable (e.g., PDFs).
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    content: str = Field(..., description="Raw text content extracted from the source")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Custom metadata (title, author, creation_date, etc.)",
    )
    source_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier (UUID or SHA-256 hash of content)",
    )
    mime_type: str = Field(..., description="MIME type of the source file")
    page_count: int | None = Field(None, description="Number of pages in the document")

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        """Validate that content is not empty or whitespace-only.

        Args:
            v: Content string to validate.

        Returns:
            The validated content.

        Raises:
            ValueError: If content is empty or whitespace-only.
        """
        if not v or not v.strip():
            raise ValueError("Content cannot be empty or whitespace-only")
        return v


class NodePosition(BaseModel):
    """Position metadata for a knowledge node within its source document.

    This model tracks where a node's content originated within the source,
    enabling citation and context reconstruction.

    Attributes:
        start_char: Character offset where the node starts in the source.
        end_char: Character offset where the node ends in the source.
        page: Page number if applicable (1-indexed).
        section: Section identifier or header chain (e.g., "Chapter 1 > Section 1.2").
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    start_char: int = Field(
        ..., ge=0, description="Starting character offset in source"
    )
    end_char: int = Field(..., ge=0, description="Ending character offset in source")
    page: int | None = Field(None, ge=1, description="Page number (1-indexed)")
    section: str | None = Field(None, description="Section identifier or header chain")

    @field_validator("end_char")
    @classmethod
    def end_after_start(cls, v: int, info: ValidationInfo) -> int:
        """Validate that end_char is after start_char.

        Args:
            v: end_char value.
            info: Validation context containing other field values.

        Returns:
            The validated end_char.

        Raises:
            ValueError: If end_char <= start_char.
        """
        if "start_char" in info.data and v <= info.data["start_char"]:
            raise ValueError("end_char must be greater than start_char")
        return v


class KnowledgeNode(BaseModel):
    """A fragmented chunk of text ready for vector indexing.

    This is the fundamental unit of retrieval in RAGMark. Each node represents
    a semantically coherent chunk of text with optional pre-computed embeddings
    and rich metadata for filtering and analysis.

    Attributes:
        node_id: Unique identifier for this node (UUID v4).
        content: Text content of this chunk.
        source_id: Reference to the parent SourceDoc.
        metadata: Inherited metadata from source plus computed metadata.
        position: Position information within the source document.
        dense_vector: Optional pre-computed dense embedding.
        sparse_vector: Optional pre-computed sparse embedding (token_id -> weight).
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    node_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique node identifier (UUID v4)",
    )
    content: str = Field(..., description="Text content of this chunk")
    source_id: str = Field(..., description="Reference to parent SourceDoc")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata inherited from source plus computed metadata",
    )
    position: NodePosition = Field(..., description="Position within source document")
    dense_vector: list[float] | None = Field(
        None,
        description="Pre-computed dense embedding vector",
    )
    sparse_vector: dict[int, float] | None = Field(
        None,
        description="Pre-computed sparse embedding (token_id -> weight)",
    )

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        """Validate that content is not empty or whitespace-only.

        Args:
            v: Content string to validate.

        Returns:
            The validated content.

        Raises:
            ValueError: If content is empty or whitespace-only.
        """
        if not v or not v.strip():
            raise ValueError("Node content cannot be empty or whitespace-only")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Enrich metadata with computed fields after model initialization.

        This automatically adds char_count, word_count, and created_at
        to metadata if they don't already exist.
        """
        if "char_count" not in self.metadata:
            self.metadata["char_count"] = len(self.content)
        if "word_count" not in self.metadata:
            self.metadata["word_count"] = len(self.content.split())
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = datetime.now(timezone.utc).isoformat()


class VectorPayload(BaseModel):
    """Payload structure for vector database storage.

    This model represents how a KnowledgeNode is serialized for storage
    in vector databases, separating vector data from metadata.

    Attributes:
        node_id: Unique identifier matching the KnowledgeNode.
        dense_vector: Dense embedding vector.
        sparse_vector: Optional sparse embedding.
        content: Original text content.
        metadata: Full metadata dictionary.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    node_id: str = Field(..., description="Unique node identifier")
    dense_vector: list[float] = Field(..., description="Dense embedding vector")
    sparse_vector: dict[int, float] | None = Field(
        None,
        description="Optional sparse embedding",
    )
    content: str = Field(..., description="Original text content")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Full metadata dictionary",
    )
