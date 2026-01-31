"""Unit tests for RAGMark data schemas (SourceDoc)."""

import json
import re
import unicodedata
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from ragmark.schemas.documents import (
    KnowledgeNode,
    NodePosition,
    SourceDoc,
    VectorPayload,
)
from ragmark.schemas.evaluation import AuditReport, CaseResult, SystemInfo, TrialCase
from ragmark.schemas.generation import GenerationResult, TokenUsage
from ragmark.schemas.retrieval import RetrievedNode, SearchResult, TraceContext


@pytest.mark.unit
class TestSourceDoc:
    """Test suite for the SourceDoc model."""

    def test_source_doc_should_generate_id_when_missing(
        self, doc_factory: Callable[..., SourceDoc]
    ) -> None:
        """
        Given: A SourceDoc initialized via factory without explicit source_id.
        When: The object is instantiated.
        Then: A valid source_id (SHA-256 hex) is automatically generated.
        """
        doc = doc_factory(source_id=None)

        assert len(doc.source_id) == 64
        assert re.fullmatch(r"^[a-f0-9]{64}$", doc.source_id) is not None

    def test_source_doc_should_produce_deterministic_ids_for_identical_content(
        self, doc_factory: Callable[..., SourceDoc]
    ) -> None:
        """
        Given: Two distinct SourceDocs with strictly identical content.
        When: The objects are instantiated independently via the factory.
        Then: Their source_ids must be identical (Deduplication).
        """

        content = "RAGMark Integrity Check: Consistency over Vanity."

        doc1 = doc_factory(content=content, source_id=None)
        doc2 = doc_factory(content=content, source_id=None)

        assert doc1.source_id == doc2.source_id
        assert len(doc1.source_id) == 64

    def test_source_doc_should_raise_error_when_content_is_empty(
        self, doc_factory: Callable[..., SourceDoc]
    ) -> None:
        """
        Given: Empty content or whitespace-only.
        When: The SourceDoc is instantiated.
        Then: An explicit ValidationError is raised.
        """

        invalid_content = "   "

        with pytest.raises(ValidationError) as exc_info:
            doc_factory(content=invalid_content)

        assert "Content cannot be empty" in str(exc_info.value)

    @pytest.mark.rag_edge_case
    def test_source_doc_should_preserve_complex_graphemes(
        self, doc_factory: Callable[..., SourceDoc]
    ) -> None:
        """
        Given: Content with complex ZWJ (Zero Width Joiner) emojis.
        When: The SourceDoc is processed.
        Then: The integrity of the character sequence is preserved.
        """

        complex_emoji = "👨‍👩‍👧‍👦"
        content = f"Emoji test: {complex_emoji}"

        doc = doc_factory(content=content)

        assert complex_emoji in doc.content
        assert doc.content == content

    @pytest.mark.unit
    @pytest.mark.rag_edge_case
    def test_source_doc_should_handle_mixed_unicode_normalization(
        self, doc_factory: Callable[..., SourceDoc]
    ) -> None:
        """
        Given: Text combining NFC (precomposed é) and NFD (e + ´ separated).
        When: The SourceDoc is created.
        Then: Both forms must be preserved without forced normalization.
        """

        nfc_text = "café"
        nfd_text = unicodedata.normalize("NFD", "café")
        content = f"{nfc_text} vs {nfd_text}"

        doc = doc_factory(content=content)

        assert doc.content == content
        assert nfc_text in doc.content
        assert nfd_text in doc.content

    @pytest.mark.unit
    @pytest.mark.rag_edge_case
    def test_source_doc_should_reject_bom_in_content(
        self, doc_factory: Callable[..., SourceDoc]
    ) -> None:
        """
        Given: Content starting with a UTF-8 BOM (Byte Order Mark).
        When: The SourceDoc is validated.
        Then: The BOM should not be rejected by schema (cleaning expected upstream).
        """

        content_with_bom = "\ufeffRAGMark test content"

        doc = doc_factory(content=content_with_bom)

        assert doc.content == content_with_bom

    @pytest.mark.unit
    @pytest.mark.rag_edge_case
    def test_source_doc_should_handle_ideographic_whitespace(
        self, doc_factory: Callable[..., SourceDoc]
    ) -> None:
        """
        Given: Content with ideographic spaces (U+3000).
        When: Non-empty content validation is performed.
        Then: Ideographic spaces must not be considered empty.
        """

        ideographic_space = "\u3000"
        content = f"Text{ideographic_space}with{ideographic_space}ideographic spaces"

        doc = doc_factory(content=content)

        assert ideographic_space in doc.content
        assert doc.content == content

    @pytest.mark.unit
    def test_source_doc_should_validate_metadata_types(self) -> None:
        """
        Given: A SourceDoc with varied metadata types.
        When: The document is created.
        Then: Metadata must be preserved with their correct types.
        """

        doc = SourceDoc(
            content="Test content",
            metadata={
                "title": "Test",
                "tags": ["rag", "test"],
                "confidence": 0.95,
                "is_valid": True,
            },
            mime_type="text/plain",
            page_count=5,
        )

        assert doc.metadata["title"] == "Test"
        assert doc.page_count == 5
        assert doc.metadata["tags"] == ["rag", "test"]
        assert doc.metadata["confidence"] == 0.95
        assert doc.metadata["is_valid"] is True

    @pytest.mark.unit
    def test_source_doc_should_forbid_extra_fields(self) -> None:
        """
        Given: An attempt to create a SourceDoc with extra fields.
        When: Pydantic validates the strict model.
        Then: A ValidationError must be raised.
        """
        invalid_data = {
            "content": "Test",
            "mime_type": "text/plain",
            "unexpected_field": "should fail",
        }

        with pytest.raises(ValidationError) as exc_info:
            SourceDoc(**invalid_data)  # type: ignore[arg-type]

        assert "Extra inputs are not permitted" in str(exc_info.value)


@pytest.mark.unit
class TestNodePosition:
    """Test suite for the NodePosition model."""

    def test_node_position_should_accept_valid_offsets(self) -> None:
        """
        Given: Valid character offsets (start < end).
        When: A NodePosition is created.
        Then: The object must be valid without error.
        """

        position = NodePosition(
            start_char=0,
            end_char=100,
            page=1,
            section="Introduction",
        )

        assert position.start_char == 0
        assert position.end_char == 100
        assert position.page == 1
        assert position.section == "Introduction"

    def test_node_position_should_reject_end_before_start(self) -> None:
        """
        Given: An end_char less than or equal to start_char.
        When: The NodePosition is validated.
        Then: An explicit ValidationError must be raised.
        """
        with pytest.raises(ValidationError) as exc_info:
            NodePosition(start_char=100, end_char=50, page=1, section="Intro")

        assert "end_char must be greater than start_char" in str(exc_info.value)

    def test_node_position_should_reject_equal_start_and_end(self) -> None:
        """
        Given: An end_char equal to start_char (empty node).
        When: The NodePosition is validated.
        Then: A ValidationError must be raised.
        """
        with pytest.raises(ValidationError) as exc_info:
            NodePosition(start_char=50, end_char=50, page=1, section="Intro")

        assert "end_char must be greater than start_char" in str(exc_info.value)

    def test_node_position_should_accept_optional_fields_as_none(self) -> None:
        """
        Given: A NodePosition without page or section.
        When: The object is created.
        Then: Optional fields must be None.
        """

        position = NodePosition(start_char=0, end_char=100, page=None, section=None)

        assert position.page is None
        assert position.section is None

    def test_node_position_should_reject_negative_start_char(self) -> None:
        """
        Given: A negative start_char.
        When: The NodePosition is validated.
        Then: A ValidationError must be raised.
        """

        with pytest.raises(ValidationError) as exc_info:
            NodePosition(start_char=-1, end_char=100, page=1, section="Intro")

        assert "greater than or equal to 0" in str(exc_info.value)

    def test_node_position_should_reject_zero_indexed_page(self) -> None:
        """
        Given: A page number of 0 (pages start at 1).
        When: The NodePosition is validated.
        Then: A ValidationError must be raised.
        """

        with pytest.raises(ValidationError) as exc_info:
            NodePosition(start_char=0, end_char=100, page=0, section="Intro")

        assert "greater than or equal to 1" in str(exc_info.value)


@pytest.mark.unit
class TestKnowledgeNode:
    """Test suite for the KnowledgeNode model."""

    def test_knowledge_node_should_auto_generate_node_id(self) -> None:
        """
        Given: A KnowledgeNode created without explicit node_id.
        When: The object is instantiated.
        Then: A valid UUID node_id must be generated automatically.
        """
        node = KnowledgeNode(
            content="Test content",
            source_id="src-001",
            position=NodePosition(start_char=0, end_char=12, page=1, section="Intro"),
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={1: 0.5, 5: 0.3, 10: 0.8},
        )

        assert len(node.node_id) == 36
        assert node.node_id.count("-") == 4

    def test_knowledge_node_should_enrich_metadata_on_init(self) -> None:
        """
        Given: A KnowledgeNode created without computed metadata.
        When: model_post_init is called automatically.
        Then: char_count, word_count, and created_at must be added.
        """

        node = KnowledgeNode(
            content="This is a test.",
            source_id="src-001",
            position=NodePosition(start_char=0, end_char=15, page=1, section="Intro"),
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={1: 0.5, 5: 0.3},
        )

        assert "char_count" in node.metadata
        assert node.metadata["char_count"] == 15
        assert "word_count" in node.metadata
        assert node.metadata["word_count"] == 4
        assert "created_at" in node.metadata
        created_at = datetime.fromisoformat(node.metadata["created_at"])
        assert created_at.tzinfo == timezone.utc

    def test_knowledge_node_should_not_override_existing_metadata(self) -> None:
        """
        Given: A KnowledgeNode with char_count already defined.
        When: model_post_init is executed.
        Then: The existing value must not be overwritten.
        """

        node = KnowledgeNode(
            content="Test",
            source_id="src-001",
            position=NodePosition(start_char=0, end_char=4, page=1, section="Intro"),
            metadata={"char_count": 999},
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={1: 0.5, 5: 0.3},
        )

        assert node.metadata["char_count"] == 999

    def test_knowledge_node_should_reject_empty_content(self) -> None:
        """
        Given: Empty content or whitespace-only.
        When: The KnowledgeNode is validated.
        Then: An explicit ValidationError must be raised.
        """

        with pytest.raises(ValidationError) as exc_info:
            KnowledgeNode(
                content="   ",
                source_id="src-001",
                position=NodePosition(
                    start_char=0, end_char=3, page=1, section="Intro"
                ),
                dense_vector=[0.1, 0.2, 0.3],
                sparse_vector={1: 0.5, 5: 0.3},
            )

        assert "Node content cannot be empty" in str(exc_info.value)

    def test_knowledge_node_should_accept_dense_vector(
        self, node_factory: Callable[..., KnowledgeNode]
    ) -> None:
        """
        Given: A KnowledgeNode with a pre-computed dense vector.
        When: The object is created.
        Then: The vector must be stored correctly.
        """

        dense_vec = [0.1, 0.2, 0.3]

        node = KnowledgeNode(
            content="Test content",
            source_id="src-001",
            position=NodePosition(start_char=0, end_char=12, page=1, section="Intro"),
            sparse_vector=None,
            dense_vector=dense_vec,
        )

        assert node.dense_vector == dense_vec
        assert node.dense_vector is not None
        assert len(node.dense_vector) == 3

    def test_knowledge_node_should_accept_sparse_vector(self) -> None:
        """
        Given: A KnowledgeNode with a sparse vector.
        When: The object is created.
        Then: The sparse_vector must be stored with the correct structure.
        """

        sparse_vec: dict[int, float] = {1: 0.5, 5: 0.3, 10: 0.8}

        node = KnowledgeNode(
            content="Test content",
            source_id="src-001",
            position=NodePosition(start_char=0, end_char=12, page=1, section="Intro"),
            sparse_vector=sparse_vec,
            dense_vector=None,
        )

        assert node.sparse_vector == sparse_vec

    @pytest.mark.rag_edge_case
    def test_knowledge_node_should_preserve_unicode_in_content(self) -> None:
        """
        Given: A KnowledgeNode with complex Unicode content.
        When: The object is created.
        Then: The content must be preserved exactly.
        """

        unicode_content = "日本語 テスト 🎌 émoji café"

        node = KnowledgeNode(
            content=unicode_content,
            source_id="src-001",
            position=NodePosition(
                start_char=0, end_char=len(unicode_content), page=1, section="Intro"
            ),
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={1: 0.5, 5: 0.3},
        )

        assert node.content == unicode_content
        assert "🎌" in node.content
        assert "日本語" in node.content

    def test_knowledge_node_should_forbid_extra_fields(self) -> None:
        """
        Given: An attempt to create a KnowledgeNode with an unknown field.
        When: Pydantic validates the strict model.
        Then: A ValidationError must be raised.
        """
        invalid_data = {
            "content": "Test",
            "source_id": "src-001",
            "position": NodePosition(start_char=0, end_char=4, page=1, section="Intro"),
            "dense_vector": [0.1, 0.2, 0.3],
            "sparse_vector": {1: 0.5, 5: 0.3},
            "unknown_field": "fail",
        }

        with pytest.raises(ValidationError) as exc_info:
            KnowledgeNode(**invalid_data)  # type: ignore[arg-type]

        assert "Extra inputs are not permitted" in str(exc_info.value)


@pytest.mark.unit
class TestVectorPayload:
    """Test suite for the VectorPayload model."""

    def test_vector_payload_should_accept_valid_data(self) -> None:
        """
        Given: Valid data for a VectorPayload.
        When: The object is created.
        Then: All properties must be correctly assigned.
        """

        dense_vec = [0.1, 0.2, 0.3, 0.4]
        sparse_vec: dict[int, float] = {1: 0.5, 5: 0.3}
        metadata: dict[str, Any] = {"source": "test", "page": 1}

        payload = VectorPayload(
            node_id="node-123",
            dense_vector=dense_vec,
            sparse_vector=sparse_vec,
            content="Test content",
            metadata=metadata,
        )

        assert payload.node_id == "node-123"
        assert payload.dense_vector == dense_vec
        assert payload.sparse_vector == sparse_vec
        assert payload.content == "Test content"
        assert payload.metadata == metadata

    def test_vector_payload_should_allow_none_sparse_vector(self) -> None:
        """
        Given: A VectorPayload without sparse_vector.
        When: The object is created.
        Then: sparse_vector must be None.
        """

        payload = VectorPayload(
            node_id="node-456",
            dense_vector=[0.1, 0.2],
            sparse_vector=None,
            content="Test",
        )

        assert payload.sparse_vector is None

    def test_vector_payload_should_require_dense_vector(self) -> None:
        """
        Given: An attempt to create a VectorPayload without dense_vector.
        When: Pydantic validates the model.
        Then: A ValidationError must be raised.
        """
        invalid_data = {
            "node_id": "node-789",
            "content": "Test",
        }

        with pytest.raises(ValidationError) as exc_info:
            VectorPayload(**invalid_data)  # type: ignore[arg-type]

        assert "dense_vector" in str(exc_info.value).lower()

    def test_vector_payload_should_use_empty_dict_for_default_metadata(self) -> None:
        """
        Given: A VectorPayload created without explicit metadata.
        When: The object is instantiated.
        Then: metadata must be an empty dictionary by default.
        """

        payload = VectorPayload(
            node_id="node-000",
            dense_vector=[0.5],
            content="Minimal payload",
            sparse_vector=None,
        )

        assert payload.metadata == {}

    def test_vector_payload_should_forbid_extra_fields(self) -> None:
        """
        Given: An attempt to create a VectorPayload with an unknown field.
        When: Pydantic validates the strict model.
        Then: A ValidationError must be raised.
        """
        invalid_data = {
            "node_id": "node-999",
            "dense_vector": [0.1],
            "content": "Test",
            "extra_field": "invalid",
        }

        with pytest.raises(ValidationError) as exc_info:
            VectorPayload(**invalid_data)  # type: ignore[arg-type]

        assert "Extra inputs are not permitted" in str(exc_info.value)


@pytest.mark.unit
class TestTokenUsage:
    """Test suite for the TokenUsage model."""

    def test_token_usage_should_accept_valid_token_counts(self) -> None:
        """Test TokenUsage with valid token counts.

        Given: Valid values for prompt, completion, and total tokens.
        When: A TokenUsage object is created.
        Then: All properties are correctly assigned.
        """

        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_token_usage_should_accept_zero_token_counts(self) -> None:
        """Test TokenUsage accepts zero tokens.

        Given: Token counters set to zero.
        When: A TokenUsage object is created.
        Then: Zero values are accepted (ge=0 constraint).
        """

        usage = TokenUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_token_usage_should_reject_negative_prompt_tokens(self) -> None:
        """Test TokenUsage rejects negative prompt_tokens.

        Given: A negative number for prompt_tokens.
        When: The TokenUsage is validated.
        Then: A ValidationError is raised with explicit message.
        """
        with pytest.raises(ValidationError) as exc_info:
            TokenUsage(
                prompt_tokens=-1,
                completion_tokens=50,
                total_tokens=49,
            )

        assert "greater than or equal to 0" in str(exc_info.value)

    def test_token_usage_should_reject_negative_completion_tokens(self) -> None:
        """Test TokenUsage rejects negative completion_tokens.

        Given: A negative number for completion_tokens.
        When: The TokenUsage is validated.
        Then: A ValidationError is raised.
        """

        with pytest.raises(ValidationError) as exc_info:
            TokenUsage(
                prompt_tokens=100,
                completion_tokens=-10,
                total_tokens=90,
            )

        assert "greater than or equal to 0" in str(exc_info.value)

    def test_token_usage_should_reject_negative_total_tokens(self) -> None:
        """Test TokenUsage rejects negative total_tokens.

        Given: A negative number for total_tokens.
        When: The TokenUsage is validated.
        Then: A ValidationError is raised.
        """

        with pytest.raises(ValidationError) as exc_info:
            TokenUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=-1,
            )

        assert "greater than or equal to 0" in str(exc_info.value)

    def test_token_usage_should_accept_large_token_counts(self) -> None:
        """Test TokenUsage with very large token counts.

        Given: Large token counts (e.g., for long context models).
        When: A TokenUsage object is created.
        Then: Large values are accepted without upper bound constraint.
        """

        usage = TokenUsage(
            prompt_tokens=100000,
            completion_tokens=50000,
            total_tokens=150000,
        )

        assert usage.prompt_tokens == 100000
        assert usage.completion_tokens == 50000
        assert usage.total_tokens == 150000

    def test_token_usage_should_forbid_extra_fields(self) -> None:
        """Test strict mode prevents unknown fields.

        Given: An attempt to create TokenUsage with extra field.
        When: Pydantic validates the strict model.
        Then: A ValidationError is raised.
        """
        invalid_data = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "extra_field": "invalid",
        }

        with pytest.raises(ValidationError) as exc_info:
            TokenUsage(**invalid_data)  # type: ignore[arg-type]

        assert "Extra inputs are not permitted" in str(exc_info.value)


@pytest.mark.unit
class TestGenerationResult:
    """Test suite for the GenerationResult model."""

    def test_generation_result_should_accept_stop_finish_reason(self) -> None:
        """Test GenerationResult with stop finish reason.

        Given: A GenerationResult with finish_reason='stop'.
        When: The object is created.
        Then: All properties are correctly assigned.
        """

        result = GenerationResult(
            text="Generated answer to the question.",
            usage=TokenUsage(
                prompt_tokens=50,
                completion_tokens=20,
                total_tokens=70,
            ),
            finish_reason="stop",
        )

        assert result.text == "Generated answer to the question."
        assert result.usage.prompt_tokens == 50
        assert result.usage.completion_tokens == 20
        assert result.usage.total_tokens == 70
        assert result.finish_reason == "stop"

    def test_generation_result_should_accept_length_finish_reason(self) -> None:
        """Test GenerationResult with length finish reason.

        Given: A GenerationResult with finish_reason='length' (truncated).
        When: The object is created.
        Then: The finish_reason is accepted.
        """

        result = GenerationResult(
            text="Truncated text...",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=100, total_tokens=110),
            finish_reason="length",
        )

        assert result.finish_reason == "length"

    def test_generation_result_should_accept_error_finish_reason(self) -> None:
        """Test GenerationResult with error finish reason.

        Given: A GenerationResult with finish_reason='error'.
        When: The object is created.
        Then: The finish_reason is accepted and empty text is valid.
        """

        result = GenerationResult(
            text="",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=0, total_tokens=10),
            finish_reason="error",
        )

        assert result.finish_reason == "error"
        assert result.text == ""

    def test_generation_result_should_reject_invalid_finish_reason(self) -> None:
        """Test GenerationResult rejects invalid finish reasons.

        Given: An invalid finish_reason (not in Literal type).
        When: The GenerationResult is validated.
        Then: A ValidationError is raised.
        """
        invalid_data = {
            "text": "Test",
            "usage": TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            "finish_reason": "timeout",
        }

        with pytest.raises(ValidationError) as exc_info:
            GenerationResult(**invalid_data)  # type: ignore[arg-type]

        assert "finish_reason" in str(exc_info.value).lower()

    def test_generation_result_should_accept_empty_text(self) -> None:
        """Test GenerationResult accepts empty text.

        Given: An empty text string (error or empty generation case).
        When: The GenerationResult is created.
        Then: Empty text is accepted without validation error.
        """

        result = GenerationResult(
            text="",
            usage=TokenUsage(prompt_tokens=50, completion_tokens=0, total_tokens=50),
            finish_reason="error",
        )

        assert result.text == ""
        assert result.usage.completion_tokens == 0

    def test_generation_result_should_accept_long_generated_text(self) -> None:
        """Test GenerationResult with very long generated text.

        Given: A GenerationResult with a long text output.
        When: The object is created.
        Then: Long text is accepted without length constraint.
        """

        long_text = "Generated response. " * 2000

        result = GenerationResult(
            text=long_text,
            usage=TokenUsage(
                prompt_tokens=100, completion_tokens=5000, total_tokens=5100
            ),
            finish_reason="stop",
        )

        assert len(result.text) > 20000
        assert result.text == long_text

    def test_generation_result_should_forbid_extra_fields(self) -> None:
        """Test strict mode prevents unknown fields.

        Given: An attempt to create GenerationResult with extra field.
        When: Pydantic validates the strict model.
        Then: A ValidationError is raised.
        """

        invalid_data = {
            "text": "Test",
            "usage": TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            "finish_reason": "stop",
            "model_name": "gpt-4",
        }

        with pytest.raises(ValidationError) as exc_info:
            GenerationResult(**invalid_data)  # type: ignore[arg-type]

        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_generation_result_should_require_all_fields(self) -> None:
        """Test GenerationResult requires all mandatory fields.

        Given: An attempt to create GenerationResult without required fields.
        When: Pydantic validates the model.
        Then: A ValidationError is raised mentioning missing fields.
        """

        invalid_data = {
            "text": "Test",
        }

        with pytest.raises(ValidationError) as exc_info:
            GenerationResult(**invalid_data)  # type: ignore[arg-type]

        error_str = str(exc_info.value).lower()
        assert "usage" in error_str
        assert "finish_reason" in error_str

    def test_generation_result_should_validate_literal_finish_reasons(self) -> None:
        """Test all valid Literal finish_reason values.

        Given: All three valid finish_reason values from the Literal type.
        When: GenerationResults are created with each value.
        Then: All values are accepted without error.
        """

        stop_result = GenerationResult(
            text="Complete",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            finish_reason="stop",
        )
        length_result = GenerationResult(
            text="Truncated",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=100, total_tokens=110),
            finish_reason="length",
        )
        error_result = GenerationResult(
            text="Error",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=1, total_tokens=11),
            finish_reason="error",
        )

        assert stop_result.finish_reason == "stop"
        assert length_result.finish_reason == "length"
        assert error_result.finish_reason == "error"


@pytest.mark.unit
class TestTrialCase:
    """Test suite for the TrialCase model."""

    def test_trial_case_should_accept_answer_ground_truth_only(self) -> None:
        """Test TrialCase with generation evaluation only.

        Given: A TrialCase with ground_truth_answer but no node IDs.
        When: The object is created.
        Then: Validation passes as at least one ground truth is provided.
        """

        case = TrialCase(
            question="What is RAG?",
            ground_truth_answer="RAG stands for Retrieval-Augmented Generation",
            ground_truth_node_ids=None,
        )

        assert case.question == "What is RAG?"
        assert (
            case.ground_truth_answer == "RAG stands for Retrieval-Augmented Generation"
        )
        assert case.ground_truth_node_ids is None

    def test_trial_case_should_accept_node_ids_ground_truth_only(self) -> None:
        """Test TrialCase with retrieval evaluation only.

        Given: A TrialCase with ground_truth_node_ids but no answer.
        When: The object is created.
        Then: Validation passes as at least one ground truth is provided.
        """

        case = TrialCase(
            question="Find documents about RAG",
            ground_truth_answer=None,
            ground_truth_node_ids=["node-1", "node-2", "node-3"],
        )

        assert case.question == "Find documents about RAG"
        assert case.ground_truth_node_ids == ["node-1", "node-2", "node-3"]
        assert case.ground_truth_answer is None

    def test_trial_case_should_accept_both_ground_truths(self) -> None:
        """Test TrialCase with both evaluation types.

        Given: A TrialCase with both answer and node ID ground truths.
        When: The object is created.
        Then: Both ground truths are stored correctly.
        """

        case = TrialCase(
            question="Comprehensive test",
            ground_truth_answer="Expected answer",
            ground_truth_node_ids=["node-1"],
        )

        assert case.ground_truth_answer == "Expected answer"
        assert case.ground_truth_node_ids == ["node-1"]

    def test_trial_case_should_reject_missing_both_ground_truths(self) -> None:
        """Test TrialCase validation requires at least one ground truth.

        Given: A TrialCase without any ground truth.
        When: The model_post_init validation runs.
        Then: A ValueError is raised with explicit message.
        """

        with pytest.raises(ValueError) as exc_info:
            TrialCase(
                question="No ground truth provided",
                ground_truth_node_ids=None,
                ground_truth_answer=None,
            )

        assert "At least one of ground_truth_answer or ground_truth_node_ids" in str(
            exc_info.value
        )

    def test_trial_case_should_reject_empty_question(self) -> None:
        """Test TrialCase rejects empty questions.

        Given: An empty or whitespace-only question string.
        When: The field validator runs.
        Then: A ValueError is raised.
        """

        with pytest.raises(ValueError) as exc_info:
            TrialCase(
                question="   ",
                ground_truth_answer="Answer",
                ground_truth_node_ids=["node-1"],
            )

        assert "Question cannot be empty" in str(exc_info.value)

    def test_trial_case_should_auto_generate_case_id(self) -> None:
        """Test TrialCase generates UUID when case_id not provided.

        Given: A TrialCase created without explicit case_id.
        When: The object is instantiated.
        Then: A valid UUID v4 is generated automatically.
        """

        case = TrialCase(
            question="Test",
            ground_truth_answer="Answer",
            ground_truth_node_ids=["node-1"],
        )

        assert len(case.case_id) == 36
        assert case.case_id.count("-") == 4

    def test_trial_case_should_accept_custom_case_id(self) -> None:
        """Test TrialCase with user-provided case_id.

        Given: A TrialCase with explicit case_id.
        When: The object is created.
        Then: The custom case_id is preserved.
        """

        case = TrialCase(
            case_id="custom-case-001",
            question="Test",
            ground_truth_answer="Answer",
            ground_truth_node_ids=["node-1"],
        )

        assert case.case_id == "custom-case-001"

    def test_trial_case_should_store_metadata(self) -> None:
        """Test TrialCase metadata storage.

        Given: A TrialCase with custom metadata.
        When: The object is created.
        Then: Metadata is stored with correct types.
        """

        case = TrialCase(
            question="Test",
            ground_truth_answer="Answer",
            ground_truth_node_ids=["node-1"],
            metadata={"difficulty": "hard", "category": "technical", "score": 0.95},
        )

        assert case.metadata["difficulty"] == "hard"
        assert case.metadata["category"] == "technical"
        assert case.metadata["score"] == 0.95

    def test_trial_case_should_load_from_jsonl_file(self, tmp_path: Path) -> None:
        """Test loading multiple TrialCases from JSONL file.

        Given: A JSONL file with multiple trial cases (one per line).
        When: TrialCase.load_cases() is called.
        Then: All cases are loaded correctly.
        """

        jsonl_file = tmp_path / "cases.jsonl"
        jsonl_content = """{"question": "Q1", "ground_truth_answer": "A1"}
{"question": "Q2", "ground_truth_node_ids": ["n1", "n2"]}
{"question": "Q3", "ground_truth_answer": "A3", "metadata": {"difficulty": "easy"}}"""
        jsonl_file.write_text(jsonl_content, encoding="utf-8")

        cases = TrialCase.load_cases(jsonl_file)

        assert len(cases) == 3
        assert cases[0].question == "Q1"
        assert cases[0].ground_truth_answer == "A1"
        assert cases[1].question == "Q2"
        assert cases[1].ground_truth_node_ids == ["n1", "n2"]
        assert cases[2].metadata["difficulty"] == "easy"

    def test_trial_case_should_load_from_json_array_file(self, tmp_path: Path) -> None:
        """Test loading TrialCases from JSON array file.

        Given: A JSON file containing an array of trial cases.
        When: TrialCase.load_cases() is called.
        Then: All cases in the array are loaded.
        """

        json_file = tmp_path / "cases.json"
        json_content: list[dict[str, Any]] = [
            {"question": "Q1", "ground_truth_answer": "A1"},
            {"question": "Q2", "ground_truth_node_ids": ["n1"]},
        ]
        json_file.write_text(json.dumps(json_content), encoding="utf-8")

        cases = TrialCase.load_cases(json_file)

        assert len(cases) == 2
        assert cases[0].question == "Q1"
        assert cases[1].question == "Q2"

    def test_trial_case_should_load_from_json_single_object(
        self, tmp_path: Path
    ) -> None:
        """Test loading single TrialCase from JSON object.

        Given: A JSON file with a single trial case object (not array).
        When: TrialCase.load_cases() is called.
        Then: A list with one case is returned.
        """

        json_file = tmp_path / "single_case.json"
        json_content = {
            "question": "Solo question",
            "ground_truth_answer": "Solo answer",
        }
        json_file.write_text(json.dumps(json_content), encoding="utf-8")

        cases = TrialCase.load_cases(json_file)

        assert len(cases) == 1
        assert cases[0].question == "Solo question"

    def test_trial_case_should_skip_empty_lines_in_jsonl(self, tmp_path: Path) -> None:
        """Test JSONL loading ignores blank lines.

        Given: A JSONL file with empty/whitespace lines.
        When: TrialCase.load_cases() is called.
        Then: Empty lines are skipped, only valid cases loaded.
        """

        jsonl_file = tmp_path / "cases_with_blanks.jsonl"
        content = """{"question": "Q1", "ground_truth_answer": "A1"}

{"question": "Q2", "ground_truth_answer": "A2"}

"""
        jsonl_file.write_text(content, encoding="utf-8")

        cases = TrialCase.load_cases(jsonl_file)

        assert len(cases) == 2

    def test_trial_case_load_should_raise_on_missing_file(self, tmp_path: Path) -> None:
        """Test load_cases raises FileNotFoundError for missing files.

        Given: A path to a non-existent file.
        When: TrialCase.load_cases() is called.
        Then: FileNotFoundError is raised with the path in message.
        """

        missing_file = tmp_path / "does_not_exist.json"

        with pytest.raises(FileNotFoundError) as exc_info:
            TrialCase.load_cases(missing_file)

        assert "Trial cases file not found" in str(exc_info.value)
        assert str(missing_file) in str(exc_info.value)

    def test_trial_case_load_should_raise_on_unsupported_format(
        self, tmp_path: Path
    ) -> None:
        """Test load_cases rejects unsupported file formats.

        Given: A file with unsupported extension (.txt, .csv, etc.).
        When: TrialCase.load_cases() is called.
        Then: ValueError is raised with the file suffix.
        """

        txt_file = tmp_path / "cases.txt"
        txt_file.write_text("Some content", encoding="utf-8")

        with pytest.raises(ValueError) as exc_info:
            TrialCase.load_cases(txt_file)

        assert "Unsupported file format" in str(exc_info.value)
        assert ".txt" in str(exc_info.value)

    def test_trial_case_should_forbid_extra_fields(self) -> None:
        """Test strict mode prevents unknown fields.

        Given: An attempt to create TrialCase with extra field.
        When: Pydantic validates the strict model.
        Then: A ValidationError is raised.
        """

        invalid_data = {
            "question": "Test",
            "ground_truth_answer": "Answer",
            "unknown_field": "invalid",
        }

        with pytest.raises(ValidationError) as exc_info:
            TrialCase(**invalid_data)  # type: ignore[arg-type]

        assert "Extra inputs are not permitted" in str(exc_info.value)


@pytest.mark.unit
class TestCaseResult:
    """Test suite for the CaseResult model."""

    def test_case_result_should_accept_complete_data(
        self, trace_factory: Callable[..., TraceContext]
    ) -> None:
        """Test CaseResult with all fields populated.

        Given: A CaseResult with answer, trace, generation result, and metrics.
        When: The object is created.
        Then: All properties are correctly assigned.
        """

        trace = trace_factory(query="Test query")
        gen_result = GenerationResult(
            text="Generated answer",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            finish_reason="stop",
        )
        metrics = {"recall@5": 0.8, "faithfulness": 0.95}

        result = CaseResult(
            case_id="case-001",
            predicted_answer="Generated answer",
            trace=trace,
            generation_result=gen_result,
            case_metrics=metrics,
        )

        assert result.case_id == "case-001"
        assert result.predicted_answer == "Generated answer"
        assert result.trace.query == "Test query"
        assert result.generation_result is not None
        assert result.generation_result.text == "Generated answer"
        assert result.case_metrics["recall@5"] == 0.8
        assert result.case_metrics["faithfulness"] == 0.95

    def test_case_result_should_accept_none_predicted_answer(
        self, trace_factory: Callable[..., TraceContext]
    ) -> None:
        """Test CaseResult when generation was not performed.

        Given: A CaseResult without predicted answer (retrieval-only eval).
        When: The object is created.
        Then: predicted_answer is None.
        """

        result = CaseResult(
            case_id="case-002",
            trace=trace_factory(),
            predicted_answer=None,
            generation_result=None,
        )

        assert result.predicted_answer is None
        assert result.generation_result is None

    def test_case_result_should_use_default_empty_metrics(
        self, trace_factory: Callable[..., TraceContext]
    ) -> None:
        """Test CaseResult with default empty metrics.

        Given: A CaseResult without case_metrics specified.
        When: The object is created.
        Then: case_metrics defaults to empty dict.
        """

        result = CaseResult(
            case_id="case-003",
            trace=trace_factory(),
            predicted_answer="Some answer",
            generation_result=GenerationResult(
                text="",
                usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                finish_reason="error",
            ),
        )

        assert result.case_metrics == {}

    def test_case_result_should_forbid_extra_fields(
        self, trace_factory: Callable[..., TraceContext]
    ) -> None:
        """Test strict mode prevents unknown fields.

        Given: An attempt to create CaseResult with extra field.
        When: Pydantic validates the strict model.
        Then: A ValidationError is raised.
        """

        invalid_data = {
            "case_id": "case-999",
            "trace": trace_factory(),
            "predicted_answer": "Some answer",
            "generation_result": GenerationResult(
                text="",
                usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                finish_reason="error",
            ),
            "extra": "invalid",
        }

        with pytest.raises(ValidationError) as exc_info:
            CaseResult(**invalid_data)  # type: ignore[arg-type]

        assert "Extra inputs are not permitted" in str(exc_info.value)


@pytest.mark.unit
class TestSystemInfo:
    """Test suite for the SystemInfo model."""

    def test_system_info_should_accept_valid_configuration(self) -> None:
        """Test SystemInfo with realistic system details.

        Given: SystemInfo with valid platform, CPU, RAM, and GPU data.
        When: The object is created.
        Then: All properties are correctly assigned.
        """

        info = SystemInfo(
            python_version="3.10.12",
            ragmark_version="0.1.0",
            platform="Linux-x86_64",
            cpu_count=8,
            ram_gb=16.0,
            gpu_info="NVIDIA RTX 3090",
        )

        assert info.python_version == "3.10.12"
        assert info.ragmark_version == "0.1.0"
        assert info.platform == "Linux-x86_64"
        assert info.cpu_count == 8
        assert info.ram_gb == 16.0
        assert info.gpu_info == "NVIDIA RTX 3090"

    def test_system_info_should_accept_none_gpu_info(self) -> None:
        """Test SystemInfo without GPU information.

        Given: SystemInfo for a CPU-only system (no GPU).
        When: The object is created with gpu_info=None.
        Then: gpu_info is None.
        """

        info = SystemInfo(
            python_version="3.11.0",
            ragmark_version="0.1.0",
            platform="Darwin-arm64",
            cpu_count=10,
            ram_gb=32.0,
            gpu_info=None,
        )

        assert info.gpu_info is None

    def test_system_info_should_reject_zero_cpu_count(self) -> None:
        """Test SystemInfo rejects invalid CPU count.

        Given: A cpu_count of 0 (invalid, ge=1 constraint).
        When: The SystemInfo is validated.
        Then: A ValidationError is raised.
        """

        with pytest.raises(ValidationError) as exc_info:
            SystemInfo(
                python_version="3.10.0",
                ragmark_version="0.1.0",
                platform="Test",
                cpu_count=0,
                ram_gb=8.0,
                gpu_info="NVIDIA RTX 3090",
            )

        assert "greater than or equal to 1" in str(exc_info.value)

    def test_system_info_should_reject_zero_ram(self) -> None:
        """Test SystemInfo rejects zero RAM.

        Given: A ram_gb of 0 (invalid, gt=0 constraint).
        When: The SystemInfo is validated.
        Then: A ValidationError is raised.
        """

        with pytest.raises(ValidationError) as exc_info:
            SystemInfo(
                python_version="3.10.0",
                ragmark_version="0.1.0",
                platform="Test",
                cpu_count=4,
                ram_gb=0.0,
                gpu_info="NVIDIA RTX 3090",
            )

        assert "greater than 0" in str(exc_info.value)

    def test_system_info_should_reject_negative_ram(self) -> None:
        """Test SystemInfo rejects negative RAM.

        Given: A negative ram_gb value.
        When: The SystemInfo is validated.
        Then: A ValidationError is raised.
        """

        with pytest.raises(ValidationError) as exc_info:
            SystemInfo(
                python_version="3.10.0",
                ragmark_version="0.1.0",
                platform="Test",
                cpu_count=4,
                ram_gb=-1.0,
                gpu_info="NVIDIA RTX 3090",
            )

        assert "greater than 0" in str(exc_info.value)

    def test_system_info_should_forbid_extra_fields(self) -> None:
        """Test strict mode prevents unknown fields.

        Given: An attempt to create SystemInfo with extra field.
        When: Pydantic validates the strict model.
        Then: A ValidationError is raised.
        """

        invalid_data = {
            "python_version": "3.10.0",
            "ragmark_version": "0.1.0",
            "platform": "Test",
            "cpu_count": 4,
            "ram_gb": 8.0,
            "disk_gb": 500,
        }

        with pytest.raises(ValidationError) as exc_info:
            SystemInfo(**invalid_data)  # type: ignore[arg-type]

        assert "Extra inputs are not permitted" in str(exc_info.value)


@pytest.mark.unit
class TestAuditReport:
    """Test suite for the AuditReport model."""

    def test_audit_report_should_accept_complete_benchmark_data(
        self, case_result_factory: Callable[..., CaseResult]
    ) -> None:
        """Test AuditReport with full benchmark results.

        Given: An AuditReport with metrics, case results, and system info.
        When: The object is created.
        Then: All properties are correctly assigned.
        """

        system_info = SystemInfo(
            python_version="3.10.0",
            ragmark_version="0.1.0",
            platform="Test",
            cpu_count=4,
            ram_gb=8.0,
            gpu_info="NVIDIA RTX 3090",
        )
        case_results = [
            case_result_factory(case_id="case-1"),
            case_result_factory(case_id="case-2"),
        ]
        metrics = {"recall@5": 0.85, "mrr": 0.72, "faithfulness": 0.90}

        report = AuditReport(
            experiment_profile_hash="abc123def456",
            duration_seconds=120.5,
            metrics=metrics,
            per_case_results=case_results,
            system_info=system_info,
        )

        assert report.experiment_profile_hash == "abc123def456"
        assert report.duration_seconds == 120.5
        assert report.metrics["recall@5"] == 0.85
        assert len(report.per_case_results) == 2
        assert report.system_info.cpu_count == 4

    def test_audit_report_should_auto_generate_report_id(self) -> None:
        """Test AuditReport generates UUID for report_id.

        Given: An AuditReport created without explicit report_id.
        When: The object is instantiated.
        Then: A valid UUID v4 is generated.
        """

        report = AuditReport(
            experiment_profile_hash="hash123",
            duration_seconds=60.0,
            system_info=SystemInfo(
                python_version="3.10.0",
                ragmark_version="0.1.0",
                platform="Test",
                cpu_count=2,
                ram_gb=4.0,
                gpu_info="NVIDIA RTX 3090",
            ),
        )

        assert len(report.report_id) == 36
        assert report.report_id.count("-") == 4

    def test_audit_report_should_auto_generate_creation_timestamp(self) -> None:
        """Test AuditReport generates creation timestamp.

        Given: An AuditReport created without explicit created_at.
        When: The object is instantiated.
        Then: A UTC timestamp is generated automatically.
        """

        before = datetime.now(timezone.utc)
        report = AuditReport(
            experiment_profile_hash="hash123",
            duration_seconds=60.0,
            system_info=SystemInfo(
                python_version="3.10.0",
                ragmark_version="0.1.0",
                platform="Test",
                cpu_count=2,
                ram_gb=4.0,
                gpu_info="NVIDIA RTX 3090",
            ),
        )
        after = datetime.now(timezone.utc)

        assert isinstance(report.created_at, datetime)
        assert report.created_at.tzinfo == timezone.utc
        assert before <= report.created_at <= after

    def test_audit_report_should_reject_negative_duration(self) -> None:
        """Test AuditReport rejects negative duration.

        Given: A negative duration_seconds value.
        When: The AuditReport is validated.
        Then: A ValidationError is raised.
        """

        with pytest.raises(ValidationError) as exc_info:
            AuditReport(
                experiment_profile_hash="hash",
                duration_seconds=-10.0,
                system_info=SystemInfo(
                    python_version="3.10.0",
                    ragmark_version="0.1.0",
                    platform="Test",
                    cpu_count=2,
                    ram_gb=4.0,
                    gpu_info="NVIDIA RTX 3090",
                ),
            )

        assert "greater than or equal to 0" in str(exc_info.value)

    def test_audit_report_should_export_to_json_file(self, tmp_path: Path) -> None:
        """Test AuditReport.to_json() exports valid JSON.

        Given: An AuditReport with data.
        When: to_json() is called with a file path.
        Then: A valid JSON file is created with correct content.
        """

        report = AuditReport(
            experiment_profile_hash="hash123",
            duration_seconds=100.0,
            metrics={"recall@5": 0.9},
            system_info=SystemInfo(
                python_version="3.10.0",
                ragmark_version="0.1.0",
                platform="Test",
                cpu_count=4,
                ram_gb=8.0,
                gpu_info="NVIDIA RTX 3090",
            ),
        )
        json_file = tmp_path / "report.json"

        report.to_json(json_file, indent=2)

        assert json_file.exists()
        loaded = json.loads(json_file.read_text(encoding="utf-8"))
        assert loaded["experiment_profile_hash"] == "hash123"
        assert loaded["duration_seconds"] == 100.0
        assert loaded["metrics"]["recall@5"] == 0.9

    def test_audit_report_should_convert_to_dataframe(
        self, node_factory: Callable[..., KnowledgeNode]
    ) -> None:
        """Test AuditReport.to_dataframe() creates valid DataFrame.

        Given: An AuditReport with multiple case results.
        When: to_dataframe() is called.
        Then: A pandas DataFrame with one row per case is returned.
        """

        trace1 = TraceContext(
            query="Query 1",
            retrieved_nodes=[RetrievedNode(node=node_factory(), score=0.9, rank=1)],
            reranked=True,
        )
        trace2 = TraceContext(
            query="Query 2",
            retrieved_nodes=[
                RetrievedNode(node=node_factory(), score=0.8, rank=1),
                RetrievedNode(node=node_factory(), score=0.7, rank=2),
            ],
            reranked=False,
        )
        case_results = [
            CaseResult(
                case_id="case-1",
                predicted_answer="Answer 1",
                trace=trace1,
                case_metrics={"recall@5": 1.0},
                generation_result=GenerationResult(
                    text="Answer 1",
                    usage=TokenUsage(
                        prompt_tokens=100, completion_tokens=50, total_tokens=150
                    ),
                    finish_reason="stop",
                ),
            ),
            CaseResult(
                case_id="case-2",
                predicted_answer="Answer 2",
                trace=trace2,
                generation_result=GenerationResult(
                    text="Answer 2",
                    usage=TokenUsage(
                        prompt_tokens=50, completion_tokens=20, total_tokens=70
                    ),
                    finish_reason="stop",
                ),
                case_metrics={"recall@5": 0.8, "latency": 120},
            ),
        ]
        report = AuditReport(
            experiment_profile_hash="hash",
            duration_seconds=200.0,
            per_case_results=case_results,
            system_info=SystemInfo(
                python_version="3.10.0",
                ragmark_version="0.1.0",
                platform="Test",
                cpu_count=2,
                ram_gb=4.0,
                gpu_info="NVIDIA RTX 3090",
            ),
        )

        df = report.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "case_id" in df.columns
        assert "predicted_answer" in df.columns
        assert "query" in df.columns
        assert "num_retrieved" in df.columns
        assert "reranked" in df.columns
        assert "recall@5" in df.columns
        assert df.iloc[0]["case_id"] == "case-1"
        assert df.iloc[0]["num_retrieved"] == 1
        assert df.iloc[1]["case_id"] == "case-2"
        assert df.iloc[1]["num_retrieved"] == 2
        assert "prompt_tokens" in df.columns
        assert df.iloc[1]["prompt_tokens"] == 50

    def test_audit_report_should_forbid_extra_fields(self) -> None:
        """Test strict mode prevents unknown fields.

        Given: An attempt to create AuditReport with extra field.
        When: Pydantic validates the strict model.
        Then: A ValidationError is raised.
        """

        invalid_data = {
            "experiment_profile_hash": "hash",
            "duration_seconds": 100.0,
            "system_info": SystemInfo(
                python_version="3.10.0",
                ragmark_version="0.1.0",
                platform="Test",
                cpu_count=2,
                ram_gb=4.0,
                gpu_info="NVIDIA RTX 3090",
            ),
            "extra_field": "invalid",
        }

        with pytest.raises(ValidationError) as exc_info:
            AuditReport(**invalid_data)  # type: ignore[arg-type]

        assert "Extra inputs are not permitted" in str(exc_info.value)


@pytest.mark.unit
class TestSearchResult:
    """Test suite for the SearchResult model."""

    def test_search_result_should_accept_minimal_data_without_node(self) -> None:
        """Test SearchResult creation with only required fields.

        Given: A SearchResult without the optional full node.
        When: The object is instantiated.
        Then: Required properties are assigned and node is None.
        """

        result = SearchResult(
            node_id="node-123",
            score=0.85,
            node=None,
        )

        assert result.node_id == "node-123"
        assert result.score == 0.85
        assert result.node is None

    def test_search_result_should_store_full_knowledge_node_when_provided(
        self, node_factory: Callable[..., KnowledgeNode]
    ) -> None:
        """Test SearchResult with complete KnowledgeNode.

        Given: A SearchResult with the full KnowledgeNode included.
        When: The object is created.
        Then: The node is stored correctly and accessible.
        """

        node = node_factory(content="Test content for retrieval")

        result = SearchResult(
            node_id=node.node_id,
            score=0.92,
            node=node,
        )

        assert result.node is not None
        assert result.node.content == "Test content for retrieval"
        assert result.node_id == node.node_id

    def test_search_result_should_accept_negative_scores(self) -> None:
        """Test SearchResult with negative similarity scores.

        Given: A negative score value (valid for some distance metrics).
        When: The SearchResult is created.
        Then: The negative score is accepted without validation error.
        """

        result = SearchResult(node_id="n1", score=-0.5, node=None)

        assert result.score == -0.5

    def test_search_result_should_accept_scores_above_one(self) -> None:
        """Test SearchResult with scores greater than 1.0.

        Given: A score value above 1.0 (valid for some metrics).
        When: The SearchResult is created.
        Then: The score is accepted without constraint violation.
        """

        result = SearchResult(node_id="n1", score=1.5, node=None)

        assert result.score == 1.5

    def test_search_result_should_forbid_extra_fields(self) -> None:
        """Test strict mode prevents unknown fields.

        Given: An attempt to create SearchResult with an extra field.
        When: Pydantic validates the strict model.
        Then: A ValidationError is raised with explicit message.
        """

        invalid_data = {
            "node_id": "node-999",
            "score": 0.8,
            "extra_field": "invalid",
        }

        with pytest.raises(ValidationError) as exc_info:
            SearchResult(**invalid_data)  # type: ignore[arg-type]

        assert "Extra inputs are not permitted" in str(exc_info.value)


@pytest.mark.unit
class TestRetrievedNode:
    """Test suite for the RetrievedNode model."""

    def test_retrieved_node_should_accept_valid_ranking_data(
        self, node_factory: Callable[..., KnowledgeNode]
    ) -> None:
        """Test RetrievedNode with all valid properties.

        Given: A RetrievedNode with valid node, score, and rank.
        When: The object is instantiated.
        Then: All properties are correctly assigned.
        """

        node = node_factory(content="Retrieved content")

        retrieved = RetrievedNode(
            node=node,
            score=0.95,
            rank=1,
        )

        assert retrieved.node.content == "Retrieved content"
        assert retrieved.score == 0.95
        assert retrieved.rank == 1

    def test_retrieved_node_should_accept_rank_one_as_minimum(
        self, node_factory: Callable[..., KnowledgeNode]
    ) -> None:
        """Test RetrievedNode with minimum allowed rank.

        Given: A rank of 1 (minimum value, ge=1 constraint).
        When: The RetrievedNode is created.
        Then: The rank is accepted without error.
        """

        retrieved = RetrievedNode(
            node=node_factory(),
            score=1.0,
            rank=1,
        )

        assert retrieved.rank == 1

    def test_retrieved_node_should_reject_rank_zero(
        self, node_factory: Callable[..., KnowledgeNode]
    ) -> None:
        """Test RetrievedNode rejects zero rank.

        Given: A rank of 0 (invalid, ranking is 1-indexed).
        When: The RetrievedNode is validated.
        Then: A ValidationError is raised with clear message.
        """

        with pytest.raises(ValidationError) as exc_info:
            RetrievedNode(
                node=node_factory(),
                score=0.9,
                rank=0,
            )

        assert "greater than or equal to 1" in str(exc_info.value)

    def test_retrieved_node_should_reject_negative_rank(
        self, node_factory: Callable[..., KnowledgeNode]
    ) -> None:
        """Test RetrievedNode rejects negative ranks.

        Given: A negative rank value.
        When: The RetrievedNode is validated.
        Then: A ValidationError is raised.
        """

        with pytest.raises(ValidationError) as exc_info:
            RetrievedNode(
                node=node_factory(),
                score=0.9,
                rank=-1,
            )

        assert "greater than or equal to 1" in str(exc_info.value)

    def test_retrieved_node_should_accept_high_rank_values(
        self, node_factory: Callable[..., KnowledgeNode]
    ) -> None:
        """Test RetrievedNode with large rank values.

        Given: A high rank value (e.g., result #1000).
        When: The RetrievedNode is created.
        Then: The rank is accepted without upper bound constraint.
        """

        retrieved = RetrievedNode(
            node=node_factory(),
            score=0.1,
            rank=1000,
        )

        assert retrieved.rank == 1000

    def test_retrieved_node_should_forbid_extra_fields(
        self, node_factory: Callable[..., KnowledgeNode]
    ) -> None:
        """Test strict mode prevents unknown fields.

        Given: An attempt to create RetrievedNode with extra field.
        When: Pydantic validates the strict model.
        Then: A ValidationError is raised.
        """

        invalid_data = {
            "node": node_factory(),
            "score": 0.9,
            "rank": 1,
            "position": "first",
        }

        with pytest.raises(ValidationError) as exc_info:
            RetrievedNode(**invalid_data)  # type: ignore[arg-type]

        assert "Extra inputs are not permitted" in str(exc_info.value)


@pytest.mark.unit
class TestTraceContext:
    """Test suite for the TraceContext model."""

    def test_trace_context_should_use_defaults_for_optional_fields(self) -> None:
        """Test TraceContext with minimal required data.

        Given: A TraceContext with only the query (optional fields omitted).
        When: The object is created.
        Then: Default values are applied correctly.
        """

        trace = TraceContext(
            query="What is RAGMark?",
            reranked=False,
        )

        assert trace.query == "What is RAGMark?"
        assert trace.retrieved_nodes == []
        assert trace.retrieval_metadata == {}
        assert trace.reranked is False

    def test_trace_context_should_store_complete_retrieval_data(
        self, node_factory: Callable[..., KnowledgeNode]
    ) -> None:
        """Test TraceContext with full retrieval information.

        Given: A TraceContext with nodes, metadata, and reranking flag.
        When: The object is instantiated.
        Then: All properties are correctly assigned.
        """

        retrieved_nodes = [
            RetrievedNode(node=node_factory(content="First"), score=0.95, rank=1),
            RetrievedNode(node=node_factory(content="Second"), score=0.85, rank=2),
        ]
        metadata: dict[str, Any] = {"strategy": "hybrid", "latency_ms": 150}

        trace = TraceContext(
            query="Test query",
            retrieved_nodes=retrieved_nodes,
            retrieval_metadata=metadata,
            reranked=True,
        )

        assert trace.query == "Test query"
        assert len(trace.retrieved_nodes) == 2
        assert trace.retrieved_nodes[0].score == 0.95
        assert trace.retrieved_nodes[1].score == 0.85
        assert trace.retrieval_metadata["strategy"] == "hybrid"
        assert trace.retrieval_metadata["latency_ms"] == 150
        assert trace.reranked is True

    def test_trace_context_should_preserve_node_order(
        self, node_factory: Callable[..., KnowledgeNode]
    ) -> None:
        """Test TraceContext preserves retrieval order.

        Given: Retrieved nodes in a specific ranking order.
        When: The TraceContext is created.
        Then: The order of nodes is preserved exactly.
        """

        nodes = [
            RetrievedNode(node=node_factory(content="A"), score=0.9, rank=1),
            RetrievedNode(node=node_factory(content="B"), score=0.8, rank=2),
            RetrievedNode(node=node_factory(content="C"), score=0.7, rank=3),
        ]

        trace = TraceContext(query="Test", retrieved_nodes=nodes, reranked=False)

        assert trace.retrieved_nodes[0].node.content == "A"
        assert trace.retrieved_nodes[0].rank == 1
        assert trace.retrieved_nodes[1].node.content == "B"
        assert trace.retrieved_nodes[1].rank == 2
        assert trace.retrieved_nodes[2].node.content == "C"
        assert trace.retrieved_nodes[2].rank == 3

    def test_trace_context_should_accept_empty_query(self) -> None:
        """Test TraceContext with empty query string.

        Given: An empty query string (edge case).
        When: The TraceContext is created.
        Then: The empty query is accepted without validation error.
        """

        trace = TraceContext(query="", reranked=False)

        assert trace.query == ""

    def test_trace_context_should_accept_empty_retrieved_nodes(self) -> None:
        """Test TraceContext when no nodes were retrieved.

        Given: An empty list of retrieved_nodes (no results).
        When: The TraceContext is created.
        Then: The empty list is accepted.
        """

        trace = TraceContext(
            query="Obscure query with no results",
            retrieved_nodes=[],
            reranked=False,
        )

        assert len(trace.retrieved_nodes) == 0
        assert trace.retrieved_nodes == []

    def test_trace_context_should_forbid_extra_fields(self) -> None:
        """Test strict mode prevents unknown fields.

        Given: An attempt to create TraceContext with extra field.
        When: Pydantic validates the strict model.
        Then: A ValidationError is raised.
        """

        invalid_data = {
            "query": "Test",
            "unknown_field": "invalid",
        }

        with pytest.raises(ValidationError) as exc_info:
            TraceContext(**invalid_data)  # type: ignore[arg-type]

        assert "Extra inputs are not permitted" in str(exc_info.value)
