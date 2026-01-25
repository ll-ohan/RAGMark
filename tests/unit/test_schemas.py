"""Unit tests for Pydantic schemas.

This module tests all data models including validation, computed fields,
and serialization.
"""

from datetime import datetime
from pathlib import Path

import pytest

from ragmark.schemas.documents import (
    KnowledgeNode,
    NodePosition,
    SourceDoc,
)
from ragmark.schemas.evaluation import (
    AuditReport,
    CaseResult,
    SystemInfo,
    TrialCase,
)
from ragmark.schemas.generation import GenerationResult
from ragmark.schemas.retrieval import (
    RetrievedNode,
    TraceContext,
)


class TestSourceDoc:
    """Tests for SourceDoc schema."""

    def test_valid_source_doc(self) -> None:
        """Test creating a valid SourceDoc."""
        doc = SourceDoc(
            content="Sample content",
            metadata={"title": "Test"},
            mime_type="text/plain",
            page_count=10,
        )
        assert doc.content == "Sample content"
        assert doc.metadata["title"] == "Test"
        assert doc.page_count == 10
        assert doc.source_id  # Should be auto-generated

    def test_empty_content_validation(self) -> None:
        """Test that empty content is rejected."""
        with pytest.raises(ValueError, match="Content cannot be empty"):
            SourceDoc(content="", mime_type="text/plain", page_count=3)

    def test_whitespace_only_content(self) -> None:
        """Test that whitespace-only content is rejected."""
        with pytest.raises(ValueError, match="Content cannot be empty"):
            SourceDoc(content="   \n\t  ", mime_type="text/plain", page_count=3)

    def test_strict_mode_rejects_extra_fields(self) -> None:
        """Test that extra fields are rejected in strict mode."""
        with pytest.raises(ValueError):
            SourceDoc(
                content="Sample",
                mime_type="text/plain",
                extra_field="not allowed",  # type: ignore
                page_count=None,
            )


class TestNodePosition:
    """Tests for NodePosition schema."""

    def test_valid_position(self) -> None:
        """Test creating a valid NodePosition."""
        pos = NodePosition(
            start_char=0,
            end_char=100,
            page=1,
            section="Introduction",
        )
        assert pos.start_char == 0
        assert pos.end_char == 100

    def test_end_char_validation(self) -> None:
        """Test that end_char must be greater than start_char."""
        with pytest.raises(
            ValueError, match="end_char must be greater than start_char"
        ):
            NodePosition(start_char=100, end_char=50, page=None, section=None)

        with pytest.raises(
            ValueError, match="end_char must be greater than start_char"
        ):
            NodePosition(start_char=100, end_char=100, page=None, section=None)

    def test_optional_fields(self) -> None:
        """Test that page and section are optional."""
        pos = NodePosition(start_char=0, end_char=100, page=None, section=None)
        assert pos.page is None
        assert pos.section is None


class TestKnowledgeNode:
    """Tests for KnowledgeNode schema."""

    def test_valid_node(self) -> None:
        """Test creating a valid KnowledgeNode."""
        node = KnowledgeNode(
            content="Sample chunk",
            source_id="source-123",
            position=NodePosition(start_char=0, end_char=12, page=None, section=None),
            dense_vector=None,
            sparse_vector=None,
        )
        assert node.content == "Sample chunk"
        assert node.node_id  # Auto-generated UUID

    def test_auto_enrichment(self) -> None:
        """Test that metadata is automatically enriched."""
        node = KnowledgeNode(
            content="This is a test chunk with multiple words",
            source_id="source-123",
            position=NodePosition(start_char=0, end_char=40, page=None, section=None),
            dense_vector=None,
            sparse_vector=None,
        )

        # Check computed metadata
        assert node.metadata["char_count"] == 40
        assert node.metadata["word_count"] == 8
        assert "created_at" in node.metadata

        # Verify created_at is a valid ISO datetime
        datetime.fromisoformat(node.metadata["created_at"])

    def test_metadata_not_overwritten(self) -> None:
        """Test that existing metadata is not overwritten."""
        node = KnowledgeNode(
            content="Test",
            source_id="source-123",
            metadata={"char_count": 999, "custom": "value"},
            position=NodePosition(start_char=0, end_char=4, page=None, section=None),
            dense_vector=None,
            sparse_vector=None,
        )

        # Custom char_count should be preserved
        assert node.metadata["char_count"] == 999
        assert node.metadata["custom"] == "value"

    def test_empty_content_validation(self) -> None:
        """Test that empty content is rejected."""
        with pytest.raises(
            ValueError, match="Node content cannot be empty or whitespace-only"
        ):
            KnowledgeNode(
                content="",
                source_id="source-123",
                position=NodePosition(
                    start_char=0, end_char=1, page=None, section=None
                ),
                dense_vector=None,
                sparse_vector=None,
            )

    def test_optional_vectors(self) -> None:
        """Test that vectors are optional."""
        node = KnowledgeNode(
            content="Test",
            source_id="source-123",
            position=NodePosition(start_char=0, end_char=4, page=None, section=None),
            dense_vector=None,
            sparse_vector=None,
        )
        assert node.dense_vector is None
        assert node.sparse_vector is None

    def test_with_vectors(self) -> None:
        """Test creating a node with embeddings."""
        node = KnowledgeNode(
            content="Test",
            source_id="source-123",
            position=NodePosition(start_char=0, end_char=4, page=None, section=None),
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={1: 0.5, 5: 0.3},
        )
        assert node.dense_vector is not None
        assert node.sparse_vector is not None
        assert len(node.dense_vector) == 3
        assert node.sparse_vector[1] == 0.5


class TestTrialCase:
    """Tests for TrialCase schema."""

    def test_valid_with_answer(self) -> None:
        """Test creating a trial case with ground truth answer."""
        case = TrialCase(
            question="What is RAG?",
            ground_truth_answer="Retrieval Augmented Generation",
            ground_truth_node_ids=None,
        )
        assert case.question == "What is RAG?"
        assert case.ground_truth_answer == "Retrieval Augmented Generation"

    def test_valid_with_node_ids(self) -> None:
        """Test creating a trial case with ground truth node IDs."""
        case = TrialCase(
            question="What is RAG?",
            ground_truth_answer=None,
            ground_truth_node_ids=["node-1", "node-2"],
        )
        assert case.ground_truth_node_ids is not None
        assert len(case.ground_truth_node_ids) == 2

    def test_at_least_one_ground_truth_required(self) -> None:
        """Test that at least one ground truth is required."""
        with pytest.raises(ValueError, match="At least one of ground_truth"):
            TrialCase(question="What is RAG?")  # type: ignore

    def test_empty_question_validation(self) -> None:
        """Test that empty questions are rejected."""
        with pytest.raises(ValueError, match="Question cannot be empty"):
            TrialCase(
                question="",
                ground_truth_answer="Answer",
                ground_truth_node_ids=["node-1", "node-2"],
            )

    def test_load_cases_json(self, tmp_path: Path) -> None:
        """Test loading trial cases from JSON file."""
        cases_file = tmp_path / "cases.json"
        cases_file.write_text(
            """[
            {
                "question": "Q1?",
                "ground_truth_answer": "A1"
            },
            {
                "question": "Q2?",
                "ground_truth_node_ids": ["n1", "n2"]
            }
        ]"""
        )

        cases = TrialCase.load_cases(cases_file)
        assert len(cases) == 2
        assert cases[0].question == "Q1?"
        assert cases[1].ground_truth_node_ids == ["n1", "n2"]

    def test_load_cases_jsonl(self, tmp_path: Path) -> None:
        """Test loading trial cases from JSONL file."""
        cases_file = tmp_path / "cases.jsonl"
        cases_file.write_text(
            '{"question": "Q1?", "ground_truth_answer": "A1"}\n'
            '{"question": "Q2?", "ground_truth_node_ids": ["n1"]}\n'
        )

        cases = TrialCase.load_cases(cases_file)
        assert len(cases) == 2

    def test_load_cases_file_not_found(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError):
            TrialCase.load_cases(tmp_path / "nonexistent.json")

    def test_load_cases_unsupported_format(self, tmp_path: Path) -> None:
        """Test that ValueError is raised for unsupported formats."""
        cases_file = tmp_path / "cases.txt"
        cases_file.write_text("Not JSON")

        with pytest.raises(ValueError, match="Unsupported file format"):
            TrialCase.load_cases(cases_file)


class TestAuditReport:
    """Tests for AuditReport schema."""

    def test_to_json(self, tmp_path: Path) -> None:
        """Test exporting report to JSON."""
        report = AuditReport(
            experiment_profile_hash="abc123",
            duration_seconds=10.5,
            system_info=SystemInfo(
                python_version="3.11",
                ragmark_version="0.1.0",
                platform="Linux",
                cpu_count=8,
                ram_gb=16.0,
                gpu_info="gpu info",
            ),
        )

        output_path = tmp_path / "report.json"
        report.to_json(output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "abc123" in content
        assert "10.5" in content

    def test_to_dataframe_empty(self) -> None:
        """Test converting empty report to DataFrame."""
        report = AuditReport(
            experiment_profile_hash="abc123",
            duration_seconds=10.0,
            system_info=SystemInfo(
                python_version="3.11",
                ragmark_version="0.1.0",
                platform="Linux",
                cpu_count=8,
                ram_gb=16.0,
                gpu_info=None,
            ),
        )

        df = report.to_dataframe()
        assert len(df) == 0

    def test_to_dataframe_with_results(
        self, sample_knowledge_node: KnowledgeNode
    ) -> None:
        """Test converting report with results to DataFrame."""
        from ragmark.schemas.generation import TokenUsage

        trace = TraceContext(
            query="Test query",
            retrieved_nodes=[
                RetrievedNode(node=sample_knowledge_node, score=0.9, rank=1)
            ],
            reranked=False,
        )

        result = CaseResult(
            case_id="case-1",
            predicted_answer="Test answer",
            trace=trace,
            generation_result=GenerationResult(
                text="Test answer",
                usage=TokenUsage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ),
                finish_reason="stop",
            ),
            case_metrics={"recall@5": 1.0},
        )

        report = AuditReport(
            experiment_profile_hash="abc123",
            duration_seconds=10.0,
            per_case_results=[result],
            system_info=SystemInfo(
                python_version="3.11",
                ragmark_version="0.1.0",
                platform="Linux",
                cpu_count=8,
                ram_gb=16.0,
                gpu_info=None,
            ),
        )

        df = report.to_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["case_id"] == "case-1"
        assert df.iloc[0]["recall@5"] == 1.0
