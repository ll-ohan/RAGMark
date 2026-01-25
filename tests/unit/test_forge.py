"""Unit tests for Forge pipeline.

This module tests the ForgeRunner which chains ingestion and fragmentation.
"""

from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from ragmark.exceptions import IngestionError
from ragmark.forge.runner import ForgeRunner
from ragmark.schemas.documents import KnowledgeNode, NodePosition, SourceDoc


class TestForgeRunner:
    """Tests for ForgeRunner."""

    @pytest.fixture
    def mock_ingestor(self) -> MagicMock:
        """Create a mock ingestor that returns test documents."""
        ingestor = MagicMock()

        ingestor.ingest_batch.return_value = iter(
            [
                SourceDoc(
                    content="Doc 1",
                    mime_type="text/plain",
                    source_id="s1",
                    page_count=None,
                ),
                SourceDoc(
                    content="Doc 2",
                    mime_type="text/plain",
                    source_id="s2",
                    page_count=None,
                ),
            ]
        )

        return ingestor

    @pytest.fixture
    def mock_fragmenter(self) -> MagicMock:
        """Create a mock fragmenter that returns test nodes."""
        fragmenter = MagicMock()
        fragmenter.chunk_size = 256
        fragmenter.overlap = 64

        def fragment_batch_generator(
            docs: list[SourceDoc],
        ) -> Generator[KnowledgeNode, Any, Any]:
            for doc in docs:
                # Yield 2 nodes per document
                for i in range(2):
                    yield KnowledgeNode(
                        content=f"{doc.content} - Chunk {i}",
                        source_id=doc.source_id,
                        position=NodePosition(
                            start_char=i * 10,
                            end_char=(i + 1) * 10,
                            page=None,
                            section=None,
                        ),
                        dense_vector=None,
                        sparse_vector=None,
                    )

        fragmenter.fragment_batch.side_effect = fragment_batch_generator

        return fragmenter

    def test_run_success(
        self, mock_ingestor: MagicMock, mock_fragmenter: MagicMock, tmp_path: Path
    ) -> None:
        """Test successful pipeline execution."""
        runner = ForgeRunner(mock_ingestor, mock_fragmenter, fail_fast=True)

        sources = [tmp_path / "doc1.txt", tmp_path / "doc2.txt"]
        nodes = runner.run(sources)

        # Should generate 2 docs * 2 chunks = 4 nodes
        assert len(nodes) == 4
        assert all(isinstance(n, KnowledgeNode) for n in nodes)

    def test_run_lazy_is_generator(
        self, mock_ingestor: MagicMock, mock_fragmenter: MagicMock, tmp_path: Path
    ) -> None:
        """Test that run_lazy returns a generator."""
        runner = ForgeRunner(mock_ingestor, mock_fragmenter)

        sources = [tmp_path / "doc1.txt"]
        result = runner.run_lazy(sources)

        # Should be a generator
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_run_lazy_streaming(
        self, mock_ingestor: MagicMock, mock_fragmenter: MagicMock, tmp_path: Path
    ) -> None:
        """Test that run_lazy yields nodes incrementally."""
        runner = ForgeRunner(mock_ingestor, mock_fragmenter)

        sources = [tmp_path / "doc1.txt", tmp_path / "doc2.txt"]
        nodes = list(runner.run_lazy(sources))

        assert len(nodes) == 4

    def test_fail_fast_true(self, mock_fragmenter: MagicMock, tmp_path: Path) -> None:
        """Test that errors are raised when fail_fast=True."""
        # Create an ingestor that raises an error
        mock_ingestor = MagicMock()
        mock_ingestor.ingest_batch.side_effect = IngestionError("Test error")

        runner = ForgeRunner(mock_ingestor, mock_fragmenter, fail_fast=True)

        sources = [tmp_path / "doc1.txt"]

        with pytest.raises(IngestionError, match="Test error"):
            list(runner.run_lazy(sources))

    def test_fail_fast_false(
        self,
        mock_fragmenter: MagicMock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that errors are logged and skipped when fail_fast=False."""
        # Create an ingestor that raises an error
        mock_ingestor = MagicMock()
        mock_ingestor.ingest_batch.side_effect = IngestionError("Test error")

        runner = ForgeRunner(mock_ingestor, mock_fragmenter, fail_fast=False)

        sources = [tmp_path / "doc1.txt"]

        # Should not raise, but should log warning
        nodes = list(runner.run_lazy(sources))

        # No nodes should be produced due to error
        assert len(nodes) == 0
        assert "Skipping failed document" in caplog.text

    def test_pipeline_chaining(self, tmp_path: Path) -> None:
        """Test that ingestor and fragmenter are properly chained."""
        mock_ingestor = MagicMock()
        mock_fragmenter = MagicMock()

        # Setup return values
        doc = SourceDoc(content="Test", mime_type="text/plain", page_count=None)
        mock_ingestor.ingest_batch.return_value = iter([doc])

        node = KnowledgeNode(
            content="Test chunk",
            source_id="test",
            position=NodePosition(start_char=0, end_char=10, page=None, section=None),
            dense_vector=None,
            sparse_vector=None,
        )
        mock_fragmenter.fragment_batch.return_value = iter([node])

        runner = ForgeRunner(mock_ingestor, mock_fragmenter)
        sources = [tmp_path / "test.txt"]

        nodes = runner.run(sources)

        # Verify both were called
        mock_ingestor.ingest_batch.assert_called_once()
        mock_fragmenter.fragment_batch.assert_called_once()

        assert len(nodes) == 1
        assert nodes[0].content == "Test chunk"

    def test_logging(
        self,
        mock_ingestor: MagicMock,
        mock_fragmenter: MagicMock,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that pipeline logs progress information."""
        import logging

        caplog.set_level(logging.INFO)

        runner = ForgeRunner(mock_ingestor, mock_fragmenter)
        sources = [tmp_path / "doc1.txt"]

        list(runner.run_lazy(sources))

        assert "Starting Forge pipeline" in caplog.text
        assert "Forge pipeline complete" in caplog.text
