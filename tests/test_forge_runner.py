"""Unit tests for Forge pipeline."""

import logging
import unicodedata
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import BinaryIO

import pytest

from ragmark.config.profile import ExperimentProfile, FragmenterConfig, IngestorConfig
from ragmark.exceptions import FragmentationError, IngestionError
from ragmark.forge.fragmenters import BaseFragmenter
from ragmark.forge.ingestors import BaseIngestor
from ragmark.forge.runner import ForgeRunner
from ragmark.schemas.documents import KnowledgeNode, NodePosition, SourceDoc


class FakeIngestor(BaseIngestor):
    """Fake implementation of an Ingestor for testing purposes.

    Behaviors:
    - Filename 'ingest_error': Raises IngestionError.
    - Filename 'runtime_error': Raises RuntimeError (unexpected exception).
    - Filename 'fragment_error': Returns content triggering the Fragmenter error.
    - Default: Returns valid SourceDoc.
    """

    @classmethod
    def from_config(cls, config: IngestorConfig) -> "FakeIngestor":
        return cls()

    @property
    def supported_formats(self) -> set[str]:
        return {".txt", ".md", ".fake"}

    def ingest(self, source: Path | bytes | BinaryIO) -> SourceDoc:
        source_str = str(source)

        if "ingest_error" in source_str:
            raise IngestionError(f"Simulated ingestion failure for {source_str}")

        if "runtime_error" in source_str:
            raise RuntimeError(f"Simulated unexpected failure for {source_str}")

        content = f"Content from {source_str}"
        if "fragment_error" in source_str:
            content = "TRIGGER_FRAGMENT_ERROR"

        return SourceDoc(
            content=content,
            mime_type="text/plain",
            source_id=f"id_{Path(source_str).name}",
            metadata={"source.original_path": source_str},
            page_count=2,
        )


class FakeFragmenter(BaseFragmenter):
    """Fake implementation of a Fragmenter.

    Behaviors:
    - Content 'TRIGGER_FRAGMENT_ERROR': Raises FragmentationError.
    - Default: Splits content into exactly 2 nodes.
    """

    @classmethod
    def from_config(cls, config: FragmenterConfig) -> "FakeFragmenter":
        return cls()

    @property
    def chunk_size(self) -> int:
        return 100

    @property
    def overlap(self) -> int:
        return 0

    def fragment(self, doc: SourceDoc) -> list[KnowledgeNode]:
        if doc.content == "TRIGGER_FRAGMENT_ERROR":
            raise FragmentationError(
                f"Simulated fragmentation failure for doc {doc.source_id}"
            )

        nodes: list[KnowledgeNode] = []
        for i in range(2):
            node_content = f"{doc.content} | Chunk {i}"
            nodes.append(
                KnowledgeNode(
                    node_id=f"{doc.source_id}_chunk_{i}",
                    content=node_content,
                    source_id=doc.source_id,
                    metadata={**doc.metadata, "chunk_index": i},
                    dense_vector=[0.1] * 384,
                    sparse_vector=None,
                    position=NodePosition(
                        start_char=i * 10,
                        end_char=(i + 1) * 10,
                        page=1,
                        section="Test Section",
                    ),
                )
            )
        return nodes


@pytest.mark.unit
class TestForgeRunner:
    """Tests for ForgeRunner logic using Fake components and real configurations."""

    @pytest.fixture
    def fake_ingestor(self) -> FakeIngestor:
        return FakeIngestor()

    @pytest.fixture
    def fake_fragmenter(self) -> FakeFragmenter:
        return FakeFragmenter()

    def test_run_should_produce_valid_nodes_when_input_is_valid(
        self,
        fake_ingestor: FakeIngestor,
        fake_fragmenter: FakeFragmenter,
        source_file_factory: Callable[[str, str], Path],
    ) -> None:
        """Processes two valid files into a flat list of KnowledgeNodes.

        Given: Two valid text files.
        When: The ForgeRunner processes them with fail_fast=True.
        Then: It should yield exactly 4 KnowledgeNodes with strictly validated content and metadata.
        """
        runner = ForgeRunner(fake_ingestor, fake_fragmenter, fail_fast=True)
        doc1 = source_file_factory("doc1.txt", "Doc 1 content")
        doc2 = source_file_factory("doc2.txt", "Doc 2 content")

        nodes = runner.run([doc1, doc2])

        assert len(nodes) == 4

        node_0 = nodes[0]
        assert node_0.source_id == "id_doc1.txt"
        assert node_0.content == f"Content from {doc1} | Chunk 0"
        assert node_0.metadata["source.original_path"] == str(doc1)
        assert isinstance(node_0.position, NodePosition)

        node_3 = nodes[3]
        assert node_3.source_id == "id_doc2.txt"
        assert node_3.content == f"Content from {doc2} | Chunk 1"

    def test_run_lazy_should_return_generator_and_yield_typed_objects(
        self,
        fake_ingestor: FakeIngestor,
        fake_fragmenter: FakeFragmenter,
        source_file_factory: Callable[[str, str], Path],
    ) -> None:
        """Verifies that run_lazy returns an iterator yielding typed objects.

        Given: A valid source file.
        When: run_lazy is called.
        Then: It should return an Iterator that yields strictly typed KnowledgeNode objects.
        """
        runner = ForgeRunner(fake_ingestor, fake_fragmenter)
        source = source_file_factory("doc1.txt", "Content")

        result = runner.run_lazy([source])

        assert isinstance(result, Iterator)
        first_node = next(result)
        assert isinstance(first_node, KnowledgeNode)

    def test_run_should_raise_ingestion_error_immediately_when_fail_fast_is_true(
        self,
        fake_ingestor: FakeIngestor,
        fake_fragmenter: FakeFragmenter,
        source_file_factory: Callable[[str, str], Path],
    ) -> None:
        """Ensures immediate failure upon ingestion errors in strict mode.

        Given: A source file that triggers an IngestionError.
        When: The pipeline runs with fail_fast=True.
        Then: An IngestionError should be raised immediately and preserve strict error messaging.
        """
        runner = ForgeRunner(fake_ingestor, fake_fragmenter, fail_fast=True)
        bad_source = source_file_factory("doc_ingest_error.txt", "Bad content")

        with pytest.raises(IngestionError) as exc_info:
            list(runner.run_lazy([bad_source]))

        assert "Simulated ingestion failure" in str(exc_info.value)
        assert str(bad_source) in str(exc_info.value)

    def test_run_should_raise_fragmentation_error_immediately_when_fail_fast_is_true(
        self,
        fake_ingestor: FakeIngestor,
        fake_fragmenter: FakeFragmenter,
        source_file_factory: Callable[[str, str], Path],
    ) -> None:
        """Ensures immediate failure upon fragmentation errors in strict mode.

        Given: A source file that passes ingestion but triggers a FragmentationError.
        When: The pipeline runs with fail_fast=True.
        Then: A FragmentationError should be raised immediately.
        """
        runner = ForgeRunner(fake_ingestor, fake_fragmenter, fail_fast=True)
        bad_source = source_file_factory(
            "doc_fragment_error.txt", "TRIGGER_FRAGMENT_ERROR"
        )

        with pytest.raises(FragmentationError) as exc_info:
            list(runner.run_lazy([bad_source]))

        assert "Simulated fragmentation failure" in str(exc_info.value)

    def test_run_should_log_and_skip_errors_when_fail_fast_is_false(
        self,
        fake_ingestor: FakeIngestor,
        fake_fragmenter: FakeFragmenter,
        source_file_factory: Callable[[str, str], Path],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verifies error skipping and logging in permissive mode.

        Given: A source file that triggers an IngestionError.
        When: The pipeline runs with fail_fast=False.
        Then: The error should be logged as a warning, and the process should continue (yielding 0 nodes).
        """
        runner = ForgeRunner(fake_ingestor, fake_fragmenter, fail_fast=False)
        bad_source = source_file_factory("doc_ingest_error.txt", "Bad content")

        logger = logging.getLogger("ragmark.forge.runner")
        logger.propagate = True

        with caplog.at_level(logging.WARNING, logger="ragmark.forge.runner"):
            nodes = list(runner.run_lazy([bad_source]))

        assert len(nodes) == 0
        assert "Skipping failed document" in caplog.text
        assert str(bad_source) in caplog.text

    def test_run_should_handle_mixed_results_correctly_when_fail_fast_is_false(
        self,
        fake_ingestor: FakeIngestor,
        fake_fragmenter: FakeFragmenter,
        source_file_factory: Callable[[str, str], Path],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verifies partial success in a batch processing scenario.

        Given: A mix of files causing IngestionError, FragmentationError, and Success.
        When: The pipeline runs with fail_fast=False.
        Then: It should yield nodes only for the successful doc and log errors for the others.
        """
        runner = ForgeRunner(fake_ingestor, fake_fragmenter, fail_fast=False)

        doc_a = source_file_factory("doc_a_ingest_error.txt", "Fail Ingest")
        doc_b = source_file_factory(
            "doc_b_fragment_error.txt", "TRIGGER_FRAGMENT_ERROR"
        )
        doc_c = source_file_factory("doc_c_success.txt", "Success content")

        sources = [doc_a, doc_b, doc_c]

        logger = logging.getLogger("ragmark.forge.runner")
        logger.propagate = True

        with caplog.at_level(logging.INFO, logger="ragmark.forge.runner"):
            nodes = list(runner.run_lazy(sources))

        assert len(nodes) == 2

        for node in nodes:
            assert "doc_c_success" in node.content
            assert node.source_id == "id_doc_c_success.txt"

        log_text = caplog.text
        assert f"Skipping failed document: source={doc_a}" in log_text
        assert "IngestionError" in log_text
        assert f"Skipping failed document: source={doc_b}" in log_text
        assert "FragmentationError" in log_text

        assert "Forge pipeline complete" in log_text
        assert "documents=2" in log_text
        assert "errors=2" in log_text

    def test_run_should_handle_unexpected_exceptions_gracefully(
        self,
        fake_ingestor: FakeIngestor,
        fake_fragmenter: FakeFragmenter,
        source_file_factory: Callable[[str, str], Path],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verifies graceful handling of generic runtime exceptions in permissive mode.

        Given: A document triggering an unexpected RuntimeError.
        When: The pipeline runs with fail_fast=False.
        Then: It should catch the generic Exception, log it with traceback, and continue.
        """
        runner = ForgeRunner(fake_ingestor, fake_fragmenter, fail_fast=False)
        bad_source = source_file_factory("doc_runtime_error.txt", "Unexpected Fail")

        logger = logging.getLogger("ragmark.forge.runner")
        logger.propagate = True

        with caplog.at_level(logging.DEBUG, logger="ragmark.forge.runner"):
            list(runner.run_lazy([bad_source]))

        assert "Skipping document" in caplog.text
        assert "due to unexpected error" in caplog.text
        assert "Simulated unexpected failure" in caplog.text

    def test_run_should_raise_unexpected_error_immediately_when_fail_fast_is_true(
        self,
        fake_ingestor: FakeIngestor,
        fake_fragmenter: FakeFragmenter,
        source_file_factory: Callable[[str, str], Path],
    ) -> None:
        """Ensures that unexpected exceptions bubble up immediately in strict mode.

        Given: A document triggering an unexpected RuntimeError.
        When: The pipeline runs with fail_fast=True.
        Then: The RuntimeError should propagate immediately, halting execution.
        """
        runner = ForgeRunner(fake_ingestor, fake_fragmenter, fail_fast=True)
        bad_source = source_file_factory("doc_runtime_error.txt", "Unexpected Fail")

        with pytest.raises(RuntimeError) as exc_info:
            list(runner.run_lazy([bad_source]))

        assert "Simulated unexpected failure" in str(exc_info.value)

    @pytest.mark.rag_edge_case
    def test_run_should_handle_unicode_filenames_correctly(
        self,
        fake_ingestor: FakeIngestor,
        fake_fragmenter: FakeFragmenter,
        source_file_factory: Callable[[str, str], Path],
    ) -> None:
        """Verifies robust handling of complex Unicode filenames including NFD normalization and ZWJ sequences.

        Given: A file with NFD characters (decomposed accent) and a ZWJ sequence (Family emoji).
        When: The pipeline runs.
        Then: It should process the file without encoding errors and preserve the filename in source_id.
        """
        runner = ForgeRunner(fake_ingestor, fake_fragmenter, fail_fast=True)

        nfd_part = unicodedata.normalize("NFD", "café")
        zwj_part = "👨‍👩‍👧‍👦"

        filename = f"doc_{nfd_part}_{zwj_part}.txt"
        doc = source_file_factory(filename, "Unicode Content")

        nodes = runner.run([doc])

        assert len(nodes) == 2

        assert nodes[0].source_id == f"id_{filename}"
        assert nodes[0].metadata["source.original_path"] == str(doc)

        assert zwj_part in nodes[0].metadata["source.original_path"]

    def test_run_should_yield_nothing_when_sources_is_empty(
        self, fake_ingestor: FakeIngestor, fake_fragmenter: FakeFragmenter
    ) -> None:
        """Verifies behavior with empty input.

        Given: An empty list of source files.
        When: The pipeline runs.
        Then: It should complete successfully returning an empty list/iterator without errors.
        """
        runner = ForgeRunner(fake_ingestor, fake_fragmenter)

        nodes = runner.run([])

        assert isinstance(nodes, list)
        assert len(nodes) == 0

    def test_from_profile_should_create_valid_runner_instance_when_profile_is_valid(
        self,
    ) -> None:
        """Verifies factory instantiation from configuration profile.

        Given: A valid ExperimentProfile with default configurations.
        When: ForgeRunner.from_profile is called.
        Then: It should return a fully initialized ForgeRunner instance with concrete Ingestor and Fragmenter.
        """
        profile = ExperimentProfile(
            ingestor=IngestorConfig(), fragmenter=FragmenterConfig()
        )

        runner = ForgeRunner.from_profile(profile)

        assert isinstance(runner, ForgeRunner)

        assert isinstance(runner.ingestor, BaseIngestor)
        assert isinstance(runner.fragmenter, BaseFragmenter)
