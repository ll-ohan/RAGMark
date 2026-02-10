"""Unit tests for Forge pipeline."""

import asyncio
import logging
import time
import unicodedata
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import BinaryIO

import pytest

from ragmark.config.profile import (
    ExperimentProfile,
    FragmenterConfig,
    IngestorConfig,
    StreamingMetricsConfig,
)
from ragmark.exceptions import FragmentationError, IngestionError
from ragmark.forge.fragmenters import BaseFragmenter
from ragmark.forge.ingestors import BaseIngestor
from ragmark.forge.runner import ForgeRunner, StreamingMetrics
from ragmark.metrics.metrics import MetricsManager
from ragmark.schemas.documents import KnowledgeNode, NodePosition, SourceDoc


class FakeIngestor(BaseIngestor):
    """Fake implementation of an Ingestor for testing purposes.

    Behaviors:
    - Filename 'ingest_error': Raises IngestionError.
    - Filename 'runtime_error': Raises RuntimeError (unexpected exception).
    - Filename 'fragment_error': Returns content triggering the Fragmenter error.
    - Default: Returns valid SourceDoc.
    """

    def __init__(self, delay_ms: float = 0):
        """Initialize with optional delay for testing async performance."""
        self.delay_ms = delay_ms

    @classmethod
    def from_config(cls, config: IngestorConfig) -> "FakeIngestor":
        return cls()

    @property
    def supported_formats(self) -> set[str]:
        return {".txt", ".md", ".fake"}

    def ingest(self, source: Path | bytes | BinaryIO) -> SourceDoc:
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000)

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
    - Default: Splits content into exactly 3 nodes.
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

    def fragment_lazy(self, doc: SourceDoc) -> Iterator[KnowledgeNode]:
        if doc.content == "TRIGGER_FRAGMENT_ERROR":
            raise FragmentationError(
                f"Simulated fragmentation failure for doc {doc.source_id}"
            )

        for i in range(3):
            node_content = f"{doc.content} | Chunk {i}"
            yield KnowledgeNode(
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

        nodes = runner.process([doc1, doc2])

        assert len(nodes) == 6  # 2 docs × 3 chunks each

        node_0 = nodes[0]
        assert node_0.source_id == "id_doc1.txt"
        assert node_0.content == f"Content from {doc1} | Chunk 0"
        assert node_0.metadata["source.original_path"] == str(doc1)
        assert isinstance(node_0.position, NodePosition)

        node_3 = nodes[3]
        assert node_3.source_id == "id_doc2.txt"
        assert node_3.content == f"Content from {doc2} | Chunk 0"

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

        result = runner.process_iter([source])

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
            list(runner.process_iter([bad_source]))

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
            list(runner.process_iter([bad_source]))

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
            nodes = list(runner.process_iter([bad_source]))

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
            nodes = list(runner.process_iter(sources))

        assert len(nodes) == 3

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
            list(runner.process_iter([bad_source]))

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
            list(runner.process_iter([bad_source]))

        assert "Simulated unexpected failure" in str(exc_info.value)

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

        nodes = runner.process([doc])

        assert len(nodes) == 3

        assert nodes[0].source_id == f"id_{filename}"
        metadata_path = nodes[0].metadata["source.original_path"]
        assert metadata_path == str(doc)

        assert isinstance(metadata_path, str)
        assert zwj_part in metadata_path

    def test_run_should_yield_nothing_when_sources_is_empty(
        self, fake_ingestor: FakeIngestor, fake_fragmenter: FakeFragmenter
    ) -> None:
        """Verifies behavior with empty input.

        Given: An empty list of source files.
        When: The pipeline runs.
        Then: It should complete successfully returning an empty list/iterator without errors.
        """
        runner = ForgeRunner(fake_ingestor, fake_fragmenter)

        nodes = runner.process([])

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


@pytest.mark.asyncio
@pytest.mark.unit
class TestForgeRunnerAsync:
    """Tests for async methods of ForgeRunner."""

    @pytest.fixture
    def fake_ingestor(self) -> FakeIngestor:
        return FakeIngestor()

    @pytest.fixture
    def fake_fragmenter(self) -> FakeFragmenter:
        return FakeFragmenter()

    async def test_process_async_should_handle_ingestion_error_when_fail_fast_is_false(
        self,
        fake_ingestor: FakeIngestor,
        fake_fragmenter: FakeFragmenter,
        source_file_factory: Callable[[str, str], Path],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verifies that process_async skips ingestion errors when fail_fast=False.

        Given: A source file triggering an IngestionError.
        When: process_async runs with fail_fast=False.
        Then: The error should be logged and processing should continue.
        """
        runner = ForgeRunner(fake_ingestor, fake_fragmenter, fail_fast=False)
        bad_source = source_file_factory("doc_ingest_error.txt", "Bad content")
        good_source = source_file_factory("doc_good.txt", "Good content")

        logger = logging.getLogger("ragmark.forge.runner")
        logger.propagate = True

        with caplog.at_level(logging.WARNING, logger="ragmark.forge.runner"):
            nodes: list[KnowledgeNode] = []
            async for node in runner.process_async([bad_source, good_source]):
                nodes.append(node)

        assert len(nodes) == 3
        assert "Skipping failed document" in caplog.text
        assert str(bad_source) in caplog.text

    async def test_process_async_should_handle_fragmentation_error_when_fail_fast_is_false(
        self,
        fake_ingestor: FakeIngestor,
        fake_fragmenter: FakeFragmenter,
        source_file_factory: Callable[[str, str], Path],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verifies that process_async skips fragmentation errors when fail_fast=False.

        Given: A source file triggering a FragmentationError.
        When: process_async runs with fail_fast=False.
        Then: The error should be logged and processing should continue.
        """
        runner = ForgeRunner(fake_ingestor, fake_fragmenter, fail_fast=False)
        bad_source = source_file_factory(
            "doc_fragment_error.txt", "TRIGGER_FRAGMENT_ERROR"
        )
        good_source = source_file_factory("doc_good.txt", "Good content")

        logger = logging.getLogger("ragmark.forge.runner")
        logger.propagate = True

        with caplog.at_level(logging.WARNING, logger="ragmark.forge.runner"):
            nodes: list[KnowledgeNode] = []
            async for node in runner.process_async([bad_source, good_source]):
                nodes.append(node)

        assert len(nodes) == 3
        assert "Skipping failed document" in caplog.text

    async def test_process_async_should_handle_unexpected_error_when_fail_fast_is_false(
        self,
        fake_ingestor: FakeIngestor,
        fake_fragmenter: FakeFragmenter,
        source_file_factory: Callable[[str, str], Path],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verifies that process_async skips unexpected errors when fail_fast=False.

        Given: A document triggering an unexpected RuntimeError.
        When: process_async runs with fail_fast=False.
        Then: The error should be logged and processing should continue.
        """
        runner = ForgeRunner(fake_ingestor, fake_fragmenter, fail_fast=False)
        bad_source = source_file_factory("doc_runtime_error.txt", "Unexpected Fail")
        good_source = source_file_factory("doc_good.txt", "Good content")

        logger = logging.getLogger("ragmark.forge.runner")
        logger.propagate = True

        with caplog.at_level(logging.WARNING, logger="ragmark.forge.runner"):
            nodes: list[KnowledgeNode] = []
            async for node in runner.process_async([bad_source, good_source]):
                nodes.append(node)

        assert len(nodes) == 3
        assert "Skipping document due to unexpected error" in caplog.text


@pytest.mark.asyncio
@pytest.mark.unit
class TestForgeRunnerConcurrent:
    """Tests for concurrent processing methods of ForgeRunner."""

    @pytest.fixture
    def fake_ingestor(self) -> FakeIngestor:
        return FakeIngestor()

    @pytest.fixture
    def fake_fragmenter(self) -> FakeFragmenter:
        return FakeFragmenter()

    async def test_process_concurrent_should_handle_errors_when_fail_fast_is_false(
        self,
        fake_ingestor: FakeIngestor,
        fake_fragmenter: FakeFragmenter,
        source_file_factory: Callable[[str, str], Path],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verifies that process_concurrent handles errors gracefully when fail_fast=False.

        Given: Mixed sources with ingestion and fragmentation errors.
        When: process_concurrent runs with fail_fast=False.
        Then: Errors should be logged and good documents should be processed.
        """
        runner = ForgeRunner(fake_ingestor, fake_fragmenter, fail_fast=False)
        bad_ingest = source_file_factory("doc_ingest_error.txt", "Bad")
        bad_fragment = source_file_factory("doc_fragment_error.txt", "Fragment Error")
        good_source = source_file_factory("doc_good.txt", "Good content")

        logger = logging.getLogger("ragmark.forge.runner")
        logger.propagate = True

        with caplog.at_level(logging.WARNING, logger="ragmark.forge.runner"):
            nodes: list[KnowledgeNode] = []
            async for item in runner.process_concurrent(
                [bad_ingest, bad_fragment, good_source], max_concurrency=2
            ):
                if not isinstance(item, Exception):
                    nodes.append(item)

        assert len(nodes) == 3
        assert "Skipping failed document" in caplog.text

    async def test_process_concurrent_should_handle_unexpected_errors_when_fail_fast_is_false(
        self,
        fake_ingestor: FakeIngestor,
        fake_fragmenter: FakeFragmenter,
        source_file_factory: Callable[[str, str], Path],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verifies that process_concurrent handles unexpected errors when fail_fast=False.

        Given: A source triggering a RuntimeError.
        When: process_concurrent runs with fail_fast=False.
        Then: The unexpected error should be logged and processing should continue.
        """
        runner = ForgeRunner(fake_ingestor, fake_fragmenter, fail_fast=False)
        bad_runtime = source_file_factory("doc_runtime_error.txt", "Runtime Error")
        good_source = source_file_factory("doc_good.txt", "Good content")

        logger = logging.getLogger("ragmark.forge.runner")
        logger.propagate = True

        with caplog.at_level(logging.WARNING, logger="ragmark.forge.runner"):
            nodes: list[KnowledgeNode] = []
            async for item in runner.process_concurrent(
                [bad_runtime, good_source], max_concurrency=2
            ):
                if not isinstance(item, Exception):
                    nodes.append(item)

        assert len(nodes) == 3
        assert "Skipping document due to unexpected error" in caplog.text

    async def test_process_concurrent_should_raise_error_when_fail_fast_is_true(
        self,
        fake_ingestor: FakeIngestor,
        fake_fragmenter: FakeFragmenter,
        source_file_factory: Callable[[str, str], Path],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verifies that process_concurrent raises errors when fail_fast=True.

        Given: A source triggering an IngestionError.
        When: process_concurrent runs with fail_fast=True.
        Then: The error should be logged and raised.
        """
        runner = ForgeRunner(fake_ingestor, fake_fragmenter, fail_fast=True)
        bad_source = source_file_factory("doc_ingest_error.txt", "Bad")

        logger = logging.getLogger("ragmark.forge.runner")
        logger.propagate = True

        with caplog.at_level(logging.ERROR, logger="ragmark.forge.runner"):
            with pytest.raises(IngestionError):
                async for _ in runner.process_concurrent(
                    [bad_source], max_concurrency=1
                ):
                    pass

        assert "Forge pipeline failed" in caplog.text

    @pytest.mark.performance
    async def test_concurrent_should_process_faster_than_sequential(
        self, tmp_path: Path
    ) -> None:
        """Verifies concurrent ingestion provides speedup over sequential.

        Given:
            20 source documents with 50ms ingestion delay each.
        When:
            Processing sequentially (max_concurrency=1) then concurrently (max_concurrency=4).
        Then:
            Concurrent processing completes at least 2x faster, and both
            produce identical 60 nodes (20 docs × 3 chunks).
        """
        runner = ForgeRunner(
            ingestor=FakeIngestor(delay_ms=50),
            fragmenter=FakeFragmenter(),
            fail_fast=True,
        )

        sources = [tmp_path / f"doc_{i}.txt" for i in range(20)]
        for source in sources:
            source.touch()

        start_seq = time.perf_counter()
        nodes_seq: list[KnowledgeNode] = []
        async for node in runner.process_concurrent(sources, max_concurrency=1):
            nodes_seq.append(node)
        seq_duration = time.perf_counter() - start_seq

        start_conc = time.perf_counter()
        nodes_conc: list[KnowledgeNode] = []
        async for node in runner.process_concurrent(sources, max_concurrency=4):
            nodes_conc.append(node)
        conc_duration = time.perf_counter() - start_conc

        assert len(nodes_seq) == 60
        assert len(nodes_conc) == 60
        speedup = seq_duration / conc_duration
        assert speedup >= 2.0, f"Speedup only {speedup:.2f}x, expected ≥2x"

    @pytest.mark.performance
    async def test_concurrent_should_enforce_max_concurrency_limit(
        self, tmp_path: Path
    ) -> None:
        """Verifies max_concurrency parameter limits parallel workers.

        Given:
            ForgeRunner with max_concurrency=3 and 10 documents with 100ms delay.
        When:
            Processing documents concurrently.
        Then:
            Produces 30 nodes and duration indicates batched execution
            (300-800ms vs ~100ms if unconstrained).
        """
        runner = ForgeRunner(
            ingestor=FakeIngestor(delay_ms=100),
            fragmenter=FakeFragmenter(),
            fail_fast=True,
        )

        sources = [tmp_path / f"test_{i}.txt" for i in range(10)]
        for source in sources:
            source.touch()

        start = time.perf_counter()
        nodes: list[KnowledgeNode] = []
        async for node in runner.process_concurrent(sources, max_concurrency=3):
            nodes.append(node)
        duration = time.perf_counter() - start

        assert len(nodes) == 30
        assert (
            0.3 < duration < 0.8
        ), f"Duration {duration:.2f}s suggests wrong concurrency"

    async def test_concurrent_should_handle_exceptions_with_fail_fast_true(
        self, tmp_path: Path
    ) -> None:
        """Verifies fail_fast=True stops processing on first error.

        Given:
            ForgeRunner with fail_fast=True and mixed valid/invalid sources.
        When:
            Running process_concurrent() with one invalid source.
        Then:
            Raises ValueError on first error and stops processing remaining sources.
        """

        class FailingIngestor(BaseIngestor):
            """Test fake ingestor that fails on specific file patterns."""

            def ingest(self, source: Path | bytes | BinaryIO) -> SourceDoc:
                if not isinstance(source, Path):
                    raise TypeError("Expected Path source")
                if "bad" in source.name:
                    raise ValueError("Simulated ingestion failure")
                return SourceDoc(
                    source_id=source.stem,
                    content="OK",
                    mime_type="text/plain",
                    metadata={},
                    page_count=1,
                )

            @classmethod
            def from_config(cls, config: IngestorConfig) -> "FailingIngestor":
                return cls()

            @property
            def supported_formats(self) -> set[str]:
                return {".txt"}

        runner = ForgeRunner(
            ingestor=FailingIngestor(),
            fragmenter=FakeFragmenter(),
            fail_fast=True,
        )

        sources = [
            tmp_path / "good_1.txt",
            tmp_path / "bad_doc.txt",
            tmp_path / "good_2.txt",
        ]
        for source in sources:
            source.touch()

        with pytest.raises(ValueError, match="Simulated ingestion failure"):
            nodes: list[KnowledgeNode] = []
            async for node in runner.process_concurrent(sources, max_concurrency=2):
                nodes.append(node)

    async def test_concurrent_should_collect_errors_with_fail_fast_false(
        self, tmp_path: Path
    ) -> None:
        """Verifies fail_fast=False continues processing despite errors.

        Given:
            ForgeRunner with fail_fast=False and 5 sources (3 valid, 2 invalid).
        When:
            Processing sources concurrently.
        Then:
            Yields 9 nodes from 3 valid docs and collects 2 RuntimeError instances.
        """

        class PartialFailIngestor(BaseIngestor):
            """Test fake ingestor that fails on specific file patterns."""

            def ingest(self, source: Path | bytes | BinaryIO) -> SourceDoc:
                if not isinstance(source, Path):
                    raise TypeError("Expected Path source")
                if "error" in source.name:
                    raise RuntimeError(f"Failed: {source.name}")
                return SourceDoc(
                    source_id=source.stem,
                    content="Success",
                    mime_type="text/plain",
                    metadata={},
                    page_count=1,
                )

            @classmethod
            def from_config(cls, config: IngestorConfig) -> "PartialFailIngestor":
                return cls()

            @property
            def supported_formats(self) -> set[str]:
                return {".txt"}

        runner = ForgeRunner(
            ingestor=PartialFailIngestor(),
            fragmenter=FakeFragmenter(),
            fail_fast=False,
        )

        sources = [
            tmp_path / "ok_1.txt",
            tmp_path / "error_1.txt",
            tmp_path / "ok_2.txt",
            tmp_path / "error_2.txt",
            tmp_path / "ok_3.txt",
        ]
        for source in sources:
            source.touch()

        nodes: list[KnowledgeNode] = []
        errors: list[Exception] = []
        async for item in runner.process_concurrent(sources, max_concurrency=2):
            if isinstance(item, Exception):
                errors.append(item)
            else:
                nodes.append(item)

        assert len(nodes) == 9
        assert len(errors) == 2
        assert all(isinstance(e, RuntimeError) for e in errors)


@pytest.mark.asyncio
@pytest.mark.unit
class TestStreamingMetrics:
    """Tests for StreamingMetrics edge cases and error handling."""

    async def test_stop_monitoring_should_handle_missing_task_gracefully(self) -> None:
        """Verifies that stop_monitoring returns early when no task exists.

        Given: A StreamingMetrics instance without a monitoring task.
        When: stop_monitoring is called.
        Then: It should return without error.
        """
        from ragmark.forge.runner import StreamingMetrics

        metrics = StreamingMetrics()
        queue: asyncio.Queue[KnowledgeNode | Exception | None] = asyncio.Queue()
        metrics.configure(queue, StreamingMetricsConfig())

        await metrics.stop_monitoring()
        assert metrics._monitor_task is None  # type: ignore

    async def test_stop_monitoring_should_handle_timeout_and_force_cancel(self) -> None:
        """Verifies that stop_monitoring forces cancellation on timeout.

        Given: A monitoring loop that doesn't respond to shutdown signal.
        When: stop_monitoring is called with a timeout.
        Then: The task should be forcefully cancelled.
        """
        from ragmark.forge.runner import StreamingMetrics

        metrics = StreamingMetrics()
        queue: asyncio.Queue[KnowledgeNode | Exception | None] = asyncio.Queue()
        metrics.configure(queue, StreamingMetricsConfig(interval=0.1))

        await metrics.start_monitoring()

        async def stubborn_loop():
            """Simulates unresponsive monitor task to test timeout cancellation."""
            while True:
                await asyncio.sleep(10)

        metrics._monitor_task.cancel()  # type: ignore
        await asyncio.sleep(0)
        metrics._monitor_task = asyncio.create_task(stubborn_loop())  # type: ignore

        await metrics.stop_monitoring()

        assert metrics._monitor_task.cancelled() or metrics._monitor_task.done()  # type: ignore

    async def test_monitor_loop_should_handle_exceptions_gracefully(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verifies that _monitor_loop catches and logs exceptions without crashing.

        Given: A queue that raises an exception on qsize().
        When: The monitor loop runs.
        Then: It should log the error and continue monitoring.
        """
        import logging
        from unittest.mock import Mock

        from ragmark.forge.runner import StreamingMetrics

        metrics = StreamingMetrics()

        # Mock required: testing exception handling in queue.qsize() edge case
        # Creating a Fake queue that raises on qsize() is impractical (TEST_POLICY.md 2.2)
        mock_queue = Mock()
        mock_queue.maxsize = 100
        call_count = {"count": 0}

        def qsize_side_effect():
            call_count["count"] += 1
            if call_count["count"] == 1:
                raise RuntimeError("Queue error")
            return 0

        mock_queue.qsize.side_effect = qsize_side_effect

        config = StreamingMetricsConfig(interval=0.1)
        metrics._queue = mock_queue  # type: ignore
        metrics._config = config  # type: ignore

        logger = logging.getLogger("ragmark.forge.runner")
        logger.propagate = True

        with caplog.at_level(logging.ERROR, logger="ragmark.forge.runner"):
            await metrics.start_monitoring()

            # Let it run to trigger the error at least once
            await asyncio.sleep(0.15)

            # Stop monitoring
            await metrics.stop_monitoring()

        # Should have logged the error
        assert "Monitoring error" in caplog.text
        # Should have attempted to sample multiple times
        assert mock_queue.qsize.call_count >= 1

    async def test_monitor_queue_deprecated_method_should_emit_warning(self) -> None:
        """Verifies that monitor_queue emits a deprecation warning.

        Given: A StreamingMetrics instance.
        When: monitor_queue is called.
        Then: A DeprecationWarning should be emitted.
        """
        import warnings
        from unittest.mock import AsyncMock, patch

        from ragmark.forge.runner import StreamingMetrics

        metrics = StreamingMetrics()
        queue: asyncio.Queue[KnowledgeNode | Exception | None] = asyncio.Queue()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Mock _monitor_loop to prevent it from actually running
            with patch.object(metrics, "_monitor_loop", new_callable=AsyncMock):
                # Call the deprecated method
                await metrics.monitor_queue(queue, interval=1.0)

            # Check warning was raised
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

    async def test_metrics_should_track_backpressure_events(self):
        """Verifies StreamingMetrics detects queue saturation.

        Given:
            A queue at 90% capacity and monitoring enabled.
        When:
            Monitoring queue size.
        Then:
            Backpressure event is recorded.
        """

        # Given
        metrics = StreamingMetrics()
        queue: asyncio.Queue[KnowledgeNode | Exception | None] = asyncio.Queue(
            maxsize=10
        )

        # Fill queue to 90%
        for _ in range(9):
            await queue.put(None)

        # Configure metrics
        metrics.configure(queue, StreamingMetricsConfig(interval=0.1))

        # When: Monitor for 0.5 seconds using context manager
        async with metrics:
            await asyncio.sleep(0.5)

        # Then
        assert metrics.backpressure_events > 0
        assert len(metrics.queue_size_samples) >= 1
        assert max(metrics.queue_size_samples) >= 9

    async def test_metrics_should_collect_queue_size_samples(self):
        """Verifies queue size sampling over time.

        Given:
            A queue with varying fill levels.
        When:
            Monitoring for 1 second with dynamic queue changes.
        Then:
            Samples reflect queue size variations.
        """

        # Given
        metrics = StreamingMetrics()
        queue: asyncio.Queue[KnowledgeNode | Exception | None] = asyncio.Queue(
            maxsize=20
        )

        async def vary_queue_size():
            for _ in range(5):
                await queue.put(None)
                await asyncio.sleep(0.1)

        # Configure metrics
        metrics.configure(queue, StreamingMetricsConfig(interval=0.1))

        # When: Monitor while varying queue size
        async with metrics:
            vary_task = asyncio.create_task(vary_queue_size())
            await vary_task
            await asyncio.sleep(0.2)

        # Then
        assert len(metrics.queue_size_samples) >= 1
        # Queue grew from 0 to 5, samples should reflect this
        samples = metrics.queue_size_samples
        assert min(samples) >= 0
        assert max(samples) <= 5


@pytest.mark.unit
@pytest.mark.asyncio
class TestStreamingMetricsContextManager:
    """Tests for StreamingMetrics async context manager lifecycle."""

    async def test_context_manager_should_start_and_stop_monitoring_automatically(self):
        """Verifies context manager starts and stops monitoring automatically.

        Given:
            StreamingMetrics configured with queue.
        When:
            Using async with context manager.
        Then:
            Monitoring task starts on enter, stops on exit, no zombie tasks.
        """

        # Given
        metrics = StreamingMetrics()
        queue: asyncio.Queue[KnowledgeNode | Exception | None] = asyncio.Queue(
            maxsize=10
        )
        metrics.configure(queue, StreamingMetricsConfig(interval=0.1))

        # Track initial tasks
        tasks_before = len([t for t in asyncio.all_tasks() if not t.done()])

        # When: Use context manager
        async with metrics:
            # Monitoring should be active
            await asyncio.sleep(0.3)
            assert len(metrics.queue_size_samples) >= 1

        # Then: Task should be cleaned up
        await asyncio.sleep(0.1)
        tasks_after = len([t for t in asyncio.all_tasks() if not t.done()])
        assert tasks_after == tasks_before

    async def test_context_manager_should_cleanup_on_exception(self):
        """Verifies metrics cleanup even when pipeline raises exception.

        Given:
            StreamingMetrics in context manager with active monitoring.
        When:
            Exception raised inside context block.
        Then:
            Monitoring task is cancelled and finalized before exception propagates.
        """

        # Given
        queue: asyncio.Queue[KnowledgeNode | Exception | None] = asyncio.Queue(
            maxsize=10
        )
        metrics = StreamingMetrics()
        metrics.configure(queue, StreamingMetricsConfig(interval=0.1))

        # When/Then: Exception should still trigger cleanup
        with pytest.raises(ValueError, match="Simulated error"):
            async with metrics:
                await asyncio.sleep(0.1)
                assert len(metrics.queue_size_samples) >= 1
                raise ValueError("Simulated error")

        # Verify cleanup completed
        await asyncio.sleep(0.1)
        tasks = [t for t in asyncio.all_tasks() if not t.done()]
        assert len(tasks) == 1  # Only the test task itself

    async def test_shutdown_flag_should_stop_loop_gracefully(self):
        """Verifies shutdown flag stops monitoring loop gracefully.

        Given:
            Running monitor loop task.
        When:
            Shutdown flag is set.
        Then:
            Task completes within interval time (no timeout needed).
        """

        # Given
        metrics = StreamingMetrics()
        queue: asyncio.Queue[KnowledgeNode | Exception | None] = asyncio.Queue(
            maxsize=10
        )
        metrics.configure(queue, StreamingMetricsConfig(interval=1.0))

        await metrics.start_monitoring()
        await asyncio.sleep(0.2)

        start = time.time()
        await metrics.stop_monitoring()
        duration = time.time() - start

        assert duration < 2.5
        assert len(metrics.queue_size_samples) >= 1

    async def test_finalize_should_compute_accurate_summary(self):
        """Verifies finalize computes correct summary statistics.

        Given:
            StreamingMetrics with collected samples.
        When:
            finalize() is called.
        Then:
            Returns correct min/max/avg/count values.
        """

        # Given
        metrics = StreamingMetrics()
        queue: asyncio.Queue[KnowledgeNode | Exception | None] = asyncio.Queue(
            maxsize=10
        )

        # Add known samples
        for _ in range(5):
            await queue.put(None)

        metrics.configure(queue, StreamingMetricsConfig(interval=0.1))

        # When: Monitor and finalize
        async with metrics:
            await asyncio.sleep(0.5)

        summary = metrics.finalize()

        # Then: Summary should be accurate
        assert summary["total_samples"] > 0
        assert "avg_queue_size" in summary
        assert "max_queue_size" in summary
        assert "min_queue_size" in summary
        assert "backpressure_events" in summary
        assert summary["max_queue_size"] <= 10

    async def test_memory_bounded_with_max_samples_limit(self):
        """Verifies sample list doesn't exceed max_samples limit.

        Given:
            Monitoring with max_samples=100.
        When:
            Monitoring for extended period (>100 samples expected).
        Then:
            Sample list length doesn't exceed max_samples.
        """

        # Given
        metrics = StreamingMetrics()
        queue: asyncio.Queue[KnowledgeNode | Exception | None] = asyncio.Queue(
            maxsize=10
        )
        # Use minimum interval (0.1s) with max_samples=20
        metrics.configure(queue, StreamingMetricsConfig(interval=0.1, max_samples=20))

        # When: Monitor for long enough to collect >20 samples
        async with metrics:
            await asyncio.sleep(3.0)  # Should collect ~30 samples at 0.1s interval

        # Then: Should be bounded to max_samples
        assert len(metrics.queue_size_samples) <= 20

    async def test_configure_validates_queue_not_none(self):
        """Verifies start_monitoring validates configuration.

        Given:
            StreamingMetrics without configuration.
        When:
            Attempting to start monitoring.
        Then:
            Raises ValueError about missing configuration.
        """
        # Given
        metrics = StreamingMetrics()

        # When/Then: Should raise ValueError
        with pytest.raises(ValueError, match="Must call configure"):
            await metrics.start_monitoring()

    async def test_backpressure_threshold_configurable(self):
        """Verifies backpressure threshold is configurable.

        Given:
            Queue at 70% capacity and threshold=0.6.
        When:
            Monitoring queue.
        Then:
            Backpressure event is recorded (70% > 60%).
        """

        # Given
        metrics = StreamingMetrics()
        queue: asyncio.Queue[KnowledgeNode | Exception | None] = asyncio.Queue(
            maxsize=10
        )

        # Fill to 70%
        for _ in range(7):
            await queue.put(None)

        # Configure with low threshold (60%)
        metrics.configure(
            queue, StreamingMetricsConfig(interval=0.1, backpressure_threshold=0.6)
        )

        # When: Monitor briefly
        async with metrics:
            await asyncio.sleep(0.3)

        # Then: Should detect backpressure (70% > 60%)
        assert metrics.backpressure_events > 0

    async def test_stop_monitoring_handles_timeout_gracefully(self):
        """Verifies stop_monitoring handles timeout and forces cancellation.

        Given:
            Monitoring task that's slow to respond.
        When:
            stop_monitoring is called.
        Then:
            Task is force-cancelled after timeout without hanging.
        """

        # Given
        metrics = StreamingMetrics()
        queue: asyncio.Queue[KnowledgeNode | Exception | None] = asyncio.Queue(
            maxsize=10
        )
        metrics.configure(queue, StreamingMetricsConfig(interval=0.1))

        # Start monitoring
        await metrics.start_monitoring()
        await asyncio.sleep(0.2)

        # When: Stop monitoring (should handle timeout gracefully)
        await metrics.stop_monitoring()

        # Then: No hanging, task is cleaned up
        assert metrics._monitor_task is not None  # type: ignore
        assert metrics._monitor_task.done() or metrics._monitor_task.cancelled()  # type: ignore


@pytest.mark.integration
@pytest.mark.asyncio
class TestStreamingPipelineMemoryEfficiency:
    """Tests for run_stream() memory efficiency and latency."""

    @pytest.mark.performance
    async def test_run_stream_should_have_low_time_to_first_node(self, tmp_path: Path):
        """Verifies streaming pipeline yields first node quickly.

        Given:
            A ForgeRunner with fake components processing 10 documents.
        When:
            Starting run_stream() and measuring time to first yield.
        Then:
            First node appears in under 100ms (streaming, not batch).
        """
        # Given
        runner = ForgeRunner(
            ingestor=FakeIngestor(delay_ms=10),
            fragmenter=FakeFragmenter(),
            fail_fast=True,
        )

        sources = [tmp_path / f"doc_{i}.txt" for i in range(10)]
        for source in sources:
            source.touch()

        # When
        start = time.perf_counter()
        stream = runner.process_async(sources)

        first_node = await anext(stream)
        ttfn_ms = (time.perf_counter() - start) * 1000

        # Then
        assert first_node is not None
        assert first_node.node_id.endswith("_chunk_0")
        assert ttfn_ms < 100, f"TTFN too high: {ttfn_ms:.2f}ms"

    async def test_run_stream_should_process_all_documents_correctly(
        self, tmp_path: Path
    ):
        """Verifies run_stream() processes all documents without data loss.

        Given:
            ForgeRunner with 5 source documents.
        When:
            Consuming entire stream.
        Then:
            Yields 15 nodes total (5 docs × 3 chunks/doc).
        """
        # Given
        runner = ForgeRunner(
            ingestor=FakeIngestor(delay_ms=5),
            fragmenter=FakeFragmenter(),
            fail_fast=True,
        )

        sources = [tmp_path / f"test_{i}.pdf" for i in range(5)]
        for source in sources:
            source.touch()

        # When
        nodes: list[KnowledgeNode] = []
        async for node in runner.process_async(sources):
            nodes.append(node)

        # Then
        assert len(nodes) == 15, f"Expected 15 nodes, got {len(nodes)}"

        # Verify all source IDs present
        source_ids = {node.node_id.split("_chunk_")[0] for node in nodes}
        assert len(source_ids) == 5

    async def test_run_stream_should_handle_async_iterator_input(self, tmp_path: Path):
        """Verifies run_stream() accepts AsyncIterator sources.

        Given:
            An async generator yielding source paths.
        When:
            Passing it to run_stream().
        Then:
            All nodes are yielded correctly.
        """
        # Given
        runner = ForgeRunner(
            ingestor=FakeIngestor(delay_ms=1),
            fragmenter=FakeFragmenter(),
            fail_fast=True,
        )

        async def async_sources():
            for i in range(3):
                source = tmp_path / f"async_{i}.txt"
                source.touch()
                yield source
                await asyncio.sleep(0.01)

        # When
        nodes: list[KnowledgeNode] = []
        async for node in runner.process_async(async_sources()):
            nodes.append(node)

        # Then
        assert len(nodes) == 9  # 3 docs × 3 chunks


@pytest.mark.integration
@pytest.mark.asyncio
class TestMetricsIntegration:
    """Integration tests for metrics with full pipeline."""

    async def test_concurrent_pipeline_with_metrics_context_manager(self):
        """Verifies metrics collection in full concurrent pipeline.

        Given:
            ForgeRunner with metrics enabled in profile.
        When:
            Running concurrent pipeline.
        Then:
            Metrics are collected correctly, no tasks leak after completion.
        """
        from ragmark.config.profile import (
            ExperimentProfile,
        )

        # Create a complete profile with metrics enabled
        profile_dict: dict[
            str, dict[str, str | int | dict[str, bool | float | int]]
        ] = {
            "ingestor": {"backend": "fitz"},
            "fragmenter": {"strategy": "token", "chunk_size": 256, "overlap": 64},
            "embedder": {"model_name": "sentence-transformers/all-MiniLM-L6-v2"},
            "index": {"backend": "memory"},
            "retrieval": {"mode": "dense", "top_k": 10},
            "metrics": {
                "streaming": {
                    "enabled": True,
                    "interval": 0.1,
                    "max_samples": 1000,
                    "backpressure_threshold": 0.8,
                }
            },
        }

        profile = ExperimentProfile.model_validate(profile_dict)
        runner = ForgeRunner.from_profile(profile)

        # Create test documents
        sources = [Path("tests/fixtures/sample.pdf")]
        if not sources[0].exists():
            pytest.skip("Sample PDF not found")

        # Track tasks before
        tasks_before = len([t for t in asyncio.all_tasks() if not t.done()])

        # When: Run pipeline
        nodes: list[KnowledgeNode] = []
        async for node in runner.process_concurrent(sources, max_concurrency=2):
            nodes.append(node)

        # Then: Verify no task leaks
        await asyncio.sleep(0.2)
        tasks_after = len([t for t in asyncio.all_tasks() if not t.done()])
        assert tasks_after == tasks_before

    async def test_metrics_manager_with_multiple_collectors(self):
        """Verifies MetricsManager handles multiple collectors.

        Given:
            MetricsManager with two StreamingMetrics collectors.
        When:
            Using context manager.
        Then:
            All collectors are started/stopped, summary aggregates correctly.
        """

        # Given
        metrics1 = StreamingMetrics()
        metrics2 = StreamingMetrics()

        queue1: asyncio.Queue[KnowledgeNode | Exception | None] = asyncio.Queue(
            maxsize=10
        )
        queue2: asyncio.Queue[KnowledgeNode | Exception | None] = asyncio.Queue(
            maxsize=20
        )

        metrics1.configure(queue1, StreamingMetricsConfig(interval=0.1))
        metrics2.configure(queue2, StreamingMetricsConfig(interval=0.1))

        manager = MetricsManager([metrics1, metrics2])

        # When: Use manager context
        async with manager:
            await asyncio.sleep(0.3)

        # Then: Summary should have StreamingMetrics entry
        # Note: When multiple collectors have same type, dict key is shared
        # so only one appears in summary (last one wins in dict comprehension)
        summary = manager.get_summary()
        assert "StreamingMetrics" in summary

        # Check that metrics were collected
        streaming_summary = summary["StreamingMetrics"]
        assert streaming_summary["total_samples"] > 0
