"""Integrated pipeline for ingestion and fragmentation.

This module provides the ForgeRunner class which chains ingestion
and fragmentation into a single streaming pipeline.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Iterable, Iterator
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, cast

from ragmark.config.profile import ExperimentProfile, StreamingMetricsConfig
from ragmark.forge.factory import (
    FragmenterFactory,
    IngestorFactory,
    QuestionGeneratorFactory,
)
from ragmark.forge.fragmenters import BaseFragmenter
from ragmark.forge.ingestors import BaseIngestor
from ragmark.forge.question_generator import BaseQuestionGenerator
from ragmark.logger import get_logger
from ragmark.metrics.base import MonitoringMetric
from ragmark.metrics.reporting import export_monitoring_summary
from ragmark.schemas.documents import KnowledgeNode

logger = get_logger(__name__)


class ForgeRunner:
    """Integrated pipeline for document ingestion and fragmentation.

    This class chains an ingestor and fragmenter into a single pipeline,
    handling the complete preprocessing workflow from raw documents to
    indexed knowledge nodes.

    Attributes:
        ingestor: Document ingestion backend.
        fragmenter: Text fragmentation strategy.
        fail_fast: If True, stop on first error. If False, skip failed documents.
        question_generator: Optional synthetic QA generator.
    """

    def __init__(
        self,
        ingestor: BaseIngestor,
        fragmenter: BaseFragmenter,
        fail_fast: bool = True,
        profile: ExperimentProfile | None = None,
        question_generator: BaseQuestionGenerator | None = None,
        monitoring: MonitoringMetric | None = None,
    ):
        """Initialize the forge runner.

        Args:
            ingestor: Ingestor instance to use.
            fragmenter: Fragmenter instance to use.
            fail_fast: Whether to stop on first error or continue.
            profile: Optional experiment profile for metrics configuration.
            question_generator: Optional question generator for synthetic QA.
            monitoring: Optional monitoring orchestrator.
        """
        self.ingestor = ingestor
        self.fragmenter = fragmenter
        self.fail_fast = fail_fast
        self.profile = profile
        self.question_generator = question_generator
        self.monitoring = monitoring

    def _handle_document_error(
        self,
        source: Path,
        error: Exception,
        is_expected: bool = False,
    ) -> None:
        """Handle document processing error based on fail_fast mode.

        Args:
            source: Failed source file path.
            error: Exception raised during processing.
            is_expected: True for Ingestion/FragmentationError, False otherwise.
        """

        if is_expected:
            if self.fail_fast:
                logger.error(
                    "Forge pipeline failed: source=%s, error=%s", source, error
                )
                logger.debug("Pipeline failure details: %s", error, exc_info=True)
                raise
            else:
                logger.warning(
                    "Skipping failed document: source=%s, error_type=%s",
                    source,
                    error.__class__.__name__,
                )
                logger.debug("Skip reason: %s", error, exc_info=True)
        else:
            if self.fail_fast:
                logger.error("Unexpected pipeline error: source=%s", source)
                logger.debug("Unexpected error details: %s", error, exc_info=True)
                raise
            else:
                logger.warning(
                    "Skipping document due to unexpected error: source=%s", source
                )
                logger.debug("Unexpected error details: %s", error, exc_info=True)

    def _log_progress(
        self,
        current_index: int,
        doc_count: int,
        node_count: int,
        interval: int = 10,
    ) -> None:
        """Log processing progress at regular intervals.

        Args:
            current_index: Current document index (0-based).
            doc_count: Successfully processed documents.
            node_count: Nodes generated.
            interval: Logging frequency in documents.
        """
        if current_index % interval == 0:
            logger.debug(
                "Processing progress: documents=%d, success=%d, nodes=%d",
                current_index,
                doc_count,
                node_count,
            )

    def _log_completion(
        self,
        start_time: float,
        doc_count: int,
        node_count: int,
        error_count: int,
        pipeline_type: str = "sync",
    ) -> None:
        """Log pipeline completion summary with metrics.

        Args:
            start_time: Start timestamp from time.time().
            doc_count: Successfully processed documents.
            node_count: Generated nodes.
            error_count: Errors encountered.
            pipeline_type: Pipeline variant for logging.
        """
        duration = time.time() - start_time

        pipeline_name = {
            "sync": "Forge pipeline",
            "async": "Async Forge pipeline",
            "concurrent": "Concurrent Forge pipeline",
        }.get(pipeline_type, "Forge pipeline")

        logger.info(
            "%s complete: documents=%d, nodes=%d, duration=%.2fs, errors=%d",
            pipeline_name,
            doc_count,
            node_count,
            duration,
            error_count,
        )

    @classmethod
    def from_profile(cls, profile: ExperimentProfile) -> "ForgeRunner":
        """Create ForgeRunner from experiment profile.

        Args:
            profile: Complete experiment configuration.

        Returns:
            Configured ForgeRunner instance.
        """
        logger.debug("Initializing ForgeRunner from profile...")

        ingestor = IngestorFactory.create(profile.ingestor)

        fragmenter = FragmenterFactory.create(profile.fragmenter)

        question_generator = None
        if profile.question_generator and profile.question_generator.enabled:
            logger.debug(
                "Creating question generator: backend=%s",
                profile.question_generator.backend,
            )
            question_generator = QuestionGeneratorFactory.create(
                profile.question_generator
            )

        monitoring = None
        if (
            profile.metrics
            and profile.metrics.monitoring
            and profile.metrics.monitoring.enabled
        ):
            monitoring = MonitoringMetric(enabled=True)

        return cls(
            ingestor=ingestor,
            fragmenter=fragmenter,
            fail_fast=profile.fail_fast,
            profile=profile,
            question_generator=question_generator,
            monitoring=monitoring,
        )

    def process(
        self,
        sources: list[Path],
        monitoring: MonitoringMetric | None = None,
    ) -> list[KnowledgeNode]:
        """Process documents and return all knowledge nodes in memory.

        Suitable for small to medium document sets where memory is not constrained.

        Args:
            sources: Source file paths.
            monitoring: Optional monitoring orchestrator.

        Returns:
            All generated knowledge nodes.

        Raises:
            IngestionError: If ingestion fails (when fail_fast=True).
            FragmentationError: If fragmentation fails (when fail_fast=True).
        """
        return list(self.process_iter(sources, monitoring=monitoring))

    def process_iter(
        self,
        sources: Iterable[Path],
        monitoring: MonitoringMetric | None = None,
    ) -> Iterator[KnowledgeNode]:
        """Process documents with streaming output (O(1) memory per node).

        Error handling respects fail_fast mode:
        - fail_fast=True: Raises on first error
        - fail_fast=False: Logs warning and continues

        Args:
            sources: Source file paths.
            monitoring: Optional monitoring orchestrator.

        Yields:
            Knowledge nodes as generated.

        Raises:
            IngestionError: If ingestion fails (when fail_fast=True).
            FragmentationError: If fragmentation fails (when fail_fast=True).
        """
        from ragmark.exceptions import FragmentationError, IngestionError

        start_time = time.time()
        doc_count = 0
        node_count = 0
        error_count = 0

        logger.info("Starting Forge pipeline")

        monitor = monitoring or self.monitoring

        for i, source in enumerate(sources):
            self._log_progress(i, doc_count, node_count)

            try:
                logger.debug("Ingesting document: source=%s", source)
                if monitor:
                    with monitor.stage("parsing"):
                        doc = self.ingestor.ingest(source)
                else:
                    doc = self.ingestor.ingest(source)
                doc_count += 1

                logger.debug("Fragmenting document: source=%s", source)
                if monitor:
                    with monitor.stage("chunking"):
                        for node in self.fragmenter.fragment_lazy(doc):
                            yield node
                            node_count += 1
                else:
                    for node in self.fragmenter.fragment_lazy(doc):
                        yield node
                        node_count += 1

            except (IngestionError, FragmentationError) as e:
                error_count += 1
                self._handle_document_error(source, e, is_expected=True)
                if not self.fail_fast:
                    continue

            except Exception as e:
                error_count += 1
                self._handle_document_error(source, e, is_expected=False)
                if not self.fail_fast:
                    continue

        self._log_completion(start_time, doc_count, node_count, error_count, "sync")
        export_monitoring_summary(monitor, artifact_prefix="forge")

    async def process_async(
        self,
        sources: AsyncIterator[Path] | Iterable[Path],
        monitoring: MonitoringMetric | None = None,
    ) -> AsyncIterator[KnowledgeNode]:
        """Process documents with async streaming (O(1) memory, non-blocking I/O).

        Uses asyncio.to_thread for I/O-bound ingestion to prevent event loop blocking.
        Optionally enriches nodes with synthetic QA pairs if question_generator is configured.

        Args:
            sources: Source file paths as async or sync iterable.
            monitoring: Optional monitoring orchestrator.

        Yields:
            Knowledge nodes as generated, optionally enriched with synthetic QA.

        Raises:
            IngestionError: If ingestion fails (when fail_fast=True).
            FragmentationError: If fragmentation fails (when fail_fast=True).
            QuestionGenerationError: If QA generation fails (when fail_fast=True).
        """
        from ragmark.exceptions import (
            FragmentationError,
            IngestionError,
            QuestionGenerationError,
        )

        start_time = time.time()
        doc_count = 0
        node_count = 0
        qa_enriched_count = 0
        error_count = 0

        monitor = monitoring or self.monitoring

        logger.info(
            "Starting async Forge pipeline with QA generation=%s",
            bool(self.question_generator),
        )

        source_iter = self._to_async_iter(sources)

        async def node_stream() -> AsyncIterator[KnowledgeNode]:
            """Create node stream from ingestion and fragmentation."""
            nonlocal doc_count, error_count
            i = 0
            async for source in source_iter:
                self._log_progress(i, doc_count, node_count)

                try:
                    logger.debug("Ingesting document: source=%s", source)

                    if monitor:
                        async with monitor.stage("parsing"):
                            doc = await asyncio.to_thread(self.ingestor.ingest, source)
                    else:
                        doc = await asyncio.to_thread(self.ingestor.ingest, source)
                    doc_count += 1

                    logger.debug("Fragmenting document: source=%s", source)
                    if monitor:
                        async with monitor.stage("chunking"):
                            for node in self.fragmenter.fragment_lazy(doc):
                                yield node
                    else:
                        for node in self.fragmenter.fragment_lazy(doc):
                            yield node

                except (IngestionError, FragmentationError) as e:
                    error_count += 1
                    self._handle_document_error(source, e, is_expected=True)
                    if self.fail_fast:
                        raise

                except Exception as e:
                    error_count += 1
                    self._handle_document_error(source, e, is_expected=False)
                    if self.fail_fast:
                        raise

                i += 1

        if self.question_generator:
            batch_size = getattr(self.question_generator, "_batch_size", 4)
            logger.debug("Applying QA generation: batch_size=%d", batch_size)

            try:
                enriched_stream = self.question_generator.generate_stream_async(
                    node_stream(), batch_size=batch_size
                )

                if monitor:
                    async with monitor.stage("qa_generation"):
                        async for enriched_node in enriched_stream:
                            if "synthetic_qa" in enriched_node.metadata:
                                qa_enriched_count += 1
                            yield enriched_node
                            node_count += 1
                else:
                    async for enriched_node in enriched_stream:
                        if "synthetic_qa" in enriched_node.metadata:
                            qa_enriched_count += 1
                        yield enriched_node
                        node_count += 1

            except QuestionGenerationError as e:
                error_count += 1
                logger.error("QA generation failed in pipeline: %s", e.message)
                logger.debug("QA generation error details: %s", e, exc_info=True)
                if self.fail_fast:
                    raise
        else:
            async for node in node_stream():
                yield node
                node_count += 1

        logger.info(
            "Async Forge pipeline complete: documents=%d, nodes=%d, qa_enriched=%d, "
            "duration=%.2fs, errors=%d",
            doc_count,
            node_count,
            qa_enriched_count,
            time.time() - start_time,
            error_count,
        )
        export_monitoring_summary(monitor, artifact_prefix="forge_async")

    async def process_concurrent(
        self,
        sources: Iterable[Path],
        max_concurrency: int = 4,
        monitoring: MonitoringMetric | None = None,
    ) -> AsyncIterator[KnowledgeNode]:
        """Process documents with concurrent ingestion (3-5x speedup, bounded memory).

        Parallel workers accelerate I/O-bound ingestion. Queue-based backpressure
        prevents unbounded memory growth.

        Args:
            sources: Source file paths.
            max_concurrency: Worker pool size.
            monitoring: Optional monitoring orchestrator.

        Yields:
            Knowledge nodes as generated.

        Raises:
            IngestionError: If ingestion fails (when fail_fast=True).
            FragmentationError: If fragmentation fails (when fail_fast=True).
        """
        from ragmark.exceptions import FragmentationError, IngestionError

        start_time = time.time()
        doc_count = 0
        node_count = 0
        error_count = 0

        logger.info(
            "Starting concurrent Forge pipeline: max_concurrency=%d", max_concurrency
        )

        monitor = monitoring or self.monitoring
        semaphore = asyncio.Semaphore(max_concurrency)

        queue: asyncio.Queue[KnowledgeNode | Exception | None] = asyncio.Queue(
            maxsize=100
        )

        metrics: StreamingMetrics | None = None
        if self.profile and self.profile.metrics and self.profile.metrics.streaming:
            streaming_config = self.profile.metrics.streaming
            if streaming_config.enabled:
                metrics = StreamingMetrics()
                metrics.configure(queue, streaming_config)

        async def worker(source: Path) -> None:
            """Process single document with concurrency limit."""
            nonlocal doc_count, error_count

            async with semaphore:
                try:
                    logger.debug("Ingesting document: source=%s", source)

                    if monitor:
                        async with monitor.stage("parsing"):
                            doc = await asyncio.to_thread(self.ingestor.ingest, source)
                    else:
                        doc = await asyncio.to_thread(self.ingestor.ingest, source)
                    doc_count += 1

                    logger.debug("Fragmenting document: source=%s", source)
                    if monitor:
                        async with monitor.stage("chunking"):
                            for node in self.fragmenter.fragment_lazy(doc):
                                await queue.put(node)
                    else:
                        for node in self.fragmenter.fragment_lazy(doc):
                            await queue.put(node)

                except (IngestionError, FragmentationError) as e:
                    error_count += 1
                    if self.fail_fast:
                        logger.error(
                            "Forge pipeline failed: source=%s, error=%s", source, e
                        )
                        logger.debug("Pipeline failure details: %s", e, exc_info=True)
                    else:
                        logger.warning(
                            "Skipping failed document: source=%s, error_type=%s",
                            source,
                            e.__class__.__name__,
                        )
                        logger.debug("Skip reason: %s", e, exc_info=True)
                    await queue.put(e)

                except Exception as e:
                    error_count += 1
                    if self.fail_fast:
                        logger.error("Unexpected pipeline error: source=%s", source)
                        logger.debug("Unexpected error details: %s", e, exc_info=True)
                    else:
                        logger.warning(
                            "Skipping document due to unexpected error: source=%s",
                            source,
                        )
                        logger.debug("Unexpected error details: %s", e, exc_info=True)
                    await queue.put(e)

        tasks = [asyncio.create_task(worker(src)) for src in sources]

        async def finalize() -> None:
            await asyncio.gather(*tasks, return_exceptions=not self.fail_fast)
            await queue.put(None)

        asyncio.create_task(finalize())

        async with AsyncExitStack() as stack:
            if metrics:
                await stack.enter_async_context(metrics)

            while True:
                item = await queue.get()

                if item is None:
                    break

                if isinstance(item, Exception):
                    if self.fail_fast:
                        raise item
                    # When not failing fast, we yield the error to the caller.
                    # We don't increment node_count for errors.
                else:
                    node_count += 1

                yield cast(KnowledgeNode, item)

        self._log_completion(
            start_time, doc_count, node_count, error_count, "concurrent"
        )
        export_monitoring_summary(monitor, artifact_prefix="forge_concurrent")

    async def _to_async_iter(
        self, sources: AsyncIterator[Path] | Iterable[Path]
    ) -> AsyncIterator[Path]:
        """Convert sync or async iterable to async iterator.

        Args:
            sources: Source paths as async iterator or sync iterable.

        Yields:
            Source paths one at a time.
        """
        if isinstance(sources, AsyncIterator):
            async for source in sources:
                yield source
        else:
            for source in sources:
                yield source


class StreamingMetrics(MonitoringMetric):
    """Metrics tracker for streaming pipeline health.

    Monitor queue sizes to detect backpressure when consumers fall behind
    producers, enabling performance tuning.

    Attributes:
        queue_size_samples: Historical queue size measurements.
        backpressure_events: Count of threshold exceedances.
    """

    def __init__(self) -> None:
        """Initialize metrics tracker.

        Call configure() before entering context manager.
        """
        super().__init__(enabled=True)
        self.queue_size_samples: list[int] = []
        self.backpressure_events: int = 0
        self._shutdown_flag: asyncio.Event | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[KnowledgeNode | Exception | None] | None = None
        self._config: StreamingMetricsConfig | None = None

    def configure(
        self,
        queue: asyncio.Queue[KnowledgeNode | Exception | None],
        config: StreamingMetricsConfig,
    ) -> None:
        """Configure metrics with queue and settings.

        Must be called before entering context manager.

        Args:
            queue: Queue to monitor for backpressure.
            config: Monitoring parameters.
        """
        self._queue = queue
        self._config = config

    async def start_monitoring(self) -> None:
        """Start background queue monitoring task.

        Raises:
            ValueError: If configure() was not called.
        """
        if self._queue is None or self._config is None:
            raise ValueError("Must call configure() before starting monitoring")

        self._shutdown_flag = asyncio.Event()
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.debug("Queue monitoring started: interval=%s", self._config.interval)

    async def stop_monitoring(self) -> None:
        """Stop monitoring task with timeout fallback.

        Wait for graceful completion with 2s timeout, then force cancellation.
        """
        if not self._monitor_task or not self._shutdown_flag:
            return

        self._shutdown_flag.set()

        try:
            await asyncio.wait_for(self._monitor_task, timeout=2.0)
            logger.debug("Monitoring task completed gracefully")
        except asyncio.TimeoutError:
            logger.warning("Monitoring task timeout, forcing cancellation")
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                logger.debug("Monitoring task cancelled")

    async def _monitor_loop(self) -> None:
        """Internal monitoring loop with bounded execution.

        Sample queue size, detect backpressure, and enforce memory bounds.
        """
        assert self._shutdown_flag is not None
        assert self._queue is not None
        assert self._config is not None

        while not self._shutdown_flag.is_set():
            try:
                size = self._queue.qsize()
                self.queue_size_samples.append(size)

                if len(self.queue_size_samples) > self._config.max_samples:
                    self.queue_size_samples = self.queue_size_samples[
                        -self._config.max_samples :
                    ]

                if size > self._queue.maxsize * self._config.backpressure_threshold:
                    self.backpressure_events += 1
                    logger.warning(
                        "Backpressure detected: queue=%d/%d, consumer_slow=true",
                        size,
                        self._queue.maxsize,
                    )

                try:
                    await asyncio.wait_for(
                        self._shutdown_flag.wait(), timeout=self._config.interval
                    )
                    break
                except asyncio.TimeoutError:
                    continue

            except Exception as e:
                logger.error("Monitoring error: %s", e, exc_info=True)

    def finalize(self) -> dict[str, Any]:
        """Compute summary statistics from collected samples.

        Returns:
            Summary metrics dictionary.
        """
        if not self.queue_size_samples:
            return {
                "total_samples": 0,
                "backpressure_events": self.backpressure_events,
            }

        return {
            "total_samples": len(self.queue_size_samples),
            "avg_queue_size": sum(self.queue_size_samples)
            / len(self.queue_size_samples),
            "max_queue_size": max(self.queue_size_samples),
            "min_queue_size": min(self.queue_size_samples),
            "backpressure_events": self.backpressure_events,
        }

    async def monitor_queue(
        self, queue: asyncio.Queue[Any], interval: float = 1.0
    ) -> None:
        """Monitor queue size for backpressure detection (deprecated).

        This method is provided for backward compatibility with existing tests.
        New code should use the context manager pattern instead.

        Deprecated:
            Use async context manager instead:
            >>> async with StreamingMetrics() as metrics:
            ...     metrics.configure(queue, StreamingMetricsConfig(interval=1.0))

        Args:
            queue: Queue to monitor.
            interval: Sampling interval in seconds.
        """
        import warnings

        warnings.warn(
            "monitor_queue() is deprecated. Use context manager pattern instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        from ragmark.config.profile import StreamingMetricsConfig

        self.configure(queue, StreamingMetricsConfig(interval=interval))

        await self._monitor_loop()
