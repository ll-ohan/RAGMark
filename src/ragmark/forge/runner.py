"""Integrated pipeline for ingestion and fragmentation.

This module provides the ForgeRunner class which chains ingestion
and fragmentation into a single streaming pipeline.
"""

import time
from collections.abc import Iterable, Iterator
from pathlib import Path

from ragmark.config.profile import ExperimentProfile
from ragmark.forge.factory import FragmenterFactory, IngestorFactory
from ragmark.forge.fragmenters import BaseFragmenter
from ragmark.forge.ingestors import BaseIngestor
from ragmark.logger import get_logger
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
    """

    def __init__(
        self,
        ingestor: BaseIngestor,
        fragmenter: BaseFragmenter,
        fail_fast: bool = True,
    ):
        """Initialize the forge runner.

        Args:
            ingestor: Ingestor instance to use.
            fragmenter: Fragmenter instance to use.
            fail_fast: Whether to stop on first error or continue.
        """
        self.ingestor = ingestor
        self.fragmenter = fragmenter
        self.fail_fast = fail_fast

    @classmethod
    def from_profile(cls, profile: ExperimentProfile) -> "ForgeRunner":
        """Create a ForgeRunner directly from an experiment profile.

        This factory method simplifies initialization by extracting the relevant
        sub-configurations (ingestor, fragmenter, fail_fast) from the profile.

        Args:
            profile: The complete experiment configuration.

        Returns:
            A configured ForgeRunner instance.
        """
        logger.debug("Initializing ForgeRunner from profile...")

        ingestor = IngestorFactory.create(profile.ingestor)

        fragmenter = FragmenterFactory.create(profile.fragmenter)

        return cls(
            ingestor=ingestor,
            fragmenter=fragmenter,
            fail_fast=profile.fail_fast,
        )

    def run(self, sources: list[Path]) -> list[KnowledgeNode]:
        """Process documents and return all knowledge nodes.

        This method loads all results into memory and is suitable for
        small to medium document sets.

        Args:
            sources: The source file paths.

        Returns:
            List of all generated knowledge nodes.

        Raises:
            IngestionError: If ingestion fails (when fail_fast=True).
            FragmentationError: If fragmentation fails (when fail_fast=True).
        """
        return list(self.run_lazy(sources))

    def run_lazy(self, sources: Iterable[Path]) -> Iterator[KnowledgeNode]:
        """Process documents with streaming output (O(1) memory).

        This method uses generators throughout the pipeline, yielding
        nodes as they are created without loading everything into memory.

        The error handling respects fail_fast:
        - If fail_fast=True: Raises on first error
        - If fail_fast=False: Logs warning and continues with next file

        Args:
            sources: The source file paths.

        Yields:
            KnowledgeNode instances as they are created.

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

        for i, source in enumerate(sources):
            # Audit Spec: Log every 10 documents to avoid flooding
            if i % 10 == 0:
                logger.debug(
                    "Processing progress: documents=%d, success=%d, nodes=%d",
                    i,
                    doc_count,
                    node_count,
                )

            try:
                logger.debug("Ingesting document: source=%s", source)
                doc = self.ingestor.ingest(source)
                doc_count += 1

                logger.debug("Fragmenting document: source=%s", source)
                nodes = self.fragmenter.fragment(doc)

                for node in nodes:
                    yield node
                    node_count += 1

            except (IngestionError, FragmentationError) as e:
                error_count += 1
                if self.fail_fast:
                    logger.error(
                        "Forge pipeline failed: source=%s, error=%s", source, e
                    )
                    logger.debug("Pipeline failure details: %s", e, exc_info=True)
                    raise
                else:
                    logger.warning(
                        "Skipping failed document: source=%s, error_type=%s",
                        source,
                        e.__class__.__name__,
                    )
                    logger.debug("Skip reason: %s", e, exc_info=True)
                    continue

            except Exception as e:
                error_count += 1
                if self.fail_fast:
                    logger.error("Unexpected pipeline error: source=%s", source)
                    logger.debug("Unexpected error details: %s", e, exc_info=True)
                    raise
                else:
                    logger.warning(
                        "Skipping document due to unexpected error: source=%s", source
                    )
                    logger.debug("Unexpected error details: %s", e, exc_info=True)
                    continue

        duration = time.time() - start_time
        logger.info(
            "Forge pipeline complete: documents=%d, nodes=%d, duration=%.2fs, errors=%d",
            doc_count,
            node_count,
            duration,
            error_count,
        )
