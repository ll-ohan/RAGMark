"""Integrated pipeline for ingestion and fragmentation.

This module provides the ForgeRunner class which chains ingestion
and fragmentation into a single streaming pipeline.
"""

import logging
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

from ragmark.config.profile import ExperimentProfile
from ragmark.forge.factory import FragmenterFactory, IngestorFactory
from ragmark.forge.fragmenters import BaseFragmenter
from ragmark.forge.ingestors import BaseIngestor
from ragmark.schemas.documents import KnowledgeNode

logger = logging.getLogger(__name__)


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
            sources: List of source file paths.

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

        Args:
            sources: Iterable of source file paths.

        Yields:
            KnowledgeNode instances as they are created.

        Raises:
            IngestionError: If ingestion fails (when fail_fast=True).
            FragmentationError: If fragmentation fails (when fail_fast=True).
        """
        start_time = time.time()
        doc_count: int | str = 0
        node_count = 0

        logger.info("Starting Forge pipeline")

        try:
            # Chain ingestion and fragmentation
            docs = self.ingestor.ingest_batch(sources)
            nodes = self.fragmenter.fragment_batch(docs)

            for node in nodes:
                yield node
                node_count += 1

                # Track document transitions (approximate)
                if node_count % 100 == 0:
                    logger.debug(f"Processed {node_count} nodes so far")

            doc_count = len(list(sources)) if isinstance(sources, list) else "unknown"

        except Exception as e:
            if self.fail_fast:
                logger.error(f"Forge pipeline failed: {e}")
                raise
            else:
                logger.warning(f"Skipping failed document: {e}")

        duration = time.time() - start_time
        logger.info(
            f"Forge pipeline complete: {doc_count} documents → {node_count} nodes "
            f"in {duration:.2f}s"
        )
