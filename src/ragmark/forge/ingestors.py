"""Abstract base class and interfaces for document ingestion.

This module defines the contract that all ingestion backends must implement,
ensuring consistent handling of document extraction across different formats
and libraries.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

from ragmark.schemas.documents import SourceDoc

if TYPE_CHECKING:
    from ragmark.config.profile import IngestorConfig


class BaseIngestor(ABC):
    """Abstract base class for document ingestion backends.

    Ingestors are responsible for extracting raw text content from various
    file formats (PDF, DOCX, HTML, etc.) and packaging it into SourceDoc
    instances with appropriate metadata.

    All implementations must support streaming ingestion via generators
    to enable memory-efficient batch processing.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: "IngestorConfig") -> "BaseIngestor":
        """Instantiate ingestor from configuration.

        Args:
            config: IngestorConfig instance.

        Returns:
            Configured ingestor instance.
        """
        pass

    @abstractmethod
    def ingest(self, source: Path | bytes | BinaryIO) -> SourceDoc:
        """Ingest a single document source.

        Args:
            source: Document source (file path, bytes, or file-like object).

        Returns:
            Extracted SourceDoc with content and metadata.

        Raises:
            IngestionError: If document extraction fails.
            UnsupportedFormatError: If file format is not supported.
        """
        pass

    def ingest_batch(self, sources: Iterable[Path]) -> Iterator[SourceDoc]:
        """Ingest multiple documents with streaming output.

        This method processes documents one at a time and yields results
        immediately, ensuring O(1) memory consumption regardless of batch size.

        Args:
            sources: Iterable of file paths to ingest.

        Yields:
            SourceDoc instances as they are extracted.

        Raises:
            IngestionError: If any document extraction fails (behavior depends
                on fail_fast configuration).
        """
        for source_path in sources:
            yield self.ingest(source_path)

    @property
    @abstractmethod
    def supported_formats(self) -> set[str]:
        """Get the set of supported file extensions.

        Returns:
            Set of file extensions (including the dot, e.g., {'.pdf', '.docx'}).
        """
        pass

    def supports_format(self, path: Path) -> bool:
        """Check if this ingestor supports a given file format.

        Args:
            path: File path to check.

        Returns:
            True if the file extension is supported.
        """
        return path.suffix.lower() in self.supported_formats
