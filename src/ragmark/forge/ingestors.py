"""Abstract base class and interfaces for document ingestion.

This module defines the contract that all ingestion backends must implement,
ensuring consistent handling of document extraction across different formats
and libraries.
"""

import errno
import os
import shutil
import tempfile
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, cast

from ragmark.logger import get_logger
from ragmark.schemas.documents import SourceDoc

if TYPE_CHECKING:
    from ragmark.config.profile import IngestorConfig

logger = get_logger(__name__)


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
        """Instantiate an ingestor based on the provided configuration.

        Args:
            config: The ingestion configuration.

        Returns:
            An initialized ingestor instance.
        """
        pass

    @abstractmethod
    def ingest(self, source: Path | bytes | BinaryIO) -> SourceDoc:
        """Ingest a single document source.

        Args:
            source: The input document.

        Returns:
            The extracted document content and metadata.

        Raises:
            IngestionError: If the document cannot be processed.
        """
        pass

    def ingest_batch(self, sources: Iterable[Path]) -> Iterator[SourceDoc]:
        """Ingest multiple documents with streaming output.

        Args:
            sources: An iterable of file paths to process.

        Yields:
            SourceDoc objects as they are successfully ingested.
        """
        for source_path in sources:
            yield self.ingest(source_path)

    @property
    @abstractmethod
    def supported_formats(self) -> set[str]:
        """Get the set of supported file extensions.

        Returns:
            A set of lowercase file extensions (e.g., {'.pdf', '.docx'}).
        """
        pass

    def supports_format(self, path: Path) -> bool:
        """Check if this ingestor supports a given file format.

        Args:
            path: The file path to check.

        Returns:
            True if the file extension is supported, False otherwise.
        """
        return path.suffix.lower() in self.supported_formats


class FitzIngestor(BaseIngestor):
    """PyMuPDF-based document ingestor for fast PDF extraction.

    This ingestor uses the PyMuPDF (fitz) library to extract text from
    PDFs and other supported formats. It's lightweight and fast, suitable
    for PDFs with selectable text (not scanned documents).

    Supported formats: PDF, XPS, EPUB, MOBI, FB2.

    Attributes:
        extract_images: Whether to extract images from the document.
    """

    def __init__(self, extract_images: bool = False):
        """Initialize the FitzIngestor.

        Args:
            extract_images: Whether to extract images from the document.
        """
        self.extract_images = extract_images

    @classmethod
    def from_config(cls, config: "IngestorConfig") -> "FitzIngestor":
        """Create a FitzIngestor instance from configuration.

        Args:
            config: The ingestion configuration.

        Returns:
            An initialized FitzIngestor.
        """
        return cls(extract_images=config.options.get("extract_images", False))

    def ingest(self, source: Path | bytes | BinaryIO) -> SourceDoc:
        """Ingest a document using PyMuPDF.

        Extract text page by page with page markers for position tracking.
        Extract metadata including title, author, page count, and TOC.

        Args:
            source: The input document.

        Returns:
            The extracted document content and metadata.

        Raises:
            IngestionError: If PyMuPDF is not installed, document is corrupted,
                encrypted, or extraction fails.
        """
        from ragmark.exceptions import IngestionError

        source_type = type(source).__name__
        logger.debug("Ingestion started: source_type=%s", source_type)

        # Lazy import fitz to avoid heavy dependency at module level
        try:
            import fitz
        except ImportError as e:
            logger.error("PyMuPDF not installed")
            logger.debug("Import error details: %s", e, exc_info=True)
            raise IngestionError(
                "PyMuPDF (fitz) not installed. Install with: pip install ragmark[ingest]",
                cause=e,
            ) from e

        doc: Any = None
        source_path: str | None = None
        temp_file_path: str | None = None

        try:
            if isinstance(source, Path):
                try:
                    logger.debug("Opening document from path: %s", source)
                    doc = cast(Any, fitz.open(source))
                    source_path = str(source)

                    # fitz.open() returns an object even for invalid files (check validity via bool)
                    if doc is not None and not bool(doc):
                        corruption_detail = RuntimeError(
                            f"Document has invalid structure: bool(doc)={bool(doc)}, "
                            f"page_count={len(doc)}, xref_length={doc.xref_length()}"
                        )
                        raise IngestionError(
                            f"Document at {source} appears to be corrupted or has invalid structure",
                            source_path=str(source),
                            cause=corruption_detail,
                        ) from corruption_detail
                except IngestionError as e:
                    logger.error("Document ingestion failed: source=%s", source)
                    logger.debug("Ingestion error details: %s", e, exc_info=True)
                    raise
                except Exception as e:
                    logger.error("Failed to open document: path=%s", source)
                    logger.debug("Open error details: %s", e, exc_info=True)
                    raise IngestionError(
                        f"Failed to open file at {source}",
                        source_path=str(source),
                        cause=e,
                    ) from e

            elif isinstance(source, bytes | bytearray | memoryview):
                try:
                    logger.debug(
                        "Writing bytes to temporary file: size=%d", len(source)
                    )
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp:
                        tmp.write(source)
                        tmp.flush()
                        temp_file_path = tmp.name

                    logger.debug(
                        "Opening document from temporary file: %s", temp_file_path
                    )
                    doc = cast(Any, fitz.open(temp_file_path))
                    source_path = "<bytes>"

                    if doc is not None and not bool(doc):
                        corruption_detail = RuntimeError(
                            f"Document has invalid structure: bool(doc)={bool(doc)}, "
                            f"page_count={len(doc)}, xref_length={doc.xref_length()}"
                        )
                        raise IngestionError(
                            "Provided bytes contain corrupted document or have invalid structure",
                            source_path=source_path,
                            cause=corruption_detail,
                        ) from corruption_detail
                except OSError as e:
                    msg = "Failed to write bytes to temporary file"
                    if e.errno == errno.ENOSPC:
                        msg = "Disk full: failed to write temporary ingestion file"
                    raise IngestionError(msg, cause=e) from e
                except IngestionError:
                    raise

            elif hasattr(source, "read"):
                try:
                    logger.debug("Streaming content to temporary file")
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp:
                        # Cast to BinaryIO to satisfy type checkers after hasattr check
                        shutil.copyfileobj(source, tmp)
                        tmp.flush()
                        temp_file_path = tmp.name

                    logger.debug(
                        "Opening document from stream: temp_file=%s", temp_file_path
                    )
                    doc = cast(Any, fitz.open(temp_file_path))
                    source_path = "<stream>"

                    if doc is not None and not bool(doc):
                        corruption_detail = RuntimeError(
                            f"Document has invalid structure: bool(doc)={bool(doc)}, "
                            f"page_count={len(doc)}, xref_length={doc.xref_length()}"
                        )
                        raise IngestionError(
                            "Stream contains corrupted document or has invalid structure",
                            source_path=source_path,
                            cause=corruption_detail,
                        ) from corruption_detail
                except (ValueError, OSError) as e:
                    # ValueError occurs if stream is closed
                    msg = "Failed to stream content to temporary file"
                    if isinstance(e, ValueError) and "closed" in str(e):
                        msg = "Attempted to read from a closed stream"
                    elif isinstance(e, OSError) and e.errno == errno.ENOSPC:
                        msg = "Disk full: failed to stream temporary ingestion file"

                    raise IngestionError(msg, cause=e) from e
                except IngestionError:
                    raise

            if doc is None:
                logger.error("Document initialization failed")
                raise IngestionError(
                    "Invalid document source: unable to initialize document object"
                )

            if doc.is_encrypted:
                logger.warning("Document is encrypted: source=%s", source_path)
                raise IngestionError(
                    "Document is encrypted (password protected). Please decrypt before ingestion.",
                    source_path=source_path,
                )

            page_count = len(doc)
            logger.debug(
                "Document opened: pages=%d, source=%s", page_count, source_path
            )

            pages_text: list[str] = []
            has_content = False

            try:
                logger.debug("Extracting text from %d pages", page_count)
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    page_text = cast(str, page.get_text("text", sort=True))
                    if page_text.strip():
                        has_content = True
                    pages_text.append(page_text)
            except Exception as e:
                logger.error("Text extraction failed: source=%s", source_path)
                logger.debug("Extraction error details: %s", e, exc_info=True)
                raise IngestionError(
                    "Failed to extract text from document pages",
                    source_path=source_path,
                    cause=e,
                ) from e

            if not has_content:
                logger.warning("No text content found: source=%s", source_path)
                raise IngestionError(
                    "No text content extracted from document (document might be scanned or empty)",
                    source_path=source_path,
                )

            logger.debug(
                "Normalizing extracted text: chars=%d", len("\n".join(pages_text))
            )
            raw_content = "\n".join(pages_text).strip()
            content = unicodedata.normalize("NFC", raw_content).replace("\x00", "")

            logger.debug("Extracting metadata: source=%s", source_path)
            metadata: dict[str, Any] = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "page_count": len(doc),
                "source": source_path,
            }

            try:
                toc = doc.get_toc()
                if toc:
                    metadata["toc"] = [
                        {"level": level, "title": title, "page": page}
                        for level, title, page in toc
                    ]
                    logger.debug("TOC extracted: entries=%d", len(toc))
            except Exception as e:
                logger.warning("Failed to extract TOC: source=%s", source_path)
                logger.debug("TOC extraction error: %s", e, exc_info=True)

            content_size = len(content)
            logger.info(
                "Ingestion completed: source=%s, pages=%d, content_size=%d",
                source_path,
                page_count,
                content_size,
            )

            return SourceDoc(
                content=content,
                metadata=metadata,
                mime_type="application/pdf",
                page_count=len(doc),
            )

        except IngestionError:
            raise
        except fitz.FileDataError as e:
            logger.error("Corrupted document format: source=%s", source_path)
            logger.debug("Format error details: %s", e, exc_info=True)
            raise IngestionError(
                f"Invalid or corrupted document format: {e}",
                source_path=source_path,
                cause=e,
            ) from e
        except Exception as e:
            logger.error("Unexpected ingestion failure: source=%s", source_path)
            logger.debug("Unexpected error details: %s", e, exc_info=True)
            raise IngestionError(
                f"Unexpected ingestion failure: {e}",
                source_path=source_path,
                cause=e,
            ) from e
        finally:
            if doc:
                try:
                    doc.close()
                    logger.debug("Document closed: source=%s", source_path)
                except Exception as e:
                    logger.warning("Failed to close document: %s", source_path)
                    logger.debug("Close error details: %s", e, exc_info=True)

            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.debug("Temporary file cleaned up: %s", temp_file_path)
                except OSError as e:
                    logger.warning(
                        "Failed to clean up temporary file: path=%s",
                        temp_file_path,
                    )
                    logger.debug("Cleanup error details: %s", e, exc_info=True)

    @property
    def supported_formats(self) -> set[str]:
        """Get the set of supported file extensions.

        Returns:
            A set of supported extensions.
        """
        return {".pdf", ".xps", ".epub", ".mobi", ".fb2"}
