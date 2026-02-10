"""Unit and Integration tests for FitzIngestor adhering to strict quality policy."""

import errno
import gc
import logging
import os
import tracemalloc
import unicodedata
from collections.abc import Callable, Iterator
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ragmark.config.profile import IngestorConfig
from ragmark.exceptions import IngestionError
from ragmark.forge.ingestors import FitzIngestor


class TestFitzIngestor:
    """Test suite ensuring robustness, correct error handling, and format compliance."""

    @pytest.mark.unit
    def test_initialization_should_set_default_attributes(self) -> None:
        """Verifies that the ingestor initializes with safe defaults.

        Given:
            No arguments provided to constructor.
        When:
            FitzIngestor is instantiated.
        Then:
            extract_images should be False.
        """
        ingestor = FitzIngestor()
        assert ingestor.extract_images is False

    @pytest.mark.unit
    def test_initialization_should_raise_import_error_if_fitz_missing(self) -> None:
        """Verifies strict dependency checking when the underlying library is missing.

        Given:
            The environment lacks the 'fitz' (PyMuPDF) module.
        When:
            ingest() is called (triggering the lazy import).
        Then:
            It should raise an IngestionError wrapping the ImportError.
        """
        with patch.dict("sys.modules", {"fitz": None}):
            ingestor = FitzIngestor()

            with pytest.raises(IngestionError) as exc_info:
                ingestor.ingest(b"dummy data")

            assert "PyMuPDF (fitz) not installed" in str(exc_info.value)

    @pytest.mark.unit
    def test_supported_formats_should_match_exact_specification(self) -> None:
        """Verifies that the ingestor supports exactly the specified set of formats.

        Given:
            A new FitzIngestor instance.
        When:
            Accessing supported_formats.
        Then:
            It returns the exact set of expected extensions (No more, no less).
        """
        expected_formats = {".pdf", ".xps", ".epub", ".mobi", ".fb2"}
        assert FitzIngestor().supported_formats == expected_formats

    @pytest.mark.unit
    def test_supports_format_should_be_case_insensitive(self) -> None:
        """Verifies that file extension checking ignores case.

        Given:
            Paths with uppercase and mixed-case extensions.
        When:
            supports_format() is called.
        Then:
            It returns True for .PDF and False for unsupported formats.
        """
        ingestor = FitzIngestor()

        assert ingestor.supports_format(Path("doc.pdf")) is True
        assert ingestor.supports_format(Path("DOC.PDF")) is True

        assert ingestor.supports_format(Path("image.png")) is False
        assert ingestor.supports_format(Path("notes.txt")) is False

    @pytest.mark.unit
    def test_from_config_should_instantiate_ingestor(self) -> None:
        """Verifies instantiation via the configuration object.

        Given:
            A valid IngestorConfig.
        When:
            Calling from_config.
        Then:
            A valid FitzIngestor instance is returned.
        """
        config = IngestorConfig(backend="fitz")
        ingestor = FitzIngestor.from_config(config)
        assert isinstance(ingestor, FitzIngestor)

    @pytest.mark.unit
    def test_ingest_bytes_should_raise_error_when_doc_structure_is_invalid_but_opens(
        self,
    ) -> None:
        """Validates detection of 'Zombie Document' via the bytes branch.

        Given:
            Input of type bytes (b"dummy").
            fitz.open() returns an object that evaluates to False (invalid structure).
        When:
            ingest() is called with bytes.
        Then:
            IngestionError is raised with RuntimeError cause and structure details.
        """
        ingestor = FitzIngestor()

        mock_doc = MagicMock()
        mock_doc.__bool__.return_value = False
        mock_doc.__len__.return_value = 0
        mock_doc.xref_length.return_value = 0

        with patch("fitz.open", return_value=mock_doc):
            with pytest.raises(IngestionError) as exc_info:
                ingestor.ingest(b"dummy pdf bytes")

            assert "invalid structure" in str(exc_info.value).lower()
            assert "bool(doc)=false" in str(exc_info.value).lower()

            assert isinstance(exc_info.value.__cause__, RuntimeError)

    @pytest.mark.unit
    def test_ingest_stream_should_raise_error_when_doc_structure_is_invalid_but_opens(
        self,
    ) -> None:
        """Validates detection of 'Zombie Document' via the stream (IO) branch.

        Given:
            A stream input (BytesIO).
            fitz.open() returns an object that evaluates to False.
        When:
            ingest() is called with stream.
        Then:
            An IngestionError is raised, capturing the manually generated RuntimeError.
        """
        ingestor = FitzIngestor()
        stream = BytesIO(b"dummy stream data")

        mock_doc = MagicMock()
        mock_doc.__bool__.return_value = False
        mock_doc.__len__.return_value = 0
        mock_doc.xref_length.return_value = 123

        with patch("fitz.open", return_value=mock_doc):
            with patch("tempfile.NamedTemporaryFile") as mock_temp:
                mock_temp.return_value.__enter__.return_value.name = "/tmp/fake.pdf"

                with pytest.raises(IngestionError) as exc_info:
                    ingestor.ingest(stream)

                error_msg = str(exc_info.value).lower()
                assert "invalid structure" in error_msg
                assert "xref_length=123" in error_msg
                assert isinstance(exc_info.value.__cause__, RuntimeError)

    @pytest.mark.integration
    def test_ingest_should_extract_content_and_metadata_from_bytes(
        self, pdf_bytes_factory: Callable[..., bytes]
    ) -> None:
        """Verifies complete ingestion flow from memory bytes.

        Given:
            Valid PDF bytes containing specific text and metadata.
        When:
            ingest() is called.
        Then:
            Content matches input, metadata is extracted, and mime-type is correct.
        """
        expected_content = "Integration Test Content"
        expected_title = "RAGMark Report"
        pdf_data = pdf_bytes_factory(content=expected_content, title=expected_title)

        ingestor = FitzIngestor()
        doc = ingestor.ingest(pdf_data)

        assert expected_content in doc.content
        assert doc.metadata.get("title") == expected_title
        assert doc.mime_type == "application/pdf"
        assert doc.page_count == 1

    @pytest.mark.integration
    def test_ingest_should_process_file_from_path_object(
        self, tmp_path: Path, pdf_bytes_factory: Callable[..., bytes]
    ) -> None:
        """Verifies ingestion when provided a pathlib.Path object.

        Given:
            A valid PDF file saved on disk.
        When:
            ingest() is called with the Path object.
        Then:
            The document is processed successfully.
        """
        pdf_content = pdf_bytes_factory(content="Path Test")
        pdf_path = tmp_path / "test_doc.pdf"
        pdf_path.write_bytes(pdf_content)

        ingestor = FitzIngestor()
        doc = ingestor.ingest(pdf_path)

        assert "Path Test" in doc.content

    @pytest.mark.integration
    def test_ingest_should_process_content_from_open_file_stream(
        self, pdf_bytes_factory: Callable[..., bytes]
    ) -> None:
        """Verifies ingestion from an open binary stream (BytesIO).

        Given:
            An open BytesIO stream containing valid PDF data.
        When:
            ingest() is called with the stream.
        Then:
            The content is extracted successfully.
        """
        pdf_data = pdf_bytes_factory(content="Stream Test")
        stream = BytesIO(pdf_data)

        ingestor = FitzIngestor()
        doc = ingestor.ingest(stream)

        assert "Stream Test" in doc.content

    @pytest.mark.integration
    def test_ingest_should_cleanup_temporary_file_after_stream_processing(
        self, pdf_bytes_factory: Callable[..., bytes]
    ) -> None:
        """Verifies that temporary files created for streams are deleted after processing.

        Given:
            An open binary stream (BytesIO).
        When:
            ingest() completes successfully.
        Then:
            The temporary file should be unlinked (deleted) by the ingestor.
        """
        pdf_data = pdf_bytes_factory(content="Cleanup Test")
        stream = BytesIO(pdf_data)
        ingestor = FitzIngestor()

        with patch("os.unlink", wraps=os.unlink) as mock_unlink:
            ingestor.ingest(stream)

            assert (
                mock_unlink.called
            ), "os.unlink was not called to clean up the temp file."

            deleted_path_arg = mock_unlink.call_args[0][0]
            assert not Path(
                deleted_path_arg
            ).exists(), "Temporary file still exists on disk."

    @pytest.mark.integration
    def test_ingest_batch_should_yield_documents_lazily(
        self, tmp_path: Path, pdf_bytes_factory: Callable[..., bytes]
    ) -> None:
        """Verifies that ingest_batch processes multiple files as a generator.

        Given:
            Two valid PDF files on disk.
        When:
            ingest_batch is called with the list of paths.
        Then:
            It yields SourceDoc objects corresponding to each input file.
        """
        doc1_path = tmp_path / "doc1.pdf"
        doc2_path = tmp_path / "doc2.pdf"

        doc1_path.write_bytes(pdf_bytes_factory(content="Batch Content A"))
        doc2_path.write_bytes(pdf_bytes_factory(content="Batch Content B"))

        ingestor = FitzIngestor()

        result_generator = ingestor.ingest_batch([doc1_path, doc2_path])

        assert isinstance(result_generator, Iterator)

        results = list(result_generator)
        assert len(results) == 2
        assert results[0].content.strip().startswith("Batch Content A")
        assert results[1].content.strip().startswith("Batch Content B")

    @pytest.mark.integration
    def test_ingest_should_extract_table_of_contents(
        self, pdf_bytes_factory: Callable[..., bytes]
    ) -> None:
        """Verifies that the table of contents is extracted into metadata.

        Given:
            A PDF with defined outline/bookmarks (Chapters).
        When:
            ingest() is called.
        Then:
            The doc.metadata['toc'] field should contain the chapter titles.
        """
        chapters = ["Chapter 1: Intro", "Chapter 2: Methods"]
        pdf_data = pdf_bytes_factory(content="Text", pages=2, toc=chapters)

        ingestor = FitzIngestor()
        doc = ingestor.ingest(pdf_data)

        assert "toc" in doc.metadata
        toc_str = str(doc.metadata["toc"])
        assert "Chapter 1: Intro" in toc_str
        assert "Chapter 2: Methods" in toc_str

    @pytest.mark.integration
    def test_ingest_should_log_warning_and_continue_when_toc_extraction_fails(
        self, pdf_bytes_factory: Callable[..., bytes], caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verifies graceful degradation when TOC extraction fails (Graceful Failure).

        Given:
            A valid PDF where the TOC extraction method raises an unexpected exception.
        When:
            ingest() is called.
        Then:
            The document is successfully ingested (content present).
            A warning is logged indicating TOC failure.
        """
        pdf_data = pdf_bytes_factory(content="Valid Content")
        ingestor = FitzIngestor()

        logger = logging.getLogger("ragmark.forge.ingestors")
        logger.propagate = True

        with caplog.at_level(logging.WARNING, logger="ragmark.forge.ingestors"):
            with patch(
                "fitz.Document.get_toc",
                side_effect=Exception("Corrupt Outline Structure"),
            ):
                doc = ingestor.ingest(pdf_data)

        assert "Valid Content" in doc.content
        assert "Failed to extract TOC" in caplog.text

    @pytest.mark.integration
    def test_ingest_should_correctly_handle_unicode_and_emojis(
        self, pdf_bytes_factory: Callable[..., bytes]
    ) -> None:
        """Verifies robust extraction of CJK and Emojis using specialized fonts."""
        ingestor = FitzIngestor()

        ideographic_space = "\u3000"
        cjk_content = f"RAGMark Test: 日本語と中国語{ideographic_space}混合这两种语言可能是种族主义"
        cjk_font_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/SubsetOTF/JP/NotoSansJP-Regular.otf"

        cjk_pdf = pdf_bytes_factory(
            content=cjk_content, font_url=cjk_font_url, font_name="NotoSansJP"
        )
        doc_cjk = ingestor.ingest(cjk_pdf)
        assert "日本語" in doc_cjk.content
        assert "中国語" in doc_cjk.content

        emoji_content = "Status: ⛴⌕ Done ☺ Checked ⎙"  # colored complex emoji are not usable in standard fonts compatibility with fitz
        emoji_font_url = "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSansSymbols/NotoSansSymbols-Regular.ttf"

        emoji_pdf = pdf_bytes_factory(
            content=emoji_content, font_url=emoji_font_url, font_name="NotoEmoji"
        )
        doc_emoji = ingestor.ingest(emoji_pdf)
        assert "⛴" in doc_emoji.content
        assert "⌕" in doc_emoji.content
        assert "☺" in doc_emoji.content
        assert "⎙" in doc_emoji.content
        assert "Done" in doc_emoji.content

        nfd_text = unicodedata.normalize("NFD", "Journal Aéronautique")
        nfd_font_url = "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf"
        nfd_pdf = pdf_bytes_factory(
            content=nfd_text, font_url=nfd_font_url, font_name="NotoSans"
        )
        doc_nfd = ingestor.ingest(nfd_pdf)
        assert "Aéronautique" in doc_nfd.content

    @pytest.mark.integration
    def test_ingest_should_raise_ingestion_error_when_pdf_is_encrypted(
        self, pdf_bytes_factory: Callable[..., bytes]
    ) -> None:
        """Verifies detection and rejection of password-protected PDFs.

        Given:
            A PDF file encrypted with a password.
        When:
            ingest() is called without credentials.
        Then:
            IngestionError is raised.
            Note: This is a logic check (is_encrypted), so no system __cause__ is expected.
        """
        encrypted_pdf = pdf_bytes_factory(
            content="Secret", encrypt=True, password="user123"
        )
        ingestor = FitzIngestor()

        with pytest.raises(IngestionError) as exc_info:
            ingestor.ingest(encrypted_pdf)

        error_msg = str(exc_info.value).lower()
        assert "encrypted" in error_msg or "password" in error_msg
        assert exc_info.value.__cause__ is None

    @pytest.mark.integration
    def test_ingest_should_raise_ingestion_error_with_cause_when_file_is_corrupted(
        self, tmp_path: Path
    ) -> None:
        """Verifies that binary corruption raises a strict IngestionError.

        Given:
            A file with a valid PDF header but corrupted binary body.
        When:
            ingest() is called.
        Then:
            IngestionError is raised preserving the original stack trace (fitz.RuntimeError).
        """
        corrupt_pdf = tmp_path / "bad.pdf"
        corrupt_pdf.write_bytes(b"%PDF-1.4\n1 0 obj\n<< CORRUPT STREAM >>")

        ingestor = FitzIngestor()

        with pytest.raises(IngestionError) as exc_info:
            ingestor.ingest(corrupt_pdf)

        assert exc_info.value.__cause__ is not None

    @pytest.mark.integration
    def test_ingest_should_raise_error_when_document_has_no_selectable_text(
        self, pdf_bytes_factory: Callable[..., bytes]
    ) -> None:
        """Verifies validation of empty or scanned documents.

        Given:
            A valid PDF structure but with no text content.
        When:
            ingest() is called.
        Then:
            IngestionError is raised indicating no content found.
        """
        empty_pdf = pdf_bytes_factory(content=None, pages=1)
        ingestor = FitzIngestor()

        with pytest.raises(IngestionError) as exc_info:
            ingestor.ingest(empty_pdf)

        assert (
            "no content" in str(exc_info.value).lower()
            or "empty" in str(exc_info.value).lower()
        )

    @pytest.mark.unit
    def test_ingest_should_raise_error_when_doc_structure_is_invalid_but_opens(
        self,
    ) -> None:
        """Verifies 'Zombie Document' handling (doc exists but is invalid).

        Covers the logic: `if doc is not None and not bool(doc): raise ...`

        Given:
            A mocked document that opens successfully but evaluates to False (0 pages/invalid).
        When:
            ingest() is called.
        Then:
            IngestionError is raised with specific message about invalid structure.
        """
        ingestor = FitzIngestor()

        mock_doc = MagicMock()
        mock_doc.__bool__.return_value = False
        mock_doc.__len__.return_value = 0

        with patch("fitz.open", return_value=mock_doc):
            with pytest.raises(IngestionError) as exc_info:
                ingestor.ingest(Path("zombie.pdf"))

            assert "invalid structure" in str(exc_info.value).lower()

    @pytest.mark.unit
    def test_ingest_should_handle_exception_during_page_text_extraction(self) -> None:
        """Verifies error handling inside the page iteration loop.

        Given:
            A document where accessing a specific page raises a system error.
        When:
            ingest() processes the pages.
        Then:
            IngestionError is raised preserving the original cause.
        """
        ingestor = FitzIngestor()

        mock_doc = MagicMock()
        mock_doc.__bool__.return_value = True
        mock_doc.is_encrypted = False
        mock_doc.__len__.return_value = 1

        mock_page = MagicMock()
        mock_page.get_text.side_effect = ValueError("Font decoding failure")
        mock_doc.__getitem__.return_value = mock_page

        with patch("fitz.open", return_value=mock_doc):
            with pytest.raises(IngestionError) as exc_info:
                ingestor.ingest(Path("partial_corrupt.pdf"))

            assert "Failed to extract text" in str(exc_info.value)
            assert isinstance(exc_info.value.__cause__, ValueError)

    @pytest.mark.unit
    def test_ingest_should_catch_generic_exceptions_at_top_level(self) -> None:
        """Verifies the global catch-all handler for unexpected errors.

        Given:
            An unexpected non-fitz exception (e.g., MemoryError, SystemError).
        When:
            ingest() execution begins.
        Then:
            It catches the exception and wraps it in IngestionError.
        """
        ingestor = FitzIngestor()

        with patch("fitz.open", side_effect=Exception("Catastrophic Failure")):
            with pytest.raises(IngestionError) as exc_info:
                ingestor.ingest(b"doomy data")

            assert "Unexpected ingestion failure" in str(exc_info.value)
            assert isinstance(exc_info.value.__cause__, Exception)

    @pytest.mark.unit
    def test_ingest_should_raise_error_on_closed_stream(self) -> None:
        """Verifies handling of invalid IO streams.

        Given:
            A BytesIO stream that has been closed.
        When:
            ingest() is called.
        Then:
            IngestionError is raised caused by ValueError.
        """
        ingestor = FitzIngestor()
        stream = BytesIO(b"dummy")
        stream.close()

        with pytest.raises(IngestionError) as exc_info:
            ingestor.ingest(stream)

        assert isinstance(exc_info.value.__cause__, ValueError)
        assert "closed" in str(exc_info.value).lower()

    @pytest.mark.integration
    def test_ingest_should_raise_error_when_disk_full(
        self, pdf_bytes_factory: Callable[..., bytes]
    ) -> None:
        """Verifies error handling when writing temporary files fails (ENOSPC).

        Given:
            The system reports no space left on device (ENOSPC).
        When:
            ingest() tries to write the stream to a temp file.
        Then:
            IngestionError is raised identifying the IO issue.
        """
        valid_pdf = pdf_bytes_factory(content="data")
        ingestor = FitzIngestor()

        with patch("tempfile.NamedTemporaryFile") as mock_temp:
            mock_file = MagicMock()
            mock_temp.return_value.__enter__.return_value = mock_file
            mock_file.write.side_effect = OSError(
                errno.ENOSPC, "No space left on device"
            )

            with pytest.raises(IngestionError) as exc_info:
                ingestor.ingest(valid_pdf)

            assert (
                "disk full" in str(exc_info.value).lower()
                or "space" in str(exc_info.value).lower()
            )
            assert isinstance(exc_info.value.__cause__, OSError)
            assert exc_info.value.__cause__.errno == errno.ENOSPC

    @pytest.mark.integration
    def test_cleanup_should_log_warning_when_permission_denied(
        self, pdf_bytes_factory: Callable[..., bytes], caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verifies that failure to delete temp file doesn't crash ingestion but logs warning.

        Given:
            A successful ingestion where os.unlink fails (PermissionError).
        When:
            ingest() completes.
        Then:
            The document is returned, but a WARNING is logged.
        """
        pdf_data = pdf_bytes_factory(content="Clean me up")
        ingestor = FitzIngestor()

        logger = logging.getLogger("ragmark.forge.ingestors")
        logger.propagate = True

        with caplog.at_level(logging.WARNING, logger="ragmark.forge.ingestors"):
            with patch("os.unlink", side_effect=PermissionError("Access denied")):
                doc = ingestor.ingest(pdf_data)
                assert "Clean me up" in doc.content

        assert "Failed to clean up" in caplog.text

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.performance
    def test_ingest_should_handle_large_files_without_excessive_memory_growth(
        self, pdf_bytes_factory: Callable[..., bytes], tmp_path: Path
    ) -> None:
        """Verifies memory stability when processing large dense PDFs.

        Given:
            A generated dense PDF (~50 pages).
        When:
            ingest() is processed while monitoring tracemalloc.
        Then:
            Memory growth should remain within defined threshold (50MB overhead).
        """
        large_pdf_bytes = pdf_bytes_factory(content="Lorem Ipsum " * 200, pages=50)
        large_pdf_path = tmp_path / "large_stress_test.pdf"
        large_pdf_path.write_bytes(large_pdf_bytes)

        gc.collect()
        tracemalloc.start()
        start_snapshot = tracemalloc.take_snapshot()

        ingestor = FitzIngestor()
        doc = ingestor.ingest(large_pdf_path)

        end_snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

        top_stats = end_snapshot.compare_to(start_snapshot, "lineno")
        total_growth = sum(stat.size_diff for stat in top_stats)
        total_growth_mb = total_growth / (1024 * 1024)

        assert doc.page_count == 50
        assert total_growth_mb < 50, f"Memory spike detected: {total_growth_mb:.2f} MB"
