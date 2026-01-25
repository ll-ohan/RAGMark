"""Unit tests for exception hierarchy.

This module tests all custom exceptions, their inheritance,
and string formatting.
"""

from ragmark.exceptions import (
    ConfigError,
    ConfigOverrideError,
    EvaluationError,
    FragmentationError,
    GenerationError,
    IndexError,
    IngestionError,
    RAGMarkError,
    RetrievalError,
    UnsupportedBackendError,
)


class TestRAGMarkError:
    """Tests for base RAGMarkError."""

    def test_basic_error(self) -> None:
        """Test creating a basic error."""
        error = RAGMarkError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.cause is None

    def test_error_with_cause(self) -> None:
        """Test error with underlying cause."""
        cause = ValueError("Original error")
        error = RAGMarkError("Wrapper error", cause=cause)

        assert "Wrapper error" in str(error)
        assert "Original error" in str(error)
        assert error.cause is cause

    def test_inheritance(self) -> None:
        """Test that RAGMarkError is an Exception."""
        error = RAGMarkError("Test")
        assert isinstance(error, Exception)


class TestConfigError:
    """Tests for ConfigError."""

    def test_basic_config_error(self) -> None:
        """Test creating a basic config error."""
        error = ConfigError("Invalid configuration")
        assert str(error) == "Invalid configuration"
        assert error.field_path is None

    def test_with_field_path(self) -> None:
        """Test config error with field path."""
        error = ConfigError("Invalid value", field_path="retrieval.top_k")

        assert "Invalid value" in str(error)
        assert "retrieval.top_k" in str(error)
        assert error.field_path == "retrieval.top_k"

    def test_with_cause_and_path(self) -> None:
        """Test config error with both cause and field path."""
        cause = ValueError("Must be positive")
        error = ConfigError(
            "Validation failed",
            field_path="fragmenter.chunk_size",
            cause=cause,
        )

        error_str = str(error)
        assert "Validation failed" in error_str
        assert "fragmenter.chunk_size" in error_str
        assert "Must be positive" in error_str

    def test_inheritance(self) -> None:
        """Test that ConfigError inherits from RAGMarkError."""
        error = ConfigError("Test")
        assert isinstance(error, RAGMarkError)
        assert isinstance(error, Exception)


class TestConfigOverrideError:
    """Tests for ConfigOverrideError."""

    def test_inheritance(self) -> None:
        """Test that ConfigOverrideError inherits from ConfigError."""
        error = ConfigOverrideError("Invalid override")
        assert isinstance(error, ConfigError)
        assert isinstance(error, RAGMarkError)


class TestUnsupportedBackendError:
    """Tests for UnsupportedBackendError."""

    def test_without_install_extra(self) -> None:
        """Test error message without install extra."""
        error = UnsupportedBackendError("custom_db")
        assert "custom_db" in str(error)
        assert "not available" in str(error)
        assert error.backend == "custom_db"
        assert error.install_extra is None

    def test_with_install_extra(self) -> None:
        """Test error message with install extra."""
        error = UnsupportedBackendError("qdrant", install_extra="qdrant")

        error_str = str(error)
        assert "qdrant" in error_str
        assert "pip install ragmark[qdrant]" in error_str
        assert error.backend == "qdrant"
        assert error.install_extra == "qdrant"

    def test_inheritance(self) -> None:
        """Test inheritance from RAGMarkError."""
        error = UnsupportedBackendError("test")
        assert isinstance(error, RAGMarkError)


class TestIngestionError:
    """Tests for IngestionError."""

    def test_without_source_path(self) -> None:
        """Test ingestion error without source path."""
        error = IngestionError("Failed to parse document")
        assert str(error) == "Failed to parse document"
        assert error.source_path is None

    def test_with_source_path(self) -> None:
        """Test ingestion error with source path."""
        error = IngestionError(
            "OCR failed",
            source_path="/path/to/document.pdf",
        )

        error_str = str(error)
        assert "OCR failed" in error_str
        assert "/path/to/document.pdf" in error_str
        assert error.source_path == "/path/to/document.pdf"

    def test_with_cause_and_path(self) -> None:
        """Test with both cause and source path."""
        cause = OSError("File not readable")
        error = IngestionError(
            "Cannot read file",
            source_path="/path/to/file.pdf",
            cause=cause,
        )

        error_str = str(error)
        assert "Cannot read file" in error_str
        assert "/path/to/file.pdf" in error_str
        assert "File not readable" in error_str


class TestOtherExceptions:
    """Tests for remaining exception types."""

    def test_fragmentation_error(self) -> None:
        """Test FragmentationError."""
        error = FragmentationError("Chunking failed")
        assert isinstance(error, RAGMarkError)
        assert str(error) == "Chunking failed"

    def test_index_error(self) -> None:
        """Test IndexError."""
        error = IndexError("Connection to Qdrant failed")
        assert isinstance(error, RAGMarkError)

    def test_retrieval_error(self) -> None:
        """Test RetrievalError."""
        error = RetrievalError("Search failed")
        assert isinstance(error, RAGMarkError)

    def test_generation_error(self) -> None:
        """Test GenerationError."""
        error = GenerationError("Model not loaded")
        assert isinstance(error, RAGMarkError)

    def test_evaluation_error(self) -> None:
        """Test EvaluationError."""
        error = EvaluationError("Metric calculation failed")
        assert isinstance(error, RAGMarkError)
