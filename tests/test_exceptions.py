"""Unit tests for exception hierarchy and string formatting.

This module validates that custom exceptions correctly handle messages,
attributes, and cause chaining according to RAGMark standards.
"""

import pytest

from ragmark.exceptions import (
    ConfigError,
    ConfigOverrideError,
    EmbeddingError,
    EvaluationError,
    FragmentationError,
    GenerationError,
    IndexError,
    IngestionError,
    RAGMarkError,
    RetrievalError,
    UnsupportedBackendError,
)


@pytest.mark.unit
class TestRAGMarkError:
    """Tests for the base RAGMarkError class behavior."""

    def test_initialization_should_store_message_and_default_attributes(self) -> None:
        """Test basic instantiation of RAGMarkError.

        Given: A simple error message.
        When: The exception is instantiated.
        Then: The string representation matches the message and cause is None.
        """
        message = "Something went wrong"
        error = RAGMarkError(message)

        assert str(error) == message
        assert error.message == message
        assert error.cause is None

    def test_initialization_with_cause_should_store_cause_attribute(self) -> None:
        """Test instantiation with an explicit cause argument.

        Given: An original ValueError.
        When: RAGMarkError is instantiated with 'cause'.
        Then: The wrapper error contains both messages and references the cause.
        """
        original_exc = ValueError("Original error")
        wrapper_exc = RAGMarkError("Wrapper error", cause=original_exc)

        assert "Wrapper error" in str(wrapper_exc)
        assert "Original error" in str(wrapper_exc)
        assert wrapper_exc.cause is original_exc

    def test_raising_error_from_cause_should_preserve_stack_trace(self) -> None:
        """Test strict exception chaining using 'raise from'.

        This validates the policy requirement regarding '__cause__' preservation.

        Given: An underlying ValueError.
        When: RAGMarkError is raised using 'from'.
        Then: The caught exception has a valid __cause__.
        """
        with pytest.raises(RAGMarkError) as exc_info:
            try:
                raise ValueError("Deep error")
            except ValueError as e:
                raise RAGMarkError("Surface error") from e

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert str(exc_info.value.__cause__) == "Deep error"


class TestConfigError:
    """Tests for configuration-related exceptions."""

    def test_error_should_format_message_without_field_path(self) -> None:
        """Test ConfigError without a specific field path.

        Given: An error message only.
        When: ConfigError is instantiated.
        Then: The string representation is exactly the message.
        """
        msg = "Invalid configuration"
        error = ConfigError(msg)

        assert str(error) == msg
        assert error.field_path is None

    def test_error_should_include_field_path_in_string_representation(self) -> None:
        """Test ConfigError formatting with a field path.

        Given: An error message and a dot-notation field path.
        When: ConfigError is instantiated.
        Then: Both the message and field path are present in the string representation.
        """
        error = ConfigError("Invalid value", field_path="retrieval.top_k")
        error_str = str(error)

        assert "Invalid value" in error_str
        assert "retrieval.top_k" in error_str
        assert error.field_path == "retrieval.top_k"

    def test_inheritance_should_maintain_hierarchy(self) -> None:
        """Verify ConfigError inheritance chain.

        Given: A ConfigOverrideError instance.
        When: Checking instance types.
        Then: It should be an instance of ConfigError, RAGMarkError, and Exception.
        """
        error = ConfigOverrideError("Override failed")
        assert isinstance(error, ConfigError)
        assert isinstance(error, RAGMarkError)
        assert isinstance(error, Exception)


@pytest.mark.unit
class TestUnsupportedBackendError:
    """Tests for backend compatibility exceptions."""

    def test_error_should_suggest_pip_install_when_extra_provided(self) -> None:
        """Test error message formatting with installation suggestion.

        Given: A backend name and its corresponding install extra.
        When: UnsupportedBackendError is instantiated.
        Then: The message includes the pip install command.
        """
        backend = "qdrant"
        error = UnsupportedBackendError(backend, install_extra="qdrant")

        error_str = str(error)
        assert backend in error_str
        assert "pip install ragmark[qdrant]" in error_str
        assert error.backend == backend
        assert error.install_extra == "qdrant"

    def test_error_should_not_suggest_install_when_extra_missing(self) -> None:
        """Test error message formatting without installation suggestion.

        Given: A backend name only.
        When: UnsupportedBackendError is instantiated.
        Then: The message indicates unavailability without install instructions.
        """
        backend = "custom_db"
        error = UnsupportedBackendError(backend)

        assert backend in str(error)
        assert "not available" in str(error)
        assert error.install_extra is None


class TestIngestionError:
    """Tests for document ingestion exceptions."""

    def test_error_should_include_source_path_and_cause(self) -> None:
        """Test IngestionError with full context details.

        Given: A message, a source path, and an underlying OS error.
        When: IngestionError is instantiated.
        Then: All three components are visible in the string representation.
        """
        cause = OSError("File not readable")
        path = "/path/to/file.pdf"
        error = IngestionError("Cannot read file", source_path=path, cause=cause)

        error_str = str(error)
        assert "Cannot read file" in error_str
        assert path in error_str
        assert "File not readable" in error_str
        assert error.source_path == path


@pytest.mark.parametrize(
    ("exception_class", "default_message"),
    [
        (FragmentationError, "Chunking failed"),
        (IndexError, "Connection failed"),
        (RetrievalError, "Search failed"),
        (EmbeddingError, "Model loading failed"),
        (GenerationError, "LLM error"),
        (EvaluationError, "Metric error"),
    ],
)
def test_standard_exceptions_should_inherit_from_ragmark_error(
    exception_class: type[RAGMarkError], default_message: str
) -> None:
    """Test inheritance for simple exception types.

    This parameterized test replaces redundant individual test methods.

    Given: A specific RAGMark exception subclass.
    When: An instance is created.
    Then: It is an instance of RAGMarkError and retains its message.
    """
    error = exception_class(default_message)
    assert isinstance(error, RAGMarkError)
    assert str(error) == default_message
