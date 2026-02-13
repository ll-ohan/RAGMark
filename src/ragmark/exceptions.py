"""Core exception hierarchy for RAGMark.

This module defines all custom exceptions used throughout the framework.
All exceptions inherit from RAGMarkError for unified error handling.
"""


class RAGMarkError(Exception):
    """Base exception for all RAGMark errors.

    All custom exceptions in the framework inherit from this class,
    allowing users to catch all RAGMark-specific errors with a single
    except clause.
    """

    def __init__(self, message: str, cause: Exception | None = None):
        """Initialize the exception.

        Args:
            message: Human-readable error description.
            cause: Optional underlying exception that triggered this error.
        """
        super().__init__(message)
        self.message = message
        self.cause = cause

    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.cause:
            return f"{self.message} (caused by: {self.cause})"
        return self.message


class ConfigError(RAGMarkError):
    """Raised when configuration validation fails.

    This includes invalid YAML files, missing required fields,
    type mismatches, or logical inconsistencies in configuration.
    """

    def __init__(
        self,
        message: str,
        field_path: str | None = None,
        cause: Exception | None = None,
    ):
        """Initialize the configuration error.

        Args:
            message: Human-readable error description.
            field_path: Optional JSON-path to the problematic field (e.g., "retrieval.top_k").
            cause: Optional underlying exception.
        """
        super().__init__(message, cause)
        self.field_path = field_path

    def __str__(self) -> str:
        """Return string representation including field path."""
        base_msg = super().__str__()
        if self.field_path:
            return f"{base_msg} (field: {self.field_path})"
        return base_msg


class ConfigOverrideError(ConfigError):
    """Raised when applying invalid configuration overrides.

    This occurs when attempting to override non-existent fields
    or with incompatible types.
    """

    pass


class UnsupportedBackendError(RAGMarkError):
    """Raised when attempting to use an unavailable backend.

    This typically occurs when optional dependencies for a specific
    backend (e.g., qdrant-client, pymilvus) are not installed.
    """

    def __init__(self, backend: str, install_extra: str | None = None):
        """Initialize the unsupported backend error.

        Args:
            backend: Name of the unavailable backend.
            install_extra: Optional pip extra to install (e.g., "qdrant").
        """
        message = f"Backend '{backend}' is not available"
        if install_extra:
            message += f". Install with: pip install ragmark[{install_extra}]"
        super().__init__(message)
        self.backend = backend
        self.install_extra = install_extra


class IngestionError(RAGMarkError):
    """Raised when document ingestion fails.

    This includes file read errors, parsing failures, OCR errors,
    or unsupported file formats.
    """

    def __init__(
        self,
        message: str,
        source_path: str | None = None,
        cause: Exception | None = None,
    ):
        """Initialize the ingestion error.

        Args:
            message: Human-readable error description.
            source_path: Optional path to the problematic source file.
            cause: Optional underlying exception.
        """
        super().__init__(message, cause)
        self.source_path = source_path

    def __str__(self) -> str:
        """Return string representation including source path."""
        base_msg = super().__str__()
        if self.source_path:
            return f"{base_msg} (source: {self.source_path})"
        return base_msg


class FragmentationError(RAGMarkError):
    """Raised when text fragmentation fails.

    This includes tokenization errors, semantic chunking failures,
    or invalid chunk size configurations.
    """

    pass


class IndexError(RAGMarkError):
    """Raised when vector index operations fail.

    This includes connection errors to vector databases, insertion failures,
    search errors, or collection management issues.
    """

    pass


class RetrievalError(RAGMarkError):
    """Raised when retrieval operations fail.

    This includes search failures, reranking errors,
    or invalid retrieval configurations.
    """

    pass


class EmbeddingError(RAGMarkError):
    """Raised when embedding computation fails.

    This includes model loading errors, encoding failures,
    dimension mismatches, or batch processing errors.
    """

    pass


class GenerationError(RAGMarkError):
    """Raised when LLM generation fails.

    This includes model loading errors, inference failures, context
    window overflow, or invalid generation parameters.
    """

    pass


class EvaluationError(RAGMarkError):
    """Raised when evaluation operations fail.

    This includes metric calculation errors, missing ground truth data,
    or judge LLM failures.
    """

    pass


class ArenaError(RAGMarkError):
    """Raised when arena orchestration fails.

    This includes configuration grid errors, pipeline execution failures
    during comparative benchmarking, or caching errors.
    """

    pass


class QuestionGenerationError(RAGMarkError):
    """Raised when synthetic QA generation fails.

    This includes LLM generation failures, parsing errors from structured
    output, validation failures, or batch processing errors.
    """

    def __init__(
        self,
        message: str,
        node_id: str | None = None,
        cause: Exception | None = None,
    ):
        """Initialize the question generation error.

        Args:
            message: Human-readable error description.
            node_id: Optional ID of the knowledge node that failed.
            cause: Optional underlying exception.
        """
        super().__init__(message, cause)
        self.node_id = node_id

    def __str__(self) -> str:
        """Return string representation including node ID."""
        base_msg = super().__str__()
        if self.node_id:
            return f"{base_msg} (node_id: {self.node_id})"
        return base_msg
