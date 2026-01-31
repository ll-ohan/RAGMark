"""Unit tests for forge factory classes and from_config() pattern."""

from collections.abc import Iterator

import pytest

from ragmark.config.profile import (
    FragmenterConfig,
    IngestorConfig,
)
from ragmark.exceptions import UnsupportedBackendError
from ragmark.forge.factory import FragmenterFactory, IngestorFactory
from ragmark.forge.ingestors import FitzIngestor
from ragmark.schemas.documents import KnowledgeNode, NodePosition, SourceDoc


@pytest.mark.unit
class TestIngestorFactory:
    """Tests for IngestorFactory compliance and error handling."""

    def test_create_should_raise_not_implemented_error_when_backend_is_marker(
        self,
    ) -> None:
        """
        Given: An IngestorConfig with backend set to 'marker'.
        When: The factory creates the ingestor.
        Then: It should raise NotImplementedError as MarkerIngestor is not ready.
        """
        config = IngestorConfig(backend="marker")

        with pytest.raises(NotImplementedError) as exc_info:
            IngestorFactory.create(config)

        assert "MarkerIngestor will be implemented" in str(exc_info.value)

    def test_create_should_raise_unsupported_backend_error_when_backend_is_unknown(
        self,
    ) -> None:
        """
        Given: An IngestorConfig with an unknown backend string.
        When: The factory creates the ingestor.
        Then: It should raise UnsupportedBackendError with the unknown backend name.
        """
        config = IngestorConfig.model_construct(backend="unknown")

        with pytest.raises(UnsupportedBackendError) as exc_info:
            IngestorFactory.create(config)

        assert "unknown" in str(exc_info.value)

    def test_create_should_return_configured_fitz_ingestor_when_options_provided(
        self,
    ) -> None:
        """
        Given: An IngestorConfig with 'fitz' backend and specific options.
        When: The factory creates the ingestor.
        Then: It should return a FitzIngestor instance with options applied.
        """
        options: dict[str, bool | list[str]] = {
            "extract_images": True,
            "languages": ["en", "fr"],
        }
        config = IngestorConfig(backend="fitz", options=options)

        ingestor = IngestorFactory.create(config)

        assert isinstance(ingestor, FitzIngestor)
        assert hasattr(ingestor, "extract_images")
        assert ingestor.extract_images is True


@pytest.mark.unit
class TestFragmenterFactory:
    """Tests for FragmenterFactory logic."""

    def test_create_should_raise_not_implemented_error_when_strategy_is_semantic(
        self,
    ) -> None:
        """
        Given: A FragmenterConfig with 'semantic' strategy.
        When: The factory creates the fragmenter.
        Then: It should raise NotImplementedError.
        """
        config = FragmenterConfig(strategy="semantic", chunk_size=512, overlap=128)

        with pytest.raises(NotImplementedError) as exc_info:
            FragmenterFactory.create(config)

        assert "SemanticFragmenter will be implemented" in str(exc_info.value)

    def test_create_should_raise_not_implemented_error_when_strategy_is_markdown(
        self,
    ) -> None:
        """
        Given: A FragmenterConfig with 'markdown' strategy.
        When: The factory creates the fragmenter.
        Then: It should raise NotImplementedError.
        """
        config = FragmenterConfig(strategy="markdown", chunk_size=1024, overlap=256)

        with pytest.raises(NotImplementedError) as exc_info:
            FragmenterFactory.create(config)

        assert "MarkdownFragmenter will be implemented" in str(exc_info.value)

    def test_create_should_raise_not_implemented_error_when_strategy_is_recursive(
        self,
    ) -> None:
        """
        Given: A FragmenterConfig with 'recursive' strategy.
        When: The factory creates the fragmenter.
        Then: It should raise NotImplementedError.
        """
        config = FragmenterConfig(strategy="recursive", chunk_size=512, overlap=100)

        with pytest.raises(NotImplementedError) as exc_info:
            FragmenterFactory.create(config)

        assert "RecursiveFragmenter will be implemented" in str(exc_info.value)

    def test_create_should_raise_value_error_when_strategy_is_unknown(self) -> None:
        """
        Given: A FragmenterConfig with an unknown strategy.
        When: The factory creates the fragmenter.
        Then: It should raise ValueError indicating the strategy is unknown.
        """
        config = FragmenterConfig.model_construct(
            strategy="unknown", chunk_size=256, overlap=64
        )

        with pytest.raises(ValueError) as exc_info:
            FragmenterFactory.create(config)

        assert "Unknown fragmentation strategy" in str(exc_info.value)

    def test_create_should_return_configured_token_fragmenter_when_valid(self) -> None:
        """
        Given: A valid FragmenterConfig with 'token' strategy.
        When: The factory creates the fragmenter.
        Then: It should return a TokenFragmenter with correct chunk_size.
        """
        config = FragmenterConfig(
            strategy="token",
            chunk_size=256,
            overlap=64,
            options={"tokenizer": "cl100k_base"},
        )

        fragmenter = FragmenterFactory.create(config)

        assert fragmenter.__class__.__name__ == "TokenFragmenter"
        assert fragmenter.chunk_size == 256


@pytest.mark.unit
class TestRegistryPattern:
    """Tests for custom backend registry pattern."""

    def test_register_ingestor_should_allow_custom_backend(self) -> None:
        """
        Given: A custom ingestor class registered with a unique name.
        When: The factory creates an ingestor using that name.
        Then: It should return an instance of the custom class.
        """
        from ragmark.forge import BaseIngestor, register_ingestor

        class CustomIngestor(BaseIngestor):
            @classmethod
            def from_config(cls, config: IngestorConfig) -> "CustomIngestor":
                return cls()

            def ingest(self, source: object) -> SourceDoc:
                return SourceDoc(
                    content="custom", mime_type="text/plain", page_count=None
                )

            def ingest_batch(self, sources: object) -> Iterator[SourceDoc]:
                yield SourceDoc(
                    content="custom", mime_type="text/plain", page_count=None
                )

            @property
            def supported_formats(self) -> set[str]:
                return {".custom"}

        register_ingestor("custom_test", CustomIngestor)

        config = IngestorConfig.model_construct(backend="custom_test")

        ingestor = IngestorFactory.create(config)

        assert isinstance(ingestor, CustomIngestor)

    def test_register_fragmenter_should_allow_custom_strategy(self) -> None:
        """
        Given: A custom fragmenter class registered with a unique name.
        When: The factory creates a fragmenter using that name.
        Then: It should return an instance of the custom class.
        """
        from ragmark.forge import BaseFragmenter, register_fragmenter

        class CustomFragmenter(BaseFragmenter):
            @classmethod
            def from_config(cls, config: FragmenterConfig) -> "CustomFragmenter":
                return cls()

            def fragment(self, doc: SourceDoc) -> list[KnowledgeNode]:
                return []

            def fragment_batch(self, docs: object) -> Iterator[KnowledgeNode]:
                yield KnowledgeNode(
                    content="test",
                    source_id="test",
                    metadata={},
                    position=NodePosition(
                        start_char=0, end_char=4, page=None, section=None
                    ),
                    dense_vector=[0.1] * 384,
                    sparse_vector=None,
                )

            @property
            def chunk_size(self) -> int:
                return 100

            @property
            def overlap(self) -> int:
                return 0

        register_fragmenter("custom_strategy", CustomFragmenter)

        config = FragmenterConfig.model_construct(
            strategy="custom_strategy", chunk_size=100, overlap=0
        )

        fragmenter = FragmenterFactory.create(config)

        assert isinstance(fragmenter, CustomFragmenter)

    def test_registry_should_take_precedence_over_builtin_backends(self) -> None:
        """
        Given: A custom ingestor registered with the same name as a builtin.
        When: The factory creates an ingestor using that name.
        Then: It should return the custom implementation, not the builtin.
        """
        from ragmark.forge import BaseIngestor, register_ingestor

        class OverrideFitzIngestor(BaseIngestor):
            @classmethod
            def from_config(cls, config: IngestorConfig) -> "OverrideFitzIngestor":
                return cls()

            def ingest(self, source: object) -> SourceDoc:
                return SourceDoc(
                    content="override", mime_type="text/plain", page_count=None
                )

            def ingest_batch(self, sources: object) -> Iterator[SourceDoc]:
                yield SourceDoc(
                    content="override", mime_type="text/plain", page_count=None
                )

            @property
            def supported_formats(self) -> set[str]:
                return {".override"}

        register_ingestor("fitz", OverrideFitzIngestor)

        config = IngestorConfig(backend="fitz")

        ingestor = IngestorFactory.create(config)

        assert isinstance(ingestor, OverrideFitzIngestor)
