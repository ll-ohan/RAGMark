"""Unit tests for factory classes.

This module tests all factory classes that instantiate components
based on configuration.
"""

from unittest.mock import MagicMock, patch

import pytest

from ragmark.config.profile import EmbedderConfig, IndexConfig, RetrievalConfig
from ragmark.exceptions import UnsupportedBackendError
from ragmark.index.factory import IndexFactory, register_index_backend
from ragmark.retrieval.factory import RetrieverFactory


class TestIndexFactory:
    """Tests for IndexFactory."""

    def test_create_memory_index(self) -> None:
        """Test creating a memory index."""
        # Patch the import path in the backends module
        with patch("ragmark.index.backends.MemoryIndex") as MockIndex:
            mock_instance = MagicMock()
            MockIndex.from_config.return_value = mock_instance

            config = IndexConfig(
                backend="memory",
                collection_name="test_collection",
                embedding_dim=384,
                connection=None,
            )

            result = IndexFactory.create(config)

            MockIndex.from_config.assert_called_once_with(config, embedder=None)
            assert result is mock_instance

    def test_create_with_embedder_config(self) -> None:
        """Test creating index with embedder configuration."""
        with patch("ragmark.index.backends.MemoryIndex") as MockIndex:
            with patch(
                "ragmark.index.embedders.EmbedderFactory.create"
            ) as MockEmbedderFactory:
                mock_embedder = MagicMock()
                mock_instance = MagicMock()
                MockEmbedderFactory.return_value = mock_embedder
                MockIndex.from_config.return_value = mock_instance

                config = IndexConfig(
                    backend="memory", embedding_dim=384, connection=None
                )
                embedder_config = EmbedderConfig(model_name="test-model")

                IndexFactory.create(config, embedder_config=embedder_config)

                MockEmbedderFactory.assert_called_once_with(embedder_config)
                call_kwargs = MockIndex.from_config.call_args.kwargs
                assert call_kwargs["embedder"] is mock_embedder

    def test_create_with_embedder_instance(self) -> None:
        """Test creating index with pre-configured embedder."""
        with patch("ragmark.index.backends.MemoryIndex") as MockIndex:
            mock_instance = MagicMock()
            MockIndex.from_config.return_value = mock_instance

            config = IndexConfig(backend="memory", embedding_dim=384, connection=None)
            mock_embedder = MagicMock()

            IndexFactory.create(config, embedder=mock_embedder)

            call_kwargs = MockIndex.from_config.call_args.kwargs
            assert call_kwargs["embedder"] is mock_embedder

    def test_unsupported_backend(self) -> None:
        """Test that UnsupportedBackendError is raised for unknown backends."""
        # Use a literal that's not in the allowed list - need to bypass Pydantic validation
        with patch.object(
            IndexConfig,
            "model_validate",
            return_value=MagicMock(backend="nonexistent", embedding_dim=384),
        ):
            config = MagicMock()
            config.backend = "nonexistent"
            config.embedding_dim = 384
            config.collection_name = "test"
            config.connection = None

            with pytest.raises(UnsupportedBackendError, match="nonexistent"):
                IndexFactory.create(config)

    def test_register_custom_backend(self) -> None:
        """Test registering a custom index backend."""
        from typing import Any

        from ragmark.index.base import VectorIndex
        from ragmark.index.embedders import BaseEmbedder
        from ragmark.schemas.documents import KnowledgeNode
        from ragmark.schemas.retrieval import SearchResult

        class CustomIndex(VectorIndex):
            """Test custom index implementation."""

            def __init__(
                self,
                config: IndexConfig | None = None,
                embedder: BaseEmbedder | None = None,
                **kwargs: str | None,
            ):
                self.config = config
                self.embedder = embedder

            @classmethod
            def from_config(
                cls, config: IndexConfig, embedder: BaseEmbedder | None = None
            ) -> "CustomIndex":
                return cls(config=config, embedder=embedder)

            async def add(self, nodes: list[KnowledgeNode]) -> None:
                pass

            async def search(
                self,
                query_vector: list[float] | dict[int, float],
                top_k: int = 10,
                filters: dict[str, Any] | None = None,
            ) -> list[SearchResult]:
                return []

            async def search_hybrid(
                self,
                dense_vector: list[float],
                sparse_vector: dict[int, float],
                top_k: int = 10,
                alpha: float = 0.0,
            ) -> list[SearchResult]:
                return []

            async def delete(self, node_ids: list[str]) -> int:
                return 0

            async def count(self) -> int:
                return 0

            async def clear(self) -> None:
                pass

            async def exists(self, node_id: str) -> bool:
                return False

            async def __aenter__(self) -> "CustomIndex":
                """Async context manager entry."""
                return self

            async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
                """Async context manager exit."""
                pass

        register_index_backend("custom", CustomIndex)

        # Create a mock config
        config = MagicMock()
        config.backend = "custom"
        config.embedding_dim = 384

        index = IndexFactory.create(config)

        assert isinstance(index, CustomIndex)


class TestRetrieverFactory:
    """Tests for RetrieverFactory."""

    @pytest.fixture
    def mock_index(self) -> MagicMock:
        """Create a mock VectorIndex."""
        return MagicMock()

    def test_create_dense_retriever(self, mock_index: MagicMock) -> None:
        """Test creating a dense retriever."""
        with patch(
            "ragmark.retrieval.strategies.DenseRetriever.from_config"
        ) as mock_from_config:
            config = RetrievalConfig(mode="dense", top_k=10, alpha=None, reranker=None)

            RetrieverFactory.create(config, mock_index)

            mock_from_config.assert_called_once_with(config, mock_index)

    def test_create_sparse_retriever(self, mock_index: MagicMock) -> None:
        """Test creating a sparse retriever."""
        with patch(
            "ragmark.retrieval.strategies.SparseRetriever.from_config"
        ) as mock_from_config:
            config = RetrievalConfig(mode="sparse", top_k=10, alpha=None, reranker=None)

            RetrieverFactory.create(config, mock_index)

            mock_from_config.assert_called_once_with(config, mock_index)

    def test_create_hybrid_retriever(self, mock_index: MagicMock) -> None:
        """Test creating a hybrid retriever."""
        with patch(
            "ragmark.retrieval.strategies.HybridRetriever.from_config"
        ) as mock_from_config:
            config = RetrievalConfig(mode="hybrid", top_k=10, alpha=0.5, reranker=None)

            RetrieverFactory.create(config, mock_index)

            mock_from_config.assert_called_once_with(config, mock_index)

    def test_hybrid_without_alpha_raises_error(self, mock_index: MagicMock) -> None:
        """Test that hybrid mode without alpha raises ValueError."""
        # Bypass Pydantic validation that already checks this
        config = MagicMock()
        config.mode = "hybrid"
        config.top_k = 10
        config.alpha = None
        config.reranker = None

        # HybridRetriever.from_config will raise the error
        with patch(
            "ragmark.retrieval.strategies.HybridRetriever.from_config",
            side_effect=ValueError("alpha is required for hybrid retrieval"),
        ):
            with pytest.raises(ValueError, match="alpha is required"):
                RetrieverFactory.create(config, mock_index)

    def test_create_with_reranker(self, mock_index: MagicMock) -> None:
        """Test creating retriever with reranker."""
        from ragmark.config.profile import RerankerConfig

        with patch(
            "ragmark.retrieval.rerankers.CrossEncoderRefiner.from_config"
        ) as mock_refiner_from_config:
            with patch(
                "ragmark.retrieval.strategies.RefinedRetriever.from_config"
            ) as mock_refined_from_config:
                reranker_config = RerankerConfig(
                    model_name="cross-encoder/test",
                    top_k=5,
                )
                config = RetrievalConfig(
                    mode="dense",
                    top_k=10,
                    alpha=None,
                    reranker=reranker_config,
                )

                mock_refiner = MagicMock()
                mock_refiner_from_config.return_value = mock_refiner

                RetrieverFactory.create(config, mock_index)

                # Verify refiner was created from config
                mock_refiner_from_config.assert_called_once_with(reranker_config)
                # Verify RefinedRetriever.from_config was called with refiner
                mock_refined_from_config.assert_called_once_with(
                    config, mock_index, refiner=mock_refiner
                )

    def test_unknown_mode_raises_error(self, mock_index: MagicMock) -> None:
        """Test that unknown retrieval mode raises ValueError."""
        # Bypass Pydantic validation
        config = MagicMock()
        config.mode = "unknown_mode"
        config.top_k = 10
        config.reranker = None
        config.alpha = None

        with pytest.raises(ValueError, match="Unknown retrieval mode"):
            RetrieverFactory.create(config, mock_index)


class TestEmbedderFactory:
    """Tests for EmbedderFactory delegation to from_config()."""

    def test_factory_delegates_to_from_config(self) -> None:
        """Test that EmbedderFactory.create() delegates to from_config()."""
        from ragmark.index.embedders import EmbedderFactory

        config = EmbedderConfig(
            model_name="test-model",
            device="cpu",
            batch_size=32,
        )

        with patch(
            "ragmark.index.backends.SentenceTransformerEmbedder.from_config"
        ) as mock_from_config:
            mock_embedder = MagicMock()
            mock_from_config.return_value = mock_embedder

            result = EmbedderFactory.create(config)

            # Verify from_config was called with the config
            mock_from_config.assert_called_once_with(config)
            assert result is mock_embedder

    def test_factory_raises_on_missing_dependencies(self) -> None:
        """Test that factory raises UnsupportedBackendError when dependencies missing."""
        from ragmark.index.embedders import EmbedderFactory

        config = EmbedderConfig(model_name="test-model")

        # Simulate ImportError when trying to import from backends module
        with patch.dict(
            "sys.modules",
            {"ragmark.index.backends": None},
        ):
            with pytest.raises(UnsupportedBackendError, match="sentence-transformers"):
                EmbedderFactory.create(config)
