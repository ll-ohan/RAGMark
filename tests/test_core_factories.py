"""Unit tests for factory classes."""

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ragmark.config.profile import EmbedderConfig, IndexConfig, RetrievalConfig
from ragmark.exceptions import UnsupportedBackendError
from ragmark.index.base import VectorIndex
from ragmark.index.embedders import BaseEmbedder, EmbedderFactory
from ragmark.index.factory import _INDEX_REGISTRY, IndexFactory, register_index_backend
from ragmark.retrieval.factory import RetrieverFactory
from ragmark.retrieval.strategies import (
    DenseRetriever,
    HybridRetriever,
    RefinedRetriever,
    SparseRetriever,
)
from ragmark.schemas.documents import KnowledgeNode
from ragmark.schemas.retrieval import SearchResult


class FakeEmbedder(BaseEmbedder):
    """Real implementation of BaseEmbedder for testing."""

    def __init__(self, config: EmbedderConfig | None = None):
        self.config = config

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.embedding_dim for _ in texts]

    def embed_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        """Concrete implementation returning empty sparse vectors for testing."""
        return [{} for _ in texts]

    @property
    def embedding_dim(self) -> int:
        return 384

    @classmethod
    def from_config(cls, config: EmbedderConfig) -> "FakeEmbedder":
        return cls(config)


class FakeVectorIndex(VectorIndex):
    """Real implementation of VectorIndex for testing."""

    def __init__(
        self,
        config: IndexConfig | None = None,
        embedder: BaseEmbedder | None = None,
        **kwargs: Any,
    ):
        self.config = config
        self.embedder = embedder

    @classmethod
    def from_config(
        cls, config: IndexConfig, embedder: BaseEmbedder | None = None
    ) -> "FakeVectorIndex":
        return cls(config=config, embedder=embedder)

    async def add(self, nodes: list[KnowledgeNode]) -> None:
        pass

    async def search(
        self, query_vector: Any, top_k: int = 10, filters: Any = None
    ) -> list[SearchResult]:
        return []

    async def search_hybrid(
        self, dense_vector: Any, sparse_vector: Any, top_k: int = 10, alpha: float = 0.0
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

    async def __aenter__(self) -> "FakeVectorIndex":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


@pytest.mark.unit
class TestIndexFactory:
    """Tests for IndexFactory logic and backend registration."""

    def test_create_should_return_memory_index_when_backend_is_memory(
        self,
        index_config_factory: Callable[..., IndexConfig],
    ) -> None:
        """Verifies that the factory initializes MemoryIndex correctly.

        Given:
            A valid IndexConfig with backend='memory'.
        When:
            IndexFactory.create is called.
        Then:
            The result is an instance of MemoryIndex.
        """
        config = index_config_factory(backend="memory")
        from ragmark.index.backends import MemoryIndex

        result = IndexFactory.create(config)

        assert isinstance(result, MemoryIndex)

    def test_create_should_inject_embedder_when_embedder_config_provided(
        self,
        index_config_factory: Callable[..., IndexConfig],
        embedder_config_factory: Callable[..., EmbedderConfig],
    ) -> None:
        """Verifies embedder creation and injection via config using a Fake backend.

        Given:
            An IndexConfig and an EmbedderConfig.
        When:
            IndexFactory.create is called with both configs.
        Then:
            The factory creates the embedder and passes it to the index backend.
        """

        config = index_config_factory(backend="custom_backend")
        embedder_config = embedder_config_factory(model_name="test-model")

        original_registry = _INDEX_REGISTRY.copy()
        register_index_backend("custom_backend", FakeVectorIndex)

        with patch(
            "ragmark.index.embedders.EmbedderFactory.create"
        ) as MockEmbedderFactory:
            fake_embedder = FakeEmbedder()
            MockEmbedderFactory.return_value = fake_embedder

            try:
                result = IndexFactory.create(config, embedder_config=embedder_config)

                MockEmbedderFactory.assert_called_once_with(embedder_config)
                assert isinstance(result, FakeVectorIndex)
                assert result.embedder is fake_embedder

            finally:
                _INDEX_REGISTRY.clear()
                _INDEX_REGISTRY.update(original_registry)

    def test_create_should_use_provided_embedder_instance_when_passed_explicitly(
        self,
        index_config_factory: Callable[..., IndexConfig],
    ) -> None:
        """Verifies that a pre-instantiated embedder takes precedence.

        Given:
            An IndexConfig and a pre-existing embedder instance.
        When:
            IndexFactory.create is called with the embedder instance.
        Then:
            The backend is initialized with the provided embedder instance.
        """

        config = index_config_factory(backend="custom_backend")
        fake_embedder = FakeEmbedder()

        original_registry = _INDEX_REGISTRY.copy()
        register_index_backend("custom_backend", FakeVectorIndex)
        try:
            result = IndexFactory.create(config, embedder=fake_embedder)

            assert isinstance(result, FakeVectorIndex)
            assert result.embedder is fake_embedder

        finally:
            _INDEX_REGISTRY.clear()
            _INDEX_REGISTRY.update(original_registry)

    def test_create_should_raise_unsupported_backend_error_when_backend_unknown(
        self,
    ) -> None:
        """Verifies proper error handling for unknown backends.

        Given:
            A config with a non-existent backend type (bypassing Pydantic validation).
        When:
            IndexFactory.create is called.
        Then:
            UnsupportedBackendError is raised with a descriptive message.
        """
        config = IndexConfig.model_construct(
            backend="nonexistent_backend_xyz", connection=None, collection_name="test"
        )

        with pytest.raises(UnsupportedBackendError, match="nonexistent_backend_xyz"):
            IndexFactory.create(config)

    def test_create_should_chain_exception_when_backend_dependency_is_missing(
        self, index_config_factory: Callable[..., IndexConfig]
    ) -> None:
        """Verifies that missing optional dependencies raise a chained error.

        Given:
            A config for a backend ('qdrant') whose dependency is missing.
        When:
            IndexFactory.create is called.
        Then:
            UnsupportedBackendError is raised, and its __cause__ is an ImportError.
        """

        config = index_config_factory(
            backend="qdrant", connection={"host": "localhost"}
        )

        import ragmark.index.backends as backends_module

        original_qdrant = getattr(backends_module, "QdrantIndex", None)

        if hasattr(backends_module, "QdrantIndex"):
            delattr(backends_module, "QdrantIndex")

        try:
            with pytest.raises(UnsupportedBackendError) as exc_info:
                IndexFactory.create(config)

            assert exc_info.value.backend == "qdrant"
            assert isinstance(exc_info.value.__cause__, ImportError)

        finally:
            if original_qdrant is not None:
                backends_module.QdrantIndex = original_qdrant  # type: ignore[misc]

    def test_register_should_allow_custom_backend_integration(
        self, index_config_factory: Callable[..., IndexConfig]
    ) -> None:
        """Verifies that new backends can be registered and instantiated.

        Given:
            A custom implementation of VectorIndex.
        When:
            The class is registered via register_index_backend and requested via config.
        Then:
            IndexFactory creates an instance of the custom class.
        """

        original_registry = _INDEX_REGISTRY.copy()

        try:
            register_index_backend("custom_backend", FakeVectorIndex)
            config = index_config_factory(backend="custom_backend")
            index = IndexFactory.create(config)

            assert isinstance(index, FakeVectorIndex)
            assert index.config == config

        finally:
            _INDEX_REGISTRY.clear()
            _INDEX_REGISTRY.update(original_registry)


@pytest.mark.unit
class TestRetrieverFactory:
    """Tests for RetrieverFactory strategy selection."""

    def test_create_should_return_dense_retriever_when_mode_is_dense(
        self, retrieval_config_factory: Callable[..., RetrievalConfig]
    ) -> None:
        """Verifies dense retriever instantiation.

        Given:
            RetrievalConfig with mode='dense'.
        When:
            RetrieverFactory.create is called.
        Then:
            DenseRetriever is returned.
        """

        config = retrieval_config_factory(mode="dense")
        fake_embedder = FakeEmbedder()
        fake_index = FakeVectorIndex(embedder=fake_embedder)

        result = RetrieverFactory.create(config, fake_index)

        assert isinstance(result, DenseRetriever)

    def test_create_should_return_sparse_retriever_when_mode_is_sparse(
        self, retrieval_config_factory: Callable[..., RetrievalConfig]
    ) -> None:
        """Verifies sparse retriever instantiation.

        Given:
            RetrievalConfig with mode='sparse'.
        When:
            RetrieverFactory.create is called.
        Then:
            SparseRetriever is returned.
        """

        config = retrieval_config_factory(mode="sparse")
        fake_index = FakeVectorIndex()

        result = RetrieverFactory.create(config, fake_index)

        assert isinstance(result, SparseRetriever)

    def test_create_should_return_hybrid_retriever_when_mode_is_hybrid(
        self, retrieval_config_factory: Callable[..., RetrievalConfig]
    ) -> None:
        """Verifies hybrid retriever instantiation.

        Given:
            RetrievalConfig with mode='hybrid' and a valid alpha.
        When:
            RetrieverFactory.create is called.
        Then:
            HybridRetriever is returned.
        """

        config = retrieval_config_factory(mode="hybrid", alpha=0.5)
        fake_index = FakeVectorIndex()

        result = RetrieverFactory.create(config, fake_index)

        assert isinstance(result, HybridRetriever)

    def test_create_should_raise_value_error_when_hybrid_alpha_is_missing(
        self, retrieval_config_factory: Callable[..., RetrievalConfig]
    ) -> None:
        """Verifies validation logic for hybrid retrieval parameters.

        Given:
            RetrievalConfig with mode='hybrid' but alpha=None.
        When:
            RetrieverFactory.create is called.
        Then:
            ValueError is raised.
        """
        config = retrieval_config_factory(mode="hybrid", alpha=0.0)
        config.alpha = None

        fake_index = FakeVectorIndex()

        with pytest.raises(ValueError, match="alpha is required"):
            RetrieverFactory.create(config, fake_index)

    def test_create_should_wrap_retriever_when_reranker_is_configured(
        self, retrieval_config_factory: Callable[..., RetrievalConfig]
    ) -> None:
        """Verifies that Reranker configuration triggers RefinedRetriever.

        Given:
            RetrievalConfig with a valid reranker configuration.
        When:
            RetrieverFactory.create is called.
        Then:
            RefinedRetriever is created, wrapping the base retriever.
        """

        from ragmark.config.profile import RerankerConfig

        reranker_config = RerankerConfig(model_name="cross-encoder/test", top_k=5)
        config = retrieval_config_factory(mode="dense", reranker=reranker_config)
        fake_embedder = FakeEmbedder()
        fake_index = FakeVectorIndex(embedder=fake_embedder)

        with patch(
            "ragmark.retrieval.rerankers.CrossEncoderRefiner.from_config"
        ) as mock_refiner_factory:
            mock_refiner = MagicMock()
            mock_refiner_factory.return_value = mock_refiner

            result = RetrieverFactory.create(config, fake_index)

            mock_refiner_factory.assert_called_once_with(reranker_config)
            assert isinstance(result, RefinedRetriever)
            assert isinstance(result.base_retriever, DenseRetriever)

    def test_create_should_raise_error_when_mode_is_unknown(
        self,
    ) -> None:
        """Verifies error handling for invalid retrieval modes.

        Given:
            Config with an unknown retrieval mode.
        When:
            RetrieverFactory.create is called.
        Then:
            ValueError is raised.
        """

        config = RetrievalConfig.model_construct(
            mode="unknown_mode", top_k=10, reranker=None, alpha=None
        )
        fake_index = FakeVectorIndex()

        with pytest.raises(ValueError, match="Unknown retrieval mode"):
            RetrieverFactory.create(config, fake_index)


@pytest.mark.unit
class TestEmbedderFactory:
    """Tests for EmbedderFactory delegation and error handling."""

    def test_create_should_delegate_to_from_config_when_config_is_valid(
        self, embedder_config_factory: Callable[..., EmbedderConfig]
    ) -> None:
        """Verifies that the factory delegates creation to the specific embedder class.

        Note: We patch SentenceTransformerEmbedder here because it involves
        heavy I/O (model download), which is allowed by the policy.
        """

        config = embedder_config_factory(model_name="test-model")

        with patch(
            "ragmark.index.embedders.SentenceTransformerEmbedder.from_config"
        ) as mock_from_config:
            mock_embedder_instance = FakeEmbedder(config)
            mock_from_config.return_value = mock_embedder_instance

            result = EmbedderFactory.create(config)

            mock_from_config.assert_called_once_with(config)
            assert result is mock_embedder_instance

    def test_create_should_propagate_exception_when_embedder_instantiation_fails(
        self, embedder_config_factory: Callable[..., EmbedderConfig]
    ) -> None:
        """Verifies that the factory propagates exceptions from the underlying embedder.

        Given:
            A valid config but a backend that fails to initialize.
        When:
            EmbedderFactory.create() is called.
        Then:
            The specific exception is raised and not swallowed.
        """

        config = embedder_config_factory()
        simulated_error = UnsupportedBackendError("Model specific error")

        with patch(
            "ragmark.index.embedders.SentenceTransformerEmbedder.from_config",
            side_effect=simulated_error,
        ):
            with pytest.raises(UnsupportedBackendError) as exc_info:
                EmbedderFactory.create(config)

            assert exc_info.value is simulated_error

    @pytest.mark.parametrize("batch_size", [8, 16, 32, 64])
    def test_create_should_configure_batch_size_correctly(
        self, batch_size: int, embedder_config_factory: Callable[..., EmbedderConfig]
    ) -> None:
        """Verifies that the factory honors different batch size configurations.

        Given:
            An EmbedderConfig with a specific batch_size.
        When:
            EmbedderFactory.create() is called.
        Then:
            The resulting embedder is initialized with that specific batch_size.
        """

        config = embedder_config_factory(batch_size=batch_size)

        with patch(
            "ragmark.index.embedders.SentenceTransformerEmbedder.from_config"
        ) as mock_from_config:
            EmbedderFactory.create(config)

            actual_call_config = mock_from_config.call_args[0][0]
            assert actual_call_config.batch_size == batch_size

    @pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
    def test_create_should_configure_device_correctly(
        self, device: str, embedder_config_factory: Callable[..., EmbedderConfig]
    ) -> None:
        """Verifies that the factory honors different compute device configurations.

        Given:
            An EmbedderConfig with a specific device (cpu, cuda, mps).
        When:
            EmbedderFactory.create() is called.
        Then:
            The resulting embedder is initialized with that specific device.
        """

        config = embedder_config_factory(device=device)

        with patch(
            "ragmark.index.embedders.SentenceTransformerEmbedder.from_config"
        ) as mock_from_config:
            EmbedderFactory.create(config)

            actual_call_config = mock_from_config.call_args[0][0]
            assert actual_call_config.device == device
