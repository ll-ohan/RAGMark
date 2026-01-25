"""Unit tests for retriever from_config() pattern.

This module tests the from_config() implementation for all retriever strategies
and the RetrieverFactory delegation.
"""

from unittest.mock import MagicMock, patch

import pytest

from ragmark.config.profile import RerankerConfig, RetrievalConfig
from ragmark.retrieval.factory import RetrieverFactory
from ragmark.retrieval.strategies import (
    DenseRetriever,
    HybridRetriever,
    RefinedRetriever,
    SparseRetriever,
)


class TestDenseRetrieverFromConfig:
    """Tests for DenseRetriever.from_config()."""

    def test_from_config_basic(self) -> None:
        """Test basic instantiation from config."""
        config = RetrievalConfig(mode="dense", top_k=10, alpha=None, reranker=None)
        mock_index = MagicMock()

        retriever = DenseRetriever.from_config(config, mock_index)

        assert retriever.index is mock_index
        assert retriever.top_k == 10

    def test_from_config_custom_top_k(self) -> None:
        """Test with custom top_k values."""
        for top_k in [5, 10, 20, 50]:
            config = RetrievalConfig(
                mode="dense", top_k=top_k, alpha=None, reranker=None
            )
            mock_index = MagicMock()

            retriever = DenseRetriever.from_config(config, mock_index)

            assert retriever.top_k == top_k

    def test_from_config_ignores_refiner(self) -> None:
        """Test that refiner parameter is ignored for DenseRetriever."""
        config = RetrievalConfig(mode="dense", top_k=10, alpha=None, reranker=None)
        mock_index = MagicMock()
        mock_refiner = MagicMock()

        # Should not raise even with refiner passed
        retriever = DenseRetriever.from_config(config, mock_index, refiner=mock_refiner)

        assert retriever.index is mock_index
        assert retriever.top_k == 10


class TestSparseRetrieverFromConfig:
    """Tests for SparseRetriever.from_config()."""

    def test_from_config_basic(self) -> None:
        """Test basic instantiation from config."""
        config = RetrievalConfig(mode="sparse", top_k=15, alpha=None, reranker=None)
        mock_index = MagicMock()

        retriever = SparseRetriever.from_config(config, mock_index)

        assert retriever.index is mock_index
        assert retriever.top_k == 15

    def test_from_config_ignores_refiner(self) -> None:
        """Test that refiner parameter is ignored for SparseRetriever."""
        config = RetrievalConfig(mode="sparse", top_k=10, alpha=None, reranker=None)
        mock_index = MagicMock()
        mock_refiner = MagicMock()

        retriever = SparseRetriever.from_config(
            config, mock_index, refiner=mock_refiner
        )

        assert retriever.index is mock_index


class TestHybridRetrieverFromConfig:
    """Tests for HybridRetriever.from_config()."""

    def test_from_config_basic(self) -> None:
        """Test basic instantiation from config."""
        config = RetrievalConfig(mode="hybrid", top_k=10, alpha=0.5, reranker=None)
        mock_index = MagicMock()

        retriever = HybridRetriever.from_config(config, mock_index)

        assert retriever.index is mock_index
        assert retriever.top_k == 10
        assert retriever.alpha == 0.5

    def test_from_config_custom_alpha(self) -> None:
        """Test with various alpha values."""
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            config = RetrievalConfig(
                mode="hybrid", top_k=10, alpha=alpha, reranker=None
            )
            mock_index = MagicMock()

            retriever = HybridRetriever.from_config(config, mock_index)

            assert retriever.alpha == alpha

    def test_from_config_raises_without_alpha(self) -> None:
        """Test that ValueError is raised when alpha is None."""
        # Use MagicMock to bypass Pydantic validation
        config = MagicMock()
        config.mode = "hybrid"
        config.top_k = 10
        config.alpha = None
        mock_index = MagicMock()

        with pytest.raises(ValueError, match="alpha is required for hybrid retrieval"):
            HybridRetriever.from_config(config, mock_index)


class TestRefinedRetrieverFromConfig:
    """Tests for RefinedRetriever.from_config()."""

    def test_from_config_with_dense_base(self) -> None:
        """Test creating RefinedRetriever with dense base retriever."""
        config = RetrievalConfig(mode="dense", top_k=10, alpha=None, reranker=None)
        mock_index = MagicMock()
        mock_refiner = MagicMock()

        retriever = RefinedRetriever.from_config(
            config, mock_index, refiner=mock_refiner
        )

        assert isinstance(retriever.base_retriever, DenseRetriever)
        assert retriever.base_retriever.index is mock_index
        assert retriever.refiner is mock_refiner

    def test_from_config_with_sparse_base(self) -> None:
        """Test creating RefinedRetriever with sparse base retriever."""
        config = RetrievalConfig(mode="sparse", top_k=10, alpha=None, reranker=None)
        mock_index = MagicMock()
        mock_refiner = MagicMock()

        retriever = RefinedRetriever.from_config(
            config, mock_index, refiner=mock_refiner
        )

        assert isinstance(retriever.base_retriever, SparseRetriever)
        assert retriever.refiner is mock_refiner

    def test_from_config_with_hybrid_base(self) -> None:
        """Test creating RefinedRetriever with hybrid base retriever."""
        config = RetrievalConfig(mode="hybrid", top_k=10, alpha=0.7, reranker=None)
        mock_index = MagicMock()
        mock_refiner = MagicMock()

        retriever = RefinedRetriever.from_config(
            config, mock_index, refiner=mock_refiner
        )

        assert isinstance(retriever.base_retriever, HybridRetriever)
        assert retriever.base_retriever.alpha == 0.7

    def test_from_config_raises_without_refiner(self) -> None:
        """Test that ValueError is raised when refiner is None."""
        config = RetrievalConfig(mode="dense", top_k=10, alpha=None, reranker=None)
        mock_index = MagicMock()

        with pytest.raises(ValueError, match="RefinedRetriever requires a refiner"):
            RefinedRetriever.from_config(config, mock_index, refiner=None)

    def test_from_config_raises_with_unknown_mode(self) -> None:
        """Test that ValueError is raised for unknown retrieval mode."""
        config = RetrievalConfig(mode="dense", top_k=10, alpha=None, reranker=None)
        # Bypass validation by changing mode after creation
        config.mode = "unknown"  # type: ignore
        mock_index = MagicMock()
        mock_refiner = MagicMock()

        with pytest.raises(ValueError, match="Unknown retrieval mode"):
            RefinedRetriever.from_config(config, mock_index, refiner=mock_refiner)


class TestRetrieverFactory:
    """Tests for RetrieverFactory delegation to from_config()."""

    def test_factory_creates_dense_retriever(self) -> None:
        """Test factory creates DenseRetriever via from_config()."""
        config = RetrievalConfig(mode="dense", top_k=10, alpha=None, reranker=None)
        mock_index = MagicMock()

        with patch(
            "ragmark.retrieval.strategies.DenseRetriever.from_config"
        ) as mock_from_config:
            mock_retriever = MagicMock()
            mock_from_config.return_value = mock_retriever

            result = RetrieverFactory.create(config, mock_index)

            mock_from_config.assert_called_once_with(config, mock_index)
            assert result is mock_retriever

    def test_factory_creates_sparse_retriever(self) -> None:
        """Test factory creates SparseRetriever via from_config()."""
        config = RetrievalConfig(mode="sparse", top_k=10, alpha=None, reranker=None)
        mock_index = MagicMock()

        with patch(
            "ragmark.retrieval.strategies.SparseRetriever.from_config"
        ) as mock_from_config:
            mock_retriever = MagicMock()
            mock_from_config.return_value = mock_retriever

            result = RetrieverFactory.create(config, mock_index)

            mock_from_config.assert_called_once_with(config, mock_index)
            assert result is mock_retriever

    def test_factory_creates_hybrid_retriever(self) -> None:
        """Test factory creates HybridRetriever via from_config()."""
        config = RetrievalConfig(mode="hybrid", top_k=10, alpha=0.5, reranker=None)
        mock_index = MagicMock()

        with patch(
            "ragmark.retrieval.strategies.HybridRetriever.from_config"
        ) as mock_from_config:
            mock_retriever = MagicMock()
            mock_from_config.return_value = mock_retriever

            result = RetrieverFactory.create(config, mock_index)

            mock_from_config.assert_called_once_with(config, mock_index)
            assert result is mock_retriever

    def test_factory_creates_refined_retriever_with_reranker(self) -> None:
        """Test factory creates RefinedRetriever when reranker is configured."""
        reranker_config = RerankerConfig(
            model_name="cross-encoder/test",
            device="cpu",
            top_k=5,
        )
        config = RetrievalConfig(
            mode="dense", top_k=10, reranker=reranker_config, alpha=None
        )
        mock_index = MagicMock()

        with patch(
            "ragmark.retrieval.rerankers.CrossEncoderRefiner.from_config"
        ) as mock_refiner_from_config:
            with patch(
                "ragmark.retrieval.strategies.RefinedRetriever.from_config"
            ) as mock_refined_from_config:
                mock_refiner = MagicMock()
                mock_retriever = MagicMock()
                mock_refiner_from_config.return_value = mock_refiner
                mock_refined_from_config.return_value = mock_retriever

                result = RetrieverFactory.create(config, mock_index)

                # Verify refiner was created from config
                mock_refiner_from_config.assert_called_once_with(reranker_config)

                # Verify RefinedRetriever.from_config was called with refiner
                mock_refined_from_config.assert_called_once_with(
                    config, mock_index, refiner=mock_refiner
                )
                assert result is mock_retriever

    def test_factory_raises_for_unknown_mode(self) -> None:
        """Test factory raises ValueError for unknown retrieval mode."""
        config = RetrievalConfig(mode="dense", top_k=10, alpha=0.5, reranker=None)
        config.mode = "unknown"  # type: ignore
        mock_index = MagicMock()

        with pytest.raises(ValueError, match="Unknown retrieval mode"):
            RetrieverFactory.create(config, mock_index)
