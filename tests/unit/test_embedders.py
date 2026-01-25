"""Unit tests for embedder from_config() pattern.

This module tests the from_config() implementation for embedders
and the EmbedderFactory delegation.
"""

from unittest.mock import MagicMock, patch

import pytest

from ragmark.config.profile import EmbedderConfig
from ragmark.exceptions import UnsupportedBackendError
from ragmark.index.embedders import EmbedderFactory


class TestEmbedderFactory:
    """Tests for EmbedderFactory delegation to from_config()."""

    def test_factory_delegates_to_from_config(self) -> None:
        """Test that EmbedderFactory.create() delegates to from_config()."""
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
        config = EmbedderConfig(model_name="test-model")

        # Simulate ImportError when trying to import from backends module
        with patch.dict(
            "sys.modules",
            {"ragmark.index.backends": None},
        ):
            # Force re-import to trigger ImportError
            with pytest.raises(UnsupportedBackendError, match="sentence-transformers"):
                EmbedderFactory.create(config)

    def test_factory_with_different_batch_sizes(self) -> None:
        """Test factory creates embedder with various batch sizes."""
        for batch_size in [8, 16, 32, 64]:
            config = EmbedderConfig(
                model_name="test-model",
                device="cpu",
                batch_size=batch_size,
            )

            with patch(
                "ragmark.index.backends.SentenceTransformerEmbedder.from_config"
            ) as mock_from_config:
                mock_embedder = MagicMock()
                mock_from_config.return_value = mock_embedder

                result = EmbedderFactory.create(config)

                assert result is mock_embedder
                call_args = mock_from_config.call_args[0]
                assert call_args[0].batch_size == batch_size

    def test_factory_with_different_devices(self) -> None:
        """Test factory with various device configurations."""
        for device in ["cpu", "cuda", "mps"]:
            config = EmbedderConfig(
                model_name="test-model",
                device=device,
                batch_size=32,
            )

            with patch(
                "ragmark.index.backends.SentenceTransformerEmbedder.from_config"
            ) as mock_from_config:
                mock_embedder = MagicMock()
                mock_from_config.return_value = mock_embedder

                result = EmbedderFactory.create(config)

                assert result is mock_embedder
                call_args = mock_from_config.call_args[0]
                assert call_args[0].device == device
