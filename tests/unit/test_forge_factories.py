"""Unit tests for forge factory classes and from_config() pattern.

This module tests the from_config() implementation for forge components
(ingestors, fragmenters, ForgeRunner) and their factories.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ragmark.config.profile import (
    EvaluationConfig,
    ExperimentProfile,
    FragmenterConfig,
    GeneratorConfig,
    IngestorConfig,
)
from ragmark.exceptions import UnsupportedBackendError
from ragmark.forge.factory import FragmenterFactory, IngestorFactory
from ragmark.forge.runner import ForgeRunner


class TestIngestorFactory:
    """Tests for IngestorFactory."""

    def test_create_fitz_ingestor_raises_not_implemented(self) -> None:
        """Test that fitz backend raises NotImplementedError (placeholder)."""
        config = IngestorConfig(backend="fitz")

        # For now, concrete implementations don't exist, so should raise
        with pytest.raises(
            NotImplementedError, match="FitzIngestor will be implemented"
        ):
            IngestorFactory.create(config)

    def test_create_marker_ingestor_raises_not_implemented(self) -> None:
        """Test that marker backend raises NotImplementedError (placeholder)."""
        config = IngestorConfig(backend="marker")

        with pytest.raises(
            NotImplementedError, match="MarkerIngestor will be implemented"
        ):
            IngestorFactory.create(config)

    def test_create_unknown_backend_raises_error(self) -> None:
        """Test that unknown backend raises UnsupportedBackendError."""
        config = IngestorConfig(backend="fitz")
        config.backend = "unknown"  # type: ignore  # Bypass validation

        with pytest.raises(UnsupportedBackendError, match="unknown"):
            IngestorFactory.create(config)

    def test_create_with_options(self) -> None:
        """Test factory accepts config with options."""
        config = IngestorConfig(
            backend="fitz", options={"ocr_enabled": True, "languages": ["en", "fr"]}
        )

        # Should still raise NotImplementedError but validates options are passed
        with pytest.raises(NotImplementedError):
            IngestorFactory.create(config)


class TestFragmenterFactory:
    """Tests for FragmenterFactory."""

    def test_create_token_fragmenter_raises_not_implemented(self) -> None:
        """Test that token strategy raises NotImplementedError (placeholder)."""
        config = FragmenterConfig(strategy="token", chunk_size=256, overlap=64)

        with pytest.raises(
            NotImplementedError, match="TokenFragmenter will be implemented"
        ):
            FragmenterFactory.create(config)

    def test_create_semantic_fragmenter_raises_not_implemented(self) -> None:
        """Test that semantic strategy raises NotImplementedError (placeholder)."""
        config = FragmenterConfig(strategy="semantic", chunk_size=512, overlap=128)

        with pytest.raises(
            NotImplementedError, match="SemanticFragmenter will be implemented"
        ):
            FragmenterFactory.create(config)

    def test_create_markdown_fragmenter_raises_not_implemented(self) -> None:
        """Test that markdown strategy raises NotImplementedError (placeholder)."""
        config = FragmenterConfig(strategy="markdown", chunk_size=1024, overlap=256)

        with pytest.raises(
            NotImplementedError, match="MarkdownFragmenter will be implemented"
        ):
            FragmenterFactory.create(config)

    def test_create_recursive_fragmenter_raises_not_implemented(self) -> None:
        """Test that recursive strategy raises NotImplementedError (placeholder)."""
        config = FragmenterConfig(strategy="recursive", chunk_size=512, overlap=100)

        with pytest.raises(
            NotImplementedError, match="RecursiveFragmenter will be implemented"
        ):
            FragmenterFactory.create(config)

    def test_create_unknown_strategy_raises_error(self) -> None:
        """Test that unknown strategy raises ValueError."""
        config = FragmenterConfig(strategy="token", chunk_size=256, overlap=64)
        config.strategy = "unknown"  # type: ignore  # Bypass validation

        with pytest.raises(ValueError, match="Unknown fragmentation strategy"):
            FragmenterFactory.create(config)

    def test_create_with_options(self) -> None:
        """Test factory accepts config with options."""
        config = FragmenterConfig(
            strategy="token",
            chunk_size=256,
            overlap=64,
            options={"tokenizer": "cl100k_base", "separators": ["\n\n", "\n", " "]},
        )

        # Should still raise NotImplementedError but validates options are passed
        with pytest.raises(NotImplementedError):
            FragmenterFactory.create(config)


class TestForgeRunnerFromProfile:
    """Tests for ForgeRunner.from_profile()."""

    @pytest.fixture
    def minimal_profile(self) -> ExperimentProfile:
        """Create a minimal valid ExperimentProfile for testing."""
        return ExperimentProfile(
            generator=GeneratorConfig(model_path=Path("/tmp/model.gguf")),
            evaluation=EvaluationConfig(
                trial_cases_path=Path("/tmp/cases.jsonl"),
                judge_model_path=None,
            ),
        )

    def test_from_profile_creates_both_from_factories(
        self, minimal_profile: ExperimentProfile
    ) -> None:
        """Test that ForgeRunner creates both components via factories from profile."""
        ingestor_config = IngestorConfig(
            backend="marker", options={"ocr_enabled": True}
        )
        fragmenter_config = FragmenterConfig(
            strategy="semantic", chunk_size=768, overlap=192
        )

        # Override with custom config
        profile = ExperimentProfile(
            ingestor=ingestor_config,
            fragmenter=fragmenter_config,
            fail_fast=False,
            generator=minimal_profile.generator,
            evaluation=minimal_profile.evaluation,
        )

        with patch(
            "ragmark.forge.factory.IngestorFactory.create"
        ) as mock_ingestor_factory:
            with patch(
                "ragmark.forge.factory.FragmenterFactory.create"
            ) as mock_fragmenter_factory:
                mock_ingestor = MagicMock()
                mock_fragmenter = MagicMock()
                mock_ingestor_factory.return_value = mock_ingestor
                mock_fragmenter_factory.return_value = mock_fragmenter

                runner = ForgeRunner.from_profile(profile)

                # Verify both factories were called
                mock_ingestor_factory.assert_called_once_with(ingestor_config)
                mock_fragmenter_factory.assert_called_once_with(fragmenter_config)

                assert runner.ingestor is mock_ingestor
                assert runner.fragmenter is mock_fragmenter
                assert runner.fail_fast is False

    def test_from_profile_with_fail_fast_true(
        self, minimal_profile: ExperimentProfile
    ) -> None:
        """Test ForgeRunner.from_profile() with fail_fast=True."""
        profile = ExperimentProfile(
            fail_fast=True,
            generator=minimal_profile.generator,
            evaluation=minimal_profile.evaluation,
        )

        with patch(
            "ragmark.forge.factory.IngestorFactory.create"
        ) as mock_ingestor_factory:
            with patch(
                "ragmark.forge.factory.FragmenterFactory.create"
            ) as mock_fragmenter_factory:
                mock_ingestor = MagicMock()
                mock_fragmenter = MagicMock()
                mock_ingestor_factory.return_value = mock_ingestor
                mock_fragmenter_factory.return_value = mock_fragmenter

                runner = ForgeRunner.from_profile(profile)

                assert runner.fail_fast is True

    def test_from_profile_equivalent_to_direct_init(
        self, minimal_profile: ExperimentProfile
    ) -> None:
        """Test that from_profile produces same result as direct __init__."""
        mock_ingestor = MagicMock()
        mock_fragmenter = MagicMock()
        fail_fast = True

        # Create via direct init
        runner_direct = ForgeRunner(
            ingestor=mock_ingestor, fragmenter=mock_fragmenter, fail_fast=fail_fast
        )

        # Create via from_profile
        profile = ExperimentProfile(
            fail_fast=fail_fast,
            generator=minimal_profile.generator,
            evaluation=minimal_profile.evaluation,
        )

        with patch(
            "ragmark.forge.factory.IngestorFactory.create", return_value=mock_ingestor
        ):
            with patch(
                "ragmark.forge.factory.FragmenterFactory.create",
                return_value=mock_fragmenter,
            ):
                runner_from_profile = ForgeRunner.from_profile(profile)

        # Both should have same attributes
        assert runner_from_profile.ingestor is runner_direct.ingestor
        assert runner_from_profile.fragmenter is runner_direct.fragmenter
        assert runner_from_profile.fail_fast == runner_direct.fail_fast

    def test_from_profile_with_default_values(
        self, minimal_profile: ExperimentProfile
    ) -> None:
        """Test ForgeRunner.from_profile() with default profile values."""
        # Use minimal profile which has default values for ingestor/fragmenter
        profile = minimal_profile

        with patch(
            "ragmark.forge.factory.IngestorFactory.create"
        ) as mock_ingestor_factory:
            with patch(
                "ragmark.forge.factory.FragmenterFactory.create"
            ) as mock_fragmenter_factory:
                mock_ingestor = MagicMock()
                mock_fragmenter = MagicMock()
                mock_ingestor_factory.return_value = mock_ingestor
                mock_fragmenter_factory.return_value = mock_fragmenter

                runner = ForgeRunner.from_profile(profile)

                # Default fail_fast is True
                assert runner.fail_fast is True
                # Default ingestor backend is "fitz"
                assert profile.ingestor.backend == "fitz"
                # Default fragmenter strategy is "token"
                assert profile.fragmenter.strategy == "token"

    def test_from_profile_with_custom_ingestor_config(
        self, minimal_profile: ExperimentProfile
    ) -> None:
        """Test ForgeRunner.from_profile() with custom ingestor configuration."""
        ingestor_config = IngestorConfig(backend="fitz")
        profile = ExperimentProfile(
            ingestor=ingestor_config,
            generator=minimal_profile.generator,
            evaluation=minimal_profile.evaluation,
        )

        with patch(
            "ragmark.forge.factory.IngestorFactory.create"
        ) as mock_ingestor_factory:
            with patch("ragmark.forge.factory.FragmenterFactory.create"):
                mock_ingestor = MagicMock()
                mock_ingestor_factory.return_value = mock_ingestor

                runner = ForgeRunner.from_profile(profile)

                # Verify IngestorFactory was called with the right config
                mock_ingestor_factory.assert_called_once_with(ingestor_config)
                assert runner.ingestor is mock_ingestor

    def test_from_profile_with_custom_fragmenter_config(
        self, minimal_profile: ExperimentProfile
    ) -> None:
        """Test ForgeRunner.from_profile() with custom fragmenter configuration."""
        fragmenter_config = FragmenterConfig(strategy="token", chunk_size=512)
        profile = ExperimentProfile(
            fragmenter=fragmenter_config,
            generator=minimal_profile.generator,
            evaluation=minimal_profile.evaluation,
        )

        with patch("ragmark.forge.factory.IngestorFactory.create"):
            with patch(
                "ragmark.forge.factory.FragmenterFactory.create"
            ) as mock_fragmenter_factory:
                mock_fragmenter = MagicMock()
                mock_fragmenter_factory.return_value = mock_fragmenter

                runner = ForgeRunner.from_profile(profile)

                # Verify FragmenterFactory was called with the right config
                mock_fragmenter_factory.assert_called_once_with(fragmenter_config)
                assert runner.fragmenter is mock_fragmenter
