"""Unit tests for configuration management.

This module tests the ExperimentProfile configuration system including
YAML serialization, overrides, and validation.
"""

from pathlib import Path

import pytest

from ragmark.config.profile import (
    EvaluationConfig,
    ExperimentProfile,
    FragmenterConfig,
    GeneratorConfig,
    IngestorConfig,
    RetrievalConfig,
)
from ragmark.exceptions import ConfigError, ConfigOverrideError


class TestIngestorConfig:
    """Tests for IngestorConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = IngestorConfig()
        assert config.backend == "fitz"
        assert config.options == {}

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = IngestorConfig(
            backend="marker",
            options={"ocr_enabled": True, "languages": ["en", "fr"]},
        )
        assert config.backend == "marker"
        assert config.options["ocr_enabled"] is True

    def test_strict_mode(self) -> None:
        """Test that extra fields are rejected."""
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            IngestorConfig(backend="fitz", unknown_field="value")  # type: ignore


class TestFragmenterConfig:
    """Tests for FragmenterConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = FragmenterConfig()
        assert config.strategy == "token"
        assert config.chunk_size == 256
        assert config.overlap == 64

    def test_overlap_validation(self) -> None:
        """Test that overlap must be less than chunk_size."""
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            FragmenterConfig(chunk_size=100, overlap=100)

        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            FragmenterConfig(chunk_size=100, overlap=200)

    def test_valid_overlap(self) -> None:
        """Test valid overlap values."""
        config = FragmenterConfig(chunk_size=512, overlap=128)
        assert config.chunk_size == 512
        assert config.overlap == 128


class TestRetrievalConfig:
    """Tests for RetrievalConfig."""

    def test_alpha_required_for_hybrid(self) -> None:
        """Test that alpha is required when mode is hybrid."""
        with pytest.raises(ValueError, match="alpha is required when mode is 'hybrid'"):
            RetrievalConfig(mode="hybrid", top_k=10, alpha=None, reranker=None)

    def test_alpha_optional_for_dense(self) -> None:
        """Test that alpha is optional for dense mode."""
        config = RetrievalConfig(mode="dense", top_k=10, alpha=None, reranker=None)
        assert config.alpha is None

    def test_hybrid_with_alpha(self) -> None:
        """Test hybrid mode with valid alpha."""
        config = RetrievalConfig(mode="hybrid", top_k=10, alpha=0.5, reranker=None)
        assert config.alpha == 0.5


class TestExperimentProfile:
    """Tests for ExperimentProfile."""

    def test_minimal_profile(self, tmp_path: Path) -> None:
        """Test creating a minimal profile."""
        profile = ExperimentProfile(
            generator=GeneratorConfig(model_path=tmp_path / "model.gguf"),
            evaluation=EvaluationConfig(
                trial_cases_path=tmp_path / "cases.jsonl", judge_model_path=None
            ),
        )
        assert profile.ingestor.backend == "fitz"
        assert profile.fragmenter.strategy == "token"

    def test_from_yaml_valid(self, tmp_path: Path, sample_yaml_config: str) -> None:
        """Test loading a valid YAML configuration."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(sample_yaml_config)

        profile = ExperimentProfile.from_yaml(config_path)
        assert profile.fragmenter.chunk_size == 512
        assert profile.fragmenter.overlap == 128
        assert profile.index.backend == "memory"

    def test_from_yaml_file_not_found(self, tmp_path: Path) -> None:
        """Test that FileNotFoundError is raised for missing files."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            ExperimentProfile.from_yaml(tmp_path / "nonexistent.yaml")

    def test_from_yaml_invalid_syntax(self, tmp_path: Path) -> None:
        """Test that ConfigError is raised for invalid YAML."""
        config_path = tmp_path / "invalid.yaml"
        config_path.write_text("invalid: yaml: syntax:")

        with pytest.raises(ConfigError, match="Invalid YAML syntax"):
            ExperimentProfile.from_yaml(config_path)

    def test_from_yaml_validation_error(self, tmp_path: Path) -> None:
        """Test that ConfigError is raised for validation failures."""
        config_path = tmp_path / "invalid_config.yaml"
        config_path.write_text(
            """
fragmenter:
  chunk_size: 100
  overlap: 200
"""
        )

        with pytest.raises(ConfigError, match="Configuration validation failed"):
            ExperimentProfile.from_yaml(config_path)

    def test_resolve_paths(self, tmp_path: Path, sample_yaml_config: str) -> None:
        """Test that relative paths are resolved correctly."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(sample_yaml_config)

        profile = ExperimentProfile.from_yaml(config_path)

        # The generator model_path should be resolved relative to config_path
        expected_path = tmp_path / "models" / "test-model.gguf"
        assert profile.generator.model_path == expected_path

    def test_to_yaml(self, tmp_path: Path) -> None:
        """Test exporting configuration to YAML."""
        output_path = tmp_path / "output.yaml"

        profile = ExperimentProfile(
            fragmenter=FragmenterConfig(chunk_size=512, overlap=128),
            generator=GeneratorConfig(model_path=tmp_path / "model.gguf"),
            evaluation=EvaluationConfig(
                trial_cases_path=tmp_path / "cases.jsonl", judge_model_path=None
            ),
        )

        profile.to_yaml(output_path)

        assert output_path.exists()

        # Verify round-trip
        loaded = ExperimentProfile.from_yaml(output_path)
        assert loaded.fragmenter.chunk_size == 512
        assert loaded.fragmenter.overlap == 128

    def test_with_overrides_valid(self, tmp_path: Path) -> None:
        """Test applying valid overrides."""
        profile = ExperimentProfile(
            fragmenter=FragmenterConfig(chunk_size=256, overlap=64),
            generator=GeneratorConfig(model_path=tmp_path / "model.gguf"),
            evaluation=EvaluationConfig(
                trial_cases_path=tmp_path / "cases.jsonl", judge_model_path=None
            ),
        )

        overridden = profile.with_overrides(
            {
                "fragmenter.chunk_size": 512,
                "retrieval.top_k": 20,
            }
        )

        # Original should be unchanged (immutability)
        assert profile.fragmenter.chunk_size == 256
        assert profile.retrieval.top_k == 10

        # New instance should have overrides
        assert overridden.fragmenter.chunk_size == 512
        assert overridden.retrieval.top_k == 20

    def test_with_overrides_invalid_key(self, tmp_path: Path) -> None:
        """Test that invalid override keys raise ConfigOverrideError."""
        profile = ExperimentProfile(
            generator=GeneratorConfig(model_path=tmp_path / "model.gguf"),
            evaluation=EvaluationConfig(
                trial_cases_path=tmp_path / "cases.jsonl", judge_model_path=None
            ),
        )

        with pytest.raises(ConfigOverrideError, match="Invalid override key"):
            profile.with_overrides({"nonexistent.field": 123})

    def test_with_overrides_validation_error(self, tmp_path: Path) -> None:
        """Test that invalid override values raise ConfigOverrideError."""
        profile = ExperimentProfile(
            generator=GeneratorConfig(model_path=tmp_path / "model.gguf"),
            evaluation=EvaluationConfig(
                trial_cases_path=tmp_path / "cases.jsonl", judge_model_path=None
            ),
        )

        with pytest.raises(ConfigOverrideError, match="Override validation failed"):
            profile.with_overrides({"fragmenter.chunk_size": -100})

    def test_diff(self, tmp_path: Path) -> None:
        """Test computing differences between profiles."""
        profile1 = ExperimentProfile(
            fragmenter=FragmenterConfig(chunk_size=256, overlap=64),
            generator=GeneratorConfig(model_path=tmp_path / "model.gguf"),
            evaluation=EvaluationConfig(
                trial_cases_path=tmp_path / "cases.jsonl", judge_model_path=None
            ),
        )

        profile2 = ExperimentProfile(
            fragmenter=FragmenterConfig(chunk_size=512, overlap=128),
            generator=GeneratorConfig(model_path=tmp_path / "model.gguf"),
            evaluation=EvaluationConfig(
                trial_cases_path=tmp_path / "cases.jsonl", judge_model_path=None
            ),
        )

        diffs = profile1.diff(profile2)

        assert "fragmenter.chunk_size" in diffs
        assert diffs["fragmenter.chunk_size"] == (256, 512)
        assert "fragmenter.overlap" in diffs
        assert diffs["fragmenter.overlap"] == (64, 128)

    def test_compute_hash(self, tmp_path: Path) -> None:
        """Test that hash is deterministic and changes with config."""
        profile1 = ExperimentProfile(
            fragmenter=FragmenterConfig(chunk_size=256),
            generator=GeneratorConfig(model_path=tmp_path / "model.gguf"),
            evaluation=EvaluationConfig(
                trial_cases_path=tmp_path / "cases.jsonl", judge_model_path=None
            ),
        )

        profile2 = ExperimentProfile(
            fragmenter=FragmenterConfig(chunk_size=256),
            generator=GeneratorConfig(model_path=tmp_path / "model.gguf"),
            evaluation=EvaluationConfig(
                trial_cases_path=tmp_path / "cases.jsonl", judge_model_path=None
            ),
        )

        profile3 = ExperimentProfile(
            fragmenter=FragmenterConfig(chunk_size=512),  # Different
            generator=GeneratorConfig(model_path=tmp_path / "model.gguf"),
            evaluation=EvaluationConfig(
                trial_cases_path=tmp_path / "cases.jsonl", judge_model_path=None
            ),
        )

        # Same configs should have same hash
        assert profile1.compute_hash() == profile2.compute_hash()

        # Different configs should have different hash
        assert profile1.compute_hash() != profile3.compute_hash()
