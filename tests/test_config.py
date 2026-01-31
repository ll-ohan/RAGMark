"""Unit tests for configuration management."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ragmark.config.profile import (
    EvaluationConfig,
    ExperimentProfile,
    FragmenterConfig,
    GeneratorConfig,
    IndexConfig,
    IngestorConfig,
    RerankerConfig,
    RetrievalConfig,
)
from ragmark.exceptions import ConfigError, ConfigOverrideError


@pytest.mark.unit
class TestIngestorConfig:
    """Tests for IngestorConfig validation and defaults."""

    def test_init_should_set_correct_defaults_when_no_args_provided(self) -> None:
        """
        Scenario: Initializing IngestorConfig without arguments.
        Action: Create an empty instance.
        Then: Backend should be 'fitz' and options should be an empty dict.
        """
        config = IngestorConfig()
        assert config.backend == "fitz"
        assert config.options == {}

    def test_init_should_accept_custom_values_when_provided(self) -> None:
        """
        Scenario: Initializing IngestorConfig with specific backend and options.
        Action: Create instance with 'marker' backend and OCR options.
        Then: Fields should reflect the provided values.
        """
        config = IngestorConfig(
            backend="marker",
            options={"ocr_enabled": True, "languages": ["en", "fr"]},
        )
        assert config.backend == "marker"
        assert config.options["ocr_enabled"] is True
        assert "fr" in config.options["languages"]

    def test_init_should_raise_validation_error_when_extra_fields_present(self) -> None:
        """
        Scenario: Initializing with unknown fields (strict mode).
        Action: Pass 'unknown_field' to constructor.
        Then: Should raise ValueError indicating extra inputs are forbidden.
        """
        invalid_data = {"backend": "fitz", "unknown_field": "value"}
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            IngestorConfig(**invalid_data)  # type: ignore[arg-type]


@pytest.mark.unit
class TestFragmenterConfig:
    """Tests for FragmenterConfig validation logic and boundaries."""

    def test_init_should_set_correct_defaults_when_no_args_provided(self) -> None:
        """
        Scenario: Initializing FragmenterConfig without arguments.
        Action: Create instance.
        Then: Default strategy should be 'token', chunk_size 256, overlap 64.
        """
        config = FragmenterConfig()
        assert config.strategy == "token"
        assert config.chunk_size == 256
        assert config.overlap == 64

    def test_init_should_raise_value_error_when_overlap_equals_or_exceeds_chunk_size(
        self,
    ) -> None:
        """
        Scenario: Overlap is greater than or equal to chunk_size.
        Action: Create FragmenterConfig with invalid dimensions.
        Then: ValueError should be raised with explicit message.
        """
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            FragmenterConfig(chunk_size=100, overlap=100)

        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            FragmenterConfig(chunk_size=100, overlap=200)

    def test_init_should_raise_validation_error_when_chunk_size_is_out_of_bounds(
        self,
    ) -> None:
        """
        Scenario: chunk_size is outside the allowed range [50, 8192].
        Action: Initialize FragmenterConfig with invalid values.
        Then: ValidationError should be raised checking ge and le constraints.
        """
        with pytest.raises(ValidationError) as exc_info_low:
            FragmenterConfig(chunk_size=49)
        assert "greater than or equal to 50" in str(exc_info_low.value)

        with pytest.raises(ValidationError) as exc_info_high:
            FragmenterConfig(chunk_size=8193)
        assert "less than or equal to 8192" in str(exc_info_high.value)

    def test_init_should_succeed_when_overlap_is_valid(self) -> None:
        """
        Scenario: Overlap is strictly less than chunk_size.
        Action: Create FragmenterConfig with valid dimensions.
        Then: Configuration object is created correctly.
        """
        config = FragmenterConfig(chunk_size=512, overlap=128)
        assert config.chunk_size == 512
        assert config.overlap == 128


@pytest.mark.unit
class TestIndexConfig:
    """Tests for IndexConfig validation rules (backend vs connection)."""

    def test_init_should_succeed_when_memory_backend_has_no_connection(self) -> None:
        """
        Scenario: Memory backend usually requires no connection params.
        Action: Initialize IndexConfig with default backend.
        Then: Instance is created successfully.
        """
        config = IndexConfig(backend="memory", connection=None)
        assert config.backend == "memory"

    def test_init_should_raise_value_error_when_remote_backend_lacks_connection(
        self,
    ) -> None:
        """
        Scenario: Using a remote backend (e.g., qdrant) without connection details.
        Action: Initialize IndexConfig with 'qdrant' and None connection.
        Then: ValueError should be raised by the model validator.
        """
        with pytest.raises(ValueError, match="connection"):
            IndexConfig(backend="qdrant", connection=None)

    def test_init_should_succeed_when_remote_backend_has_connection(self) -> None:
        """
        Scenario: Using a remote backend with valid connection dict.
        Action: Initialize IndexConfig with 'qdrant' and connection params.
        Then: Instance is created successfully.
        """
        config = IndexConfig(
            backend="qdrant", connection={"url": "http://localhost:6333"}
        )
        assert config.connection is not None
        assert config.connection["url"] == "http://localhost:6333"


@pytest.mark.unit
class TestRetrievalConfig:
    """Tests for RetrievalConfig logic and boundaries."""

    def test_init_should_raise_value_error_when_hybrid_mode_misses_alpha(self) -> None:
        """
        Scenario: Mode is 'hybrid' but alpha is None.
        Action: Initialize RetrievalConfig.
        Then: ValueError should be raised requiring alpha.
        """
        with pytest.raises(ValueError, match="alpha is required when mode is 'hybrid'"):
            RetrievalConfig(mode="hybrid", top_k=10, alpha=None, reranker=None)

    def test_init_should_raise_validation_error_when_top_k_exceeds_limit(self) -> None:
        """
        Scenario: top_k exceeds the maximum allowed (100).
        Action: Initialize RetrievalConfig with top_k=101.
        Then: ValidationError should be raised (le=100 constraint).
        """
        with pytest.raises(ValidationError) as exc_info:
            RetrievalConfig(mode="dense", top_k=101, alpha=None, reranker=None)
        assert "less than or equal to 100" in str(exc_info.value)

    def test_init_should_allow_none_alpha_when_mode_is_dense(self) -> None:
        """
        Scenario: Mode is 'dense'.
        Action: Initialize RetrievalConfig with alpha=None.
        Then: Config should accept None for alpha.
        """
        config = RetrievalConfig(mode="dense", top_k=10, alpha=None, reranker=None)
        assert config.alpha is None

    def test_init_should_set_alpha_when_mode_is_hybrid(self) -> None:
        """
        Scenario: Mode is 'hybrid' and alpha is provided.
        Action: Initialize RetrievalConfig.
        Then: Alpha value is correctly stored.
        """
        config = RetrievalConfig(mode="hybrid", top_k=10, alpha=0.5, reranker=None)
        assert config.alpha == 0.5


@pytest.mark.unit
class TestRerankerConfig:
    """Tests for RerankerConfig validation."""

    def test_init_should_raise_validation_error_when_top_k_is_invalid(self) -> None:
        """
        Scenario: top_k is less than the minimum allowed (1).
        Action: Initialize RerankerConfig with top_k=0.
        Then: ValidationError should be raised (ge=1 constraint).
        """
        with pytest.raises(ValidationError) as exc_info:
            RerankerConfig(model_name="bert", top_k=0)
        assert "greater than or equal to 1" in str(exc_info.value)


@pytest.mark.unit
class TestGeneratorConfig:
    """Tests for GeneratorConfig boundary values."""

    def test_init_should_raise_validation_error_when_temperature_exceeds_limit(
        self, tmp_path: Path
    ) -> None:
        """
        Scenario: Temperature provided is above the maximum allowed (2.0).
        Action: Initialize GeneratorConfig with temperature=2.1.
        Then: Pydantic ValidationError should be raised.
        """
        with pytest.raises(ValidationError) as exc_info:
            GeneratorConfig(model_path=tmp_path / "model.gguf", temperature=2.1)

        assert "less than or equal to 2" in str(exc_info.value)


@pytest.mark.unit
class TestExperimentProfile:
    """Tests for ExperimentProfile serialization, overrides, and hashing."""

    def test_init_should_create_minimal_profile_with_defaults(
        self, tmp_path: Path
    ) -> None:
        """
        Scenario: Creating a profile with only required nested configs.
        Action: Initialize ExperimentProfile with Generator and Evaluation configs.
        Then: Optional components (Ingestor, Fragmenter) should use default values.
        """
        profile = ExperimentProfile(
            generator=GeneratorConfig(model_path=tmp_path / "model.gguf"),
            evaluation=EvaluationConfig(
                trial_cases_path=tmp_path / "cases.jsonl", judge_model_path=None
            ),
        )
        assert profile.ingestor.backend == "fitz"
        assert profile.fragmenter.strategy == "token"

    def test_from_yaml_should_load_valid_config(
        self, tmp_path: Path, sample_yaml_config: str
    ) -> None:
        """
        Scenario: Loading a valid YAML configuration file.
        Action: Call ExperimentProfile.from_yaml().
        Then: All nested configurations should match YAML content.
        """
        config_path = tmp_path / "config.yaml"
        config_path.write_text(sample_yaml_config, encoding="utf-8")

        profile = ExperimentProfile.from_yaml(config_path)

        assert profile.fragmenter.chunk_size == 256
        assert profile.fragmenter.overlap == 64
        assert profile.index.backend == "memory"
        assert profile.embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_from_yaml_should_raise_file_not_found_when_path_missing(
        self, tmp_path: Path
    ) -> None:
        """
        Scenario: Providing a path to a non-existent file.
        Action: Call ExperimentProfile.from_yaml().
        Then: FileNotFoundError should be raised.
        """
        missing_path = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            ExperimentProfile.from_yaml(missing_path)

    def test_from_yaml_should_raise_config_error_when_syntax_is_invalid(
        self, tmp_path: Path
    ) -> None:
        """
        Scenario: YAML file contains syntax errors.
        Action: Call ExperimentProfile.from_yaml().
        Then: ConfigError should be raised and preserve the original YAMLError as cause.
        """
        config_path = tmp_path / "invalid.yaml"
        config_path.write_text("invalid: yaml: syntax:", encoding="utf-8")

        with pytest.raises(ConfigError, match="Invalid YAML syntax") as exc_info:
            ExperimentProfile.from_yaml(config_path)

        assert exc_info.value.__cause__ is not None

    def test_from_yaml_should_raise_config_error_when_validation_fails(
        self, tmp_path: Path
    ) -> None:
        """
        Scenario: YAML structure is valid but breaks schema rules (e.g. overlap > chunk).
        Action: Call ExperimentProfile.from_yaml().
        Then: ConfigError should be raised due to inner validation failure.
        """
        config_path = tmp_path / "invalid_config.yaml"
        config_path.write_text(
            """
fragmenter:
  chunk_size: 100
  overlap: 200
""",
            encoding="utf-8",
        )

        with pytest.raises(ConfigError, match="Configuration validation failed"):
            ExperimentProfile.from_yaml(config_path)

    def test_resolve_paths_should_resolve_relative_paths_against_config_file(
        self, tmp_path: Path, sample_yaml_config: str
    ) -> None:
        """
        Scenario: YAML contains relative paths.
        Action: Load config and check path resolution.
        Then: Paths should be resolved relative to the configuration file's directory.
        """
        config_path = tmp_path / "config.yaml"
        config_path.write_text(sample_yaml_config, encoding="utf-8")

        profile = ExperimentProfile.from_yaml(config_path)

        expected_path = tmp_path / "models" / "test-model.gguf"
        assert profile.generator is not None
        assert profile.generator.model_path == expected_path

    def test_resolve_paths_should_leave_absolute_paths_untouched(
        self, tmp_path: Path, sample_yaml_config: str
    ) -> None:
        """
        Scenario: YAML contains an absolute path.
        Action: Load config where 'trial_cases_path' is absolute.
        Then: The path should remain absolute and not be joined with config dir.
        """
        abs_path = tmp_path / "absolute" / "data.jsonl"

        modified_yaml = sample_yaml_config.replace(
            "data/trial_cases.jsonl", str(abs_path)
        )

        config_path = tmp_path / "config_abs.yaml"
        config_path.write_text(modified_yaml, encoding="utf-8")

        profile = ExperimentProfile.from_yaml(config_path)

        assert profile.evaluation is not None
        assert profile.evaluation.trial_cases_path == abs_path

    def test_resolve_paths_should_handle_lists_recursively(
        self, tmp_path: Path
    ) -> None:
        """
        Scenario: Configuration contains a list of relative paths.
        Action: Call _resolve_paths directly with a list structure.
        Then: All paths inside the list should be resolved relative to base_dir.
        """
        base_dir = tmp_path / "configs"
        data = {"extra_files": ["data/file1.txt", "data/file2.txt"]}

        resolved = ExperimentProfile._resolve_paths(data, base_dir)

        expected_path_1 = base_dir / "data/file1.txt"
        expected_path_2 = base_dir / "data/file2.txt"

        assert resolved["extra_files"][0] == expected_path_1
        assert resolved["extra_files"][1] == expected_path_2

    def test_to_yaml_should_export_correct_configuration(self, tmp_path: Path) -> None:
        """
        Scenario: Exporting a profile instance to YAML.
        Action: Call profile.to_yaml() then reload it.
        Then: The reloaded profile should match the original (Round-trip test).
        """
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

        loaded = ExperimentProfile.from_yaml(output_path)
        assert loaded.fragmenter.chunk_size == 512
        assert loaded.fragmenter.overlap == 128

    def test_with_overrides_should_apply_valid_changes_immutably(
        self, tmp_path: Path
    ) -> None:
        """
        Scenario: Applying dot-notation overrides to a profile.
        Action: Call with_overrides with valid key-values.
        Then: Return a new instance with changes; original remains untouched.
        """
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

        assert profile.fragmenter.chunk_size == 256
        assert profile.retrieval.top_k == 10

        assert overridden.fragmenter.chunk_size == 512
        assert overridden.retrieval.top_k == 20

    def test_with_overrides_should_raise_error_on_unknown_key(
        self, tmp_path: Path
    ) -> None:
        """
        Scenario: Applying override with a non-existent configuration key.
        Action: Call with_overrides.
        Then: ConfigOverrideError should be raised.
        """
        profile = ExperimentProfile(
            generator=GeneratorConfig(model_path=tmp_path / "model.gguf"),
            evaluation=EvaluationConfig(
                trial_cases_path=tmp_path / "cases.jsonl", judge_model_path=None
            ),
        )

        with pytest.raises(ConfigOverrideError, match="Invalid override key"):
            profile.with_overrides({"nonexistent.field": 123})

    def test_with_overrides_should_raise_error_on_invalid_value(
        self, tmp_path: Path
    ) -> None:
        """
        Scenario: Applying override that violates schema validation.
        Action: Set chunk_size to negative value.
        Then: ConfigOverrideError should be raised (wrapping validation error).
        """
        profile = ExperimentProfile(
            generator=GeneratorConfig(model_path=tmp_path / "model.gguf"),
            evaluation=EvaluationConfig(
                trial_cases_path=tmp_path / "cases.jsonl", judge_model_path=None
            ),
        )

        with pytest.raises(
            ConfigOverrideError, match="Override validation failed"
        ) as exc_info:
            profile.with_overrides({"fragmenter.chunk_size": -100})

        assert exc_info.value.__cause__ is not None

    def test_diff_should_identify_differences_between_profiles(
        self, tmp_path: Path
    ) -> None:
        """
        Scenario: Comparing two different profiles.
        Action: Call diff().
        Then: Returns dictionary containing only changed keys with (old, new) tuples.
        """
        base_kwargs: dict[str, Any] = {
            "generator": GeneratorConfig(model_path=tmp_path / "model.gguf"),
            "evaluation": EvaluationConfig(
                trial_cases_path=tmp_path / "cases.jsonl", judge_model_path=None
            ),
        }

        profile1 = ExperimentProfile(
            fragmenter=FragmenterConfig(chunk_size=256, overlap=64),
            **base_kwargs,
        )

        profile2 = ExperimentProfile(
            fragmenter=FragmenterConfig(chunk_size=512, overlap=128),
            **base_kwargs,
        )

        diffs = profile1.diff(profile2)

        assert "fragmenter.chunk_size" in diffs
        assert diffs["fragmenter.chunk_size"] == (256, 512)
        assert "fragmenter.overlap" in diffs
        assert diffs["fragmenter.overlap"] == (64, 128)

        assert "generator.model_path" not in diffs

    def test_compute_hash_should_be_deterministic(self, tmp_path: Path) -> None:
        """
        Scenario: Hashing profiles with same vs different configurations.
        Action: Compute hash for identical and different profiles.
        Then: Identical configs produce equal hashes; different configs produce unequal hashes.
        """
        base_kwargs: dict[str, Any] = {
            "generator": GeneratorConfig(model_path=tmp_path / "model.gguf"),
            "evaluation": EvaluationConfig(
                trial_cases_path=tmp_path / "cases.jsonl", judge_model_path=None
            ),
        }

        profile1 = ExperimentProfile(
            fragmenter=FragmenterConfig(chunk_size=256), **base_kwargs
        )
        profile2 = ExperimentProfile(
            fragmenter=FragmenterConfig(chunk_size=256), **base_kwargs
        )
        profile3 = ExperimentProfile(
            fragmenter=FragmenterConfig(chunk_size=512), **base_kwargs
        )

        assert profile1.compute_hash() == profile2.compute_hash()

        assert profile1.compute_hash() != profile3.compute_hash()
