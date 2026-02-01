"""Experiment configuration profile and sub-configurations.

This module defines the complete configuration structure for RAGMark
experiments using Pydantic V2 for validation and type safety.
"""

import hashlib
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)


class IngestorConfig(BaseModel):
    """Configuration for document ingestion.

    Attributes:
        backend: Ingestion backend to use.
        options: Backend-specific options.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    backend: Literal["marker", "fitz"] = Field(
        default="fitz",
        description="Ingestion backend (marker for OCR, fitz for fast extraction)",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Backend-specific options (e.g., ocr_enabled, languages)",
    )


class FragmenterConfig(BaseModel):
    """Configuration for text fragmentation.

    Attributes:
        strategy: Fragmentation strategy to use.
        chunk_size: Target size of each chunk (tokens or characters).
        overlap: Overlap between consecutive chunks.
        options: Strategy-specific options.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    strategy: Literal["token", "semantic", "markdown", "recursive"] = Field(
        default="token",
        description="Fragmentation strategy",
    )
    chunk_size: int = Field(
        default=256,
        ge=50,
        le=8192,
        description="Target chunk size (tokens or characters)",
    )
    overlap: int = Field(
        default=64,
        ge=0,
        description="Overlap between consecutive chunks",
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy-specific options (e.g., tokenizer, separators)",
    )

    @field_validator("overlap")
    @classmethod
    def overlap_less_than_chunk_size(cls, v: int, info: ValidationInfo) -> int:
        """Validate that overlap is less than chunk_size.

        Args:
            v: Overlap value.
            info: Validation context.

        Returns:
            Validated overlap.

        Raises:
            ValueError: If overlap >= chunk_size.
        """
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError("overlap must be less than chunk_size")
        return v


class EmbedderConfig(BaseModel):
    """Configuration for embedding models.

    Attributes:
        model_name: HuggingFace model identifier or local path.
        device: Device to use for inference (cpu, cuda, mps).
        batch_size: Batch size for embedding computation.
        rate_limit: Maximum requests per second (None for unlimited).
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    model_name: str | Path = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model identifier or local path",
    )
    device: str = Field(
        default="cpu",
        description="Device for inference (cpu, cuda, mps)",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size for embedding computation",
    )
    rate_limit: float | None = Field(
        default=None,
        ge=0.1,
        description="Maximum requests per second (None for unlimited)",
    )


class IndexConfig(BaseModel):
    """Configuration for vector index.

    Attributes:
        backend: Vector database backend.
        collection_name: Name of the collection/table.
        embedding_dim: Dimensionality of dense vectors.
        connection: Backend-specific connection parameters.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    backend: Literal["memory", "qdrant", "milvus", "lancedb", "custom_backend"] = Field(
        default="memory",
        description="Vector database backend",
    )
    collection_name: str = Field(
        default="ragmark_default",
        description="Collection or table name",
    )
    embedding_dim: int = Field(
        default=384,
        ge=64,
        le=4096,
        description="Dimensionality of dense vectors",
    )
    connection: dict[str, Any] | None = Field(
        None,
        description="Backend-specific connection parameters (host, port, api_key)",
    )

    @model_validator(mode="after")
    def connection_required_for_remote_backends(self) -> "IndexConfig":
        """Validate that connection is provided for remote backends.

        Returns:
            Validated IndexConfig instance.

        Raises:
            ValueError: If connection is None for qdrant, milvus, or lancedb.
        """
        if self.backend in ("qdrant", "milvus", "lancedb") and self.connection is None:
            raise ValueError(
                f"Connection parameters are required for backend '{self.backend}'"
            )
        return self


class RerankerConfig(BaseModel):
    """Configuration for reranking.

    Attributes:
        model_name: Cross-encoder model identifier.
        device: Device for inference.
        top_k: Number of results to keep after reranking.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model identifier",
    )
    device: str = Field(
        default="cpu",
        description="Device for inference",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        description="Number of results after reranking",
    )


class RetrievalConfig(BaseModel):
    """Configuration for retrieval.

    Attributes:
        mode: Retrieval mode (dense, sparse, or hybrid).
        top_k: Number of results to retrieve initially.
        alpha: Hybrid fusion weight (0=sparse only, 1=dense only).
        reranker: Optional reranker configuration.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    mode: Literal["dense", "sparse", "hybrid"] = Field(
        default="dense",
        description="Retrieval mode",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to retrieve",
    )
    alpha: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Hybrid fusion weight (dense vs sparse)",
    )
    reranker: RerankerConfig | None = Field(
        None,
        description="Optional reranker configuration",
    )

    @model_validator(mode="after")
    def alpha_required_for_hybrid(self) -> "RetrievalConfig":
        """Validate that alpha is provided when mode is hybrid.

        Returns:
            Validated RetrievalConfig instance.

        Raises:
            ValueError: If alpha is None when mode is hybrid.
        """
        if self.mode == "hybrid" and self.alpha is None:
            raise ValueError("alpha is required when mode is 'hybrid'")
        return self


class GeneratorConfig(BaseModel):
    """Configuration for LLM generation.

    Attributes:
        model_path: Path to the LLM model (GGUF or HF).
        context_window: Maximum context window size.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    model_path: Path = Field(
        ...,
        description="Path to the LLM model (GGUF or HuggingFace)",
    )
    context_window: int = Field(
        default=4096,
        ge=512,
        le=128000,
        description="Maximum context window size",
    )
    max_new_tokens: int = Field(
        default=512,
        ge=1,
        le=4096,
        description="Maximum tokens to generate",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )


class EvaluationConfig(BaseModel):
    """Configuration for evaluation.

    Attributes:
        metrics: List of metrics to compute.
        trial_cases_path: Path to trial cases file.
        judge_model_path: Optional path to judge LLM for generation metrics.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    metrics: list[
        Literal[
            "recall@k", "mrr", "ndcg", "precision@k", "faithfulness", "answer_relevancy"
        ]
    ] = Field(
        default_factory=lambda: cast(
            list[
                Literal[
                    "recall@k",
                    "mrr",
                    "ndcg",
                    "precision@k",
                    "faithfulness",
                    "answer_relevancy",
                ]
            ],
            ["recall@k", "mrr"],
        ),
        description="Metrics to compute",
    )
    trial_cases_path: Path = Field(
        ...,
        description="Path to trial cases JSON/JSONL file",
    )
    judge_model_path: Path | None = Field(
        None,
        description="Path to judge LLM for generation metrics",
    )


class StreamingMetricsConfig(BaseModel):
    """Configuration for streaming pipeline metrics.

    Attributes:
        enabled: Whether to enable metrics collection.
        interval: Sampling interval in seconds.
        max_samples: Memory limit for retained samples.
        backpressure_threshold: Queue fill ratio triggering warnings.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    enabled: bool = Field(
        default=True,
        description="Enable streaming metrics collection",
    )
    interval: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Sampling interval in seconds",
    )
    max_samples: int = Field(
        default=10000,
        ge=10,
        le=100000,
        description="Maximum samples to retain (memory bound)",
    )
    backpressure_threshold: float = Field(
        default=0.8,
        ge=0.5,
        le=1.0,
        description="Queue fill ratio to trigger warnings",
    )


class MetricsConfig(BaseModel):
    """Configuration for metrics collection.

    Attributes:
        streaming: Streaming pipeline metrics settings.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    streaming: StreamingMetricsConfig | None = Field(
        default_factory=StreamingMetricsConfig,
        description="Streaming pipeline metrics configuration",
    )


class ExperimentProfile(BaseModel):
    """Complete experiment configuration profile.

    This is the root configuration model that aggregates all sub-configurations
    for a RAG experiment.

    Attributes:
        ingestor: Document ingestion configuration.
        fragmenter: Text fragmentation configuration.
        embedder: Embedding model configuration.
        index: Vector index configuration.
        retrieval: Retrieval configuration.
        generator: LLM generation configuration.
        evaluation: Evaluation configuration.
        metrics: Metrics collection configuration.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    ingestor: IngestorConfig = Field(
        default_factory=IngestorConfig,
        description="Document ingestion configuration",
    )
    fragmenter: FragmenterConfig = Field(
        default_factory=FragmenterConfig,
        description="Text fragmentation configuration",
    )
    fail_fast: bool = Field(
        default=True,
        description="Whether to stop ingestion on first error",
    )
    embedder: EmbedderConfig = Field(
        default_factory=EmbedderConfig,
        description="Embedding model configuration",
    )
    index: IndexConfig = Field(
        default_factory=lambda: IndexConfig(connection=None),
        description="Vector index configuration",
    )
    retrieval: RetrievalConfig = Field(
        default_factory=lambda: RetrievalConfig(alpha=None, reranker=None),
        description="Retrieval configuration",
    )
    generator: GeneratorConfig | None = Field(
        default=None,
        description="LLM generation configuration",
    )
    evaluation: EvaluationConfig | None = Field(
        default=None,
        description="Evaluation configuration",
    )
    metrics: MetricsConfig | None = Field(
        default_factory=MetricsConfig,
        description="Metrics collection configuration",
    )

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentProfile":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Validated ExperimentProfile instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            ConfigError: If YAML is invalid or validation fails.
        """
        from ragmark.exceptions import ConfigError

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            data = cls._resolve_paths(data, path.parent)

            return cls.model_validate(data)
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML syntax: {e}") from e
        except Exception as e:
            raise ConfigError(f"Configuration validation failed: {e}") from e

    @staticmethod
    def _resolve_paths(data: Any, base_dir: Path) -> Any:
        if isinstance(data, dict):
            return {
                k: ExperimentProfile._resolve_paths(v, base_dir)
                for k, v in cast(dict[str, Any], data).items()
            }
        elif isinstance(data, list):
            return [ExperimentProfile._resolve_paths(item, base_dir) for item in data]
        elif isinstance(data, str):
            is_path = data.startswith(("./", "../", "/", "~", "\\")) or data.endswith(
                (".pt", ".gguf", ".jsonl", ".json", ".yaml", ".yml", ".txt")
            )

            if is_path:
                path = Path(data)
                if not path.is_absolute():
                    return base_dir / path
                return path
        return data

    def to_yaml(self, path: Path) -> None:
        """Export configuration to a YAML file.

        Args:
            path: Output file path.
        """

        def convert_paths_to_str(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {
                    k: convert_paths_to_str(v)
                    for k, v in cast(dict[str, Any], obj).items()
                }
            elif isinstance(obj, list):
                return [convert_paths_to_str(item) for item in obj]
            return obj

        data = self.model_dump(mode="python")
        data = convert_paths_to_str(data)

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                data,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

    def with_overrides(self, overrides: dict[str, Any]) -> "ExperimentProfile":
        """Create a new profile with partial overrides applied.

        Args:
            overrides: Dictionary of overrides using dotted notation.
                Example: {"retrieval.top_k": 20, "fragmenter.chunk_size": 512}

        Returns:
            New ExperimentProfile instance with overrides applied.

        Raises:
            ConfigOverrideError: If override key is invalid or type mismatches.
        """
        from ragmark.exceptions import ConfigOverrideError

        # Deep copy current configuration
        data = self.model_dump(mode="python")

        # Apply overrides
        last_key: str = ""
        for key, value in overrides.items():
            last_key = key
            parts = key.split(".")
            current = data
            for part in parts[:-1]:
                if part not in current:
                    raise ConfigOverrideError(
                        f"Invalid override key: {key}", field_path=key
                    )
                current = current[part]

            final_key = parts[-1]
            if final_key not in current:
                raise ConfigOverrideError(
                    f"Invalid override key: {key}", field_path=key
                )

            current[final_key] = value

        try:
            return self.model_validate(data)
        except Exception as e:
            raise ConfigOverrideError(
                f"Override validation failed: {e}", field_path=last_key
            ) from e

    def diff(self, other: "ExperimentProfile") -> dict[str, tuple[Any, Any]]:
        """Compute differences between two profiles.

        Args:
            other: Another ExperimentProfile to compare against.

        Returns:
            Dictionary mapping field paths to (self_value, other_value) tuples.
        """

        def _diff_recursive(
            d1: dict[str, Any], d2: dict[str, Any], prefix: str = ""
        ) -> dict[str, tuple[Any, Any]]:
            diffs: dict[str, tuple[Any, Any]] = {}
            all_keys = set(d1.keys()) | set(d2.keys())
            for key in all_keys:
                path = f"{prefix}.{key}" if prefix else key
                v1 = d1.get(key)
                v2 = d2.get(key)

                if isinstance(v1, dict) and isinstance(v2, dict):
                    diffs.update(
                        _diff_recursive(
                            cast(dict[str, Any], v1), cast(dict[str, Any], v2), path
                        )
                    )
                elif v1 != v2:
                    diffs[path] = (v1, v2)

            return diffs

        self_data = self.model_dump(mode="python")
        other_data = other.model_dump(mode="python")
        return _diff_recursive(self_data, other_data)

    def compute_hash(self) -> str:
        """Compute a deterministic hash of this configuration.

        Returns:
            SHA-256 hash of the configuration.
        """
        import json

        # Use json.dumps for deterministic serialization with sorted keys
        data_dict = self.model_dump(mode="json")
        data_json = json.dumps(data_dict, sort_keys=True)
        return hashlib.sha256(data_json.encode()).hexdigest()
