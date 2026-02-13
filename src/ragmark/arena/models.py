"""Data models for benchmark arena orchestration."""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from ragmark.schemas.evaluation import AuditReport


class ArenaConfig(BaseModel):
    """Configuration for a benchmark arena run.

    Attributes:
        base_profile_path: Path to the base YAML configuration.
        grid: Parameter grid with dotted keys mapping to value lists.
        parallel: Maximum concurrent config executions.
        cache_dir: Directory for arena cache storage.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    base_profile_path: str = Field(..., description="Path to base YAML config")
    grid: dict[str, list[Any]] = Field(
        default_factory=dict,
        description="Parameter grid (dotted keys -> value lists)",
    )
    parallel: int = Field(default=1, ge=1, le=16, description="Max concurrent configs")
    cache_dir: str | None = Field(None, description="Cache directory path")


class ArenaResult(BaseModel):
    """Result of a complete arena benchmark run.

    Attributes:
        arena_id: Unique identifier for this arena run.
        created_at: Timestamp of arena completion.
        reports: AuditReport per configuration variant.
        grid: The parameter grid that was used.
        summary: Aggregated summary metrics keyed by config hash.
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    arena_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique arena run identifier",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Arena completion timestamp",
    )
    reports: list[AuditReport] = Field(
        default_factory=list[AuditReport],
        description="AuditReport per configuration variant",
    )
    grid: dict[str, list[Any]] = Field(
        default_factory=dict,
        description="Parameter grid used for this run",
    )
    summary: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Aggregated metrics keyed by config hash",
    )
