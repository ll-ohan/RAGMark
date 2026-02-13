"""Benchmark arena for comparative RAG evaluation.

This package provides the orchestration layer for running parameterised
benchmark sweeps across multiple configuration variants.
"""

from ragmark.arena.cache import ArenaCache
from ragmark.arena.engine import BenchmarkArena
from ragmark.arena.models import ArenaConfig, ArenaResult
from ragmark.arena.reporting import (
    build_comparison_table,
    export_all_reports,
    to_csv,
    to_parquet,
)

__all__ = [
    "ArenaCache",
    "ArenaConfig",
    "ArenaResult",
    "BenchmarkArena",
    "build_comparison_table",
    "export_all_reports",
    "to_csv",
    "to_parquet",
]
