"""CLI entry point for running benchmark arena sweeps.

Usage::

    python scripts/run_bench.py --config profile.yaml --cases cases.jsonl --output results/
    python scripts/run_bench.py --config profile.yaml --cases cases.jsonl --output results/ \\
        --grid grid.yaml --parallel 4 --cache-dir .cache/arena
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from ragmark.arena.cache import ArenaCache
from ragmark.arena.engine import BenchmarkArena
from ragmark.arena.reporting import build_comparison_table, export_all_reports
from ragmark.config.profile import ExperimentProfile
from ragmark.logger import set_log_level
from ragmark.schemas.evaluation import AuditReport, TrialCase

app = typer.Typer(help="RAGMark benchmark arena CLI.")
console = Console()


def _load_grid(grid_path: Path) -> dict[str, list[Any]]:
    """Load a parameter grid from a YAML file.

    Args:
        grid_path: Path to a YAML file mapping dotted keys to value lists.

    Returns:
        Parsed parameter grid.

    Raises:
        typer.BadParameter: If the grid file is invalid.
    """
    try:
        with open(grid_path, encoding="utf-8") as f:
            data: dict[str, list[Any] | None] | None = yaml.safe_load(f)
    except Exception as exc:
        raise typer.BadParameter(f"Failed to parse grid YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise typer.BadParameter("Grid YAML must be a mapping of keys to lists")

    for key, values in data.items():
        if not isinstance(values, list) and values is not None:
            raise typer.BadParameter(
                f"Grid key '{key}' must map to a list, got {type(values).__name__}"
            )
    return {k: v for k, v in data.items() if v is not None}


async def _run_arena(
    profile: ExperimentProfile,
    trial_cases: list[TrialCase],
    grid: dict[str, list[Any]],
    parallel: int,
    sources: list[Path],
    cache: ArenaCache | None,
    output_dir: Path,
) -> list[AuditReport]:
    """Execute the arena run with a rich progress display.

    Args:
        profile: Base experiment configuration.
        trial_cases: Evaluation trial cases.
        grid: Parameter grid for variant generation.
        parallel: Maximum concurrent config executions.
        sources: Source document paths for forge pipeline.
        cache: Optional arena cache.
        output_dir: Directory for exported results.

    Returns:
        List of AuditReports from the arena run.
    """
    arena = BenchmarkArena(
        base_profile=profile,
        grid=grid,
        sources=sources,
        cache=cache,
    )

    configs = arena.generate_configs()
    completed_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Running {len(configs)} config(s)...", total=len(configs)
        )

        def on_complete(idx: int, report: AuditReport) -> None:
            nonlocal completed_count
            completed_count += 1
            progress.update(
                task,
                advance=1,
                description=(
                    f"Config {completed_count}/{len(configs)} done "
                    f"({report.experiment_profile_hash[:8]})"
                ),
            )

        reports = await arena.run(
            trial_cases=trial_cases,
            parallel=parallel,
            on_config_complete=on_complete,
        )

    return reports


def _print_summary(reports: list[AuditReport]) -> None:
    """Print a rich summary table to the console.

    Args:
        reports: AuditReports from the arena run.
    """
    df = build_comparison_table(reports)
    if df.empty:
        console.print("[yellow]No results to display.[/yellow]")
        return

    table = Table(title="Arena Summary")
    for col in df.columns:
        table.add_column(str(col), justify="right" if col != "config_hash" else "left")

    for _, row in df.iterrows():
        table.add_row(*[f"{v:.4f}" if isinstance(v, float) else str(v) for v in row])

    console.print(table)


@app.callback(invoke_without_command=True)
def run(
    config: Path = typer.Option(
        ..., "--config", exists=True, help="Base YAML experiment profile."
    ),
    cases: Path = typer.Option(
        ..., "--cases", exists=True, help="Trial cases JSONL or JSON file."
    ),
    output: Path = typer.Option(..., "--output", help="Output directory for results."),
    sources: list[Path] = typer.Option(
        [], "--sources", help="Source document paths for forge pipeline."
    ),
    grid: Path | None = typer.Option(
        None, "--grid", exists=True, help="Optional parameter grid YAML."
    ),
    parallel: int = typer.Option(
        1, "--parallel", min=1, max=16, help="Max concurrent configs."
    ),
    cache_dir: Path | None = typer.Option(
        None, "--cache-dir", help="Arena cache directory."
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", help="Log level (DEBUG, INFO, WARNING, ERROR)."
    ),
    fmt: str = typer.Option("csv", "--format", help="Export format: csv or parquet."),
) -> None:
    """Run a benchmark arena sweep across configuration variants."""
    # ---- Configure logging ----
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        console.print(f"[red]Invalid log level: {log_level}[/red]")
        raise SystemExit(1)
    set_log_level(log_level.upper())

    # ---- Eager validation ----
    try:
        profile = ExperimentProfile.from_yaml(config)
    except Exception as exc:
        console.print(f"[red]Invalid config: {exc}[/red]")
        raise SystemExit(1) from exc

    try:
        trial_cases = TrialCase.load_cases(cases)
    except Exception as exc:
        console.print(f"[red]Failed to load trial cases: {exc}[/red]")
        raise SystemExit(1) from exc

    if not trial_cases:
        console.print("[red]No trial cases found in file.[/red]")
        raise SystemExit(2)

    param_grid: dict[str, list[Any]] = {}
    if grid is not None:
        param_grid = _load_grid(grid)

    if fmt not in ("csv", "parquet"):
        console.print(f"[red]Unsupported format: {fmt}. Use 'csv' or 'parquet'.[/red]")
        raise SystemExit(1)

    # ---- Setup cache ----
    cache: ArenaCache | None = None
    if cache_dir is not None:
        cache = ArenaCache(cache_dir)

    # ---- Run arena ----
    console.print(
        f"[bold]RAGMark Arena[/bold] — "
        f"{len(trial_cases)} case(s), "
        f"grid params: {len(param_grid)}, "
        f"parallel: {parallel}"
    )

    try:
        reports = asyncio.run(
            _run_arena(
                profile, trial_cases, param_grid, parallel, sources, cache, output
            )
        )
    except Exception as exc:
        console.print(f"[red]Arena failed: {exc}[/red]")
        raise SystemExit(1) from exc

    # ---- Export results ----
    export_all_reports(reports, output, fmt=fmt)  # type: ignore[arg-type]
    console.print(f"\n[green]Results exported to {output}/[/green]")

    # ---- Print summary ----
    _print_summary(reports)


if __name__ == "__main__":
    app()
