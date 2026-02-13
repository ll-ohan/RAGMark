"""Export and comparison utilities for arena benchmark reports.

Provides functions for exporting AuditReports to CSV/Parquet,
batch-exporting all reports from an arena run, and building
a comparison table across configuration variants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from ragmark.logger import get_logger
from ragmark.schemas.evaluation import AuditReport

logger = get_logger(__name__)


def to_csv(report: AuditReport, path: Path) -> None:
    """Export a single AuditReport to CSV.

    Args:
        report: The report to export.
        path: Destination file path.
    """
    df = report.to_dataframe()
    df.to_csv(path, index=False)
    logger.debug("Exported report %s to CSV: %s", report.report_id[:12], path)


def to_parquet(report: AuditReport, path: Path) -> None:
    """Export a single AuditReport to Parquet.

    Args:
        report: The report to export.
        path: Destination file path.
    """
    df = report.to_dataframe()
    df.to_parquet(path, index=False)
    logger.debug("Exported report %s to Parquet: %s", report.report_id[:12], path)


def export_all_reports(
    reports: list[AuditReport],
    output_dir: Path,
    fmt: Literal["csv", "parquet"] = "csv",
) -> None:
    """Export all reports and a summary file to the output directory.

    Each report is exported individually as ``{config_hash}.{ext}``,
    plus a ``summary.csv`` with one row per configuration variant
    containing aggregated metrics.

    Args:
        reports: AuditReports from an arena run.
        output_dir: Directory to write output files into.
        fmt: Export format for individual reports.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    exporter = to_csv if fmt == "csv" else to_parquet
    ext = fmt

    for report in reports:
        filename = f"{report.experiment_profile_hash[:16]}.{ext}"
        exporter(report, output_dir / filename)

    summary = build_comparison_table(reports)
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    logger.info(
        "Exported %d reports to %s (format=%s)",
        len(reports),
        output_dir,
        fmt,
    )


def build_comparison_table(reports: list[AuditReport]) -> pd.DataFrame:
    """Build a comparison table with one row per configuration variant.

    Columns include the config hash, duration, and all aggregated
    metrics from each report.

    Args:
        reports: AuditReports from an arena run.

    Returns:
        DataFrame with one row per config variant.
    """
    rows: list[dict[str, str | float]] = []

    for report in reports:
        row: dict[str, str | float] = {
            "config_hash": report.experiment_profile_hash[:16],
            "duration_seconds": report.duration_seconds,
        }
        row.update(report.metrics)
        rows.append(row)

    return pd.DataFrame(rows)
