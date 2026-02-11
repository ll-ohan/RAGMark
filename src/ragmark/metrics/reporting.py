"""Monitoring and evaluation metrics reporting helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ragmark.logger import get_logger
from ragmark.metrics.base import MonitoringMetric
from ragmark.schemas.evaluation import AuditReport

logger = get_logger(__name__)


def export_monitoring_summary(
    monitoring: MonitoringMetric | None,
    artifact_dir: Path | None = None,
    artifact_prefix: str = "monitoring",
) -> Path | None:
    """Log monitoring summary and write JSON artifact.

    Args:
        monitoring: Monitoring orchestrator instance.
        artifact_dir: Optional directory for JSON output.
        artifact_prefix: Prefix for the artifact filename.

    Returns:
        Path to the artifact if written, otherwise None.
    """
    if monitoring is None or not monitoring.enabled:
        return None

    summary = monitoring.finalize()
    logger.info("Monitoring metrics summary: %s", summary)

    output_dir = artifact_dir or Path("output/monitoring")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = output_dir / f"{artifact_prefix}_{timestamp}.json"
    artifact_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True)
    )

    logger.info("Monitoring metrics artifact written: path=%s", artifact_path)
    return artifact_path


def export_evaluation_report(
    report: AuditReport,
    artifact_dir: Path | None = None,
    artifact_prefix: str = "evaluation",
) -> Path:
    """Log evaluation metrics and write report JSON artifact.

    Args:
        report: Evaluation report to export.
        artifact_dir: Optional directory for JSON output.
        artifact_prefix: Prefix for the artifact filename.

    Returns:
        Path to the written JSON report.
    """
    logger.info(
        "Evaluation report summary: report_id=%s, metrics=%s",
        report.report_id,
        report.metrics,
    )

    output_dir = artifact_dir or Path("output/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    created_at = report.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    timestamp = created_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = (
        output_dir / f"{artifact_prefix}_{report.report_id}_{timestamp}.json"
    )
    report.to_json(artifact_path)

    logger.info("Evaluation report artifact written: path=%s", artifact_path)
    return artifact_path
