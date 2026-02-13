"""Unit tests for arena reporting module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ragmark.arena.reporting import (
    build_comparison_table,
    export_all_reports,
    to_csv,
    to_parquet,
)
from ragmark.schemas.documents import KnowledgeNode, NodePosition
from ragmark.schemas.evaluation import AuditReport, CaseResult, SystemInfo
from ragmark.schemas.generation import GenerationResult, TokenUsage
from ragmark.schemas.retrieval import RetrievedNode, TraceContext


def _make_system_info() -> SystemInfo:
    """Build a minimal SystemInfo for test reports."""
    return SystemInfo(
        python_version="3.12.0",
        ragmark_version="0.1.0",
        platform="Linux-x86_64",
        cpu_count=4,
        ram_gb=16.0,
        gpu_info=None,
    )


def _make_case_result(
    case_id: str = "case-1",
    query: str = "test query",
    predicted_answer: str = "test answer",
) -> CaseResult:
    """Build a minimal CaseResult for test reports."""
    node = KnowledgeNode(
        node_id="n1",
        content="Test content",
        source_id="src-1",
        position=NodePosition(start_char=0, end_char=12, page=1, section="test"),
        dense_vector=[0.1, 0.2, 0.3],
        sparse_vector=None,
    )
    trace = TraceContext(
        query=query,
        retrieved_nodes=[RetrievedNode(node=node, score=0.9, rank=1)],
        reranked=False,
    )
    return CaseResult(
        case_id=case_id,
        predicted_answer=predicted_answer,
        trace=trace,
        generation_result=GenerationResult(
            text=predicted_answer,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            finish_reason="stop",
        ),
        case_metrics={"recall@5": 0.8},
    )


def _make_report(
    config_hash: str = "abc123hash",
    duration: float = 1.5,
    metrics: dict[str, float] | None = None,
    case_results: list[CaseResult] | None = None,
) -> AuditReport:
    """Build a minimal AuditReport for test purposes."""
    return AuditReport(
        experiment_profile_hash=config_hash,
        duration_seconds=duration,
        metrics=metrics or {"recall@5": 0.8, "mrr": 0.7},
        per_case_results=case_results
        if case_results is not None
        else [_make_case_result()],
        system_info=_make_system_info(),
    )


# ---------------------------------------------------------------------------
# to_csv
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToCsv:
    """Tests for to_csv export function."""

    def test_to_csv_should_produce_readable_csv_with_correct_columns(
        self, tmp_path: Path
    ) -> None:
        """A single report is written as a valid CSV with the expected columns.

        Given:
            An AuditReport with one CaseResult containing a generation_result.
        When:
            to_csv is called with a destination path.
        Then:
            The file exists, is readable by pandas, and contains the expected
            columns from AuditReport.to_dataframe (case_id, predicted_answer,
            query, num_retrieved, reranked, case metric keys, and generation
            fields).
        """
        report = _make_report()
        csv_path = tmp_path / "report.csv"

        to_csv(report, csv_path)

        assert csv_path.exists()
        df = pd.read_csv(csv_path)  # type: ignore
        assert len(df) == 1
        expected_columns = {
            "case_id",
            "predicted_answer",
            "query",
            "num_retrieved",
            "reranked",
            "recall@5",
            "prompt_tokens",
            "completion_tokens",
            "finish_reason",
        }
        assert expected_columns.issubset(set(df.columns))


# ---------------------------------------------------------------------------
# to_parquet
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToParquet:
    """Tests for to_parquet export function."""

    def test_to_parquet_should_produce_readable_parquet_with_correct_columns(
        self, tmp_path: Path
    ) -> None:
        """A single report is written as a valid Parquet file with the expected columns.

        Given:
            An AuditReport with one CaseResult containing a generation_result.
        When:
            to_parquet is called with a destination path.
        Then:
            The file exists, is readable by pandas, and contains the expected
            columns from AuditReport.to_dataframe.
        """
        report = _make_report()
        parquet_path = tmp_path / "report.parquet"

        to_parquet(report, parquet_path)

        assert parquet_path.exists()
        df = pd.read_parquet(parquet_path)  # type: ignore
        assert len(df) == 1
        expected_columns = {
            "case_id",
            "predicted_answer",
            "query",
            "num_retrieved",
            "reranked",
            "recall@5",
            "prompt_tokens",
            "completion_tokens",
            "finish_reason",
        }
        assert expected_columns.issubset(set(df.columns))


# ---------------------------------------------------------------------------
# export_all_reports
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExportAllReports:
    """Tests for export_all_reports batch export function."""

    def test_export_all_reports_should_create_correct_number_of_files(
        self, tmp_path: Path
    ) -> None:
        """One file per report plus a summary.csv are created.

        Given:
            Two AuditReports with distinct config hashes.
        When:
            export_all_reports is called with fmt="csv".
        Then:
            The output directory contains exactly three files: one CSV per
            report named by the first 16 characters of the config hash, plus
            summary.csv.
        """
        report_a = _make_report(config_hash="aaaa1111bbbb2222cccc3333")
        report_b = _make_report(config_hash="dddd4444eeee5555ffff6666", duration=2.0)
        output_dir = tmp_path / "results"

        export_all_reports([report_a, report_b], output_dir, fmt="csv")

        files = sorted(f.name for f in output_dir.iterdir())
        assert len(files) == 3
        assert "aaaa1111bbbb2222.csv" in files
        assert "dddd4444eeee5555.csv" in files
        assert "summary.csv" in files

    def test_export_all_reports_summary_should_contain_one_row_per_report(
        self, tmp_path: Path
    ) -> None:
        """The summary.csv has one row for each exported report.

        Given:
            Three AuditReports.
        When:
            export_all_reports is called.
        Then:
            summary.csv contains exactly three rows with the expected
            config_hash values (truncated to 16 characters).
        """
        reports = [
            _make_report(config_hash="hash_alpha_00001"),
            _make_report(config_hash="hash_bravo_00002", duration=2.0),
            _make_report(config_hash="hash_charlie_003", duration=3.0),
        ]
        output_dir = tmp_path / "results"

        export_all_reports(reports, output_dir)

        summary = pd.read_csv(output_dir / "summary.csv")  # type: ignore
        assert len(summary) == 3
        assert set(summary["config_hash"]) == {
            "hash_alpha_00001"[:16],
            "hash_bravo_00002"[:16],
            "hash_charlie_003"[:16],
        }

    def test_export_all_reports_should_create_output_dir_if_missing(
        self, tmp_path: Path
    ) -> None:
        """A non-existent output directory is created automatically.

        Given:
            A path to a directory that does not yet exist.
        When:
            export_all_reports is called with that path.
        Then:
            The directory is created and files are written into it.
        """
        output_dir = tmp_path / "deeply" / "nested" / "output"
        assert not output_dir.exists()

        export_all_reports([_make_report()], output_dir)

        assert output_dir.is_dir()
        assert (output_dir / "summary.csv").exists()

    def test_export_all_reports_should_support_parquet_format(
        self, tmp_path: Path
    ) -> None:
        """Individual report files use the .parquet extension when fmt="parquet".

        Given:
            A single AuditReport.
        When:
            export_all_reports is called with fmt="parquet".
        Then:
            The individual report is written as a .parquet file, and the
            summary is still a CSV.
        """
        report = _make_report(config_hash="parquettest12345678")
        output_dir = tmp_path / "parquet_out"

        export_all_reports([report], output_dir, fmt="parquet")

        files = sorted(f.name for f in output_dir.iterdir())
        assert "parquettest12345"[:16] + ".parquet" in " ".join(files)
        assert "summary.csv" in files


# ---------------------------------------------------------------------------
# build_comparison_table
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildComparisonTable:
    """Tests for build_comparison_table aggregation function."""

    def test_build_comparison_table_should_have_one_row_per_report(self) -> None:
        """Each report becomes exactly one row in the comparison table.

        Given:
            Two AuditReports with different config hashes.
        When:
            build_comparison_table is called.
        Then:
            The resulting DataFrame has exactly two rows.
        """
        reports = [
            _make_report(config_hash="hash_one_1234567890"),
            _make_report(config_hash="hash_two_1234567890", duration=3.0),
        ]

        df = build_comparison_table(reports)

        assert len(df) == 2

    def test_build_comparison_table_should_include_expected_columns(self) -> None:
        """The table contains config_hash, duration_seconds, and metric names.

        Given:
            A report with metrics "recall@5" and "mrr".
        When:
            build_comparison_table is called.
        Then:
            The DataFrame columns include config_hash, duration_seconds,
            recall@5, and mrr.
        """
        report = _make_report(metrics={"recall@5": 0.8, "mrr": 0.7})

        df = build_comparison_table([report])

        expected_columns = {"config_hash", "duration_seconds", "recall@5", "mrr"}
        assert expected_columns.issubset(set(df.columns))

    def test_build_comparison_table_should_truncate_config_hash(self) -> None:
        """Config hashes are truncated to their first 16 characters.

        Given:
            A report whose experiment_profile_hash is longer than 16 characters.
        When:
            build_comparison_table is called.
        Then:
            The config_hash column value contains only the first 16 characters.
        """
        long_hash = "abcdef1234567890extra_characters_beyond_sixteen"
        report = _make_report(config_hash=long_hash)

        df = build_comparison_table([report])

        assert df.iloc[0]["config_hash"] == long_hash[:16]

    def test_build_comparison_table_should_preserve_metric_values(self) -> None:
        """Metric values are faithfully transferred to the DataFrame.

        Given:
            A report with specific metric values.
        When:
            build_comparison_table is called.
        Then:
            The DataFrame row contains the exact metric values and
            duration_seconds.
        """
        report = _make_report(
            duration=4.2,
            metrics={"recall@5": 0.85, "mrr": 0.72},
        )

        df = build_comparison_table([report])

        row = df.iloc[0]
        assert row["duration_seconds"] == pytest.approx(4.2)  # type: ignore
        assert row["recall@5"] == pytest.approx(0.85)  # type: ignore
        assert row["mrr"] == pytest.approx(0.72)  # type: ignore

    def test_build_comparison_table_should_return_empty_dataframe_for_empty_list(
        self,
    ) -> None:
        """An empty reports list produces an empty DataFrame.

        Given:
            An empty list of AuditReports.
        When:
            build_comparison_table is called.
        Then:
            The returned DataFrame has zero rows.
        """
        df = build_comparison_table([])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
