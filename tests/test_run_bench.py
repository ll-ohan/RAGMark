"""Unit tests for the run_bench CLI entry point."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from typer.testing import CliRunner

# ---------------------------------------------------------------------------
# Import the Typer app from scripts/run_bench.py via importlib
# (the scripts/ directory is not on sys.path or in pyproject pythonpath).
# ---------------------------------------------------------------------------
_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_bench.py"
_spec = importlib.util.spec_from_file_location("run_bench", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
_run_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run_bench)
app = _run_bench.app

runner = CliRunner()


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHelp:
    """Tests for the --help flag on the CLI."""

    def test_help_should_exit_zero(self) -> None:
        """Invoking --help exits with code 0.

        Given:
            The run_bench CLI app.
        When:
            Invoked with --help.
        Then:
            The exit code is 0.
        """
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0

    def test_help_should_display_all_required_options(self) -> None:
        """The help text advertises every required CLI option.

        Given:
            The run_bench CLI app.
        When:
            Invoked with --help.
        Then:
            The output contains --config, --cases, and --output.
        """
        result = runner.invoke(app, ["--help"])

        assert "--config" in result.output
        assert "--cases" in result.output
        assert "--output" in result.output

    def test_help_should_display_all_optional_options(self) -> None:
        """The help text advertises every optional CLI option.

        Given:
            The run_bench CLI app.
        When:
            Invoked with --help.
        Then:
            The output contains --grid, --parallel, --cache-dir,
            --log-level, and --format.
        """
        result = runner.invoke(app, ["--help"])

        assert "--grid" in result.output
        assert "--parallel" in result.output
        assert "--cache-dir" in result.output
        assert "--log-level" in result.output
        assert "--format" in result.output


# ---------------------------------------------------------------------------
# Missing required arguments
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMissingRequiredArgs:
    """Tests that omitting required options produces a non-zero exit."""

    def test_missing_config_should_exit_nonzero(self, tmp_path: Path) -> None:
        """Omitting --config causes a non-zero exit code.

        Given:
            Valid --cases and --output arguments but no --config.
        When:
            The CLI is invoked.
        Then:
            The exit code is non-zero.
        """
        cases_file = tmp_path / "cases.jsonl"
        cases_file.write_text("{}\n")

        result = runner.invoke(
            app,
            [
                "--cases",
                str(cases_file),
                "--output",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code != 0

    def test_missing_cases_should_exit_nonzero(self, tmp_path: Path) -> None:
        """Omitting --cases causes a non-zero exit code.

        Given:
            Valid --config and --output arguments but no --cases.
        When:
            The CLI is invoked.
        Then:
            The exit code is non-zero.
        """
        config_file = tmp_path / "config.yaml"
        config_file.write_text("fragmenter:\n  chunk_size: 256\n")

        result = runner.invoke(
            app,
            [
                "--config",
                str(config_file),
                "--output",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code != 0

    def test_missing_output_should_exit_nonzero(self, tmp_path: Path) -> None:
        """Omitting --output causes a non-zero exit code.

        Given:
            Valid --config and --cases arguments but no --output.
        When:
            The CLI is invoked.
        Then:
            The exit code is non-zero.
        """
        config_file = tmp_path / "config.yaml"
        config_file.write_text("fragmenter:\n  chunk_size: 256\n")
        cases_file = tmp_path / "cases.jsonl"
        cases_file.write_text("{}\n")

        result = runner.invoke(
            app,
            [
                "--config",
                str(config_file),
                "--cases",
                str(cases_file),
            ],
        )

        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Non-existent file validation (exists=True)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNonExistentFile:
    """Tests that Typer's exists=True validation rejects missing files."""

    def test_nonexistent_config_should_exit_nonzero(self, tmp_path: Path) -> None:
        """Passing a non-existent path for --config causes a non-zero exit.

        Given:
            A --config path that does not exist on disk, with valid
            --cases and --output arguments.
        When:
            The CLI is invoked.
        Then:
            The exit code is non-zero and the output mentions the
            invalid path.
        """
        cases_file = tmp_path / "cases.jsonl"
        cases_file.write_text("{}\n")
        nonexistent = tmp_path / "no_such_config.yaml"

        result = runner.invoke(
            app,
            [
                "--config",
                str(nonexistent),
                "--cases",
                str(cases_file),
                "--output",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code != 0

    def test_nonexistent_cases_should_exit_nonzero(self, tmp_path: Path) -> None:
        """Passing a non-existent path for --cases causes a non-zero exit.

        Given:
            A --cases path that does not exist on disk, with valid
            --config and --output arguments.
        When:
            The CLI is invoked.
        Then:
            The exit code is non-zero.
        """
        config_file = tmp_path / "config.yaml"
        config_file.write_text("fragmenter:\n  chunk_size: 256\n")
        nonexistent = tmp_path / "no_such_cases.jsonl"

        result = runner.invoke(
            app,
            [
                "--config",
                str(config_file),
                "--cases",
                str(nonexistent),
                "--output",
                str(tmp_path / "out"),
            ],
        )

        assert result.exit_code != 0
