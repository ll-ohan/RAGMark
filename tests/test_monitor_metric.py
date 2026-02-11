"""Unit tests for monitoring metrics and timing utilities."""

from __future__ import annotations

import asyncio

import pytest

from ragmark.metrics.base import MonitoringMetric


class TrackingMonitoringMetric(MonitoringMetric):
    """Monitoring metric that tracks lifecycle calls."""

    def __init__(self, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self.started = False
        self.stopped = False
        self.finalized = False

    async def start_monitoring(self) -> None:
        self.started = True

    async def stop_monitoring(self) -> None:
        self.stopped = True

    def finalize(self) -> dict[str, int]:
        self.finalized = True
        return {"count": 1}


@pytest.mark.unit
def test_stage_timer_should_record_when_enabled() -> None:
    """Validates stage timing is recorded when monitoring is enabled.

    Given:
        An enabled MonitoringMetric instance.
    When:
        Recording a stage with the context manager.
    Then:
        The summary contains timing stats for the stage.
    """
    metric = MonitoringMetric(enabled=True)

    with metric.stage("ingest"):
        pass

    summary = metric.finalize()

    assert "ingest" in summary
    stats = summary["ingest"]
    assert stats["count"] == 1
    assert stats["total_ms"] >= 0.0
    assert stats["min_ms"] <= stats["avg_ms"] <= stats["max_ms"]


@pytest.mark.unit
def test_stage_timer_should_skip_when_disabled() -> None:
    """Validates stage timing is skipped when monitoring is disabled.

    Given:
        A disabled MonitoringMetric instance.
    When:
        Using the stage context manager.
    Then:
        No timings are recorded in the summary.
    """
    metric = MonitoringMetric(enabled=False)

    with metric.stage("retrieve"):
        pass

    summary = metric.finalize()

    assert summary == {}


@pytest.mark.unit
def test_record_should_append_timings() -> None:
    """Validates record appends timing samples.

    Given:
        An enabled MonitoringMetric instance.
    When:
        Recording multiple timings for a stage.
    Then:
        Summary reflects the correct count and total.
    """
    metric = MonitoringMetric(enabled=True)

    metric.record("rank", 10.0)
    metric.record("rank", 20.0)

    summary = metric.finalize()

    stats = summary["rank"]
    assert stats["count"] == 2
    assert stats["total_ms"] == 30.0
    assert stats["avg_ms"] == 15.0
    assert stats["min_ms"] == 10.0
    assert stats["max_ms"] == 20.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_monitoring_metric_context_manager_should_call_lifecycle() -> None:
    """Validates start/stop/finalize are called in async context manager.

    Given:
        A MonitoringMetric subclass tracking lifecycle state.
    When:
        Using async context manager.
    Then:
        start_monitoring, stop_monitoring, and finalize are called.
    """
    metric = TrackingMonitoringMetric()

    async with metric:
        await asyncio.sleep(0)

    assert metric.started is True
    assert metric.stopped is True
    assert metric.finalized is True
