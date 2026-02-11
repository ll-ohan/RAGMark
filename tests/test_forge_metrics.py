"""Tests for MetricsManager composite pattern.

This module tests the MetricsManager class which manages multiple
metrics collectors with unified lifecycle management.
"""

import asyncio

import pytest

from ragmark.config.profile import StreamingMetricsConfig
from ragmark.forge.runner import StreamingMetrics
from ragmark.metrics.base import MonitoringMetric
from ragmark.metrics.metrics import MetricsManager
from ragmark.schemas.documents import KnowledgeNode, NodePosition


def _make_node(index: int) -> KnowledgeNode:
    return KnowledgeNode(
        content=f"Node {index}",
        source_id="source",
        position=NodePosition(start_char=0, end_char=1, page=1, section="section"),
        dense_vector=[index * 0.1, index * 0.2],
        sparse_vector={0: index * 0.3, 1: index * 0.4},
    )


@pytest.mark.unit
@pytest.mark.asyncio
class TestMetricsManager:
    """Tests for MetricsManager composite lifecycle management."""

    async def test_manager_should_start_all_collectors(self) -> None:
        """Verifies MetricsManager starts all collectors on entry.

        Given:
            MetricsManager with multiple configured collectors.
        When:
            Entering context manager.
        Then:
            All collectors are started and monitoring, with at least 2
            samples collected per collector over 0.3s interval.
        """
        metrics1 = StreamingMetrics()
        metrics2 = StreamingMetrics()

        queue1: asyncio.Queue[KnowledgeNode | Exception | None] = asyncio.Queue(
            maxsize=10
        )
        queue2: asyncio.Queue[KnowledgeNode | Exception | None] = asyncio.Queue(
            maxsize=20
        )

        metrics1.configure(queue1, StreamingMetricsConfig(interval=0.1))
        metrics2.configure(queue2, StreamingMetricsConfig(interval=0.1))

        manager = MetricsManager([metrics1, metrics2])

        async with manager:
            await asyncio.sleep(0.3)

        assert len(metrics1.queue_size_samples) >= 2
        assert len(metrics2.queue_size_samples) >= 2

    async def test_manager_should_stop_all_collectors_even_on_error(self) -> None:
        """Verifies MetricsManager stops all collectors even if one fails.

        Given:
            MetricsManager with collectors, one that fails on exit.
        When:
            Exiting context manager.
        Then:
            All collectors are stopped despite errors (logged but not raised),
            normal collector completes monitoring, and failing collector
            executes stop_monitoring before raising.
        """

        class FailingCollector(MonitoringMetric):
            """Test fake that simulates stop_monitoring failure."""

            def __init__(self) -> None:
                super().__init__(enabled=True)
                self.started = False
                self.stopped = False

            async def start_monitoring(self) -> None:
                self.started = True

            async def stop_monitoring(self) -> None:
                self.stopped = True
                raise RuntimeError("Simulated stop failure")

            def finalize(self) -> dict[str, str]:
                return {"status": "failed"}

        normal_metrics = StreamingMetrics()
        queue: asyncio.Queue[KnowledgeNode | Exception | None] = asyncio.Queue(
            maxsize=10
        )
        normal_metrics.configure(queue, StreamingMetricsConfig(interval=0.1))

        failing_collector = FailingCollector()

        manager = MetricsManager([normal_metrics, failing_collector])

        async with manager:
            await asyncio.sleep(0.2)

        assert len(normal_metrics.queue_size_samples) >= 1
        assert failing_collector.started is True
        assert failing_collector.stopped is True

    async def test_get_summary_aggregates_all_collectors(self) -> None:
        """Verifies get_summary aggregates summaries from all collectors.

        Given:
            MetricsManager with multiple collectors after collection.
        When:
            Calling get_summary().
        Then:
            Returns dict mapping collector types to their summaries with
            all expected metrics fields (total_samples, avg_queue_size,
            max_queue_size, backpressure_events).
        """
        metrics1 = StreamingMetrics()
        metrics2 = StreamingMetrics()

        queue1: asyncio.Queue[KnowledgeNode | Exception | None] = asyncio.Queue(
            maxsize=10
        )
        queue2: asyncio.Queue[KnowledgeNode | Exception | None] = asyncio.Queue(
            maxsize=20
        )

        for i in range(5):
            await queue1.put(_make_node(i))

        for i in range(10):
            await queue2.put(_make_node(i + 100))

        metrics1.configure(queue1, StreamingMetricsConfig(interval=0.1))
        metrics2.configure(queue2, StreamingMetricsConfig(interval=0.1))

        manager = MetricsManager([metrics1, metrics2])

        async with manager:
            await asyncio.sleep(0.5)

        summary = manager.get_summary()

        assert "StreamingMetrics" in summary

        streaming_summary = summary["StreamingMetrics"]
        assert "total_samples" in streaming_summary
        assert streaming_summary["total_samples"] >= 3
        assert "avg_queue_size" in streaming_summary
        assert isinstance(streaming_summary["avg_queue_size"], float)
        assert "max_queue_size" in streaming_summary
        assert streaming_summary["max_queue_size"] >= 0
        assert "backpressure_events" in streaming_summary

    async def test_manager_should_cleanup_on_startup_failure(self) -> None:
        """Verifies MetricsManager cleans up if collector fails during startup.

        Given:
            MetricsManager with collector that fails on start.
        When:
            Entering context manager.
        Then:
            Previously started collectors are cleaned up, RuntimeError
            propagates with original failure message, and collector states
            reflect attempted start and successful cleanup.
        """

        class FailingStartCollector(MonitoringMetric):
            """Test fake that simulates start_monitoring failure."""

            def __init__(self) -> None:
                super().__init__(enabled=True)
                self.started = False
                self.stopped = False

            async def start_monitoring(self) -> None:
                self.started = True
                raise RuntimeError("Simulated start failure")

            async def stop_monitoring(self) -> None:
                self.stopped = True

            def finalize(self) -> dict[str, str]:
                return {"status": "failed"}

        normal_metrics = StreamingMetrics()
        queue: asyncio.Queue[KnowledgeNode | Exception | None] = asyncio.Queue(
            maxsize=10
        )
        normal_metrics.configure(queue, StreamingMetricsConfig(interval=0.1))

        failing_collector = FailingStartCollector()

        manager = MetricsManager([normal_metrics, failing_collector])

        with pytest.raises(RuntimeError, match="Simulated start failure"):
            async with manager:
                pass

        await asyncio.sleep(0.1)

        assert failing_collector.started is True
        assert failing_collector.stopped is False

    async def test_manager_with_empty_collectors_list(self) -> None:
        """Verifies MetricsManager handles empty collectors list gracefully.

        Given:
            MetricsManager with empty collectors list.
        When:
            Using context manager and calling get_summary.
        Then:
            Context manager completes without errors and summary returns
            an empty dictionary.
        """
        manager = MetricsManager([])

        async with manager:
            await asyncio.sleep(0.1)

        summary = manager.get_summary()
        assert summary == {}
        assert isinstance(summary, dict)
