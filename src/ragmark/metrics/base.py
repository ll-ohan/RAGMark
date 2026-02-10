"""Base abstractions for metrics.

Defines the common interface and two specialized branches for different
metric paradigms: evaluation (sync, stateless) and monitoring (async, stateful).
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class BaseMetric(ABC):
    """Minimal interface for all metrics.

    All metrics must have a unique name, description, and category
    for discovery and monitoring purposes.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique metric identifier.

        Examples: 'recall@5', 'streaming_backpressure', 'faithfulness'.
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this metric measures."""
        pass

    @property
    def category(self) -> str:
        """Metric category: 'monitoring' or 'evaluation'."""
        return "unknown"

    @property
    def metadata(self) -> dict[str, Any]:
        """Additional metadata (tags, version, etc.)."""
        return {}


class EvaluationMetric(BaseMetric, Generic[TInput, TOutput]):
    """Base for stateless evaluation metrics.

    Evaluation metrics are pure functions that compute a score from inputs
    without maintaining state. They are synchronous and can validate inputs
    before computation.
    """

    @property
    def category(self) -> str:
        """Evaluation metrics are always in the 'evaluation' category."""
        return "evaluation"

    @abstractmethod
    def compute(self, **inputs: TInput) -> TOutput:
        """Compute metric from inputs.

        Args:
            **inputs: Metric-specific inputs.

        Returns:
            Computed metric value.

        Raises:
            MetricValidationError: If inputs are invalid.
        """
        pass

    def validate_inputs(self, **inputs: Any) -> None:
        """Validate inputs before computation.

        Override to implement custom validation. Default implementation
        does nothing and allows all inputs to proceed.

        Args:
            **inputs: Inputs to validate.

        Raises:
            MetricValidationError: If validation fails.
        """
        pass

    def __call__(self, **inputs: TInput) -> TOutput:
        """Allow metrics to be called like functions.

        Convenience wrapper that validates inputs and calls compute().

        Args:
            **inputs: Metric-specific inputs.

        Returns:
            Computed metric value.
        """
        self.validate_inputs(**inputs)
        return self.compute(**inputs)


class MonitoringMetric(BaseMetric):
    """Unified monitoring metrics orchestrator.

    Collects stage timings and exposes context managers for consistent
    instrumentation across pipeline stages.
    """

    def __init__(self, enabled: bool = True):
        """Initialize monitoring metrics.

        Args:
            enabled: Whether monitoring is active.
        """
        self.enabled = enabled
        self._stage_timings: dict[str, list[float]] = {}

    @property
    def category(self) -> str:
        """Monitoring metrics are always in the 'monitoring' category."""
        return "monitoring"

    @property
    def name(self) -> str:
        """Return the metric identifier."""
        return "monitoring"

    @property
    def description(self) -> str:
        """Describe the monitoring metric."""
        return "Pipeline stage latency metrics"

    async def start_monitoring(self) -> None:
        """Initialize and launch background monitoring tasks.

        Raises:
            ValueError: If configuration is invalid or incomplete.
        """
        return None

    async def stop_monitoring(self) -> None:
        """Stop background tasks and release resources.

        Use timeout-based cancellation to prevent hanging when tasks
        don't respond to shutdown signals.
        """
        return None

    def finalize(self) -> dict[str, Any]:
        """Compute summary statistics from collected data.

        Returns:
            Summary statistics dictionary with metric names as keys.
        """
        summary: dict[str, Any] = {}
        for stage, timings in self._stage_timings.items():
            if not timings:
                continue
            total = sum(timings)
            summary[stage] = {
                "count": len(timings),
                "total_ms": total,
                "avg_ms": total / len(timings),
                "min_ms": min(timings),
                "max_ms": max(timings),
            }
        return summary

    def stage(self, stage_name: str) -> "_StageTimer":
        """Create a context manager to track stage latency.

        Args:
            stage_name: Name of the stage to record.

        Returns:
            Context manager that records elapsed time.
        """
        return _StageTimer(self, stage_name)

    def record(self, stage_name: str, elapsed_ms: float) -> None:
        """Record timing for a stage.

        Args:
            stage_name: Name of the stage to record.
            elapsed_ms: Elapsed time in milliseconds.
        """
        if not self.enabled:
            return
        self._stage_timings.setdefault(stage_name, []).append(elapsed_ms)

    async def __aenter__(self) -> "MonitoringMetric":
        """Start monitoring and return self.

        Raises:
            ValueError: If collector configuration is invalid.
        """
        await self.start_monitoring()
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> None:
        """Stop monitoring and finalize metrics.

        Cleanup executes even if exceptions occurred during monitoring.
        """
        await self.stop_monitoring()
        self.finalize()


class _StageTimer:
    def __init__(self, monitor: MonitoringMetric, stage_name: str):
        self._monitor = monitor
        self._stage_name = stage_name
        self._start: float | None = None

    def __enter__(self) -> "_StageTimer":
        if self._monitor.enabled:
            self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> None:
        if self._monitor.enabled and self._start is not None:
            elapsed_ms = (time.perf_counter() - self._start) * 1000
            self._monitor.record(self._stage_name, elapsed_ms)

    async def __aenter__(self) -> "_StageTimer":
        return self.__enter__()

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> None:
        self.__exit__(_exc_type, _exc_val, _exc_tb)


class MetricValidationError(ValueError):
    """Raised when metric input validation fails."""

    pass


MonitoringMetrics = MonitoringMetric
