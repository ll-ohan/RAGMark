"""Metrics collection infrastructure with async lifecycle management."""

from abc import ABC, abstractmethod
from typing import Any

from ragmark.logger import get_logger

logger = get_logger(__name__)


class BaseMetricsCollector(ABC):
    """Abstract base for metrics collectors with lifecycle management.

    Subclasses must implement start_monitoring(), stop_monitoring(), and
    finalize() to define collection behavior.
    """

    @abstractmethod
    async def start_monitoring(self) -> None:
        """Initialize resources and launch background monitoring tasks.

        Raises:
            ValueError: If configuration is invalid or incomplete.
        """
        pass

    @abstractmethod
    async def stop_monitoring(self) -> None:
        """Stop background tasks and release resources.

        Use timeout-based cancellation to prevent hanging when tasks
        don't respond to shutdown signals.
        """
        pass

    @abstractmethod
    def finalize(self) -> dict[str, Any]:
        """Compute summary statistics from collected data.

        Returns:
            Summary statistics dictionary with metric names as keys.
        """
        pass

    async def __aenter__(self) -> "BaseMetricsCollector":
        """Start monitoring and return self.

        Raises:
            ValueError: If collector configuration is invalid.
        """
        await self.start_monitoring()
        logger.debug("Metrics collector started: type=%s", type(self).__name__)
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
        summary = self.finalize()
        logger.info(
            "Metrics finalized: type=%s, summary=%s", type(self).__name__, summary
        )


class MetricsManager:
    """Composite manager coordinating multiple metrics collectors.

    All collectors start and stop together with guaranteed cleanup even if
    individual collectors fail during shutdown.

    Attributes:
        collectors: Metrics collectors under management.
    """

    def __init__(self, collectors: list[BaseMetricsCollector]):
        """Initialize manager with metrics collectors.

        Args:
            collectors: Metrics collectors to manage.
        """
        self.collectors = collectors

    async def __aenter__(self) -> "MetricsManager":
        """Start all collectors and return self.

        Raises:
            Exception: If any collector fails to start. Already-started
                collectors are cleaned up before re-raising.
        """
        started = []
        try:
            for collector in self.collectors:
                await collector.__aenter__()
                started.append(collector)

            logger.debug("MetricsManager started: collectors=%d", len(self.collectors))
            return self

        except Exception:
            for collector in started:
                try:
                    await collector.__aexit__(None, None, None)
                except Exception as cleanup_error:
                    logger.error(
                        "Collector cleanup failed during startup rollback: type=%s",
                        type(collector).__name__,
                    )
                    logger.debug(
                        "Startup rollback error details: %s",
                        cleanup_error,
                        exc_info=True,
                    )
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Stop all collectors.

        Continue cleanup even if individual collectors fail to prevent
        resource leaks.
        """
        errors = []
        for collector in self.collectors:
            try:
                await collector.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.error(
                    "Collector cleanup failed: type=%s",
                    type(collector).__name__,
                )
                logger.debug("Cleanup failure details: %s", e, exc_info=True)
                errors.append(e)

        if errors:
            logger.warning(
                "MetricsManager cleanup completed with %d errors", len(errors)
            )
        else:
            logger.debug("MetricsManager cleanup completed successfully")

    def get_summary(self) -> dict[str, dict[str, Any]]:
        """Aggregate finalized metrics from all collectors.

        Returns:
            Dictionary mapping collector class names to their summaries.
        """
        return {
            type(collector).__name__: collector.finalize()
            for collector in self.collectors
        }
