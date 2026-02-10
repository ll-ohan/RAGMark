"""Monitoring metrics collection infrastructure."""

from typing import Any

from ragmark.logger import get_logger
from ragmark.metrics.base import MonitoringMetric

logger = get_logger(__name__)


class MetricsManager:
    """Composite manager coordinating monitoring metrics.

    All metrics start and stop together with guaranteed cleanup even if
    individual metrics fail during shutdown.

    Attributes:
        collectors: Monitoring metrics under management.
    """

    def __init__(self, collectors: list[MonitoringMetric]):
        """Initialize manager with monitoring metrics.

        Args:
            collectors: Monitoring metrics to manage.
        """
        self.collectors = collectors

    async def __aenter__(self) -> "MetricsManager":
        """Start all monitoring metrics and return self.

        Raises:
            Exception: If any metric fails to start. Already-started
                metrics are cleaned up before re-raising.
        """
        started: list[MonitoringMetric] = []
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
        """Stop all monitoring metrics.

        Continue cleanup even if individual metrics fail to prevent
        resource leaks.
        """
        errors: list[Exception] = []
        for collector in self.collectors:
            try:
                await collector.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as error:
                logger.error(
                    "Collector cleanup failed: type=%s",
                    type(collector).__name__,
                )
                logger.debug("Cleanup failure details: %s", error, exc_info=True)
                errors.append(error)

        if errors:
            logger.warning(
                "MetricsManager cleanup completed with %d errors", len(errors)
            )
        else:
            logger.debug("MetricsManager cleanup completed successfully")

    def get_summary(self) -> dict[str, dict[str, Any]]:
        """Aggregate finalized metrics from all collectors.

        Returns:
            Dictionary mapping metric class names to their summaries.
        """
        return {
            type(collector).__name__: collector.finalize()
            for collector in self.collectors
        }
