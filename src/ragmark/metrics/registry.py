"""Central registry for metric discovery and instantiation."""

import re
from collections.abc import Callable
from typing import Any, TypeVar, overload

from ragmark.logger import get_logger
from ragmark.metrics.base import BaseMetric

logger = get_logger(__name__)


TMetric = TypeVar("TMetric", bound=BaseMetric)


class MetricRegistry:
    """Central registry for metric discovery and instantiation.

    Supports both direct metric registration and pattern-based factories
    for parameterized metrics like recall@k, precision@k, etc.
    """

    _metrics: dict[str, type[BaseMetric]] = {}
    _metric_by_base: dict[str, type[BaseMetric]] = {}
    _parameterized: dict[str, str] = {}
    _factories: dict[str, Callable[[str], type[BaseMetric]]] = {}

    @overload
    @classmethod
    def register(
        cls,
        metric_class: type[TMetric],
        *,
        parameter: str | None = None,
        base_name: str | None = None,
    ) -> type[TMetric]:
        ...

    @overload
    @classmethod
    def register(
        cls,
        metric_class: None = None,
        *,
        parameter: str | None = None,
        base_name: str | None = None,
    ) -> Callable[[type[TMetric]], type[TMetric]]:
        ...

    @classmethod
    def register(
        cls,
        metric_class: type[TMetric] | None = None,
        *,
        parameter: str | None = None,
        base_name: str | None = None,
    ) -> type[TMetric] | Callable[[type[TMetric]], type[TMetric]]:
        """Register a metric class.

        Usage:
            @MetricRegistry.register
            class RecallAtK(EvaluationMetric):
                ...

            @MetricRegistry.register(parameter="k", base_name="recall")
            class RecallAtK(EvaluationMetric):
                ...

        Args:
            metric_class: Metric class to register.
            parameter: Optional parameter name for parameterized metrics.
            base_name: Optional base name for parameterized metrics.

        Returns:
            The metric class (allows use as decorator).

        Raises:
            ValueError: If metric cannot be instantiated.
        """
        if metric_class is None:
            return lambda cls_to_register: cls.register(
                cls_to_register, parameter=parameter, base_name=base_name
            )

        try:
            instance = metric_class()
            name = instance.name
        except Exception as exc:
            logger.error(
                "Metric registration failed: class=%s, reason=%s",
                metric_class.__name__,
                str(exc),
            )
            logger.debug("Registration error details: %s", exc, exc_info=True)
            raise ValueError(
                f"Cannot register metric {metric_class.__name__}: "
                f"failed to instantiate for name extraction"
            ) from exc

        if name in cls._metrics:
            logger.warning(
                "Metric already registered: name=%s, old_class=%s, new_class=%s",
                name,
                cls._metrics[name].__name__,
                metric_class.__name__,
            )

        cls._metrics[name] = metric_class

        resolved_base, resolved_param = cls._resolve_parameterization(
            name, parameter, base_name
        )
        if resolved_base and resolved_param:
            cls._metric_by_base[resolved_base] = metric_class
            cls._parameterized[resolved_base] = resolved_param
            cls._register_factory(metric_class, resolved_base)

        logger.debug(
            "Metric registered: name=%s, class=%s", name, metric_class.__name__
        )
        return metric_class

    @classmethod
    def _resolve_parameterization(
        cls,
        name: str,
        parameter: str | None,
        base_name: str | None,
    ) -> tuple[str | None, str | None]:
        if parameter is None and "@" in name:
            parameter = "k"

        if base_name is None and "@" in name:
            match = re.match(r"^(\w+)@\w+$", name)
            if match:
                base_name = match.group(1)

        return base_name, parameter

    @classmethod
    def _register_factory(cls, metric_class: type[BaseMetric], base_name: str) -> None:
        """Register pattern-based factory for parameterized metrics.

        For metrics with @k pattern, allows creating any k value dynamically.

        Args:
            metric_class: Metric class to analyze for factory creation.
        """
        try:

            def factory(
                req_name: str,
                metric_cls: type[BaseMetric] = metric_class,
                base: str = base_name,
            ) -> type[BaseMetric]:
                if re.match(rf"^{base}@(\w+)$", req_name):
                    logger.debug(
                        "Creating parameterized metric: name=%s, class=%s",
                        req_name,
                        metric_cls.__name__,
                    )
                    return metric_cls
                return metric_cls

            cls._factories[base_name] = factory
            logger.debug(
                "Metric factory registered: pattern=%s, class=%s",
                base_name,
                metric_class.__name__,
            )
        except Exception as exc:
            logger.debug(
                "Failed to register factory: class=%s, error=%s",
                metric_class.__name__,
                str(exc),
            )

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> BaseMetric:
        """Get metric instance by name.

        Supports both direct lookup and factory-based creation for
        parameterized metrics.

        Args:
            name: Metric name (e.g., 'recall@5', 'mrr').

        Returns:
            Instantiated metric object.

        Raises:
            ValueError: If metric is not registered or not found.
        """
        metric_cls, resolved_kwargs = cls._resolve_metric(name, kwargs)
        logger.debug(
            "Metric instance created: name=%s, class=%s",
            name,
            metric_cls.__name__,
        )
        return metric_cls(**resolved_kwargs)

    @classmethod
    def create(cls, metric: str | type[BaseMetric], **kwargs: Any) -> BaseMetric:
        """Instantiate a metric by name or class with parameters.

        Extracts parameters from metric name (e.g., recall@5 → k=5)
        and merges with provided kwargs.

        Args:
            metric: Metric name (e.g., 'recall@5', 'mrr') or metric class.
            **kwargs: Arguments to pass to metric constructor.

        Returns:
            Instantiated metric object.

        Raises:
            ValueError: If metric is not registered.
        """
        if isinstance(metric, str):
            return cls.get(metric, **kwargs)

        logger.debug(
            "Creating metric from class: class=%s, kwargs=%s",
            metric.__name__,
            kwargs,
        )
        return metric(**kwargs)

    @classmethod
    def list_metrics(cls) -> list[str]:
        """List available metric categories.

        Returns:
            Sorted list of metric categories.
        """
        categories: set[str] = set()
        for metric_cls in cls._metrics.values():
            try:
                categories.add(metric_cls().category)
            except Exception as exc:
                logger.debug("Failed to read metric category: %s", exc, exc_info=True)

        return sorted(categories)

    @classmethod
    def get_registered(cls) -> list[str]:
        """List all registered metric names.

        Returns:
            Sorted list of registered metric names.
        """
        return sorted(cls._metrics.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered metrics and factories.

        Warning: Only use in tests.
        """
        cls._metrics.clear()
        cls._metric_by_base.clear()
        cls._parameterized.clear()
        cls._factories.clear()
        logger.debug("MetricRegistry cleared")

    @classmethod
    def _resolve_metric(
        cls, name: str, kwargs: dict[str, Any]
    ) -> tuple[type[BaseMetric], dict[str, Any]]:
        if name in cls._metrics:
            metric_cls = cls._metrics[name]
            return metric_cls, cls._resolve_parameters(name, kwargs)

        for base_name, factory in cls._factories.items():
            if re.match(rf"^{base_name}@(\w+)$", name):
                metric_cls = factory(name)
                return metric_cls, cls._resolve_parameters(name, kwargs)

        if name in cls._metric_by_base:
            metric_cls = cls._metric_by_base[name]
            return metric_cls, kwargs

        available = ", ".join(sorted(cls._metrics.keys())) or "none"
        logger.error("Metric not found: name=%s, available=%s", name, available)
        raise ValueError(f"Unknown metric: {name}. Available: {available}")

    @classmethod
    def _resolve_parameters(cls, name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        if "@" not in name:
            return kwargs

        base_name, param_value = name.split("@", 1)
        param_name = cls._parameterized.get(base_name)
        if not param_name:
            return kwargs

        if param_value == param_name:
            if param_name not in kwargs:
                raise ValueError(f"Missing parameter '{param_name}' for metric {name}")
            return kwargs

        if param_name not in kwargs:
            if param_value.isdigit():
                kwargs[param_name] = int(param_value)
            else:
                kwargs[param_name] = param_value
            logger.debug(
                "Extracted parameter for metric: name=%s, %s=%s",
                name,
                param_name,
                kwargs[param_name],
            )
        return kwargs
