"""Unit tests for MetricRegistry behaviors."""

from __future__ import annotations

from typing import Any, cast

import pytest

from ragmark.metrics.base import EvaluationMetric
from ragmark.metrics.registry import MetricRegistry


class SimpleMetric(EvaluationMetric[Any, int]):
    """Simple metric for registry tests."""

    def __init__(self, value: int = 1) -> None:
        self.value = value

    @property
    def name(self) -> str:
        return "simple"

    @property
    def description(self) -> str:
        return "Simple metric for testing"

    def compute(self, **inputs: Any) -> int:
        return self.value


class AlphaAtK(EvaluationMetric[Any, int]):
    """Parameterized metric using @k naming."""

    def __init__(self, k: int = 5) -> None:
        self.k = k

    @property
    def name(self) -> str:
        return f"alpha@{self.k}"

    @property
    def description(self) -> str:
        return "Alpha metric"

    def compute(self, **inputs: Any) -> int:
        return self.k


@pytest.fixture(autouse=True)
def restore_metric_registry() -> Any:
    """Restore registry state after each test."""
    snapshot_metrics = dict(MetricRegistry._metrics)  # type: ignore[attr-defined]
    snapshot_by_base = dict(MetricRegistry._metric_by_base)  # type: ignore[attr-defined]
    snapshot_parameterized = dict(MetricRegistry._parameterized)  # type: ignore[attr-defined]
    snapshot_factories = dict(MetricRegistry._factories)  # type: ignore[attr-defined]

    yield

    MetricRegistry._metrics = snapshot_metrics  # type: ignore[attr-defined]
    MetricRegistry._metric_by_base = snapshot_by_base  # type: ignore[attr-defined]
    MetricRegistry._parameterized = snapshot_parameterized  # type: ignore[attr-defined]
    MetricRegistry._factories = snapshot_factories  # type: ignore[attr-defined]


@pytest.mark.unit
def test_registry_register_and_get_should_create_instance() -> None:
    """Validates register and get return a working metric instance.

    Given:
        A metric class registered in the registry.
    When:
        Getting it by name.
    Then:
        The instance computes expected output.
    """
    MetricRegistry.clear()
    MetricRegistry.register(SimpleMetric)

    metric = cast(EvaluationMetric[Any, int], MetricRegistry.get("simple"))

    assert metric.name == "simple"
    assert metric.description.startswith("Simple")
    assert metric.compute() == 1


@pytest.mark.unit
def test_registry_should_resolve_parameterized_metric_name() -> None:
    """Validates parameterized metrics extract k from name.

    Given:
        A parameterized metric registered by name pattern.
    When:
        Requesting a different k via name.
    Then:
        The instance receives the extracted k value.
    """
    MetricRegistry.clear()
    MetricRegistry.register(AlphaAtK)

    metric = cast(EvaluationMetric[Any, int], MetricRegistry.get("alpha@7"))

    assert metric.name == "alpha@7"
    assert metric.compute() == 7


@pytest.mark.unit
def test_registry_create_should_accept_metric_class() -> None:
    """Validates create can instantiate from metric class.

    Given:
        A metric class.
    When:
        Creating via MetricRegistry.create with kwargs.
    Then:
        The instance reflects provided parameters.
    """
    MetricRegistry.clear()

    metric = cast(
        EvaluationMetric[Any, int], MetricRegistry.create(SimpleMetric, value=4)
    )

    assert metric.name == "simple"
    assert metric.compute() == 4


@pytest.mark.unit
def test_registry_should_raise_when_metric_instantiation_fails() -> None:
    """Validates registration fails for metrics without default constructor.

    Given:
        A metric class that requires a constructor argument.
    When:
        Registering the metric.
    Then:
        ValueError is raised with chained cause.
    """

    class RequiresArg(EvaluationMetric[Any, int]):
        def __init__(self, required: int) -> None:
            self.required = required

        @property
        def name(self) -> str:
            return "requires_arg"

        @property
        def description(self) -> str:
            return "Requires arg"

        def compute(self, **inputs: Any) -> int:
            return self.required

    MetricRegistry.clear()

    with pytest.raises(ValueError) as exc_info:
        MetricRegistry.register(RequiresArg)

    assert "failed to instantiate" in str(exc_info.value)
    assert exc_info.value.__cause__ is not None


@pytest.mark.unit
def test_registry_should_raise_for_unknown_metric() -> None:
    """Validates unknown metric requests include available list.

    Given:
        An empty registry.
    When:
        Requesting a non-existent metric.
    Then:
        ValueError is raised with available metrics detail.
    """
    MetricRegistry.clear()

    with pytest.raises(ValueError, match="Unknown metric") as exc_info:
        MetricRegistry.get("missing")

    assert "Available" in str(exc_info.value)
