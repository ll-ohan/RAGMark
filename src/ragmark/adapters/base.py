"""Base abstractions for generic adapters."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

TSource = TypeVar("TSource")
TTarget = TypeVar("TTarget")


class Adapter(ABC, Generic[TSource, TTarget]):
    """Generic adapter for transforming objects from source to target type.

    Adapters encapsulate transformation logic and can handle:
    - One-to-one transformations (source → single target)
    - One-to-many transformations (source → multiple targets)
    - Field mapping and value transformation
    - Input validation before transformation
    """

    @abstractmethod
    def adapt(self, source: TSource) -> TTarget | list[TTarget]:
        """Transform source object(s) to target object(s).

        Can return a single target or list of targets for one-to-many
        transformations. Implementations should validate source before
        transformation and log results appropriately.

        Args:
            source: Source object to transform.

        Returns:
            Transformed object or list of objects.

        Raises:
            ValueError: If source cannot be adapted.
        """
        pass

    def adapt_many(self, sources: list[TSource]) -> list[TTarget]:
        """Transform multiple source objects.

        Handles both one-to-one and one-to-many adapters by flattening
        results into a single list.

        Args:
            sources: List of source objects to transform.

        Returns:
            Flattened list of target objects.
        """
        results: list[Any] = []
        for source in sources:
            adapted = self.adapt(source)
            if isinstance(adapted, list):
                results.extend(cast(list[TTarget], adapted))
            else:
                results.append(adapted)

        return results

    def validate(self, source: TSource) -> bool:
        """Check if source can be adapted.

        Override to implement custom validation. Default implementation
        always returns True, allowing all sources to proceed.

        Args:
            source: Source object to validate.

        Returns:
            True if source is valid for adaptation, False otherwise.
        """
        return True


class FormatAdapter(ABC):
    """Abstract base for file format I/O.

    Handles serialization and deserialization for different file formats
    (JSON, JSONL, YAML, etc.), allowing transformations to work with
    various persistence formats.
    """

    @abstractmethod
    def read(self, path: Path) -> list[dict[str, Any]]:
        """Read objects from file.

        Args:
            path: Input file path.

        Returns:
            List of objects as dictionaries.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file format is invalid.
        """
        pass

    @abstractmethod
    def write(
        self,
        data: list[dict[str, Any]],
        path: Path,
        **kwargs: Any,
    ) -> None:
        """Write objects to file.

        Args:
            data: Objects to write as dictionaries.
            path: Output file path.
            **kwargs: Format-specific options (indent, etc.).

        Raises:
            IOError: If write operation fails.
        """
        pass
