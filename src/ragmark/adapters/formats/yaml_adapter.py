"""YAML format adapter for reading and writing YAML files.

Placeholder for future implementation with PyYAML support.
"""

from pathlib import Path
from typing import Any

from ragmark.adapters.base import FormatAdapter
from ragmark.logger import get_logger

logger = get_logger(__name__)


class YAMLAdapter(FormatAdapter):
    """Adapter for YAML file I/O.

    Placeholder for future implementation. When implemented, will support
    reading and writing Python objects in YAML format.
    """

    def read(self, path: Path) -> list[dict[str, Any]]:
        """Read objects from YAML file.

        Args:
            path: Input YAML file path.

        Returns:
            List of objects as dictionaries.

        Raises:
            NotImplementedError: YAML support not yet implemented.
        """
        logger.debug("YAML adapter not yet implemented: path=%s", path)
        raise NotImplementedError(
            "YAML format adapter not yet implemented. "
            "Please use JSON or JSONL format for now."
        )

    def write(
        self,
        data: list[dict[str, Any]],
        path: Path,
        **kwargs: Any,
    ) -> None:
        """Write objects to YAML file.

        Args:
            data: Objects to write as dictionaries.
            path: Output YAML file path.
            **kwargs: Additional YAML options.

        Raises:
            NotImplementedError: YAML support not yet implemented.
        """
        logger.debug(
            "YAML adapter not yet implemented: path=%s, objects=%d",
            path,
            len(data),
        )
        raise NotImplementedError(
            "YAML format adapter not yet implemented. "
            "Please use JSON or JSONL format for now."
        )
