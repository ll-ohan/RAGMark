"""JSON format adapter for reading and writing JSON files."""

import json
from pathlib import Path
from typing import Any, cast

from ragmark.adapters.base import FormatAdapter
from ragmark.logger import get_logger

logger = get_logger(__name__)


class JSONAdapter(FormatAdapter):
    """Adapter for JSON file I/O.

    Reads and writes Python objects as JSON, handling both single objects
    and lists of objects.
    """

    def read(self, path: Path) -> list[dict[str, Any]]:
        """Read objects from JSON file.

        Args:
            path: Input JSON file path.

        Returns:
            List of objects as dictionaries. If file contains a single
            object, wraps it in a list.

        Raises:
            FileNotFoundError: If file doesn't exist.
            json.JSONDecodeError: If JSON is invalid.
        """
        logger.debug("Reading JSON file: path=%s", path)

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                data_list = cast(list[dict[str, Any]], data)
                logger.debug("JSON read: objects=%d, path=%s", len(data_list), path)
                return data_list

            data_dict = cast(dict[str, Any], data)
            logger.debug("JSON read: single object wrapped, path=%s", path)
            return [data_dict]

        except FileNotFoundError:
            logger.error("JSON file not found: path=%s", path)
            raise
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON: path=%s, error=%s", path, str(exc))
            logger.debug("JSON decode error details: %s", exc, exc_info=True)
            raise exc from exc

    def write(
        self,
        data: list[dict[str, Any]],
        path: Path,
        indent: int = 2,
        **kwargs: Any,
    ) -> None:
        """Write objects to JSON file.

        Args:
            data: Objects to write as dictionaries.
            path: Output JSON file path.
            indent: JSON indentation level (default: 2).
            **kwargs: Additional arguments passed to json.dump (e.g., sort_keys).

        Raises:
            IOError: If write operation fails.
        """
        logger.debug("Writing JSON file: path=%s, objects=%d", path, len(data))

        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    data,
                    f,
                    indent=indent,
                    ensure_ascii=False,
                    **kwargs,
                )

            logger.info("JSON file written: path=%s, objects=%d", path, len(data))

        except OSError as exc:
            logger.error("JSON write failed: path=%s, error=%s", path, str(exc))
            logger.debug("JSON write error details: %s", exc, exc_info=True)
            raise
