"""JSONL (JSON Lines) format adapter for reading and writing JSONL files."""

import json
from pathlib import Path
from typing import Any

from ragmark.adapters.base import FormatAdapter
from ragmark.logger import get_logger

logger = get_logger(__name__)


class JSONLAdapter(FormatAdapter):
    """Adapter for JSONL (JSON Lines) file I/O.

    Each line in a JSONL file is a complete valid JSON object. This format
    is useful for streaming and processing large datasets line-by-line.
    """

    def read(self, path: Path) -> list[dict[str, Any]]:
        """Read objects from JSONL file.

        Reads one JSON object per line. Empty lines are skipped.

        Args:
            path: Input JSONL file path.

        Returns:
            List of objects as dictionaries.

        Raises:
            FileNotFoundError: If file doesn't exist.
            json.JSONDecodeError: If any line contains invalid JSON.
        """
        logger.debug("Reading JSONL file: path=%s", path)

        results: list[dict[str, Any]] = []

        try:
            with open(path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        obj = json.loads(line)
                        results.append(obj)
                    except json.JSONDecodeError as exc:
                        logger.error(
                            "Invalid JSON on line: path=%s, line=%d, error=%s",
                            path,
                            line_num,
                            str(exc),
                        )
                        logger.debug(
                            "JSON decode error details: %s", exc, exc_info=True
                        )
                        raise

            logger.debug("JSONL read: objects=%d, path=%s", len(results), path)
            return results

        except FileNotFoundError:
            logger.error("JSONL file not found: path=%s", path)
            raise

    def write(
        self,
        data: list[dict[str, Any]],
        path: Path,
        **kwargs: Any,
    ) -> None:
        """Write objects to JSONL file.

        Writes one JSON object per line, without indentation for compactness.

        Args:
            data: Objects to write as dictionaries.
            path: Output JSONL file path.
            **kwargs: Additional arguments passed to json.dumps.

        Raises:
            IOError: If write operation fails.
        """
        logger.debug("Writing JSONL file: path=%s, objects=%d", path, len(data))

        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding="utf-8") as f:
                for idx, item in enumerate(data):
                    try:
                        line = json.dumps(item, ensure_ascii=False, **kwargs)
                        f.write(line + "\n")
                    except (TypeError, ValueError) as exc:
                        logger.error(
                            "JSON serialization failed: path=%s, index=%d, error=%s",
                            path,
                            idx,
                            str(exc),
                        )
                        logger.debug(
                            "Serialization error details: %s", exc, exc_info=True
                        )
                        raise exc from exc

            logger.info("JSONL file written: path=%s, objects=%d", path, len(data))

        except OSError as exc:
            logger.error("JSONL write failed: path=%s, error=%s", path, str(exc))
            logger.debug("JSONL write error details: %s", exc, exc_info=True)
            raise
