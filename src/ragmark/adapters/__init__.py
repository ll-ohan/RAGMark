"""Generic adapter framework for object transformations.

Provides abstractions for transforming objects from source to target types,
with support for field mapping and multiple output formats.
"""

from ragmark.adapters.base import Adapter, FormatAdapter

__all__ = [
    "Adapter",
    "FormatAdapter",
]
