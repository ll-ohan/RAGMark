"""The Forge - Document ingestion and fragmentation pipeline.

This package provides tools for ingesting documents from various sources
and fragmenting them into knowledge nodes ready for indexing.
"""

from ragmark.forge.factory import (
    FragmenterFactory,
    IngestorFactory,
    register_fragmenter,
    register_ingestor,
)
from ragmark.forge.fragmenters import BaseFragmenter
from ragmark.forge.ingestors import BaseIngestor
from ragmark.forge.qa_exporter import QAExporter
from ragmark.forge.runner import ForgeRunner

__all__ = [
    "BaseIngestor",
    "BaseFragmenter",
    "IngestorFactory",
    "FragmenterFactory",
    "ForgeRunner",
    "QAExporter",
    "register_ingestor",
    "register_fragmenter",
]
