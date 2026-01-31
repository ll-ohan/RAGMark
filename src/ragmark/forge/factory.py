"""Factories for forge components.

This module provides factory functions for creating ingestor and fragmenter
instances from configuration objects.
"""

from ragmark.config.profile import FragmenterConfig, IngestorConfig
from ragmark.exceptions import UnsupportedBackendError
from ragmark.forge.fragmenters import BaseFragmenter
from ragmark.forge.ingestors import BaseIngestor
from ragmark.logger import get_logger

logger = get_logger(__name__)

_INGESTOR_REGISTRY: dict[str, type[BaseIngestor]] = {}
_FRAGMENTER_REGISTRY: dict[str, type[BaseFragmenter]] = {}


def register_ingestor(name: str, cls: type[BaseIngestor]) -> None:
    """Register a custom ingestor backend.

    Enables adding custom implementations without modifying the core library.

    Args:
        name: Unique backend identifier.
        cls: Implementation class.
    """
    _INGESTOR_REGISTRY[name] = cls
    logger.debug("Registered ingestor backend: %s", name)


def register_fragmenter(name: str, cls: type[BaseFragmenter]) -> None:
    """Register a custom fragmenter strategy.

    Enables adding custom implementations without modifying the core library.

    Args:
        name: Unique strategy identifier.
        cls: Implementation class.
    """
    _FRAGMENTER_REGISTRY[name] = cls
    logger.debug("Registered fragmenter strategy: %s", name)


class IngestorFactory:
    """Factory for creating ingestor instances."""

    @staticmethod
    def create(config: IngestorConfig) -> BaseIngestor:
        """Instantiate an ingestor based on the provided configuration.

        Args:
            config: The ingestion configuration.

        Returns:
            An initialized ingestor instance.

        Raises:
            UnsupportedBackendError: If the backend is not supported or installed.
            NotImplementedError: If the backend is valid but not yet implemented.
        """
        backend = config.backend

        if backend in _INGESTOR_REGISTRY:
            return _INGESTOR_REGISTRY[backend].from_config(config)

        if backend == "fitz":
            from ragmark.forge.ingestors import FitzIngestor

            return FitzIngestor.from_config(config)
        elif backend == "marker":
            try:
                raise NotImplementedError(
                    "MarkerIngestor will be implemented when concrete ingestor "
                    "implementations are added to the codebase"
                )
            except ImportError as e:
                raise UnsupportedBackendError("marker", "marker-pdf") from e
        else:
            raise UnsupportedBackendError(backend)


class FragmenterFactory:
    """Factory for creating fragmenter instances."""

    @staticmethod
    def create(config: FragmenterConfig) -> BaseFragmenter:
        """Instantiate a fragmenter based on the provided configuration.

        Args:
            config: The fragmentation configuration.

        Returns:
            An initialized fragmenter instance.

        Raises:
            ValueError: If the strategy is unknown.
            NotImplementedError: If the strategy is valid but not yet implemented.
        """
        strategy = config.strategy

        if strategy in _FRAGMENTER_REGISTRY:
            return _FRAGMENTER_REGISTRY[strategy].from_config(config)

        if strategy == "token":
            from ragmark.forge.fragmenters import TokenFragmenter

            return TokenFragmenter.from_config(config)
        elif strategy == "semantic":
            raise NotImplementedError(
                "SemanticFragmenter will be implemented when concrete fragmenter "
                "implementations are added to the codebase"
            )
        elif strategy == "markdown":
            raise NotImplementedError(
                "MarkdownFragmenter will be implemented when concrete fragmenter "
                "implementations are added to the codebase"
            )
        elif strategy == "recursive":
            raise NotImplementedError(
                "RecursiveFragmenter will be implemented when concrete fragmenter "
                "implementations are added to the codebase"
            )
        else:
            raise ValueError(f"Unknown fragmentation strategy: {strategy}")
