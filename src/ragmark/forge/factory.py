"""Factories for forge components.

This module provides factory functions for creating ingestor and fragmenter
instances from configuration objects.
"""

from ragmark.config.profile import FragmenterConfig, IngestorConfig
from ragmark.exceptions import UnsupportedBackendError
from ragmark.forge.fragmenters import BaseFragmenter
from ragmark.forge.ingestors import BaseIngestor


class IngestorFactory:
    """Factory for creating ingestor instances."""

    @staticmethod
    def create(config: IngestorConfig) -> BaseIngestor:
        """Create an ingestor from configuration.

        Args:
            config: IngestorConfig instance.

        Returns:
            Configured ingestor instance.

        Raises:
            UnsupportedBackendError: If backend is not available.
            NotImplementedError: If concrete implementations are not yet available.
        """
        backend = config.backend

        # Note: These imports will work once concrete implementations are added
        # For now, they will raise NotImplementedError
        if backend == "fitz":
            # from ragmark.forge.implementations import FitzIngestor
            # return FitzIngestor.from_config(config)
            raise NotImplementedError(
                "FitzIngestor will be implemented when concrete ingestor "
                "implementations are added to the codebase"
            )
        elif backend == "marker":
            try:
                # from ragmark.forge.implementations import MarkerIngestor
                # return MarkerIngestor.from_config(config)
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
        """Create a fragmenter from configuration.

        Args:
            config: FragmenterConfig instance.

        Returns:
            Configured fragmenter instance.

        Raises:
            ValueError: If strategy is unknown.
            NotImplementedError: If concrete implementations are not yet available.
        """
        strategy = config.strategy

        # Note: These imports will work once concrete implementations are added
        # For now, they will raise NotImplementedError
        if strategy == "token":
            # from ragmark.forge.implementations import TokenFragmenter
            # return TokenFragmenter.from_config(config)
            raise NotImplementedError(
                "TokenFragmenter will be implemented when concrete fragmenter "
                "implementations are added to the codebase"
            )
        elif strategy == "semantic":
            # from ragmark.forge.implementations import SemanticFragmenter
            # return SemanticFragmenter.from_config(config)
            raise NotImplementedError(
                "SemanticFragmenter will be implemented when concrete fragmenter "
                "implementations are added to the codebase"
            )
        elif strategy == "markdown":
            # from ragmark.forge.implementations import MarkdownFragmenter
            # return MarkdownFragmenter.from_config(config)
            raise NotImplementedError(
                "MarkdownFragmenter will be implemented when concrete fragmenter "
                "implementations are added to the codebase"
            )
        elif strategy == "recursive":
            # from ragmark.forge.implementations import RecursiveFragmenter
            # return RecursiveFragmenter.from_config(config)
            raise NotImplementedError(
                "RecursiveFragmenter will be implemented when concrete fragmenter "
                "implementations are added to the codebase"
            )
        else:
            raise ValueError(f"Unknown fragmentation strategy: {strategy}")
