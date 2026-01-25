"""Factory for creating retriever instances.

This module provides factory functions for building retrieval pipelines
based on configuration, including optional reranking.
"""

import logging

from ragmark.config.profile import RetrievalConfig
from ragmark.index.base import VectorIndex
from ragmark.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)


class RetrieverFactory:
    """Factory for creating retriever instances."""

    @staticmethod
    def create(config: RetrievalConfig, index: VectorIndex) -> BaseRetriever:
        """Create a retriever from configuration.

        Args:
            config: Retrieval configuration.
            index: Vector index to search.

        Returns:
            Configured retriever instance, optionally wrapped with reranking.

        Raises:
            ValueError: If configuration is invalid.
            UnsupportedBackendError: If required dependencies are missing.
        """
        # Create refiner if configured
        refiner = None
        if config.reranker is not None:
            from ragmark.retrieval.rerankers import CrossEncoderRefiner

            refiner = CrossEncoderRefiner.from_config(config.reranker)

        # If reranker present, return RefinedRetriever
        if refiner is not None:
            from ragmark.retrieval.strategies import RefinedRetriever

            return RefinedRetriever.from_config(config, index, refiner=refiner)

        # Otherwise, create appropriate base retriever
        mode = config.mode
        if mode == "dense":
            from ragmark.retrieval.strategies import DenseRetriever

            return DenseRetriever.from_config(config, index)

        elif mode == "sparse":
            from ragmark.retrieval.strategies import SparseRetriever

            return SparseRetriever.from_config(config, index)

        elif mode == "hybrid":
            from ragmark.retrieval.strategies import HybridRetriever

            return HybridRetriever.from_config(config, index)

        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")
