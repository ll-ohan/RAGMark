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
        """Instantiate a retriever based on the provided configuration.

        Args:
            config: The retrieval configuration.
            index: The vector index to search.

        Returns:
            The configured retriever instance, optionally wrapped with reranking.

        Raises:
            ValueError: If the configuration is invalid.
            UnsupportedBackendError: If required dependencies are missing.
        """
        refiner = None
        if config.reranker is not None:
            from ragmark.retrieval.rerankers import CrossEncoderRefiner

            refiner = CrossEncoderRefiner.from_config(config.reranker)

        if refiner is not None:
            from ragmark.retrieval.strategies import RefinedRetriever

            return RefinedRetriever.from_config(config, index, refiner=refiner)

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
