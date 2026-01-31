"""Reranker implementations.

This module contains concrete implementations of rerankers/refiners.
"""

from ragmark.config.profile import RerankerConfig
from ragmark.retrieval.base import BaseRefiner
from ragmark.schemas.retrieval import RetrievedNode


class CrossEncoderRefiner(BaseRefiner):
    """Reranks candidates using a cross-encoder model.

    Attributes:
        model_name: Name or path of the model.
        device: Computation device.
        top_k: Default number of results to return.
    """

    def __init__(self, model_name: str, device: str = "cpu", top_k: int = 10):
        """Initialize the cross-encoder refiner.

        Args:
            model_name: Name or path of the model.
            device: Computation device to use.
            top_k: Default number of top results to keep.
        """
        self.model_name = model_name
        self.device = device
        self.top_k = top_k

    @classmethod
    def from_config(cls, config: RerankerConfig) -> "CrossEncoderRefiner":
        """Instantiate the refiner from configuration.

        Args:
            config: The reranker configuration.

        Returns:
            The configured refiner instance.
        """
        return cls(
            model_name=config.model_name,
            device=config.device,
            top_k=config.top_k,
        )

    def refine(
        self, query: str, candidates: list[RetrievedNode], top_k: int
    ) -> list[RetrievedNode]:
        """Rerank retrieved candidates.

        Args:
            query: The search query.
            candidates: The list of nodes to reorder.
            top_k: The maximum number of nodes to return.

        Returns:
            The reordered list of nodes.

        Raises:
            NotImplementedError: Because this class is a placeholder for Phase 2.
        """
        raise NotImplementedError("CrossEncoderRefiner will be implemented in Phase 2")
