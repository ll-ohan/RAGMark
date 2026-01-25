"""Abstract base class and interfaces for text fragmentation.

This module defines the contract for fragmenting source documents into
smaller, semantically coherent chunks (knowledge nodes) suitable for
vector indexing and retrieval.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

from ragmark.schemas.documents import KnowledgeNode, SourceDoc

if TYPE_CHECKING:
    from ragmark.config.profile import FragmenterConfig


class BaseFragmenter(ABC):
    """Abstract base class for text fragmentation strategies.

    Fragmenters are responsible for splitting source documents into
    smaller chunks while preserving semantic coherence and respecting
    size constraints. Different strategies optimize for different goals
    (token limits, semantic boundaries, structural preservation).

    All implementations must support streaming fragmentation via generators
    to enable memory-efficient processing of large document collections.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: "FragmenterConfig") -> "BaseFragmenter":
        """Instantiate fragmenter from configuration.

        Args:
            config: FragmenterConfig instance.

        Returns:
            Configured fragmenter instance.
        """
        pass

    @abstractmethod
    def fragment(self, doc: SourceDoc) -> list[KnowledgeNode]:
        """Fragment a single source document into knowledge nodes.

        Args:
            doc: Source document to fragment.

        Returns:
            List of knowledge nodes extracted from the document.

        Raises:
            FragmentationError: If fragmentation fails.
        """
        pass

    def fragment_batch(self, docs: Iterable[SourceDoc]) -> Iterator[KnowledgeNode]:
        """Fragment multiple documents with streaming output.

        This method processes documents one at a time and yields nodes
        immediately as they are created, ensuring O(1) memory consumption.

        Args:
            docs: Iterable of source documents to fragment.

        Yields:
            KnowledgeNode instances as they are created.

        Raises:
            FragmentationError: If fragmentation fails for any document.
        """
        for doc in docs:
            yield from self.fragment(doc)

    @property
    @abstractmethod
    def chunk_size(self) -> int:
        """Get the target chunk size for this fragmenter.

        Returns:
            Target chunk size in tokens or characters (strategy-dependent).
        """
        pass

    @property
    @abstractmethod
    def overlap(self) -> int:
        """Get the overlap size between consecutive chunks.

        Returns:
            Overlap size in tokens or characters (strategy-dependent).
        """
        pass
