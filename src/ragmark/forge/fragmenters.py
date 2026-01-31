"""Abstract base class and interfaces for text fragmentation.

This module defines the contract for fragmenting source documents into
smaller, semantically coherent chunks (knowledge nodes) suitable for
vector indexing and retrieval.
"""

import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any

from ragmark.logger import get_logger
from ragmark.schemas.documents import KnowledgeNode, NodePosition, SourceDoc

if TYPE_CHECKING:
    from ragmark.config.profile import FragmenterConfig

logger = get_logger(__name__)


class BaseFragmenter(ABC):
    """Abstract base class for text fragmentation strategies.

    Fragmenters are responsible for splitting source documents into smaller
    chunks while preserving semantic coherence and respecting size constraints.
    Different strategies optimize for different goals (token limits, semantic
    boundaries, structural preservation).

    All implementations must support streaming fragmentation via generators to
    enable memory-efficient processing of large document collections.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: "FragmenterConfig") -> "BaseFragmenter":
        """Instantiate a fragmenter based on the provided configuration.

        Args:
            config: The fragmentation configuration.

        Returns:
            An initialized fragmenter instance.
        """
        pass

    @abstractmethod
    def fragment(self, doc: SourceDoc) -> list[KnowledgeNode]:
        """Fragment a single source document into knowledge nodes.

        Args:
            doc: The source document to process.

        Returns:
            A list of generated knowledge nodes.

        Raises:
            FragmentationError: If fragmentation fails.
        """
        pass

    def fragment_batch(self, docs: Iterable[SourceDoc]) -> Iterator[KnowledgeNode]:
        """Fragment multiple documents with streaming output.

        This method processes documents one at a time and yields nodes
        immediately as they are created, ensuring O(1) memory consumption.

        Args:
            docs: The source documents to process.

        Yields:
            Knowledge nodes as they are generated.

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
            The target chunk size in tokens or characters (strategy-dependent).
        """
        pass

    @property
    @abstractmethod
    def overlap(self) -> int:
        """Get the overlap size between consecutive chunks.

        Returns:
            The overlap size in tokens or characters (strategy-dependent).
        """
        pass


class TokenFragmenter(BaseFragmenter):
    """Token-based text fragmentation strategy using tiktoken.

    This fragmenter splits text into chunks based on token count using the
    tiktoken library. It uses a sliding window approach with overlap, trusting
    tiktoken's tokenization to preserve semantic coherence.

    Attributes:
        chunk_size: The target size of each chunk in tokens.
        overlap: The number of tokens to overlap between chunks.
    """

    def __init__(
        self,
        chunk_size: int = 256,
        overlap: int = 64,
        tokenizer: str = "cl100k_base",
    ):
        """Initialize the TokenFragmenter.

        Args:
            chunk_size: The target size of each chunk in tokens.
            overlap: The number of tokens to overlap between chunks.
            tokenizer: The tiktoken encoding name (e.g., 'cl100k_base').

        Raises:
            ValueError: If overlap is greater than or equal to chunk_size.
        """
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")

        self._chunk_size = chunk_size
        self._overlap = overlap
        self._tokenizer_name = tokenizer
        self._encoding: Any = None

    @classmethod
    def from_config(cls, config: "FragmenterConfig") -> "TokenFragmenter":
        """Instantiate a TokenFragmenter based on the provided configuration.

        Args:
            config: The fragmentation configuration.

        Returns:
            An initialized TokenFragmenter instance.
        """
        tokenizer = config.options.get("tokenizer", "cl100k_base")
        return cls(
            chunk_size=config.chunk_size,
            overlap=config.overlap,
            tokenizer=tokenizer,
        )

    def _get_encoding(self) -> Any:
        if self._encoding is None:
            try:
                import tiktoken
            except ImportError as e:
                raise ImportError(
                    "tiktoken not installed. Install with: pip install tiktoken"
                ) from e

            try:
                self._encoding = tiktoken.get_encoding(self._tokenizer_name)
            except KeyError as e:
                from ragmark.exceptions import FragmentationError

                raise FragmentationError(
                    f"Unknown tokenizer: {self._tokenizer_name}"
                ) from e

        return self._encoding

    def fragment(self, doc: SourceDoc) -> list[KnowledgeNode]:
        """Fragment a document into token-based chunks.

        Args:
            doc: The source document to process.

        Returns:
            A list of knowledge nodes containing token-based chunks.

        Raises:
            FragmentationError: If fragmentation fails.
        """
        logger.debug("Fragmentation started: source_id=%s", doc.source_id)

        try:
            encoding = self._get_encoding()
            tokens = encoding.encode(doc.content)
            token_count = len(tokens)

            logger.debug(
                "Document tokenized: tokens=%d, chunk_size=%d, overlap=%d",
                token_count,
                self._chunk_size,
                self._overlap,
            )

            if token_count == 0:
                from ragmark.exceptions import FragmentationError

                logger.warning(
                    "Document tokenization produced no tokens: source_id=%s",
                    doc.source_id,
                )
                raise FragmentationError("Document tokenization produced no tokens")

            nodes: list[KnowledgeNode] = []
            start_token_idx = 0
            current_char_pos = 0

            def _find_end_offset(
                source_text: str, start_pos: int, target_text: str
            ) -> int:
                # Handle cases where tokenization normalizes text (e.g. NFC vs NFD),
                # causing the decoded length to differ from the source length.
                decoded_len = len(target_text)

                if source_text[start_pos : start_pos + decoded_len] == target_text:
                    return start_pos + decoded_len

                # Try a small window scan (e.g. up to +16 chars).
                search_limit = min(start_pos + decoded_len + 16, len(source_text))

                target_norm = unicodedata.normalize("NFC", target_text)

                for i in range(decoded_len - 5, search_limit - start_pos + 1):
                    candidate_end = start_pos + i
                    candidate = source_text[start_pos:candidate_end]
                    if unicodedata.normalize("NFC", candidate) == target_norm:
                        return candidate_end

                # Fallback: trust the length if match fails (should be rare)
                return min(start_pos + decoded_len, len(source_text))

            while start_token_idx < len(tokens):
                end_token_idx = min(start_token_idx + self._chunk_size, len(tokens))
                chunk_tokens = tokens[start_token_idx:end_token_idx]

                # Decode to compare tokenizer's view against source text (normalization check)
                chunk_text_decoded = encoding.decode(chunk_tokens)

                start_char = current_char_pos
                end_char = _find_end_offset(doc.content, start_char, chunk_text_decoded)

                node_content = doc.content[start_char:end_char]

                if len(node_content) != len(chunk_text_decoded):
                    logger.debug(
                        "Normalization drift detected: position=%d, source_len=%d, decoded_len=%d",
                        start_char,
                        len(node_content),
                        len(chunk_text_decoded),
                    )

                if node_content.strip():
                    source_metadata = {
                        f"source.{k}": v for k, v in doc.metadata.items()
                    }
                    node = KnowledgeNode(
                        content=node_content,
                        source_id=doc.source_id,
                        metadata={
                            **source_metadata,
                            "token_count": len(chunk_tokens),
                            "chunk_index": len(nodes),
                        },
                        position=NodePosition(
                            start_char=start_char,
                            end_char=end_char,
                            page=None,
                            section=None,
                        ),
                        dense_vector=None,
                        sparse_vector=None,
                    )

                    nodes.append(node)

                stride_token_count = self._chunk_size - self._overlap

                # We need to know exactly how many *source characters* to advance
                # to maintain synchronization.
                if end_token_idx == len(tokens):
                    break

                tokens_to_advance = chunk_tokens[:stride_token_count]
                text_to_advance = encoding.decode(tokens_to_advance)

                next_char_pos = _find_end_offset(
                    doc.content, start_char, text_to_advance
                )

                current_char_pos = next_char_pos
                start_token_idx += stride_token_count

            logger.info(
                "Fragmentation completed: source_id=%s, nodes=%d, tokens=%d",
                doc.source_id,
                len(nodes),
                token_count,
            )
            return nodes

        except Exception as e:
            from ragmark.exceptions import FragmentationError

            logger.error("Fragmentation failed: source_id=%s", doc.source_id)
            logger.debug("Fragmentation error details: %s", e, exc_info=True)
            raise FragmentationError(f"Fragmentation failed: {e}") from e

    @property
    def chunk_size(self) -> int:
        """Get the target chunk size.

        Returns:
            The target chunk size in tokens.
        """
        return self._chunk_size

    @property
    def overlap(self) -> int:
        """Get the overlap size.

        Returns:
            The overlap size in tokens.
        """
        return self._overlap
