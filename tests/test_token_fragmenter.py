"""Unit tests for TokenFragmenter including Unicode, Byte-level, and Drift Edge Cases."""

import sys
from collections.abc import Callable
from unittest.mock import patch

import pytest

from ragmark.exceptions import FragmentationError
from ragmark.forge.fragmenters import TokenFragmenter
from ragmark.schemas.documents import SourceDoc


@pytest.mark.unit
class TestTokenFragmenter:
    """Test suite for TokenFragmenter.

    Validates tokenization logic, unicode handling, drift detection
    and error handling paths.
    """

    def test_initialization_defaults(self) -> None:
        """
        Given a default TokenFragmenter initialization.
        When inspecting the properties.
        Then the chunk size should be 256 and tokenizer 'cl100k_base'.
        """
        fragmenter = TokenFragmenter()
        assert fragmenter.chunk_size == 256
        assert fragmenter._tokenizer_name == "cl100k_base"  # type: ignore

    def test_fragment_real_content_continuity(
        self, doc_factory: Callable[..., SourceDoc]
    ) -> None:
        """
        Given a standard text document.
        When fragmented with contiguous chunks (overlap=0).
        Then the reconstructed content must strictly match the original text.
        """
        content = "The quick brown fox jumps over the lazy dog." * 5
        doc = doc_factory(content=content, source_id="std")

        fragmenter = TokenFragmenter(chunk_size=10, overlap=0)
        nodes = fragmenter.fragment(doc)

        assert len(nodes) > 0
        reconstructed = "".join(n.content for n in nodes)
        assert reconstructed == doc.content

        for i in range(len(nodes) - 1):
            assert nodes[i].position.end_char == nodes[i + 1].position.start_char

    def test_fragment_batch_generator(
        self, doc_factory: Callable[..., SourceDoc]
    ) -> None:
        """
        Given multiple source documents.
        When processed via fragment_batch.
        Then it should yield nodes for all documents ensuring generator behavior.
        """
        docs = [
            doc_factory(content="Doc 1 content", source_id="d1"),
            doc_factory(content="Doc 2 content", source_id="d2"),
        ]
        fragmenter = TokenFragmenter(chunk_size=5, overlap=0)

        nodes = list(fragmenter.fragment_batch(docs))

        assert len(nodes) >= 2
        assert set(n.source_id for n in nodes) == {"d1", "d2"}

    def test_unicode_complex_graphemes(
        self, doc_factory: Callable[..., SourceDoc]
    ) -> None:
        """
        Given text containing complex Emoji sequences (ZWJ sequences).
        When fragmented with a small chunk size.
        Then the reconstruction must match exactly, preserving grapheme clusters.
        """
        text = "Hello 👨‍👩‍👧‍👦 family"
        doc = doc_factory(content=text, source_id="emoji")

        fragmenter = TokenFragmenter(chunk_size=5, overlap=0)
        nodes = fragmenter.fragment(doc)

        reconstructed = ""
        for node in nodes:
            extracted = text[node.position.start_char : node.position.end_char]
            assert extracted == node.content
            reconstructed += node.content

        assert reconstructed == text
        assert "👨‍👩‍👧‍👦" in reconstructed

    def test_normalization_drift_forced(
        self, doc_factory: Callable[..., SourceDoc]
    ) -> None:
        """
        Given a source text in NFD (decomposed) form.
        When the tokenizer 'decodes' strictly to NFC (composed), creating a length mismatch.
        Then the drift correction logic must activate to find the correct semantic end.

        Note: Real tiktoken is usually reversible. We use a Fake to FORCE the drift path.
        """
        nfd_text = "caf" + "e\u0301"
        nfc_text = "caf" + "\u00e9"

        assert len(nfd_text) == 5
        assert len(nfc_text) == 4

        class FakeAggressiveNormalizingEncoding:
            """A Fake encoding that mimics a tokenizer normalizing everything to NFC."""

            def encode(self, text: str) -> list[int]:
                return [1, 2, 3]

            def decode(self, tokens: list[int]) -> str:
                return nfc_text

        doc = doc_factory(content=nfd_text, source_id="drift-force")

        with patch(
            "tiktoken.get_encoding", return_value=FakeAggressiveNormalizingEncoding()
        ):
            fragmenter = TokenFragmenter(
                chunk_size=10, overlap=0, tokenizer="fake_encoding"
            )
            nodes = fragmenter.fragment(doc)

            assert len(nodes) == 1
            assert nodes[0].content == nfd_text
            assert len(nodes[0].content) == 5

    def test_drift_exceeds_search_window(
        self, doc_factory: Callable[..., SourceDoc]
    ) -> None:
        """
        Given a 'Zalgo' text with massive combining accents (drift > 16 chars).
        When the tokenizer merges these into fewer tokens than chars.
        Then the drift logic must robustly find the correct end offset.
        """
        torture_char = "A" + "\u0300" * 50
        text = f"Start {torture_char} End"
        doc = doc_factory(content=text, source_id="drift")

        fragmenter = TokenFragmenter(chunk_size=5, overlap=0)
        nodes = fragmenter.fragment(doc)

        reconstructed = "".join(n.content for n in nodes)
        assert reconstructed == text
        assert len(nodes) > 1

    def test_null_bytes_handling(self, doc_factory: Callable[..., SourceDoc]) -> None:
        """
        Given text containing null bytes.
        When fragmented.
        Then the null bytes must be preserved in the content and length calculations.
        """
        text = "Data\x00Preserved"
        doc = doc_factory(content=text, source_id="nulls")

        fragmenter = TokenFragmenter(chunk_size=100, overlap=0)
        nodes = fragmenter.fragment(doc)

        assert len(nodes) == 1
        assert nodes[0].content == "Data\x00Preserved"

    def test_invalid_tokenizer_config(
        self, doc_factory: Callable[..., SourceDoc]
    ) -> None:
        """
        Given a TokenFragmenter configured with a non-existent tokenizer name.
        When fragmentation is attempted.
        Then a FragmentationError should be raised, chaining the original KeyError.
        """
        doc = doc_factory(content="Test content")
        fragmenter = TokenFragmenter(tokenizer="non_existent_tokenizer_999")

        with pytest.raises(FragmentationError) as exc_info:
            fragmenter.fragment(doc)

        assert "Unknown encoding" in str(exc_info.value)

    def test_missing_tiktoken_dependency(
        self, doc_factory: Callable[..., SourceDoc]
    ) -> None:
        """
        Given an environment where tiktoken is not installed.
        When fragmentation is attempted.
        Then a FragmentationError should be raised suggesting installation.
        """
        doc = doc_factory(content="Test content")
        fragmenter = TokenFragmenter()

        with patch.dict(sys.modules, {"tiktoken": None}):
            with pytest.raises(FragmentationError) as exc_info:
                fragmenter._encoding = None  # type: ignore
                fragmenter.fragment(doc)

        assert "tiktoken not installed" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, ImportError)

    def test_empty_string_edge_case(self) -> None:
        """
        Given an empty document content (constructed bypassing validation).
        When fragmentation is attempted.
        Then a FragmentationError with a specific message should be raised.
        """
        doc = SourceDoc.model_construct(
            content="", source_id="empty", metadata={}, mime_type="text/plain"
        )

        with pytest.raises(
            FragmentationError, match="Document tokenization produced no tokens"
        ):
            TokenFragmenter().fragment(doc)

    def test_fragment_lazy_yields_progressively(
        self, doc_factory: Callable[..., SourceDoc]
    ) -> None:
        """
        Given a document that produces multiple chunks.
        When processed with fragment_lazy.
        Then it should yield nodes one at a time, not materialize all at once.
        """
        content = "The quick brown fox jumps over the lazy dog." * 20
        doc = doc_factory(content=content, source_id="progressive")

        fragmenter = TokenFragmenter(chunk_size=10, overlap=2)
        generator = fragmenter.fragment_lazy(doc)

        assert hasattr(generator, "__iter__")
        assert hasattr(generator, "__next__")

        nodes = list(generator)
        assert len(nodes) > 1

        nodes_from_fragment = fragmenter.fragment(doc)
        assert len(nodes) == len(nodes_from_fragment)

        for lazy_node, list_node in zip(nodes, nodes_from_fragment, strict=True):
            assert lazy_node.content == list_node.content
            assert lazy_node.position.start_char == list_node.position.start_char
            assert lazy_node.position.end_char == list_node.position.end_char

    def test_fragment_lazy_equivalence_with_fragment(
        self, doc_factory: Callable[..., SourceDoc]
    ) -> None:
        """
        Given a standard document.
        When processing with both fragment() and fragment_lazy().
        Then both methods must produce identical results in content, metadata, and positions.
        """
        content = "Lorem ipsum dolor sit amet, consectetur adipiscing elit." * 30
        doc = doc_factory(content=content, source_id="equiv")

        fragmenter = TokenFragmenter(chunk_size=20, overlap=5)

        lazy_nodes = list(fragmenter.fragment_lazy(doc))
        list_nodes = fragmenter.fragment(doc)

        assert len(lazy_nodes) == len(list_nodes)

        for lazy, listed in zip(lazy_nodes, list_nodes, strict=True):
            assert lazy.content == listed.content
            assert lazy.source_id == listed.source_id
            assert lazy.position.start_char == listed.position.start_char
            assert lazy.position.end_char == listed.position.end_char
            assert lazy.metadata["token_count"] == listed.metadata["token_count"]
            assert lazy.metadata["chunk_index"] == listed.metadata["chunk_index"]

    def test_fragment_lazy_handles_unicode_normalization_drift(
        self, doc_factory: Callable[..., SourceDoc]
    ) -> None:
        """
        Given text with combining characters that may cause normalization drift.
        When processed with fragment_lazy.
        Then content must be preserved exactly, matching fragment() behavior.
        """
        text = "caf" + "e\u0301" + " " + "A" + "\u0300" * 10 + " test"
        doc = doc_factory(content=text, source_id="drift")

        fragmenter = TokenFragmenter(chunk_size=5, overlap=0)

        lazy_nodes = list(fragmenter.fragment_lazy(doc))
        list_nodes = fragmenter.fragment(doc)

        assert len(lazy_nodes) == len(list_nodes)

        lazy_reconstructed = "".join(n.content for n in lazy_nodes)
        list_reconstructed = "".join(n.content for n in list_nodes)

        assert lazy_reconstructed == text
        assert list_reconstructed == text
        assert lazy_reconstructed == list_reconstructed
