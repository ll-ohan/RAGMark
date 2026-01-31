"""Pytest configuration and standardized factories for RAGMark."""

import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import fitz
import pytest
import requests

from ragmark.config.profile import EmbedderConfig, IndexConfig, RetrievalConfig
from ragmark.index.backends import MemoryIndex
from ragmark.index.base import VectorIndex
from ragmark.index.embedders import BaseEmbedder
from ragmark.schemas.documents import KnowledgeNode, NodePosition, SourceDoc
from ragmark.schemas.evaluation import CaseResult
from ragmark.schemas.generation import GenerationResult, TokenUsage
from ragmark.schemas.retrieval import RetrievedNode, SearchResult, TraceContext


@pytest.fixture
def node_factory() -> Callable[..., KnowledgeNode]:
    """Factory to create KnowledgeNode instances with default valid data.

    Returns:
        A callable that generates KnowledgeNodes.
    """

    def _make_node(
        content: str = "Standard test node content.",
        dense_vector: list[float] | None = None,
        embedding_dim: int = 384,
        source_id: str = "src-999",
    ) -> KnowledgeNode:
        return KnowledgeNode(
            node_id=f"node-{hash(content)}",
            content=content,
            source_id=source_id,
            metadata={"source.title": "Unit Test Document"},
            position=NodePosition(
                start_char=0,
                end_char=len(content),
                page=1,
                section="Introduction",
            ),
            dense_vector=dense_vector or [0.1] * embedding_dim,
            sparse_vector=None,
        )

    return _make_node


@pytest.fixture
def retrieved_node_factory(
    node_factory: Callable[..., KnowledgeNode],
) -> Callable[..., RetrievedNode]:
    """Factory to create RetrievedNode wrappers for retrieval tests.

    Returns:
        A callable that generates RetrievedNodes.
    """

    def _make_retrieved(
        content: str = "Relevant retrieved content.", score: float = 0.95, rank: int = 1
    ) -> RetrievedNode:
        return RetrievedNode(node=node_factory(content=content), score=score, rank=rank)

    return _make_retrieved


@pytest.fixture
def doc_factory() -> Callable[..., SourceDoc]:
    """Factory to create SourceDoc instances for ingestion tests."""

    def _make_doc(
        content: str = "Standard document content.", source_id: str | None = "doc-001"
    ) -> SourceDoc:
        doc_params: dict[str, Any] = {
            "content": content,
            "metadata": {"title": "Test Document"},
            "mime_type": "text/plain",
            "page_count": 1,
        }
        if source_id is not None:
            doc_params["source_id"] = source_id
        return SourceDoc(**doc_params)

    return _make_doc


@pytest.fixture
def sample_source_doc(doc_factory: Callable[..., SourceDoc]) -> SourceDoc:
    """Fixture providing a default SourceDoc."""
    return doc_factory()


@pytest.fixture
def sample_knowledge_node(node_factory: Callable[..., KnowledgeNode]) -> KnowledgeNode:
    """Fixture providing a default KnowledgeNode."""
    return node_factory()


@pytest.fixture
def sample_yaml_config() -> str:
    """Provides a sample YAML configuration as a string."""
    return """
fragmenter:
  chunk_size: 256
  overlap: 64
index:
  backend: memory
embedder:
  model_name: sentence-transformers/all-MiniLM-L6-v2
generator:
  model_path: models/test-model.gguf
evaluation:
  trial_cases_path: data/trial_cases.jsonl
"""


@pytest.fixture
def embedder_config_factory() -> Callable[..., EmbedderConfig]:
    """Factory to create EmbedderConfig instances with customizable parameters.

    Returns:
        A callable that generates EmbedderConfig objects.
    """

    def _make_config(**kwargs: Any) -> EmbedderConfig:
        defaults: dict[str, Any] = {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
            "batch_size": 32,
        }
        config_data = {**defaults, **kwargs}
        return EmbedderConfig(**config_data)

    return _make_config


class FakeEmbedder(BaseEmbedder):
    """A concrete fake embedder for testing purposes."""

    def __init__(
        self,
        dimension: int = 384,
        side_effect: Exception | None = None,
        delay: float = 0.0,
    ):
        self.dimension = dimension
        self.side_effect = side_effect
        self.delay = delay

    @property
    def embedding_dim(self) -> int:
        """Concrete implementation of the required abstract property."""
        return self.dimension

    @classmethod
    def from_config(cls, config: Any) -> "FakeEmbedder":
        """Concrete implementation of the required factory method."""
        return cls()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Concrete implementation of the required abstract method."""
        if self.delay > 0:
            time.sleep(self.delay)

        if self.side_effect:
            raise self.side_effect

        return [[0.1] * self.dimension for _ in texts]

    def embed_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        """Concrete implementation returning empty sparse vectors for testing."""
        return [{} for _ in texts]


@pytest.fixture
def fake_embedder_factory() -> Callable[..., FakeEmbedder]:
    """Factory to create configured FakeEmbedder instances."""

    def _make_embedder(
        dimension: int = 384, side_effect: Exception | None = None, delay: float = 0.0
    ) -> FakeEmbedder:
        return FakeEmbedder(dimension=dimension, side_effect=side_effect, delay=delay)

    return _make_embedder


class FakeVectorIndex(VectorIndex):
    """A concrete in-memory index for testing purposes.

    Provides 'call_history' to inspect arguments passed to search,
    removing the need for mocks/spies.
    """

    def __init__(
        self,
        dimension: int = 384,
        search_results: list[SearchResult] | None = None,
        embedder: BaseEmbedder | None = None,
    ):
        self.dimension = dimension
        self.storage: list[SearchResult] = search_results or []
        self.embedder = embedder
        self.config = None
        self._should_fail = False
        self._fail_exception = None
        self.call_history: list[dict[str, Any]] = []

    async def search(
        self,
        query_vector: list[float] | dict[str, list[int] | list[float]],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        self.call_history.append(
            {"query_vector": query_vector, "top_k": top_k, "filters": filters}
        )

        if self._should_fail and self._fail_exception:
            raise self._fail_exception

        if isinstance(query_vector, list) and len(query_vector) != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {len(query_vector)}"
            )
        return self.storage[:top_k]

    @classmethod
    def from_config(
        cls, config: Any, embedder: BaseEmbedder | None = None
    ) -> "FakeVectorIndex":
        return cls(embedder=embedder)

    async def add(self, nodes: list[KnowledgeNode]) -> None:
        pass

    async def search_hybrid(self, *_: Any, **__: Any) -> list[SearchResult]:
        return []

    async def delete(self, node_ids: list[str]) -> int:
        return 0

    async def count(self) -> int:
        return len(self.storage)

    async def clear(self) -> None:
        self.storage = []

    async def exists(self, node_id: str) -> bool:
        return False

    async def __aexit__(self, *_: Any) -> None:
        pass


@pytest.fixture
def fake_vector_index_factory(
    fake_embedder_factory: Callable[..., FakeEmbedder],
) -> Callable[..., FakeVectorIndex]:
    """Factory to create configured FakeVectorIndex instances."""

    def _make_index(
        dimension: int = 384,
        search_results: list[SearchResult] | None = None,
        embedder: BaseEmbedder | None = None,
        use_default_embedder: bool = True,
    ) -> FakeVectorIndex:
        if embedder is None and use_default_embedder:
            embedder = fake_embedder_factory(dimension=dimension)
        return FakeVectorIndex(
            dimension=dimension, search_results=search_results, embedder=embedder
        )

    return _make_index


@pytest.fixture
def retrieval_config_factory() -> Callable[..., RetrievalConfig]:
    """Factory for creating RetrievalConfig instances."""

    def _make_config(
        mode: Literal["dense", "sparse", "hybrid"] = "dense",
        top_k: int = 10,
        **kwargs: Any,
    ) -> RetrievalConfig:
        return RetrievalConfig(
            mode=mode,
            top_k=top_k,
            alpha=kwargs.get("alpha", None),
            reranker=kwargs.get("reranker", None),
        )

    return _make_config


@pytest.fixture
def index_config_factory() -> Callable[..., IndexConfig]:
    """Factory for creating IndexConfig instances."""

    def _make_config(
        backend: Literal["memory", "qdrant", "milvus", "lancedb"] = "memory",
        collection_name: str = "test_collection",
        embedding_dim: int = 384,
        **kwargs: Any,
    ) -> IndexConfig:
        return IndexConfig(
            backend=backend,
            collection_name=collection_name,
            embedding_dim=embedding_dim,
            connection=kwargs.get("connection", None),
        )

    return _make_config


@pytest.fixture
def pdf_bytes_factory() -> Callable[..., bytes]:
    """Factory to create valid PDF bytes with custom font support for Unicode/CJK/Emoji."""

    temp_dir = tempfile.gettempdir()

    def _download_font(url: str, filename: str) -> str | None:
        font_path = os.path.join(temp_dir, filename)
        if not os.path.exists(font_path):
            try:
                print(f"Downloading font: {filename}...")
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                with open(font_path, "wb") as f:
                    f.write(response.content)
            except Exception as e:
                print(f"Warning: Could not download font {filename}: {e}")
        return font_path

    def _make_pdf(
        content: str = "Default PDF Content",
        title: str = "Test Document",
        pages: int = 1,
        font_url: str = "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf",
        font_name: str = "NotoSans",
        encrypt: bool = False,
        password: str = "user123",
        toc: list[str] | None = None,
    ) -> bytes:
        filename = font_url.split("/")[-1]
        path_to_font = _download_font(font_url, filename)

        doc: Any = fitz.open()
        for i in range(pages):
            page: Any = doc.new_page()
            if i == 0 and content:
                try:
                    page.insert_text(
                        (50, 50),
                        content,
                        fontname=font_name,
                        fontfile=path_to_font if path_to_font else None,
                    )
                except Exception as e:
                    print(f"Warning: Font insertion failed: {e}")
                    page.insert_text((50, 50), content)

        doc.set_metadata({"title": title})
        if toc:
            doc.set_toc([[1, t, i + 1] for i, t in enumerate(toc)])

        save_args: dict[str, Any] = {}
        if encrypt:
            save_args = {
                "encryption": fitz.PDF_ENCRYPT_AES_256,
                "owner_pw": "owner",
                "user_pw": password,
            }

        pdf_bytes = doc.tobytes(**save_args)
        doc.close()
        return cast(bytes, pdf_bytes)

    return _make_pdf


@pytest.fixture
def source_file_factory(tmp_path: Path) -> Callable[[str, str], Path]:
    """Factory to create temporary source files with specific content.

    This fixture ensures all file I/O is confined to the temporary directory
    managed by pytest, compliant with the Anti-Mocking Strategy for I/O.

    Args:
        filename: The name of the file to create.
        content: The text content to write into the file.

    Returns:
        The Path to the created file.
    """

    def _create(filename: str, content: str = "Default content") -> Path:
        p = tmp_path / filename
        p.write_text(content, encoding="utf-8")
        return p

    return _create


@pytest.fixture
def memory_index_factory() -> Callable[..., MemoryIndex]:
    """Fixture to provide a clean MemoryIndex instance with low dimensionality for testing."""

    def _make_memory_index(
        embedding_dim: int = 3,
        embedder: Any = None,
        distance_metric: Literal["cosine", "euclidean", "dot"] = "cosine",
    ) -> MemoryIndex:
        return MemoryIndex(
            embedding_dim=embedding_dim,
            embedder=embedder,
            distance_metric=distance_metric,
        )

    return _make_memory_index


@pytest.fixture
def case_result_factory(
    trace_factory: Callable[..., TraceContext],
) -> Callable[..., CaseResult]:
    """Factory to create CaseResult instances for audit reports."""

    def _make_case_result(case_id: str = "case-001") -> CaseResult:
        return CaseResult(
            case_id=case_id,
            predicted_answer="Test generated answer",
            trace=trace_factory(),
            generation_result=GenerationResult(
                text="Test generated answer",
                usage=TokenUsage(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ),
                finish_reason="stop",
            ),
            case_metrics={"recall@5": 1.0, "latency": 0.5},
        )

    return _make_case_result


@pytest.fixture
def trace_factory(sample_knowledge_node: KnowledgeNode) -> Callable[..., TraceContext]:
    """Factory to create TraceContext instances."""

    def _make_trace(query: str = "Test query") -> TraceContext:
        return TraceContext(
            query=query,
            retrieved_nodes=[
                RetrievedNode(node=sample_knowledge_node, score=0.9, rank=1)
            ],
            reranked=False,
        )

    return _make_trace


# Test helpers for assertion patterns


def assert_abstract_class_cannot_be_instantiated(
    cls: type[Any], error_fragment: str = "abstract"
) -> None:
    """Verify that an abstract class raises TypeError when instantiated.

    This helper centralizes the type suppression needed to test abstract
    class instantiation, reducing code duplication across test files.

    Args:
        cls: The abstract class to test.
        error_fragment: Expected fragment in the error message (default: "abstract").

    Raises:
        AssertionError: If the class doesn't raise TypeError or the error
                       message doesn't contain the expected fragment.
    """
    with pytest.raises(TypeError) as exc_info:
        cls()

    error_message = str(exc_info.value).lower()
    assert (
        error_fragment in error_message
    ), f"Expected '{error_fragment}' in error message, got: {exc_info.value}"
