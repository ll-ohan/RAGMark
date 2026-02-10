"""Contract tests for all RAGMark abstract base classes."""

import inspect
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterator
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Literal, cast

import pytest
from conftest import assert_abstract_class_cannot_be_instantiated

from ragmark.config.profile import (
    EmbedderConfig,
    FragmenterConfig,
    IndexConfig,
    IngestorConfig,
    RerankerConfig,
    RetrievalConfig,
)
from ragmark.exceptions import (
    EmbeddingError,
    FragmentationError,
    GenerationError,
    IndexError,
    IngestionError,
    RetrievalError,
)
from ragmark.forge.fragmenters import BaseFragmenter
from ragmark.forge.ingestors import BaseIngestor
from ragmark.generation.drivers import BaseLLMDriver
from ragmark.index.base import VectorIndex
from ragmark.index.embedders import BaseEmbedder
from ragmark.retrieval.base import BaseRefiner, BaseRetriever
from ragmark.schemas.documents import KnowledgeNode, NodePosition, SourceDoc
from ragmark.schemas.generation import GenerationResult, TokenUsage
from ragmark.schemas.retrieval import RetrievedNode, SearchResult, TraceContext


class MinimalVectorIndex(VectorIndex):
    """Minimal implementation of VectorIndex for contract testing.

    This implementation provides the simplest possible logic to satisfy
    the abstract interface contract. It is NOT feature-complete and is
    only used to test that the interface can be implemented.
    """

    def __init__(self, should_fail: bool = False) -> None:
        self.config = None
        self.embedder = None
        self._should_fail = should_fail
        self._storage: list[SearchResult] = []

    @classmethod
    def from_config(
        cls, config: IndexConfig, embedder: BaseEmbedder | None = None
    ) -> "MinimalVectorIndex":
        instance = cls()
        instance.config = config
        instance.embedder = embedder
        return instance

    async def add(
        self, nodes: list[KnowledgeNode], monitoring: Any | None = None
    ) -> None:
        if self._should_fail:
            try:
                raise ValueError("Simulated internal error")
            except ValueError as e:
                raise IndexError("Failed to add nodes") from e
        for node in nodes:
            self._storage.append(
                SearchResult(node_id=node.node_id, score=1.0, node=node)
            )
        return

    async def search(
        self,
        query_vector: list[float] | dict[str, list[int] | list[float]],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if self._should_fail:
            try:
                raise ValueError("Simulated search error")
            except ValueError as e:
                raise IndexError("Search failed") from e
        return self._storage[:top_k]

    async def search_hybrid(
        self,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
        top_k: int,
        alpha: float,
    ) -> list[SearchResult]:
        return self._storage[:top_k]

    async def delete(self, node_ids: list[str]) -> int:
        deleted_count = 0
        self._storage = [
            sr
            for sr in self._storage
            if sr.node_id not in node_ids
            or (deleted_count := deleted_count + 1)
            and False
        ]
        return deleted_count

    async def count(self) -> int:
        return len(self._storage)

    async def clear(self) -> None:
        self._storage = []

    async def exists(self, node_id: str) -> bool:
        return any(sr.node_id == node_id for sr in self._storage)

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> None:
        pass


class MinimalRetriever(BaseRetriever):
    """Minimal implementation of BaseRetriever for contract testing."""

    def __init__(self, index: VectorIndex, should_fail: bool = False) -> None:
        self.index = index
        self._should_fail = should_fail

    @classmethod
    def from_config(
        cls,
        config: RetrievalConfig,
        index: VectorIndex,
        refiner: BaseRefiner | None = None,
    ) -> "MinimalRetriever":
        return cls(index=index)

    async def retrieve(self, query: str, top_k: int = 10) -> TraceContext:
        if self._should_fail:
            try:
                raise ValueError("Simulated retrieval error")
            except ValueError as e:
                raise RetrievalError("Retrieval failed") from e

        return TraceContext(
            query=query,
            retrieved_nodes=[],
            reranked=False,
        )


class MinimalRefiner(BaseRefiner):
    """Minimal implementation of BaseRefiner for contract testing."""

    def __init__(self, should_fail: bool = False) -> None:
        self._should_fail = should_fail

    @classmethod
    def from_config(cls: Any, config: RerankerConfig) -> "MinimalRefiner":
        return cast("MinimalRefiner", cls())

    def refine(
        self,
        query: str,
        candidates: list[RetrievedNode],
        top_k: int,
    ) -> list[RetrievedNode]:
        if self._should_fail:
            try:
                raise ValueError("Simulated reranking error")
            except ValueError as e:
                raise RetrievalError("Reranking failed") from e
        return candidates[:top_k]


class MinimalEmbedder(BaseEmbedder):
    """Minimal implementation of BaseEmbedder for contract testing."""

    def __init__(self, dimension: int = 384, should_fail: bool = False) -> None:
        self._dimension = dimension
        self._should_fail = should_fail

    @classmethod
    def from_config(cls: Any, config: EmbedderConfig) -> "MinimalEmbedder":
        return cast("MinimalEmbedder", cls(dimension=384))

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._should_fail:
            try:
                raise ValueError("Simulated embedding error")
            except ValueError as e:
                raise EmbeddingError("Embedding computation failed") from e
        return [[0.1] * self._dimension for _ in texts]

    def embed_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        """Minimal implementation returning empty sparse vectors."""
        return [{} for _ in texts]

    @property
    def embedding_dim(self) -> int:
        return self._dimension


class MinimalFragmenter(BaseFragmenter):
    """Minimal implementation of BaseFragmenter for contract testing."""

    def __init__(
        self, chunk_size: int = 256, overlap: int = 64, should_fail: bool = False
    ) -> None:
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._should_fail = should_fail

    @classmethod
    def from_config(cls: Any, config: FragmenterConfig) -> "MinimalFragmenter":
        return cast(
            "MinimalFragmenter",
            cls(chunk_size=config.chunk_size, overlap=config.overlap),
        )

    def fragment_lazy(self, doc: SourceDoc) -> Iterator[KnowledgeNode]:
        """Yields a single KnowledgeNode from the source document.

        This minimal implementation creates exactly one node containing the
        entire document content for contract testing purposes.

        Args:
            doc: Source document to fragment.

        Yields:
            A single KnowledgeNode with the document's full content.

        Raises:
            FragmentationError: If should_fail is True during instantiation.
        """
        if self._should_fail:
            try:
                raise ValueError("Simulated fragmentation error")
            except ValueError as e:
                raise FragmentationError("Fragmentation failed") from e

        yield KnowledgeNode(
            content=doc.content,
            source_id=doc.source_id,
            metadata=doc.metadata,
            position=NodePosition(
                start_char=0,
                end_char=len(doc.content),
                page=None,
                section=None,
            ),
            dense_vector=None,
            sparse_vector=None,
        )

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def overlap(self) -> int:
        return self._overlap


class MinimalIngestor(BaseIngestor):
    """Minimal implementation of BaseIngestor for contract testing."""

    def __init__(self, should_fail: bool = False) -> None:
        self._should_fail = should_fail

    @classmethod
    def from_config(cls: Any, config: IngestorConfig) -> "MinimalIngestor":
        return cast("MinimalIngestor", cls())

    def ingest(self, source: Path | bytes | BinaryIO) -> SourceDoc:
        if self._should_fail:
            try:
                raise ValueError("Simulated ingestion error")
            except ValueError as e:
                raise IngestionError("Ingestion failed") from e

        if isinstance(source, Path):
            content = source.read_text()
        elif isinstance(source, bytes):
            content = source.decode("utf-8")
        elif isinstance(source, BytesIO):
            content = source.read().decode("utf-8")
        else:
            content = "Default content"

        return SourceDoc(
            content=content,
            metadata={"title": "Test Document"},
            mime_type="text/plain",
            page_count=1,
        )

    @property
    def supported_formats(self) -> set[str]:
        return {".txt", ".md"}


class MinimalLLMDriver(BaseLLMDriver):
    """Minimal implementation of BaseLLMDriver for contract testing."""

    def __init__(self, context_window: int = 4096, should_fail: bool = False) -> None:
        self._context_window = context_window
        self._should_fail = should_fail

    async def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.7,
        stop: list[str] | None = None,
        response_format: dict[
            str, dict[Literal["json_schema"], Any] | Literal["json_object"]
        ]
        | None = None,
    ) -> GenerationResult:
        if self._should_fail:
            try:
                raise ValueError("Simulated generation error")
            except ValueError as e:
                raise GenerationError("Generation failed") from e

        return GenerationResult(
            text="Generated response",
            usage=TokenUsage(
                prompt_tokens=len(prompt.split()),
                completion_tokens=2,
                total_tokens=len(prompt.split()) + 2,
            ),
            finish_reason="stop",
        )

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str]:
        async def _generate() -> AsyncGenerator[str, None]:
            if self._should_fail:
                try:
                    raise ValueError("Simulated streaming error")
                except ValueError as e:
                    raise GenerationError("Streaming generation failed") from e

            for chunk in ["Generated", " ", "response"]:
                yield chunk

        return _generate()

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    @property
    def context_window(self) -> int:
        return self._context_window

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> None:
        pass


@pytest.mark.unit
class TestVectorIndexContract:
    """Contract tests for the VectorIndex abstract base class."""

    def test_instantiating_vector_index_directly_should_raise_type_error(self) -> None:
        """Verifies that the abstract class cannot be instantiated.

        Given:
            The VectorIndex abstract base class.
        When:
            Attempting to instantiate it directly.
        Then:
            A TypeError is raised, preventing instantiation.
        """
        assert_abstract_class_cannot_be_instantiated(VectorIndex)

    def test_minimal_vector_index_should_be_instantiable_and_implement_contract(
        self,
    ) -> None:
        """Verifies that a minimal concrete implementation is valid.

        Given:
            A class `MinimalVectorIndex` that inherits from VectorIndex.
        When:
            The class implements all abstract methods.
        Then:
            It can be instantiated successfully.
        """
        index = MinimalVectorIndex()

        assert isinstance(index, VectorIndex)
        assert callable(index.add)
        assert callable(index.search)
        assert callable(index.search_hybrid)
        assert callable(index.delete)
        assert callable(index.count)
        assert callable(index.clear)
        assert callable(index.exists)

    @pytest.mark.asyncio
    async def test_vector_index_should_support_async_context_manager_protocol(
        self,
    ) -> None:
        """Verifies the class supports the async context manager protocol.

        Given:
            A concrete `MinimalVectorIndex` instance.
        When:
            Used in an `async with` statement.
        Then:
            The `__aenter__` method returns self and `__aexit__` is called.
        """
        index = MinimalVectorIndex()

        async with index as ctx_index:
            assert ctx_index is index

    def test_from_config_should_create_instance_with_config_and_embedder(
        self, index_config_factory: Any, fake_embedder_factory: Any
    ) -> None:
        """Verifies that from_config correctly populates the instance.

        Given:
            An IndexConfig and a fake embedder.
        When:
            Calling the `from_config` classmethod.
        Then:
            A new instance is created with the provided config and embedder.
        """
        config = index_config_factory(backend="memory", embedding_dim=384)
        embedder = fake_embedder_factory(dimension=384)

        index = MinimalVectorIndex.from_config(config, embedder)

        assert isinstance(index, VectorIndex)
        assert index.config is config
        assert index.embedder is embedder

    @pytest.mark.asyncio
    async def test_add_should_be_awaitable_and_return_none(
        self, node_factory: Any
    ) -> None:
        """Verifies the `add` method is an awaitable async method.

        Given:
            A `MinimalVectorIndex` instance and a `KnowledgeNode`.
        When:
            Calling the `add` method.
        Then:
            The method is awaitable and returns None.
        """
        index = MinimalVectorIndex()
        node = node_factory(content="Test content")

        await index.add([node])

        assert True

    @pytest.mark.asyncio
    async def test_search_should_accept_dense_and_sparse_vectors(
        self, node_factory: Any
    ) -> None:
        """Verifies `search` handles both dense and sparse vector queries.

        Given:
            A `MinimalVectorIndex` containing a node.
        When:
            Calling `search` with a dense vector (`list[float]`).
            And calling `search` with a sparse vector (`dict`).
        Then:
            Both calls succeed and return a list of `SearchResult`.
        """
        index = MinimalVectorIndex()
        node = node_factory(content="Test", dense_vector=[1.0, 0.5, 0.3])
        await index.add([node])

        results_dense = await index.search(query_vector=[1.0, 0.5, 0.3], top_k=5)
        assert isinstance(results_dense, list)
        assert all(isinstance(r, SearchResult) for r in results_dense)

        results_sparse = await index.search(
            query_vector={"indices": [1, 5, 10], "values": [0.9, 0.7, 0.5]}, top_k=5
        )
        assert isinstance(results_sparse, list)

    @pytest.mark.asyncio
    async def test_search_hybrid_should_return_search_result_list(self) -> None:
        """Verifies the signature and return type of `search_hybrid`.

        Given:
            A `MinimalVectorIndex` instance.
        When:
            Calling `search_hybrid` with dense/sparse vectors and an alpha.
        Then:
            The method returns a list of `SearchResult` instances.
        """
        index = MinimalVectorIndex()

        results = await index.search_hybrid(
            dense_vector=[1.0, 0.5], sparse_vector={1: 0.8, 2: 0.6}, top_k=5, alpha=0.7
        )

        assert isinstance(results, list)
        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_delete_should_return_integer_count_of_deleted_nodes(
        self, node_factory: Any
    ) -> None:
        """Verifies that `delete` returns the number of deleted nodes.

        Given:
            A `MinimalVectorIndex` with two nodes.
        When:
            Calling `delete` with the ID of one node.
        Then:
            The method returns an integer count of the deleted nodes.
        """
        index = MinimalVectorIndex()
        node1 = node_factory(content="Node 1")
        node2 = node_factory(content="Node 2")
        node1.node_id = "node-1"
        node2.node_id = "node-2"
        await index.add([node1, node2])

        deleted_count = await index.delete(["node-1"])

        assert isinstance(deleted_count, int)
        assert deleted_count >= 0

    @pytest.mark.asyncio
    async def test_count_should_return_integer_total_of_indexed_nodes(
        self, node_factory: Any
    ) -> None:
        """Verifies that `count` returns the total number of nodes.

        Given:
            A `MinimalVectorIndex` with three nodes.
        When:
            Calling the `count` method.
        Then:
            The method returns an integer representing the total count.
        """
        index = MinimalVectorIndex()
        nodes = [node_factory(content=f"Node {i}") for i in range(3)]
        await index.add(nodes)

        total = await index.count()

        assert isinstance(total, int)
        assert total >= 0

    @pytest.mark.asyncio
    async def test_exists_should_return_boolean_indicating_node_presence(
        self, node_factory: Any
    ) -> None:
        """Verifies that `exists` correctly checks for a node's presence.

        Given:
            A `MinimalVectorIndex` with a specific node.
        When:
            Calling `exists` with the ID of the existing node.
            And calling `exists` with a non-existent ID.
        Then:
            The method returns `True` for the existing node and `False` otherwise.
        """
        index = MinimalVectorIndex()
        node = node_factory(content="Test node")
        node.node_id = "test-node-id"
        await index.add([node])

        exists = await index.exists("test-node-id")
        not_exists = await index.exists("non-existent-id")

        assert isinstance(exists, bool)
        assert isinstance(not_exists, bool)

    def test_incomplete_implementation_should_raise_type_error_on_instantiation(
        self,
    ) -> None:
        """Verifies that incomplete implementations are rejected.

        Given:
            A class that inherits `VectorIndex` but omits an abstract method.
        When:
            Attempting to instantiate it.
        Then:
            A `TypeError` is raised.
        """

        class IncompleteVectorIndex(VectorIndex):
            pass

        assert_abstract_class_cannot_be_instantiated(IncompleteVectorIndex)

    @pytest.mark.asyncio
    async def test_failed_operation_should_raise_index_error_with_cause(self) -> None:
        """Verifies that internal errors are wrapped in `IndexError`.

        Given:
            A `MinimalVectorIndex` configured to simulate internal failures.
        When:
            An operation like `add` is called.
        Then:
            An `IndexError` is raised, and its `__cause__` attribute
            references the original `ValueError`.
        """
        index = MinimalVectorIndex(should_fail=True)

        with pytest.raises(IndexError) as exc_info:
            await index.add([])

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)


@pytest.mark.unit
class TestBaseRetrieverContract:
    """Contract tests for the BaseRetriever abstract base class."""

    def test_instantiating_base_retriever_directly_should_raise_type_error(
        self,
    ) -> None:
        """Verifies that the abstract class cannot be instantiated.

        Given:
            The BaseRetriever abstract base class.
        When:
            Attempting to instantiate it directly.
        Then:
            A TypeError is raised.
        """
        assert_abstract_class_cannot_be_instantiated(BaseRetriever)

    def test_minimal_retriever_should_be_instantiable_and_implement_contract(
        self,
    ) -> None:
        """Verifies that a minimal concrete implementation is valid.

        Given:
            A `MinimalRetriever` class that implements all abstract methods.
        When:
            Instantiating the class.
        Then:
            The instance is created successfully and methods are callable.
        """
        index = MinimalVectorIndex()
        retriever = MinimalRetriever(index=index)

        assert isinstance(retriever, BaseRetriever)
        assert callable(retriever.retrieve)
        assert callable(retriever.retrieve_batch)

    def test_from_config_should_create_instance_with_index_and_refiner(
        self, retrieval_config_factory: Callable[..., RetrievalConfig]
    ) -> None:
        """Verifies `from_config` creates an instance with dependencies.

        Given:
            A `RetrievalConfig`, a `VectorIndex`, and an optional `BaseRefiner`.
        When:
            Calling the `from_config` classmethod.
        Then:
            A retriever instance is created successfully.
        """
        config = retrieval_config_factory(mode="dense", top_k=10)
        index = MinimalVectorIndex()
        refiner = MinimalRefiner()

        retriever = MinimalRetriever.from_config(config, index, refiner)

        assert isinstance(retriever, BaseRetriever)

    @pytest.mark.asyncio
    async def test_retrieve_should_be_awaitable_and_return_trace_context(self) -> None:
        """Verifies the signature and return type of the `retrieve` method.

        Given:
            A `MinimalRetriever` instance.
        When:
            Calling the `retrieve` method.
        Then:
            The method is awaitable and returns a `TraceContext` instance.
        """
        index = MinimalVectorIndex()
        retriever = MinimalRetriever(index=index)

        trace = await retriever.retrieve("test query", top_k=5)

        assert isinstance(trace, TraceContext)
        assert trace.query == "test query"

    @pytest.mark.asyncio
    async def test_retrieve_batch_should_have_default_implementation_and_return_traces(
        self,
    ) -> None:
        """Verifies the default `retrieve_batch` implementation works correctly.

        Given:
            A `MinimalRetriever` instance.
        When:
            Calling `retrieve_batch` with a list of queries.
        Then:
            The method returns a list of `TraceContext` objects.
        """
        index = MinimalVectorIndex()
        retriever = MinimalRetriever(index=index)

        queries = ["query 1", "query 2", "query 3"]
        traces = await retriever.retrieve_batch(queries, top_k=5)

        assert isinstance(traces, list)
        assert len(traces) == 3
        assert all(isinstance(t, TraceContext) for t in traces)

    @pytest.mark.asyncio
    async def test_retrieve_batch_should_process_queries_concurrently(self) -> None:
        """Verifies the default `retrieve_batch` runs concurrently.

        Given:
            A `MinimalRetriever` with the default `retrieve_batch` implementation.
        When:
            `retrieve_batch` is called.
        Then:
            The implementation uses `asyncio.gather` to run `retrieve` calls
            concurrently.
        """
        index = MinimalVectorIndex()
        retriever = MinimalRetriever(index=index)

        queries = ["query 1", "query 2"]
        traces = await retriever.retrieve_batch(queries)

        assert len(traces) == len(queries)
        for i, trace in enumerate(traces):
            assert trace.query == queries[i]

    def test_incomplete_retriever_implementation_should_raise_type_error(self) -> None:
        """Verifies that incomplete implementations are rejected.

        Given:
            A class that inherits `BaseRetriever` but omits an abstract method.
        When:
            Attempting to instantiate it.
        Then:
            A `TypeError` is raised.
        """

        class IncompleteRetriever(BaseRetriever):
            pass

        assert_abstract_class_cannot_be_instantiated(IncompleteRetriever)

    @pytest.mark.asyncio
    async def test_failed_retrieve_should_raise_retrieval_error_with_cause(
        self,
    ) -> None:
        """Verifies that internal errors are wrapped in `RetrievalError`.

        Given:
            A `MinimalRetriever` configured to simulate internal failures.
        When:
            The `retrieve` method is called.
        Then:
            A `RetrievalError` is raised with its `__cause__` attribute
            referencing the original `ValueError`.
        """
        index = MinimalVectorIndex()
        retriever = MinimalRetriever(index=index, should_fail=True)

        with pytest.raises(RetrievalError) as exc_info:
            await retriever.retrieve("test query")

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)


@pytest.mark.unit
class TestBaseRefinerContract:
    """Contract tests for the BaseRefiner abstract base class."""

    def test_instantiating_base_refiner_directly_should_raise_type_error(self) -> None:
        """Verifies that the abstract class cannot be instantiated.

        Given:
            The BaseRefiner abstract base class.
        When:
            Attempting to instantiate it directly.
        Then:
            A TypeError is raised.
        """
        assert_abstract_class_cannot_be_instantiated(BaseRefiner)

    def test_minimal_refiner_should_be_instantiable_and_implement_contract(
        self,
    ) -> None:
        """Verifies that a minimal concrete implementation is valid.

        Given:
            A `MinimalRefiner` class that implements all abstract methods.
        When:
            Instantiating the class.
        Then:
            The instance is created successfully and `refine` is callable.
        """
        refiner = MinimalRefiner()

        assert isinstance(refiner, BaseRefiner)
        assert callable(refiner.refine)

    def test_refine_should_be_synchronous_and_return_list(
        self, retrieved_node_factory: Any
    ) -> None:
        """Verifies the `refine` method is synchronous.

        Given:
            A `MinimalRefiner` instance and a list of candidates.
        When:
            Calling the `refine` method.
        Then:
            The method is synchronous (not awaitable) and returns a list.
        """
        refiner = MinimalRefiner()
        candidates = [retrieved_node_factory(score=0.9, rank=i + 1) for i in range(5)]

        result = refiner.refine("query", candidates, top_k=3)

        assert not inspect.iscoroutine(result)
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_refine_should_accept_query_candidates_and_top_k(
        self, retrieved_node_factory: Callable[..., RetrievedNode]
    ) -> None:
        """Verifies the signature and return type of the `refine` method.

        Given:
            A `MinimalRefiner` and a list of candidate nodes.
        When:
            Calling `refine` with a query, candidates, and top_k.
        Then:
            The method accepts all parameters and returns a list of `RetrievedNode`.
        """
        refiner = MinimalRefiner()
        candidates = [retrieved_node_factory(score=0.8, rank=i + 1) for i in range(10)]

        reranked = refiner.refine("test query", candidates, top_k=5)

        assert isinstance(reranked, list)
        assert all(isinstance(n, RetrievedNode) for n in reranked)
        assert len(reranked) <= 5

    def test_from_config_should_create_instance_from_reranker_config(self) -> None:
        """Verifies `from_config` creates an instance from a `RerankerConfig`.

        Given:
            A `RerankerConfig` instance.
        When:
            Calling the `from_config` classmethod.
        Then:
            A `MinimalRefiner` instance is created.
        """
        from ragmark.config.profile import RerankerConfig

        config = RerankerConfig(model_name="test-reranker")
        refiner = MinimalRefiner.from_config(config)

        assert isinstance(refiner, BaseRefiner)

    def test_incomplete_refiner_implementation_should_raise_type_error(self) -> None:
        """Verifies that incomplete implementations are rejected.

        Given:
            A class that inherits `BaseRefiner` but omits an abstract method.
        When:
            Attempting to instantiate it.
        Then:
            A `TypeError` is raised.
        """

        class IncompleteRefiner(BaseRefiner):
            pass

        assert_abstract_class_cannot_be_instantiated(IncompleteRefiner)

    def test_failed_refine_should_raise_retrieval_error_with_cause(
        self, retrieved_node_factory: Any
    ) -> None:
        """Verifies that internal errors are wrapped in `RetrievalError`.

        Given:
            A `MinimalRefiner` configured to simulate internal failures.
        When:
            The `refine` method is called.
        Then:
            A `RetrievalError` is raised with its `__cause__` referencing the original error.
        """
        refiner = MinimalRefiner(should_fail=True)
        candidates = [retrieved_node_factory()]

        with pytest.raises(RetrievalError) as exc_info:
            refiner.refine("query", candidates, top_k=1)

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)


@pytest.mark.unit
class TestBaseEmbedderContract:
    """Contract tests for the BaseEmbedder abstract base class."""

    def test_instantiating_base_embedder_directly_should_raise_type_error(self) -> None:
        """Verifies that the abstract class cannot be instantiated.

        Given:
            The BaseEmbedder abstract base class.
        When:
            Attempting to instantiate it directly.
        Then:
            A TypeError is raised.
        """
        assert_abstract_class_cannot_be_instantiated(BaseEmbedder)

    def test_minimal_embedder_should_be_instantiable_and_implement_contract(
        self,
    ) -> None:
        """Verifies that a minimal concrete implementation is valid.

        Given:
            A `MinimalEmbedder` class that implements all abstract members.
        When:
            Instantiating the class.
        Then:
            The instance is created successfully with required members.
        """
        embedder = MinimalEmbedder(dimension=384)

        assert isinstance(embedder, BaseEmbedder)
        assert callable(embedder.embed)
        assert hasattr(embedder, "embedding_dim")

    def test_embed_should_be_synchronous(self) -> None:
        """Verifies the `embed` method is synchronous.

        Given:
            A `MinimalEmbedder` instance.
        When:
            Calling the `embed` method.
        Then:
            The method is synchronous (not awaitable) and returns a list.
        """
        embedder = MinimalEmbedder(dimension=128)

        result = embedder.embed(["text 1", "text 2"])

        assert not inspect.iscoroutine(result)
        assert isinstance(result, list)

    def test_embed_should_return_list_of_vectors(self) -> None:
        """Verifies the signature and return type of the `embed` method.

        Given:
            A `MinimalEmbedder` instance.
        When:
            Calling `embed` with a list of texts.
        Then:
            It returns a list of embedding vectors (list of lists of floats).
        """
        embedder = MinimalEmbedder(dimension=256)

        embeddings = embedder.embed(["hello", "world"])

        assert isinstance(embeddings, list)
        assert len(embeddings) == 2
        assert all(isinstance(vec, list) for vec in embeddings)
        assert all(len(vec) == 256 for vec in embeddings)

    def test_embed_sparse_should_have_default_implementation(self) -> None:
        """Verifies the default `embed_sparse` implementation works correctly.

        Given:
            A `MinimalEmbedder` instance using the default `embed_sparse`.
        When:
            Calling `embed_sparse`.
        Then:
            It returns a list of empty dictionaries by default.
        """
        embedder = MinimalEmbedder()

        sparse_embeddings = embedder.embed_sparse(["text 1", "text 2"])

        assert isinstance(sparse_embeddings, list)
        assert len(sparse_embeddings) == 2
        assert all(isinstance(sv, dict) for sv in sparse_embeddings)
        assert all(sv == {} for sv in sparse_embeddings)

    def test_supports_sparse_should_default_to_false(self) -> None:
        """Verifies the `supports_sparse` property defaults to False.

        Given:
            A `MinimalEmbedder` with the default `supports_sparse` property.
        When:
            Accessing the `supports_sparse` property.
        Then:
            It returns `False`.
        """
        embedder = MinimalEmbedder()

        assert embedder.supports_sparse is False

    def test_embedding_dim_should_be_a_property(self) -> None:
        """Verifies that `embedding_dim` is a property, not a method.

        Given:
            A `MinimalEmbedder` instance.
        When:
            Accessing `embedding_dim`.
        Then:
            It returns an integer without needing to be called.
        """
        embedder = MinimalEmbedder(dimension=512)

        dim = embedder.embedding_dim

        assert isinstance(dim, int)
        assert dim == 512

    def test_from_config_should_create_instance_from_embedder_config(
        self, embedder_config_factory: Callable[..., EmbedderConfig]
    ) -> None:
        """Verifies `from_config` creates an instance from an `EmbedderConfig`.

        Given:
            An `EmbedderConfig` instance.
        When:
            Calling the `from_config` classmethod.
        Then:
            An embedder instance is created.
        """
        config = embedder_config_factory(model_name="test-model", device="cpu")

        embedder = MinimalEmbedder.from_config(config)

        assert isinstance(embedder, BaseEmbedder)

    def test_incomplete_embedder_implementation_should_raise_type_error(self) -> None:
        """Verifies that incomplete implementations are rejected.

        Given:
            A class that inherits `BaseEmbedder` but omits an abstract member.
        When:
            Attempting to instantiate it.
        Then:
            A `TypeError` is raised.
        """

        class IncompleteEmbedder(BaseEmbedder):
            pass

        assert_abstract_class_cannot_be_instantiated(IncompleteEmbedder)

    def test_failed_embed_should_raise_embedding_error_with_cause(self) -> None:
        """Verifies that internal errors are wrapped in `EmbeddingError`.

        Given:
            A `MinimalEmbedder` configured to simulate internal failures.
        When:
            The `embed` method is called.
        Then:
            An `EmbeddingError` is raised with its `__cause__` referencing the original error.
        """
        embedder = MinimalEmbedder(should_fail=True)

        with pytest.raises(EmbeddingError) as exc_info:
            embedder.embed(["text"])

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)


@pytest.mark.unit
class TestBaseFragmenterContract:
    """Contract tests for the BaseFragmenter abstract base class."""

    def test_instantiating_base_fragmenter_directly_should_raise_type_error(
        self,
    ) -> None:
        """Verifies that the abstract class cannot be instantiated.

        Given:
            The BaseFragmenter abstract base class.
        When:
            Attempting to instantiate it directly.
        Then:
            A TypeError is raised.
        """
        assert_abstract_class_cannot_be_instantiated(BaseFragmenter)

    def test_minimal_fragmenter_should_be_instantiable_and_implement_contract(
        self,
    ) -> None:
        """Verifies that a minimal concrete implementation is valid.

        Given:
            A `MinimalFragmenter` class that implements all abstract members.
        When:
            Instantiating the class.
        Then:
            The instance is created successfully with required members.
        """
        fragmenter = MinimalFragmenter()

        assert isinstance(fragmenter, BaseFragmenter)
        assert callable(fragmenter.fragment)
        assert hasattr(fragmenter, "chunk_size")
        assert hasattr(fragmenter, "overlap")

    def test_fragment_should_return_list_of_knowledge_nodes(
        self, doc_factory: Callable[..., SourceDoc]
    ) -> None:
        """Verifies the signature and return type of the `fragment` method.

        Given:
            A `MinimalFragmenter` instance and a `SourceDoc`.
        When:
            Calling the `fragment` method.
        Then:
            It returns a list of `KnowledgeNode` instances.
        """
        fragmenter = MinimalFragmenter()
        doc = doc_factory(content="Test document content")

        nodes = fragmenter.fragment(doc)

        assert isinstance(nodes, list)
        assert all(isinstance(node, KnowledgeNode) for node in nodes)

    def test_fragment_lazy_should_yield_nodes_as_generator(
        self, doc_factory: Callable[..., SourceDoc]
    ) -> None:
        """Verifies that fragment_lazy returns a lazy iterator yielding nodes.

        Given:
            A MinimalFragmenter and a SourceDoc with specific content.
        When:
            Calling fragment_lazy.
        Then:
            It returns an Iterator that yields exactly one KnowledgeNode with
            the document's content, source_id, metadata, and position preserved.
        """
        fragmenter = MinimalFragmenter()
        doc = doc_factory(content="Test document content")

        result = fragmenter.fragment_lazy(doc)

        assert isinstance(result, Iterator)
        nodes = list(result)
        assert len(nodes) == 1
        assert nodes[0].content == "Test document content"
        assert nodes[0].source_id == doc.source_id
        assert all(
            k in nodes[0].metadata and nodes[0].metadata[k] == v
            for k, v in doc.metadata.items()
        )
        assert nodes[0].position.start_char == 0
        assert nodes[0].position.end_char == len(doc.content)

    def test_fragment_batch_should_yield_nodes_as_generator(
        self, doc_factory: Callable[..., SourceDoc]
    ) -> None:
        """Verifies that `fragment_batch` returns a generator.

        Given:
            A `MinimalFragmenter` and multiple `SourceDoc` instances.
        When:
            Calling `fragment_batch`.
        Then:
            It returns an `Iterator` that yields `KnowledgeNode` instances.
        """
        fragmenter = MinimalFragmenter()
        docs = [doc_factory(content=f"Doc {i}") for i in range(3)]

        result = fragmenter.fragment_batch(docs)

        assert isinstance(result, Iterator)
        nodes = list(result)
        assert all(isinstance(node, KnowledgeNode) for node in nodes)

    def test_fragment_batch_should_have_default_implementation(
        self, doc_factory: Callable[..., SourceDoc]
    ) -> None:
        """Verifies the default `fragment_batch` implementation works.

        Given:
            A `MinimalFragmenter` using the default `fragment_batch` method.
        When:
            Calling `fragment_batch` with a list of documents.
        Then:
            It yields nodes by calling `fragment` for each document.
        """
        fragmenter = MinimalFragmenter()
        docs = [doc_factory(content=f"Document {i} content") for i in range(2)]

        nodes = list(fragmenter.fragment_batch(docs))

        assert len(nodes) == 2

    def test_chunk_size_should_be_a_property(self) -> None:
        """Verifies that `chunk_size` is a property, not a method.

        Given:
            A `MinimalFragmenter` instance.
        When:
            Accessing the `chunk_size` attribute.
        Then:
            It returns an integer without needing to be called.
        """
        fragmenter = MinimalFragmenter(chunk_size=512)

        chunk_size = fragmenter.chunk_size

        assert isinstance(chunk_size, int)
        assert chunk_size == 512

    def test_overlap_should_be_a_property(self) -> None:
        """Verifies that `overlap` is a property, not a method.

        Given:
            A `MinimalFragmenter` instance.
        When:
            Accessing the `overlap` attribute.
        Then:
            It returns an integer without needing to be called.
        """
        fragmenter = MinimalFragmenter(overlap=128)

        overlap = fragmenter.overlap

        assert isinstance(overlap, int)
        assert overlap == 128

    def test_from_config_should_create_instance_from_fragmenter_config(self) -> None:
        """Verifies `from_config` creates an instance from a `FragmenterConfig`.

        Given:
            A `FragmenterConfig` instance.
        When:
            Calling the `from_config` classmethod.
        Then:
            A fragmenter instance is created with the correct properties.
        """
        config = FragmenterConfig(chunk_size=256, overlap=64)

        fragmenter = MinimalFragmenter.from_config(config)

        assert isinstance(fragmenter, BaseFragmenter)
        assert fragmenter.chunk_size == 256
        assert fragmenter.overlap == 64

    def test_incomplete_fragmenter_implementation_should_raise_type_error(self) -> None:
        """Verifies that incomplete implementations are rejected.

        Given:
            A class that inherits `BaseFragmenter` but omits an abstract member.
        When:
            Attempting to instantiate it.
        Then:
            A `TypeError` is raised.
        """

        class IncompleteFragmenter(BaseFragmenter):
            pass

        assert_abstract_class_cannot_be_instantiated(IncompleteFragmenter)

    def test_failed_fragment_should_raise_fragmentation_error_with_cause(
        self, doc_factory: Callable[..., SourceDoc]
    ) -> None:
        """Verifies that internal errors are wrapped in `FragmentationError`.

        Given:
            A `MinimalFragmenter` configured to simulate internal failures.
        When:
            The `fragment` method is called.
        Then:
            A `FragmentationError` is raised with its `__cause__` referencing the original error.
        """
        fragmenter = MinimalFragmenter(should_fail=True)
        doc = doc_factory(content="Test")

        with pytest.raises(FragmentationError) as exc_info:
            fragmenter.fragment(doc)

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)


@pytest.mark.unit
class TestBaseIngestorContract:
    """Contract tests for the BaseIngestor abstract base class."""

    def test_instantiating_base_ingestor_directly_should_raise_type_error(self) -> None:
        """Verifies that the abstract class cannot be instantiated.

        Given:
            The BaseIngestor abstract base class.
        When:
            Attempting to instantiate it directly.
        Then:
            A TypeError is raised.
        """
        assert_abstract_class_cannot_be_instantiated(BaseIngestor)

    def test_minimal_ingestor_should_be_instantiable_and_implement_contract(
        self,
    ) -> None:
        """Verifies that a minimal concrete implementation is valid.

        Given:
            A `MinimalIngestor` class that implements all abstract members.
        When:
            Instantiating the class.
        Then:
            The instance is created successfully with required members.
        """
        ingestor = MinimalIngestor()

        assert isinstance(ingestor, BaseIngestor)
        assert callable(ingestor.ingest)
        assert hasattr(ingestor, "supported_formats")

    def test_ingest_should_accept_path_bytes_and_binaryio(self, tmp_path: Any) -> None:
        """Verifies `ingest` handles Path, bytes, and BytesIO sources.

        Given:
            A `MinimalIngestor` instance.
        When:
            Calling `ingest` with a `Path` object.
            And calling `ingest` with a `bytes` object.
            And calling `ingest` with a `BytesIO` object.
        Then:
            All calls succeed and return a `SourceDoc` instance.
        """
        ingestor = MinimalIngestor()

        file_path = tmp_path / "test.txt"
        file_path.write_text("Test content from file")
        doc_from_path = ingestor.ingest(file_path)
        assert isinstance(doc_from_path, SourceDoc)

        doc_from_bytes = ingestor.ingest(b"Test content from bytes")
        assert isinstance(doc_from_bytes, SourceDoc)

        doc_from_io = ingestor.ingest(BytesIO(b"Test content from BytesIO"))
        assert isinstance(doc_from_io, SourceDoc)

    def test_ingest_should_return_source_doc(self, tmp_path: Any) -> None:
        """Verifies the return type of the `ingest` method.

        Given:
            A `MinimalIngestor` and a source file.
        When:
            Calling the `ingest` method.
        Then:
            It returns a `SourceDoc` instance.
        """
        ingestor = MinimalIngestor()
        file_path = tmp_path / "document.txt"
        file_path.write_text("Document content")

        doc = ingestor.ingest(file_path)

        assert isinstance(doc, SourceDoc)
        assert doc.content is not None

    def test_ingest_batch_should_yield_source_docs(self, tmp_path: Any) -> None:
        """Verifies that `ingest_batch` returns a generator of `SourceDoc`.

        Given:
            A `MinimalIngestor` and multiple source files.
        When:
            Calling `ingest_batch`.
        Then:
            It returns an `Iterator` that yields `SourceDoc` instances.
        """
        ingestor = MinimalIngestor()

        files: list[Path] = []
        for i in range(3):
            file_path = tmp_path / f"doc_{i}.txt"
            file_path.write_text(f"Content {i}")
            files.append(file_path)

        result = ingestor.ingest_batch(files)

        assert isinstance(result, Iterator)
        docs = list(result)
        assert len(docs) == 3
        assert all(isinstance(doc, SourceDoc) for doc in docs)

    def test_ingest_batch_should_have_default_implementation(
        self, tmp_path: Any
    ) -> None:
        """Verifies the default `ingest_batch` implementation works.

        Given:
            A `MinimalIngestor` using the default `ingest_batch` method.
        When:
            Calling `ingest_batch` with a list of files.
        Then:
            It yields `SourceDoc`s by calling `ingest` for each file.
        """
        ingestor = MinimalIngestor()

        files: list[Path] = []
        for i in range(2):
            file_path = tmp_path / f"file_{i}.txt"
            file_path.write_text(f"File {i}")
            files.append(file_path)

        docs = list(ingestor.ingest_batch(files))

        assert len(docs) == 2

    def test_supported_formats_should_be_a_property_returning_set(self) -> None:
        """Verifies `supported_formats` is a property returning a set.

        Given:
            A `MinimalIngestor` instance.
        When:
            Accessing the `supported_formats` attribute.
        Then:
            It returns a set of strings without being called.
        """
        ingestor = MinimalIngestor()

        formats = ingestor.supported_formats

        assert isinstance(formats, set)
        assert all(isinstance(fmt, str) for fmt in formats)

    def test_supports_format_should_check_file_suffix(self, tmp_path: Any) -> None:
        """Verifies the `supports_format` method checks file extensions.

        Given:
            A `MinimalIngestor` with a set of supported formats.
        When:
            Calling `supports_format` with various file paths.
        Then:
            It returns `True` for supported formats and `False` otherwise.
        """
        ingestor = MinimalIngestor()

        txt_file = tmp_path / "document.txt"
        md_file = tmp_path / "readme.md"
        pdf_file = tmp_path / "paper.pdf"

        assert ingestor.supports_format(txt_file) is True
        assert ingestor.supports_format(md_file) is True
        assert ingestor.supports_format(pdf_file) is False

    def test_incomplete_ingestor_implementation_should_raise_type_error(self) -> None:
        """Verifies that incomplete implementations are rejected.

        Given:
            A class that inherits `BaseIngestor` but omits an abstract member.
        When:
            Attempting to instantiate it.
        Then:
            A `TypeError` is raised.
        """

        class IncompleteIngestor(BaseIngestor):
            pass

        assert_abstract_class_cannot_be_instantiated(IncompleteIngestor)

    def test_failed_ingest_should_raise_ingestion_error_with_cause(self) -> None:
        """Verifies that internal errors are wrapped in `IngestionError`.

        Given:
            A `MinimalIngestor` configured to simulate internal failures.
        When:
            The `ingest` method is called.
        Then:
            An `IngestionError` is raised with its `__cause__` referencing the original error.
        """
        ingestor = MinimalIngestor(should_fail=True)

        with pytest.raises(IngestionError) as exc_info:
            ingestor.ingest(b"test")

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)


@pytest.mark.unit
class TestBaseLLMDriverContract:
    """Contract tests for the BaseLLMDriver abstract base class."""

    def test_instantiating_base_llm_driver_directly_should_raise_type_error(
        self,
    ) -> None:
        """Verifies that the abstract class cannot be instantiated.

        Given:
            The BaseLLMDriver abstract base class.
        When:
            Attempting to instantiate it directly.
        Then:
            A TypeError is raised.
        """
        assert_abstract_class_cannot_be_instantiated(BaseLLMDriver)

    def test_minimal_llm_driver_should_be_instantiable_and_implement_contract(
        self,
    ) -> None:
        """Verifies that a minimal concrete implementation is valid.

        Given:
            A `MinimalLLMDriver` class that implements all abstract members.
        When:
            Instantiating the class.
        Then:
            The instance is created successfully with required members.
        """
        driver = MinimalLLMDriver()

        assert isinstance(driver, BaseLLMDriver)
        assert callable(driver.generate)
        assert callable(driver.generate_stream)
        assert callable(driver.count_tokens)
        assert hasattr(driver, "context_window")

    @pytest.mark.asyncio
    async def test_base_llm_driver_should_support_async_context_manager(self) -> None:
        """Verifies the class supports the async context manager protocol.

        Given:
            A concrete `MinimalLLMDriver` instance.
        When:
            Used in an `async with` statement.
        Then:
            The `__aenter__` method returns self and `__aexit__` is called.
        """
        driver = MinimalLLMDriver()

        async with driver as ctx_driver:
            assert ctx_driver is driver

    @pytest.mark.asyncio
    async def test_generate_should_be_awaitable_and_return_generation_result(
        self,
    ) -> None:
        """Verifies the signature and return type of the `generate` method.

        Given:
            A `MinimalLLMDriver` instance.
        When:
            Calling the `generate` method.
        Then:
            The method is awaitable and returns a `GenerationResult` instance.
        """
        driver = MinimalLLMDriver()

        result = await driver.generate("Test prompt", max_tokens=50)

        assert isinstance(result, GenerationResult)
        assert result.text is not None
        assert result.usage is not None

    @pytest.mark.asyncio
    async def test_generate_should_accept_temperature_and_stop_sequences(
        self,
    ) -> None:
        """Verifies that `generate` accepts optional parameters.

        Given:
            A `MinimalLLMDriver` instance.
        When:
            Calling `generate` with `temperature` and `stop` sequences.
        Then:
            The parameters are accepted and a `GenerationResult` is returned.
        """
        driver = MinimalLLMDriver()

        result = await driver.generate(
            "Prompt", max_tokens=100, temperature=0.5, stop=["END", "\n\n"]
        )

        assert isinstance(result, GenerationResult)

    @pytest.mark.asyncio
    async def test_generate_stream_should_return_async_iterator(self) -> None:
        """Verifies that `generate_stream` returns an async iterator.

        Given:
            A `MinimalLLMDriver` instance.
        When:
            Calling the `generate_stream` method.
        Then:
            It returns an `AsyncIterator` that yields string chunks.
        """
        driver = MinimalLLMDriver()

        stream = driver.generate_stream("Prompt", max_tokens=50)

        assert inspect.isasyncgen(stream)

        chunks: list[Any] = []
        async for chunk in stream:
            chunks.append(chunk)
            assert isinstance(chunk, str)

        assert len(chunks) > 0

    def test_count_tokens_should_be_synchronous_and_return_int(self) -> None:
        """Verifies the `count_tokens` method is synchronous and returns an int.

        Given:
            A `MinimalLLMDriver` instance.
        When:
            Calling the `count_tokens` method.
        Then:
            The method is synchronous (not awaitable) and returns an integer.
        """
        driver = MinimalLLMDriver()

        token_count = driver.count_tokens("Hello world")

        assert not inspect.iscoroutine(token_count)
        assert isinstance(token_count, int)
        assert token_count >= 0

    def test_context_window_should_be_a_property(self) -> None:
        """Verifies that `context_window` is a property, not a method.

        Given:
            A `MinimalLLMDriver` instance.
        When:
            Accessing the `context_window` attribute.
        Then:
            It returns an integer without needing to be called.
        """
        driver = MinimalLLMDriver(context_window=8192)

        window_size = driver.context_window

        assert isinstance(window_size, int)
        assert window_size == 8192

    def test_incomplete_llm_driver_implementation_should_raise_type_error(self) -> None:
        """Verifies that incomplete implementations are rejected.

        Given:
            A class that inherits `BaseLLMDriver` but omits an abstract member.
        When:
            Attempting to instantiate it.
        Then:
            A `TypeError` is raised.
        """

        class IncompleteLLMDriver(BaseLLMDriver):
            pass

        assert_abstract_class_cannot_be_instantiated(IncompleteLLMDriver)

    @pytest.mark.asyncio
    async def test_failed_generate_should_raise_generation_error_with_cause(
        self,
    ) -> None:
        """Verifies that internal errors are wrapped in `GenerationError`.

        Given:
            A `MinimalLLMDriver` configured to simulate internal failures.
        When:
            The `generate` method is called.
        Then:
            A `GenerationError` is raised with its `__cause__` referencing the original error.
        """
        driver = MinimalLLMDriver(should_fail=True)

        with pytest.raises(GenerationError) as exc_info:
            await driver.generate("Prompt", max_tokens=10)

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)
