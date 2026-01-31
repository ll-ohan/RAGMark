"""Vector index backend implementations.

This module will contain concrete implementations of VectorIndex
for different backends (Memory, Qdrant, Milvus, LanceDB).
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from ragmark.config.profile import IndexConfig
from ragmark.exceptions import IndexError
from ragmark.index.base import VectorIndex
from ragmark.index.embedders import BaseEmbedder
from ragmark.logger import get_logger
from ragmark.schemas.documents import KnowledgeNode
from ragmark.schemas.retrieval import SearchResult

logger = get_logger(__name__)


class MemoryIndex(VectorIndex):
    """In-memory vector index using NumPy for fast prototyping.

    This index stores all vectors in RAM using NumPy arrays and performs
    brute-force similarity search. It is suitable for development, testing,
    and small to medium datasets.

    Features:
    - Dynamic array resizing
    - Thread-safe write operations via asyncio locks
    - Multiple distance metrics (cosine, euclidean, dot product)
    - Metadata filtering
    - Persistence via save/load

    Attributes:
        embedding_dim: Dimensionality of embedding vectors.
        collection_name: Logical collection name.
        embedder: Embedder for auto-embedding nodes.
        distance_metric: Metric for similarity search.
    """

    def __init__(
        self,
        embedding_dim: int,
        collection_name: str = "default",
        embedder: BaseEmbedder | None = None,
        distance_metric: Literal["cosine", "euclidean", "dot"] = "cosine",
    ):
        """Initialize the memory index.

        Args:
            embedding_dim: Dimensionality of embedding vectors.
            collection_name: Logical collection name.
            embedder: Embedder for auto-embedding nodes.
            distance_metric: Metric for similarity search.
        """
        self.embedding_dim = embedding_dim
        self.collection_name = collection_name
        self.embedder = embedder
        self.distance_metric = distance_metric

        self._vectors: NDArray[np.float32] = np.zeros(
            (0, embedding_dim), dtype=np.float32
        )
        self._node_ids: list[str] = []
        self._metadata: list[dict[str, Any]] = []
        self._content: list[str] = []
        self._nodes: list[KnowledgeNode] = []

        self._inverted_index: dict[int, list[tuple[int, float]]] = {}

        self._lock = asyncio.Lock()

    @classmethod
    def from_config(
        cls, config: IndexConfig, embedder: BaseEmbedder | None = None
    ) -> "MemoryIndex":
        """Create MemoryIndex from configuration.

        Args:
            config: Index configuration.
            embedder: Embedder for auto-embedding.

        Returns:
            Configured MemoryIndex instance.
        """
        connection = config.connection or {}
        return cls(
            embedding_dim=config.embedding_dim,
            collection_name=config.collection_name,
            embedder=embedder,
            distance_metric=connection.get("distance_metric", "cosine"),
        )

    async def add(self, nodes: list[KnowledgeNode]) -> None:
        """Add knowledge nodes to the index.

        If nodes lack embeddings and an embedder is configured, embeddings
        will be computed automatically.

        Args:
            nodes: Knowledge nodes to add.

        Raises:
            IndexError: If nodes lack embeddings and no embedder is configured,
                or if vector dimensions don't match.
        """
        if not nodes:
            logger.debug("No nodes to add, skipping")
            return

        logger.debug("Adding nodes to index: count=%d", len(nodes))

        vectors_to_add: list[list[float]] = []
        needs_embedding = False

        for node in nodes:
            if node.dense_vector is None:
                needs_embedding = True
                break
            vectors_to_add.append(node.dense_vector)

        if needs_embedding:
            if self.embedder is None:
                logger.error("Nodes lack embeddings and no embedder configured")
                raise IndexError(
                    "Nodes must have dense_vector or embedder must be configured"
                )

            logger.debug("Computing embeddings: count=%d", len(nodes))
            texts = [node.content for node in nodes]
            loop = asyncio.get_running_loop()
            vectors_to_add = await loop.run_in_executor(
                None, self.embedder.embed, texts
            )
            logger.debug("Embeddings computed successfully")

        for vec in vectors_to_add:
            if len(vec) != self.embedding_dim:
                logger.error(
                    "Vector dimension mismatch: expected=%d, got=%d",
                    self.embedding_dim,
                    len(vec),
                )
                raise IndexError(
                    f"Vector dimension mismatch: expected {self.embedding_dim}, "
                    f"got {len(vec)}"
                )

        async with self._lock:
            current_size = len(self._node_ids)
            new_size = current_size + len(nodes)

            logger.debug("Resizing index: current=%d, new=%d", current_size, new_size)

            new_vectors = np.zeros((new_size, self.embedding_dim), dtype=np.float32)
            if current_size > 0:
                new_vectors[:current_size] = self._vectors
            self._vectors = new_vectors

            sparse_count = 0
            for i, (node, vec) in enumerate(zip(nodes, vectors_to_add, strict=True)):
                idx = current_size + i
                self._vectors[idx] = vec
                self._node_ids.append(node.node_id)
                self._metadata.append(node.metadata)
                self._content.append(node.content)
                self._nodes.append(node)

                if node.sparse_vector:
                    sparse_count += 1
                    for token_id, weight in node.sparse_vector.items():
                        if token_id not in self._inverted_index:
                            self._inverted_index[token_id] = []
                        self._inverted_index[token_id].append((idx, weight))

        logger.info(
            "Nodes added to index: count=%d, total=%d, sparse_vectors=%d",
            len(nodes),
            new_size,
            sparse_count,
        )

    async def search(
        self,
        query_vector: list[float] | dict[str, list[int] | list[float]],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors.

        Args:
            query_vector: Dense query embedding vector.
            top_k: Number of results to return.
            filters: Metadata filters for exact match.

        Returns:
            Search results sorted by descending similarity.

        Raises:
            IndexError: If sparse vectors are provided or dimension mismatch.
        """
        logger.debug("Search started: top_k=%d, filters=%s", top_k, filters is not None)

        if isinstance(query_vector, dict):
            logger.error("Sparse vectors not supported by MemoryIndex")
            raise IndexError("MemoryIndex only supports dense vectors")

        if len(query_vector) != self.embedding_dim:
            logger.error(
                "Query vector dimension mismatch: expected=%d, got=%d",
                self.embedding_dim,
                len(query_vector),
            )
            raise IndexError(
                f"Query vector dimension mismatch: expected {self.embedding_dim}, "
                f"got {len(query_vector)}"
            )

        async with self._lock:
            count = len(self._node_ids)
            if count == 0:
                logger.warning("Search on empty index")
                return []

            vectors_snapshot = self._vectors[:count].copy()
            metadata_snapshot = self._metadata.copy()
            node_ids_snapshot = self._node_ids.copy()
            nodes_snapshot = self._nodes.copy()

        query_np = np.array(query_vector, dtype=np.float32)
        loop = asyncio.get_running_loop()
        search_results = await loop.run_in_executor(
            None,
            self._search_sync,
            query_np,
            vectors_snapshot,
            node_ids_snapshot,
            metadata_snapshot,
            nodes_snapshot,
            filters,
            top_k,
        )

        return search_results

    def _search_sync(
        self,
        query_vector: NDArray[np.float32],
        vectors: NDArray[np.float32],
        node_ids: list[str],
        metadata: list[dict[str, Any]],
        nodes: list[KnowledgeNode],
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> list[SearchResult]:
        similarities = self._compute_similarities(query_vector, vectors)

        results: list[SearchResult] = []
        for i in range(len(node_ids)):
            if filters and not self._matches_filter(metadata[i], filters):
                continue

            results.append(
                SearchResult(
                    node_id=node_ids[i],
                    score=float(similarities[i]),
                    node=nodes[i],
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _compute_similarities(
        self,
        query_vector: NDArray[np.float32],
        vectors: NDArray[np.float32] | int,
    ) -> NDArray[np.float32]:
        if isinstance(vectors, int):
            vectors = self._vectors[:vectors]

        if self.distance_metric == "cosine":
            query_norm = np.linalg.norm(query_vector)
            vector_norms = np.linalg.norm(vectors, axis=1)
            dots = np.dot(vectors, query_vector)
            similarities = dots / (vector_norms * query_norm + 1e-8)
        elif self.distance_metric == "dot":
            similarities = np.dot(vectors, query_vector)
        elif self.distance_metric == "euclidean":
            distances = np.linalg.norm(vectors - query_vector, axis=1)
            similarities = -distances
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        return cast(NDArray[np.float32], similarities)

    @staticmethod
    def _matches_filter(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        for key, value in filters.items():
            if metadata.get(key) != value:
                return False
        return True

    async def search_hybrid(
        self,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
        top_k: int,
        alpha: float,
    ) -> list[SearchResult]:
        """Perform hybrid search combining dense and sparse retrieval.

        Fuses dense and sparse similarity scores using a weighted average (alpha):
        1. Computing dense similarity scores
        2. Computing sparse scores via inverted index
        3. Normalizing both scores (Min-Max scaling to [0,1])
        4. Fusing: final_score = alpha * dense_norm + (1 - alpha) * sparse_norm

        Args:
            dense_vector: Dense query embedding.
            sparse_vector: Sparse query vector (token_id -> weight mapping).
            top_k: Number of results to return.
            alpha: Fusion weight (0=pure sparse, 1=pure dense).

        Returns:
            Search results sorted by fused score.

        Raises:
            IndexError: If vector dimensions don't match.
        """
        if len(dense_vector) != self.embedding_dim:
            raise IndexError(
                f"Dense vector dimension mismatch: expected {self.embedding_dim}, "
                f"got {len(dense_vector)}"
            )

        async with self._lock:
            count = len(self._node_ids)
            if count == 0:
                return []

            vectors_snapshot = self._vectors[:count].copy()
            node_ids_snapshot = self._node_ids.copy()
            nodes_snapshot = self._nodes.copy()
            inverted_index_snapshot = {
                k: list(v) for k, v in self._inverted_index.items()
            }

        query_dense_np = np.array(dense_vector, dtype=np.float32)
        loop = asyncio.get_running_loop()
        search_results = await loop.run_in_executor(
            None,
            self._search_hybrid_sync,
            query_dense_np,
            sparse_vector,
            vectors_snapshot,
            inverted_index_snapshot,
            node_ids_snapshot,
            nodes_snapshot,
            alpha,
            top_k,
        )

        return search_results

    def _search_hybrid_sync(
        self,
        query_dense: NDArray[np.float32],
        sparse_vector: dict[int, float],
        vectors: NDArray[np.float32],
        inverted_index: dict[int, list[tuple[int, float]]],
        node_ids: list[str],
        nodes: list[KnowledgeNode],
        alpha: float,
        top_k: int,
    ) -> list[SearchResult]:
        num_docs = len(node_ids)

        dense_scores = self._compute_similarities(query_dense, vectors)

        sparse_scores = np.zeros(num_docs, dtype=np.float32)

        for token_id, query_weight in sparse_vector.items():
            if token_id in inverted_index:
                for doc_idx, doc_weight in inverted_index[token_id]:
                    if doc_idx < num_docs:
                        sparse_scores[doc_idx] += query_weight * doc_weight

        dense_normalized = self._min_max_normalize(dense_scores)
        sparse_normalized = self._min_max_normalize(sparse_scores)

        fused_scores = alpha * dense_normalized + (1 - alpha) * sparse_normalized

        results: list[SearchResult] = []
        for i in range(num_docs):
            results.append(
                SearchResult(
                    node_id=node_ids[i],
                    score=float(fused_scores[i]),
                    node=nodes[i],
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    @staticmethod
    def _min_max_normalize(scores: NDArray[np.float32]) -> NDArray[np.float32]:
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score - min_score < 1e-8:
            # All scores are the same, return uniform distribution
            return np.ones_like(scores)
        return cast(NDArray[np.float32], (scores - min_score) / (max_score - min_score))

    async def delete(self, node_ids: list[str]) -> int:
        """Delete nodes by ID.

        Args:
            node_ids: Node IDs to delete.

        Returns:
            Number of nodes successfully deleted.
        """
        async with self._lock:
            deleted = 0
            for node_id in node_ids:
                try:
                    idx = self._node_ids.index(node_id)
                    self._node_ids.pop(idx)
                    self._metadata.pop(idx)
                    self._content.pop(idx)
                    self._nodes.pop(idx)
                    count = len(self._node_ids)
                    if idx < count:
                        self._vectors[idx:count] = self._vectors[idx + 1 : count + 1]
                    deleted += 1
                except ValueError:
                    pass

            if deleted > 0:
                self._vectors = self._vectors[: len(self._node_ids)].copy()
                self._rebuild_inverted_index()
            return deleted

    def _rebuild_inverted_index(self) -> None:
        self._inverted_index = {}
        for idx, node in enumerate(self._nodes):
            if node.sparse_vector:
                for token_id, weight in node.sparse_vector.items():
                    if token_id not in self._inverted_index:
                        self._inverted_index[token_id] = []
                    self._inverted_index[token_id].append((idx, weight))

    async def count(self) -> int:
        """Get total number of nodes in the index.

        Returns:
            Node count.
        """
        return len(self._node_ids)

    async def clear(self) -> None:
        """Clear all nodes from the index."""
        async with self._lock:
            self._vectors = np.zeros((0, self.embedding_dim), dtype=np.float32)
            self._node_ids = []
            self._metadata = []
            self._content = []
            self._nodes = []
            self._inverted_index = {}

    async def exists(self, node_id: str) -> bool:
        """Check if a node exists in the index.

        Args:
            node_id: Node ID.

        Returns:
            True if node exists.
        """
        return node_id in self._node_ids

    def save(self, path: Path) -> None:
        """Save index to disk.

        This method is synchronous and blocks the event loop.

        Args:
            path: Target directory path.
        """
        count = len(self._node_ids)
        vectors = self._vectors[:count]

        logger.debug("Saving index: path=%s, nodes=%d", path, count)

        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", vectors)

        nodes_serialized = [
            node.model_dump(exclude={"dense_vector", "sparse_vector"})
            for node in self._nodes
        ]

        inverted_index_serialized = {str(k): v for k, v in self._inverted_index.items()}

        metadata_dict: dict[str, Any] = {
            "node_ids": self._node_ids,
            "metadata": self._metadata,
            "content": self._content,
            "nodes": nodes_serialized,
            "inverted_index": inverted_index_serialized,
            "embedding_dim": self.embedding_dim,
            "distance_metric": self.distance_metric,
            "collection_name": self.collection_name,
        }
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, ensure_ascii=False, indent=2)

        logger.info("Index saved: path=%s, nodes=%d", path, count)

    @classmethod
    def load(cls, path: Path, embedder: BaseEmbedder | None = None) -> "MemoryIndex":
        """Load index from disk.

        Args:
            path: Source directory path.
            embedder: Embedder for the loaded index.

        Returns:
            Loaded MemoryIndex instance.
        """
        logger.debug("Loading index: path=%s", path)

        json_path = path / "metadata.json"
        pkl_path = path / "metadata.pkl"

        if json_path.exists():
            logger.debug("Loading from JSON format")
            with open(json_path, encoding="utf-8") as f:
                metadata_dict = json.load(f)

            inverted_index_deserialized: dict[int, list[tuple[int, float]]] = {}
            if "inverted_index" in metadata_dict:
                for k, v in metadata_dict["inverted_index"].items():
                    inverted_index_deserialized[int(k)] = [
                        (doc_idx, weight) for doc_idx, weight in v
                    ]

            nodes_deserialized = [
                KnowledgeNode(**node_data)
                for node_data in metadata_dict.get("nodes", [])
            ]

        elif pkl_path.exists():
            # Backward compatibility for old pickle format
            logger.warning("Loading from deprecated pickle format: path=%s", pkl_path)
            import pickle

            with open(pkl_path, "rb") as f:
                metadata_dict = pickle.load(f)

            inverted_index_deserialized = metadata_dict.get("inverted_index", {})
            nodes_deserialized = metadata_dict.get("nodes", [])

        else:
            logger.error("No metadata file found: path=%s", path)
            raise FileNotFoundError(
                f"No metadata file found at {path} (tried metadata.json and metadata.pkl)"
            )

        vectors = np.load(path / "vectors.npy")
        node_count = len(metadata_dict["node_ids"])

        logger.debug(
            "Creating index from loaded data: nodes=%d, dim=%d",
            node_count,
            metadata_dict["embedding_dim"],
        )

        index = cls(
            embedding_dim=metadata_dict["embedding_dim"],
            collection_name=metadata_dict.get("collection_name", "default"),
            embedder=embedder,
            distance_metric=metadata_dict.get("distance_metric", "cosine"),
        )

        index._vectors = np.zeros(
            (vectors.shape[0], metadata_dict["embedding_dim"]), dtype=np.float32
        )
        index._vectors[: vectors.shape[0]] = vectors
        index._node_ids = metadata_dict["node_ids"]
        index._metadata = metadata_dict["metadata"]
        index._content = metadata_dict["content"]
        index._nodes = nodes_deserialized
        index._inverted_index = inverted_index_deserialized

        logger.info("Index loaded: path=%s, nodes=%d", path, node_count)
        return index

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> None:
        """Cleanup (nothing to do for memory index)."""
        pass


# These will raise ImportError if optional dependencies are not installed
try:
    from qdrant_client import QdrantClient  # noqa: F401

    class QdrantIndex(VectorIndex):
        """Qdrant vector index implementation.

        Attributes:
            collection_name: Name of the Qdrant collection.
            embedding_dim: Dimensionality of embeddings.
            host: Qdrant server host.
            port: Qdrant server port.
            api_key: Optional API key for authentication.
            embedder: Optional embedder instance.
        """

        def __init__(
            self,
            collection_name: str,
            embedding_dim: int,
            host: str,
            port: int,
            api_key: str | None = None,
            embedder: BaseEmbedder | None = None,
        ):
            """Initialize Qdrant connection.

            Args:
                collection_name: Target collection name.
                embedding_dim: Vector dimensionality.
                host: Server host address.
                port: Server port number.
                api_key: Authentication key.
                embedder: Embedder instance.
            """
            pass

        @classmethod
        def from_config(
            cls, config: IndexConfig, embedder: BaseEmbedder | None = None
        ) -> "QdrantIndex":
            """Create QdrantIndex from configuration.

            Args:
                config: Index configuration.
                embedder: Embedder instance.

            Returns:
                Configured QdrantIndex instance.
            """
            connection = config.connection or {}
            return cls(
                collection_name=config.collection_name,
                embedding_dim=config.embedding_dim,
                host=connection.get("host", "localhost"),
                port=connection.get("port", 6333),
                api_key=connection.get("api_key"),
                embedder=embedder,
            )

        async def add(self, nodes: list[KnowledgeNode]) -> None:
            """Add knowledge nodes to index.

            Args:
                nodes: Nodes to index.

            Raises:
                NotImplementedError: Always.
            """
            raise NotImplementedError()

        async def search(
            self,
            query_vector: list[float] | dict[str, list[int] | list[float]],
            top_k: int = 10,
            filters: dict[str, Any] | None = None,
        ) -> list[SearchResult]:
            """Search for similar vectors using dense embeddings.

            Args:
                query_vector: Dense query embedding or sparse vector.
                top_k: Number of results to return.
                filters: Metadata filters.

            Raises:
                NotImplementedError: Always.
            """
            raise NotImplementedError()

        async def search_hybrid(
            self,
            dense_vector: list[float],
            sparse_vector: dict[int, float],
            top_k: int,
            alpha: float,
        ) -> list[SearchResult]:
            """Perform hybrid dense and sparse retrieval.

            Args:
                dense_vector: Dense query embedding.
                sparse_vector: Sparse query embedding.
                top_k: Number of results to return.
                alpha: Fusion weight.

            Raises:
                NotImplementedError: Always.
            """
            raise NotImplementedError()

        async def delete(self, node_ids: list[str]) -> int:
            """Delete nodes by ID.

            Args:
                node_ids: Node IDs to delete.

            Raises:
                NotImplementedError: Always.
            """
            raise NotImplementedError()

        async def count(self) -> int:
            """Get total number of indexed nodes.

            Raises:
                NotImplementedError: Always.
            """
            raise NotImplementedError()

        async def clear(self) -> None:
            """Delete all nodes from index.

            Raises:
                NotImplementedError: Always.
            """
            raise NotImplementedError()

        async def exists(self, node_id: str) -> bool:
            """Check if node exists in index.

            Args:
                node_id: Node ID to check.

            Raises:
                NotImplementedError: Always.
            """
            raise NotImplementedError()

        async def __aexit__(
            self,
            _exc_type: type[BaseException] | None,
            _exc_val: BaseException | None,
            _exc_tb: Any,
        ) -> None:
            pass

except ImportError:
    pass


try:
    from pymilvus import MilvusClient  # noqa: F401

    class MilvusIndex(VectorIndex):
        """Milvus vector index implementation.

        Attributes:
            collection_name: Name of the Milvus collection.
            embedding_dim: Dimensionality of embeddings.
            host: Milvus server host.
            port: Milvus server port.
            embedder: Optional embedder instance.
        """

        def __init__(
            self,
            collection_name: str,
            embedding_dim: int,
            host: str,
            port: int,
            embedder: BaseEmbedder | None = None,
        ):
            """Initialize Milvus connection.

            Args:
                collection_name: Target collection name.
                embedding_dim: Vector dimensionality.
                host: Server host address.
                port: Server port number.
                embedder: Embedder instance.
            """
            pass

        @classmethod
        def from_config(
            cls, config: IndexConfig, embedder: BaseEmbedder | None = None
        ) -> "MilvusIndex":
            """Create MilvusIndex from configuration.

            Args:
                config: Index configuration.
                embedder: Embedder instance.

            Returns:
                Configured MilvusIndex instance.
            """
            connection = config.connection or {}
            return cls(
                collection_name=config.collection_name,
                embedding_dim=config.embedding_dim,
                host=connection.get("host", "localhost"),
                port=connection.get("port", 19530),
                embedder=embedder,
            )

        async def add(self, nodes: list[KnowledgeNode]) -> None:
            """Add knowledge nodes to index.

            Args:
                nodes: Nodes to index.

            Raises:
                NotImplementedError: Always.
            """
            raise NotImplementedError()

        async def search(
            self,
            query_vector: list[float] | dict[str, list[int] | list[float]],
            top_k: int = 10,
            filters: dict[str, Any] | None = None,
        ) -> list[SearchResult]:
            """Search for similar vectors using dense embeddings.

            Args:
                query_vector: Dense query embedding or sparse vector.
                top_k: Number of results to return.
                filters: Metadata filters.

            Raises:
                NotImplementedError: Always.
            """
            raise NotImplementedError()

        async def search_hybrid(
            self,
            dense_vector: list[float],
            sparse_vector: dict[int, float],
            top_k: int,
            alpha: float,
        ) -> list[SearchResult]:
            """Perform hybrid dense and sparse retrieval.

            Args:
                dense_vector: Dense query embedding.
                sparse_vector: Sparse query embedding.
                top_k: Number of results to return.
                alpha: Fusion weight.

            Raises:
                NotImplementedError: Always.
            """
            raise NotImplementedError()

        async def delete(self, node_ids: list[str]) -> int:
            """Delete nodes by ID.

            Args:
                node_ids: Node IDs to delete.

            Raises:
                NotImplementedError: Always.
            """
            raise NotImplementedError()

        async def count(self) -> int:
            """Get total number of indexed nodes.

            Raises:
                NotImplementedError: Always.
            """
            raise NotImplementedError()

        async def clear(self) -> None:
            """Delete all nodes from index.

            Raises:
                NotImplementedError: Always.
            """
            raise NotImplementedError()

        async def exists(self, node_id: str) -> bool:
            """Check if node exists in index.

            Args:
                node_id: Node ID to check.

            Raises:
                NotImplementedError: Always.
            """
            raise NotImplementedError()

        async def __aexit__(
            self,
            _exc_type: type[BaseException] | None,
            _exc_val: BaseException | None,
            _exc_tb: Any,
        ) -> None:
            pass

except ImportError:
    pass


try:
    import lancedb  # noqa: F401

    class LanceDBIndex(VectorIndex):
        """LanceDB vector index implementation.

        Attributes:
            db_path: Path to the LanceDB database.
            table_name: Name of the table.
            embedding_dim: Dimensionality of embeddings.
            embedder: Optional embedder instance.
        """

        def __init__(
            self,
            db_path: Path,
            table_name: str,
            embedding_dim: int,
            embedder: BaseEmbedder | None = None,
        ):
            """Initialize LanceDB connection.

            Args:
                db_path: Database directory path.
                table_name: Target table name.
                embedding_dim: Vector dimensionality.
                embedder: Embedder instance.
            """
            pass

        @classmethod
        def from_config(
            cls, config: IndexConfig, embedder: BaseEmbedder | None = None
        ) -> "LanceDBIndex":
            """Create LanceDBIndex from configuration.

            Args:
                config: Index configuration.
                embedder: Embedder instance.

            Returns:
                Configured LanceDBIndex instance.
            """
            connection = config.connection or {}
            return cls(
                db_path=Path(connection.get("db_path", "./lancedb")),
                table_name=config.collection_name,
                embedding_dim=config.embedding_dim,
                embedder=embedder,
            )

        async def add(self, nodes: list[KnowledgeNode]) -> None:
            """Add knowledge nodes to index.

            Args:
                nodes: Nodes to index.

            Raises:
                NotImplementedError: Always.
            """
            raise NotImplementedError()

        async def search(
            self,
            query_vector: list[float] | dict[str, list[int] | list[float]],
            top_k: int = 10,
            filters: dict[str, Any] | None = None,
        ) -> list[SearchResult]:
            """Search for similar vectors using dense embeddings.

            Args:
                query_vector: Dense query embedding or sparse vector.
                top_k: Number of results to return.
                filters: Metadata filters.

            Raises:
                NotImplementedError: Always.
            """
            raise NotImplementedError()

        async def search_hybrid(
            self,
            dense_vector: list[float],
            sparse_vector: dict[int, float],
            top_k: int,
            alpha: float,
        ) -> list[SearchResult]:
            """Perform hybrid dense and sparse retrieval.

            Args:
                dense_vector: Dense query embedding.
                sparse_vector: Sparse query embedding.
                top_k: Number of results to return.
                alpha: Fusion weight.

            Raises:
                NotImplementedError: Always.
            """
            raise NotImplementedError()

        async def delete(self, node_ids: list[str]) -> int:
            """Delete nodes by ID.

            Args:
                node_ids: Node IDs to delete.

            Raises:
                NotImplementedError: Always.
            """
            raise NotImplementedError()

        async def count(self) -> int:
            """Get total number of indexed nodes.

            Raises:
                NotImplementedError: Always.
            """
            raise NotImplementedError()

        async def clear(self) -> None:
            """Delete all nodes from index.

            Raises:
                NotImplementedError: Always.
            """
            raise NotImplementedError()

        async def exists(self, node_id: str) -> bool:
            """Check if node exists in index.

            Args:
                node_id: Node ID to check.

            Raises:
                NotImplementedError: Always.
            """
            raise NotImplementedError()

        async def __aexit__(
            self,
            _exc_type: type[BaseException] | None,
            _exc_val: BaseException | None,
            _exc_tb: Any,
        ) -> None:
            pass

except ImportError:
    pass
