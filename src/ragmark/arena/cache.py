"""Section-specific caching for arena pipeline stages.

Avoids recomputing expensive operations (ingestion, embedding, indexing)
when only downstream configuration changes between arena variants.
"""

import hashlib
import json
import shutil
from pathlib import Path

from ragmark.config.profile import ExperimentProfile
from ragmark.index.backends import MemoryIndex
from ragmark.index.base import VectorIndex
from ragmark.index.embedders import BaseEmbedder
from ragmark.logger import get_logger
from ragmark.schemas.documents import KnowledgeNode

logger = get_logger(__name__)


class ArenaCache:
    """Section-specific caching for arena pipeline stages.

    Caches intermediate results (nodes, indexes) keyed by hashes
    of the relevant configuration sections, so that only stages
    affected by parameter changes are recomputed.

    Disk layout::

        cache_dir/
            nodes/{forge_hash}.jsonl
            indexes/{index_hash}/

    Attributes:
        cache_dir: Root directory for cache storage.
    """

    def __init__(self, cache_dir: Path) -> None:
        """Initialize the cache.

        Args:
            cache_dir: Root directory for all cached artifacts.
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "nodes").mkdir(exist_ok=True)
        (self.cache_dir / "indexes").mkdir(exist_ok=True)
        logger.debug("ArenaCache initialized: cache_dir=%s", cache_dir)

    @staticmethod
    def compute_section_hash(
        profile: ExperimentProfile,
        sections: list[str],
    ) -> str:
        """Compute deterministic hash of specific profile sections.

        Since ExperimentProfile.compute_hash() hashes the entire config,
        this helper extracts only the requested sections and hashes them
        independently.

        Args:
            profile: The experiment profile.
            sections: Section names to include (e.g., ["ingestor", "fragmenter"]).

        Returns:
            SHA-256 hex digest of the section subset.
        """
        full_dump = profile.model_dump(mode="json")
        subset = {s: full_dump[s] for s in sorted(sections) if s in full_dump}
        data_json = json.dumps(subset, sort_keys=True)
        return hashlib.sha256(data_json.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Node caching (forge stage)
    # ------------------------------------------------------------------

    def _nodes_path(self, forge_hash: str) -> Path:
        return self.cache_dir / "nodes" / f"{forge_hash}.jsonl"

    def has_nodes(self, forge_hash: str) -> bool:
        """Check if cached nodes exist for a forge hash."""
        return self._nodes_path(forge_hash).exists()

    def load_nodes(self, forge_hash: str) -> list[KnowledgeNode]:
        """Load cached nodes from JSONL file.

        Args:
            forge_hash: Hash of ingestor+fragmenter config sections.

        Returns:
            Deserialized knowledge nodes.

        Raises:
            FileNotFoundError: If cache file does not exist.
        """
        path = self._nodes_path(forge_hash)
        if not path.exists():
            raise FileNotFoundError(f"No cached nodes for hash: {forge_hash}")

        nodes: list[KnowledgeNode] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    nodes.append(KnowledgeNode.model_validate_json(line))

        logger.info(
            "Reusing cached nodes (forge_hash: %s): count=%d",
            forge_hash[:12],
            len(nodes),
        )
        return nodes

    def save_nodes(self, forge_hash: str, nodes: list[KnowledgeNode]) -> None:
        """Save nodes to JSONL cache file.

        Args:
            forge_hash: Hash of ingestor+fragmenter config sections.
            nodes: Knowledge nodes to persist.
        """
        path = self._nodes_path(forge_hash)
        with open(path, "w", encoding="utf-8") as f:
            for node in nodes:
                f.write(node.model_dump_json())
                f.write("\n")

        logger.debug(
            "Cached nodes saved: forge_hash=%s, count=%d",
            forge_hash[:12],
            len(nodes),
        )

    # ------------------------------------------------------------------
    # Index caching
    # ------------------------------------------------------------------

    def index_path(self, index_hash: str) -> Path:
        """Return the directory path for a cached index.

        Args:
            index_hash: Hash of ingestor+fragmenter+embedder+index sections.

        Returns:
            Directory path where the index is (or would be) stored.
        """
        return self.cache_dir / "indexes" / index_hash

    def has_index(self, index_hash: str) -> bool:
        """Check if a cached index exists for the given hash."""
        idx_dir = self.index_path(index_hash)
        return idx_dir.exists() and any(idx_dir.iterdir())

    async def load_index(
        self,
        index_hash: str,
        embedder: BaseEmbedder | None = None,
    ) -> VectorIndex:
        """Load a cached MemoryIndex from disk.

        Args:
            index_hash: Hash of the relevant config sections.
            embedder: Optional embedder to attach to the loaded index.

        Returns:
            Restored VectorIndex instance.

        Raises:
            FileNotFoundError: If cache directory does not exist.
        """
        idx_dir = self.index_path(index_hash)
        if not idx_dir.exists():
            raise FileNotFoundError(f"No cached index for hash: {index_hash}")

        index = await MemoryIndex.load(idx_dir, embedder=embedder)
        logger.info(
            "Reusing cached index (index_hash: %s): count=%d",
            index_hash[:12],
            await index.count(),
        )
        return index

    async def save_index(self, index_hash: str, index: VectorIndex) -> None:
        """Persist a VectorIndex to disk cache.

        Args:
            index_hash: Hash of the relevant config sections.
            index: The index to persist.
        """
        idx_dir = self.index_path(index_hash)
        idx_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(index, MemoryIndex):
            await index.save(idx_dir)
            logger.debug("Cached index saved: index_hash=%s", index_hash[:12])
        else:
            logger.warning(
                "Index caching skipped: backend=%s does not support persistence",
                type(index).__name__,
            )

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all cached artifacts."""
        for sub in ("nodes", "indexes"):
            sub_dir = self.cache_dir / sub
            if sub_dir.exists():
                shutil.rmtree(sub_dir)
                sub_dir.mkdir()

        logger.info("ArenaCache cleared: cache_dir=%s", self.cache_dir)
