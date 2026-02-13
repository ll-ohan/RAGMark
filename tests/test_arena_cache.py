"""Unit tests for ArenaCache section-specific caching."""

import logging
from pathlib import Path
from typing import Any

import pytest

from ragmark.arena.cache import ArenaCache
from ragmark.config.profile import (
    EmbedderConfig,
    ExperimentProfile,
    FragmenterConfig,
    IndexConfig,
)
from ragmark.schemas.documents import KnowledgeNode, NodePosition


def _make_node(
    node_id: str = "node-1",
    content: str = "Test content",
    source_id: str = "src-1",
    dense_vector: list[float] | None = None,
) -> KnowledgeNode:
    """Build a minimal KnowledgeNode for cache tests."""
    return KnowledgeNode(
        node_id=node_id,
        content=content,
        source_id=source_id,
        position=NodePosition(
            start_char=0, end_char=len(content), page=1, section="test"
        ),
        dense_vector=dense_vector or [0.1, 0.2, 0.3],
        sparse_vector=None,
    )


def _make_profile(**overrides: object) -> ExperimentProfile:
    """Build an ExperimentProfile with safe defaults for cache tests."""
    defaults: dict[str, Any] = dict(
        fragmenter=FragmenterConfig(chunk_size=256, overlap=64),
        index=IndexConfig(backend="memory", connection=None),
        embedder=EmbedderConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    )
    defaults.update(overrides)
    return ExperimentProfile(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# compute_section_hash
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeSectionHash:
    """Tests for ArenaCache.compute_section_hash determinism and correctness."""

    def test_compute_section_hash_should_be_deterministic(self) -> None:
        """Same profile and same sections always yield the same hash.

        Given:
            Two identical profiles and the same section list.
        When:
            compute_section_hash is called twice.
        Then:
            Both calls return the same hex digest.
        """
        profile_a = _make_profile()
        profile_b = _make_profile()

        hash_a = ArenaCache.compute_section_hash(profile_a, ["fragmenter", "embedder"])
        hash_b = ArenaCache.compute_section_hash(profile_b, ["fragmenter", "embedder"])

        assert hash_a == hash_b

    def test_compute_section_hash_should_differ_for_different_sections(self) -> None:
        """Different section selections produce distinct hashes.

        Given:
            A single profile.
        When:
            Hashing ["fragmenter"] vs ["fragmenter", "embedder"].
        Then:
            The two hashes differ.
        """
        profile = _make_profile()

        hash_fragmenter = ArenaCache.compute_section_hash(profile, ["fragmenter"])
        hash_both = ArenaCache.compute_section_hash(profile, ["fragmenter", "embedder"])

        assert hash_fragmenter != hash_both

    def test_compute_section_hash_should_return_64_char_hex(self) -> None:
        """SHA-256 digests are 64-character lowercase hex strings.

        Given:
            Any valid profile and sections.
        When:
            compute_section_hash is called.
        Then:
            The result is a 64-character string containing only hex digits.
        """
        profile = _make_profile()

        result = ArenaCache.compute_section_hash(profile, ["fragmenter"])

        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_compute_section_hash_should_be_order_independent_for_sections(
        self
    ) -> None:
        """Section order does not affect the hash because sections are sorted.

        Given:
            The same profile.
        When:
            Hashing ["embedder", "fragmenter"] vs ["fragmenter", "embedder"].
        Then:
            The hashes match.
        """
        profile = _make_profile()

        hash_ab = ArenaCache.compute_section_hash(profile, ["embedder", "fragmenter"])
        hash_ba = ArenaCache.compute_section_hash(profile, ["fragmenter", "embedder"])

        assert hash_ab == hash_ba

    def test_compute_section_hash_should_differ_for_different_profiles(self) -> None:
        """Profiles with different config values produce different hashes.

        Given:
            Two profiles with different chunk_size values.
        When:
            Hashing the same sections.
        Then:
            The hashes differ.
        """
        profile_256 = _make_profile(
            fragmenter=FragmenterConfig(chunk_size=256, overlap=64)
        )
        profile_512 = _make_profile(
            fragmenter=FragmenterConfig(chunk_size=512, overlap=64)
        )

        hash_256 = ArenaCache.compute_section_hash(profile_256, ["fragmenter"])
        hash_512 = ArenaCache.compute_section_hash(profile_512, ["fragmenter"])

        assert hash_256 != hash_512

    def test_compute_section_hash_should_ignore_missing_sections(self) -> None:
        """Non-existent section names are silently skipped.

        Given:
            A profile and a section list containing a non-existent key.
        When:
            compute_section_hash is called.
        Then:
            The hash matches a call with only the valid sections.
        """
        profile = _make_profile()

        hash_with_bogus = ArenaCache.compute_section_hash(
            profile, ["fragmenter", "nonexistent_section"]
        )
        hash_without_bogus = ArenaCache.compute_section_hash(profile, ["fragmenter"])

        assert hash_with_bogus == hash_without_bogus


# ---------------------------------------------------------------------------
# Node save / load round-trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNodeCaching:
    """Tests for save_nodes, load_nodes, and has_nodes."""

    def test_has_nodes_should_return_false_before_save(self, tmp_path: Path) -> None:
        """A freshly-created cache has no nodes for any hash.

        Given:
            An empty ArenaCache.
        When:
            Checking has_nodes for an arbitrary hash.
        Then:
            The result is False.
        """
        cache = ArenaCache(cache_dir=tmp_path / "cache")

        assert cache.has_nodes("deadbeef") is False

    def test_has_nodes_should_return_true_after_save(self, tmp_path: Path) -> None:
        """After saving, has_nodes returns True for that hash.

        Given:
            An ArenaCache with nodes saved under a known hash.
        When:
            Checking has_nodes for that hash.
        Then:
            The result is True.
        """
        cache = ArenaCache(cache_dir=tmp_path / "cache")
        nodes = [_make_node()]
        cache.save_nodes("abc123", nodes)

        assert cache.has_nodes("abc123") is True

    def test_save_and_load_nodes_should_round_trip(self, tmp_path: Path) -> None:
        """Nodes survive a save/load cycle with content preserved.

        Given:
            Two distinct KnowledgeNodes.
        When:
            Saving then loading with the same hash key.
        Then:
            The loaded list has the same length and content.
        """
        cache = ArenaCache(cache_dir=tmp_path / "cache")
        original_nodes = [
            _make_node(node_id="n1", content="First node content"),
            _make_node(node_id="n2", content="Second node content", source_id="src-2"),
        ]

        cache.save_nodes("roundtrip", original_nodes)
        loaded_nodes = cache.load_nodes("roundtrip")

        assert len(loaded_nodes) == 2
        assert loaded_nodes[0].node_id == "n1"
        assert loaded_nodes[0].content == "First node content"
        assert loaded_nodes[1].node_id == "n2"
        assert loaded_nodes[1].source_id == "src-2"

    def test_save_and_load_nodes_should_preserve_vectors(self, tmp_path: Path) -> None:
        """Dense vectors survive JSONL serialization faithfully.

        Given:
            A node with a specific dense_vector.
        When:
            Saving and loading.
        Then:
            The vector values are equal.
        """
        cache = ArenaCache(cache_dir=tmp_path / "cache")
        original = _make_node(dense_vector=[0.111, 0.222, 0.333])
        cache.save_nodes("vectors", [original])

        loaded = cache.load_nodes("vectors")

        assert loaded[0].dense_vector == [0.111, 0.222, 0.333]

    def test_load_nodes_should_raise_file_not_found_for_missing_hash(
        self, tmp_path: Path
    ) -> None:
        """Loading a hash that was never saved raises FileNotFoundError.

        Given:
            An ArenaCache with no saved nodes.
        When:
            Calling load_nodes with an unknown hash.
        Then:
            FileNotFoundError is raised.
        """
        cache = ArenaCache(cache_dir=tmp_path / "cache")

        with pytest.raises(FileNotFoundError, match="No cached nodes"):
            cache.load_nodes("no_such_hash")

    def test_save_nodes_should_create_jsonl_file(self, tmp_path: Path) -> None:
        """Saving writes a .jsonl file into the nodes subdirectory.

        Given:
            An ArenaCache.
        When:
            Saving nodes under a hash.
        Then:
            A file at cache_dir/nodes/{hash}.jsonl exists.
        """
        cache = ArenaCache(cache_dir=tmp_path / "cache")
        cache.save_nodes("filehash", [_make_node()])

        expected_path = tmp_path / "cache" / "nodes" / "filehash.jsonl"
        assert expected_path.exists()
        assert expected_path.stat().st_size > 0

    def test_save_empty_nodes_should_produce_empty_file(self, tmp_path: Path) -> None:
        """Saving an empty node list creates a file but load returns empty list.

        Given:
            An empty node list.
        When:
            Saving then loading.
        Then:
            The loaded list is empty and has_nodes returns True (file exists).
        """
        cache = ArenaCache(cache_dir=tmp_path / "cache")
        cache.save_nodes("empty", [])

        assert cache.has_nodes("empty") is True
        assert cache.load_nodes("empty") == []


# ---------------------------------------------------------------------------
# Index caching (sync-only parts)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIndexCaching:
    """Tests for has_index and index_path."""

    def test_has_index_should_return_false_for_empty_cache(
        self, tmp_path: Path
    ) -> None:
        """An empty cache has no indexes.

        Given:
            A freshly-created ArenaCache.
        When:
            Checking has_index for any hash.
        Then:
            The result is False.
        """
        cache = ArenaCache(cache_dir=tmp_path / "cache")

        assert cache.has_index("some_hash") is False

    def test_has_index_should_return_false_for_empty_directory(
        self, tmp_path: Path
    ) -> None:
        """An existing but empty index directory does not count as cached.

        Given:
            An ArenaCache where the index directory exists but is empty.
        When:
            Checking has_index.
        Then:
            The result is False because has_index checks any(dir.iterdir()).
        """
        cache = ArenaCache(cache_dir=tmp_path / "cache")
        idx_dir = cache.index_path("empty_dir")
        idx_dir.mkdir(parents=True, exist_ok=True)

        assert cache.has_index("empty_dir") is False

    def test_has_index_should_return_true_when_directory_has_files(
        self, tmp_path: Path
    ) -> None:
        """A populated index directory is considered cached.

        Given:
            An index directory containing at least one file.
        When:
            Checking has_index.
        Then:
            The result is True.
        """
        cache = ArenaCache(cache_dir=tmp_path / "cache")
        idx_dir = cache.index_path("populated")
        idx_dir.mkdir(parents=True, exist_ok=True)
        (idx_dir / "vectors.npy").write_bytes(b"data")

        assert cache.has_index("populated") is True

    def test_index_path_should_return_expected_structure(self, tmp_path: Path) -> None:
        """index_path returns cache_dir/indexes/{hash}.

        Given:
            An ArenaCache.
        When:
            Calling index_path with a hash.
        Then:
            The returned path is cache_dir / "indexes" / hash.
        """
        cache = ArenaCache(cache_dir=tmp_path / "cache")

        result = cache.index_path("myhash")

        assert result == tmp_path / "cache" / "indexes" / "myhash"


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestClear:
    """Tests for ArenaCache.clear()."""

    def test_clear_should_remove_cached_nodes(self, tmp_path: Path) -> None:
        """Clearing the cache removes previously saved node files.

        Given:
            An ArenaCache with saved nodes.
        When:
            clear() is called.
        Then:
            has_nodes returns False for the previously saved hash.
        """
        cache = ArenaCache(cache_dir=tmp_path / "cache")
        cache.save_nodes("tobecleared", [_make_node()])
        assert cache.has_nodes("tobecleared") is True

        cache.clear()

        assert cache.has_nodes("tobecleared") is False

    def test_clear_should_remove_cached_indexes(self, tmp_path: Path) -> None:
        """Clearing the cache removes previously saved index directories.

        Given:
            An ArenaCache with a populated index directory.
        When:
            clear() is called.
        Then:
            has_index returns False for the previously saved hash.
        """
        cache = ArenaCache(cache_dir=tmp_path / "cache")
        idx_dir = cache.index_path("idx_clear")
        idx_dir.mkdir(parents=True, exist_ok=True)
        (idx_dir / "data.bin").write_bytes(b"index")
        assert cache.has_index("idx_clear") is True

        cache.clear()

        assert cache.has_index("idx_clear") is False

    def test_clear_should_leave_subdirectories_existing_but_empty(
        self, tmp_path: Path
    ) -> None:
        """After clear(), the nodes/ and indexes/ directories still exist.

        Given:
            An ArenaCache with cached data.
        When:
            clear() is called.
        Then:
            cache_dir/nodes and cache_dir/indexes exist and are empty.
        """
        cache = ArenaCache(cache_dir=tmp_path / "cache")
        cache.save_nodes("temp", [_make_node()])

        cache.clear()

        nodes_dir = tmp_path / "cache" / "nodes"
        indexes_dir = tmp_path / "cache" / "indexes"
        assert nodes_dir.is_dir()
        assert indexes_dir.is_dir()
        assert list(nodes_dir.iterdir()) == []
        assert list(indexes_dir.iterdir()) == []

    def test_clear_should_be_idempotent(self, tmp_path: Path) -> None:
        """Calling clear() twice does not raise an error.

        Given:
            An already-cleared ArenaCache.
        When:
            clear() is called a second time.
        Then:
            No exception is raised and directories remain intact.
        """
        cache = ArenaCache(cache_dir=tmp_path / "cache")
        cache.clear()
        cache.clear()

        assert (tmp_path / "cache" / "nodes").is_dir()
        assert (tmp_path / "cache" / "indexes").is_dir()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCacheLogging:
    """Tests for cache hit/miss logging behaviour."""

    def test_load_nodes_should_log_info_on_cache_hit(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Reading cached nodes emits an INFO log with the forge hash prefix.

        Given:
            An ArenaCache with saved nodes.
        When:
            load_nodes is called.
        Then:
            An INFO-level log record containing "Reusing cached nodes" is emitted.
        """
        cache = ArenaCache(cache_dir=tmp_path / "cache")
        cache.save_nodes("loghash123456", [_make_node()])

        logger = logging.getLogger("ragmark.arena.cache")
        logger.propagate = True
        try:
            with caplog.at_level(logging.INFO, logger="ragmark.arena.cache"):
                cache.load_nodes("loghash123456")
        finally:
            logger.propagate = False

        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("Reusing cached nodes" in msg for msg in info_messages)
        assert any("loghash12345" in msg for msg in info_messages)

    def test_save_nodes_should_log_debug(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Saving nodes emits a DEBUG log.

        Given:
            An ArenaCache.
        When:
            save_nodes is called.
        Then:
            A DEBUG-level record containing "Cached nodes saved" is emitted.
        """
        cache = ArenaCache(cache_dir=tmp_path / "cache")

        logger = logging.getLogger("ragmark.arena.cache")
        logger.propagate = True
        try:
            with caplog.at_level(logging.DEBUG, logger="ragmark.arena.cache"):
                cache.save_nodes("debughash1234", [_make_node()])
        finally:
            logger.propagate = False

        debug_messages = [
            r.message for r in caplog.records if r.levelno == logging.DEBUG
        ]
        assert any("Cached nodes saved" in msg for msg in debug_messages)


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestArenaCacheInit:
    """Tests for ArenaCache.__init__ directory setup."""

    def test_init_should_create_cache_directory_structure(self, tmp_path: Path) -> None:
        """Initialization creates cache_dir, nodes/, and indexes/ subdirectories.

        Given:
            A non-existent target directory.
        When:
            ArenaCache is initialized with that path.
        Then:
            cache_dir, cache_dir/nodes, and cache_dir/indexes all exist.
        """
        target = tmp_path / "brand_new_cache"
        assert not target.exists()

        ArenaCache(cache_dir=target)

        assert target.is_dir()
        assert (target / "nodes").is_dir()
        assert (target / "indexes").is_dir()

    def test_init_should_be_idempotent(self, tmp_path: Path) -> None:
        """Re-initializing with the same path does not raise or lose data.

        Given:
            A cache directory already containing nodes.
        When:
            ArenaCache is initialized again on the same path.
        Then:
            Existing data is preserved.
        """
        target = tmp_path / "reuse_cache"
        cache = ArenaCache(cache_dir=target)
        cache.save_nodes("keepme", [_make_node()])

        cache2 = ArenaCache(cache_dir=target)

        assert cache2.has_nodes("keepme") is True
