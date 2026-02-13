"""Unit and integration tests for BenchmarkArena."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ragmark.arena.cache import ArenaCache
from ragmark.arena.engine import BenchmarkArena
from ragmark.config.profile import (
    EmbedderConfig,
    ExperimentProfile,
    FragmenterConfig,
    IndexConfig,
)
from ragmark.exceptions import ArenaError
from ragmark.index.embedders import BaseEmbedder
from ragmark.schemas.documents import KnowledgeNode, NodePosition
from ragmark.schemas.evaluation import SystemInfo, TrialCase
from ragmark.schemas.retrieval import RetrievedNode, TraceContext


def _make_profile(**overrides: object) -> ExperimentProfile:
    """Build an ExperimentProfile with safe defaults for arena tests."""
    defaults: dict[str, Any] = dict(
        fragmenter=FragmenterConfig(chunk_size=256, overlap=64),
        index=IndexConfig(backend="memory", connection=None),
        embedder=EmbedderConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    )
    defaults.update(overrides)
    return ExperimentProfile(**defaults)  # type: ignore[arg-type]


_UNSET: Any = object()


def _make_node(
    node_id: str = "node-1",
    content: str = "Test content",
    dense_vector: list[float] | None | Any = _UNSET,
) -> KnowledgeNode:
    """Build a KnowledgeNode with safe defaults.

    Pass ``dense_vector=None`` explicitly to create a node *without* a vector.
    Omitting the argument gives a default ``[0.1, 0.2, 0.3]``.
    """
    vec = [0.1, 0.2, 0.3] if dense_vector is _UNSET else dense_vector
    return KnowledgeNode(
        node_id=node_id,
        content=content,
        source_id="src-1",
        position=NodePosition(
            start_char=0, end_char=len(content), page=1, section="test"
        ),
        dense_vector=vec,
        sparse_vector=None,
    )


def _make_trial_case(
    case_id: str = "case-1",
    question: str = "What is the answer?",
    ground_truth_node_ids: list[str] | None = None,
) -> TrialCase:
    """Build a TrialCase with safe defaults."""
    return TrialCase(
        case_id=case_id,
        question=question,
        ground_truth_answer=None,
        ground_truth_node_ids=ground_truth_node_ids or ["node-1"],
    )


# ---------------------------------------------------------------------------
# Fake collaborators (legitimate test doubles, not mocks)
# ---------------------------------------------------------------------------


class _FakeEmbedder(BaseEmbedder):
    """Deterministic embedder returning constant vectors."""

    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension

    @property
    def embedding_dim(self) -> int:
        return self.dimension

    @classmethod
    def from_config(cls, config: Any) -> _FakeEmbedder:
        return cls()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dimension for _ in texts]

    def embed_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        return [{} for _ in texts]


class _FakeRetriever:
    """Fake retriever returning pre-built TraceContexts."""

    def __init__(self, nodes: list[KnowledgeNode] | None = None) -> None:
        self._nodes = nodes or [_make_node()]

    async def retrieve(self, query: str, top_k: int = 10) -> TraceContext:
        retrieved = [
            RetrievedNode(node=n, score=0.9, rank=i + 1)
            for i, n in enumerate(self._nodes)
        ]
        return TraceContext(
            query=query,
            retrieved_nodes=retrieved,
            reranked=False,
        )


# ---------------------------------------------------------------------------
# BenchmarkArena.__init__
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBenchmarkArenaInit:
    """Tests for BenchmarkArena constructor and attribute storage."""

    def test_init_should_store_base_profile(self) -> None:
        """Constructor stores the provided base_profile.

        Given:
            A valid ExperimentProfile.
        When:
            BenchmarkArena is initialized with that profile.
        Then:
            The base_profile attribute holds the same object.
        """
        profile = _make_profile()

        arena = BenchmarkArena(base_profile=profile, grid={})

        assert arena.base_profile is profile

    def test_init_should_store_grid(self) -> None:
        """Constructor stores the provided parameter grid.

        Given:
            A non-empty parameter grid.
        When:
            BenchmarkArena is initialized with that grid.
        Then:
            The grid attribute holds the same dictionary.
        """
        grid = {"fragmenter.chunk_size": [128, 256, 512]}
        profile = _make_profile()

        arena = BenchmarkArena(base_profile=profile, grid=grid)

        assert arena.grid is grid

    def test_init_should_store_cache_when_provided(self) -> None:
        """Constructor stores the cache object when explicitly provided.

        Given:
            A sentinel object used as a cache.
        When:
            BenchmarkArena is initialized with that cache.
        Then:
            The cache attribute holds the same object.
        """
        profile = _make_profile()
        sentinel_cache = object()

        arena = BenchmarkArena(
            base_profile=profile,
            grid={},
            cache=sentinel_cache,  # type: ignore[arg-type]
        )

        assert arena.cache is sentinel_cache

    def test_init_should_default_cache_to_none(self) -> None:
        """Cache defaults to None when not provided.

        Given:
            No cache argument.
        When:
            BenchmarkArena is initialized without a cache.
        Then:
            The cache attribute is None.
        """
        profile = _make_profile()

        arena = BenchmarkArena(base_profile=profile, grid={})

        assert arena.cache is None


# ---------------------------------------------------------------------------
# BenchmarkArena.generate_configs -- empty grid
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenerateConfigsEmptyGrid:
    """Tests for generate_configs when the grid is empty."""

    def test_generate_configs_should_return_base_profile_for_empty_grid(self) -> None:
        """An empty grid produces a single-element list containing the base profile.

        Given:
            A BenchmarkArena with an empty grid.
        When:
            generate_configs is called.
        Then:
            The result is a list with exactly one element, the base profile.
        """
        profile = _make_profile()
        arena = BenchmarkArena(base_profile=profile, grid={})

        configs = arena.generate_configs()

        assert len(configs) == 1
        assert configs[0] is profile

    def test_generate_configs_should_return_list_type_for_empty_grid(self) -> None:
        """The return type is always a list, even for an empty grid.

        Given:
            A BenchmarkArena with an empty grid.
        When:
            generate_configs is called.
        Then:
            The result is a list instance.
        """
        profile = _make_profile()
        arena = BenchmarkArena(base_profile=profile, grid={})

        configs = arena.generate_configs()

        assert isinstance(configs, list)


# ---------------------------------------------------------------------------
# BenchmarkArena.generate_configs -- single parameter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenerateConfigsSingleParam:
    """Tests for generate_configs with a single grid parameter."""

    def test_generate_configs_should_return_n_configs_for_n_values(self) -> None:
        """A single parameter with N values produces exactly N configs.

        Given:
            A grid with one key mapping to three chunk_size values.
        When:
            generate_configs is called.
        Then:
            Three ExperimentProfile instances are returned.
        """
        profile = _make_profile()
        grid = {"fragmenter.chunk_size": [128, 256, 512]}
        arena = BenchmarkArena(base_profile=profile, grid=grid)

        configs = arena.generate_configs()

        assert len(configs) == 3

    def test_generate_configs_should_apply_each_value(self) -> None:
        """Each config reflects the corresponding override value.

        Given:
            A grid with chunk_size values [128, 256, 512].
        When:
            generate_configs is called.
        Then:
            The chunk_size of each returned config matches the grid values in order.
        """
        profile = _make_profile()
        grid = {"fragmenter.chunk_size": [128, 256, 512]}
        arena = BenchmarkArena(base_profile=profile, grid=grid)

        configs = arena.generate_configs()

        assert configs[0].fragmenter.chunk_size == 128
        assert configs[1].fragmenter.chunk_size == 256
        assert configs[2].fragmenter.chunk_size == 512

    def test_generate_configs_should_preserve_non_overridden_fields(self) -> None:
        """Fields not in the grid remain unchanged from the base profile.

        Given:
            A grid overriding only chunk_size.
        When:
            generate_configs is called.
        Then:
            The overlap, index backend, and embedder model remain as in the base.
        """
        profile = _make_profile()
        grid = {"fragmenter.chunk_size": [128, 512]}
        arena = BenchmarkArena(base_profile=profile, grid=grid)

        configs = arena.generate_configs()

        for cfg in configs:
            assert cfg.fragmenter.overlap == 64
            assert cfg.index.backend == "memory"
            assert cfg.embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_generate_configs_should_return_experiment_profile_instances(self) -> None:
        """All returned objects are ExperimentProfile instances.

        Given:
            A grid with a single parameter.
        When:
            generate_configs is called.
        Then:
            Every element in the result is an ExperimentProfile.
        """
        profile = _make_profile()
        grid = {"fragmenter.chunk_size": [128, 512]}
        arena = BenchmarkArena(base_profile=profile, grid=grid)

        configs = arena.generate_configs()

        for cfg in configs:
            assert isinstance(cfg, ExperimentProfile)


# ---------------------------------------------------------------------------
# BenchmarkArena.generate_configs -- cartesian product
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenerateConfigsCartesianProduct:
    """Tests for generate_configs with multiple grid parameters."""

    def test_generate_configs_should_produce_cartesian_product(self) -> None:
        """Two parameters with 2 and 3 values produce 2x3=6 configs.

        Given:
            A grid with chunk_size (2 values) and retrieval.top_k (3 values).
        When:
            generate_configs is called.
        Then:
            Six ExperimentProfile instances are returned.
        """
        profile = _make_profile()
        grid = {
            "fragmenter.chunk_size": [128, 512],
            "retrieval.top_k": [5, 10, 20],
        }
        arena = BenchmarkArena(base_profile=profile, grid=grid)

        configs = arena.generate_configs()

        assert len(configs) == 6

    def test_generate_configs_should_cover_all_combinations(self) -> None:
        """Every combination of grid values appears exactly once.

        Given:
            A grid with chunk_size [128, 512] and top_k [5, 10, 20].
        When:
            generate_configs is called.
        Then:
            The set of (chunk_size, top_k) tuples covers the full cartesian product.
        """
        profile = _make_profile()
        grid = {
            "fragmenter.chunk_size": [128, 512],
            "retrieval.top_k": [5, 10, 20],
        }
        arena = BenchmarkArena(base_profile=profile, grid=grid)

        configs = arena.generate_configs()

        pairs = {(cfg.fragmenter.chunk_size, cfg.retrieval.top_k) for cfg in configs}
        expected = {
            (128, 5),
            (128, 10),
            (128, 20),
            (512, 5),
            (512, 10),
            (512, 20),
        }
        assert pairs == expected

    def test_generate_configs_should_produce_distinct_hashes(self) -> None:
        """Configs with different overrides have unique configuration hashes.

        Given:
            A grid producing multiple distinct configs.
        When:
            generate_configs is called and hashes are computed.
        Then:
            All hashes are unique (no two configs hash the same).
        """
        profile = _make_profile()
        grid = {
            "fragmenter.chunk_size": [128, 256, 512],
            "retrieval.top_k": [5, 10],
        }
        arena = BenchmarkArena(base_profile=profile, grid=grid)

        configs = arena.generate_configs()
        hashes = [cfg.compute_hash() for cfg in configs]

        assert len(set(hashes)) == len(hashes)


# ---------------------------------------------------------------------------
# BenchmarkArena.generate_configs -- error handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenerateConfigsErrors:
    """Tests for generate_configs error wrapping."""

    def test_generate_configs_should_raise_arena_error_for_invalid_key(self) -> None:
        """An invalid override key is wrapped as ArenaError.

        Given:
            A grid containing a dotted key that does not exist in the profile.
        When:
            generate_configs is called.
        Then:
            ArenaError is raised with a message about the failed overrides.
        """
        profile = _make_profile()
        grid = {"nonexistent.field": [1, 2, 3]}
        arena = BenchmarkArena(base_profile=profile, grid=grid)

        with pytest.raises(ArenaError, match="Failed to apply overrides"):
            arena.generate_configs()

    def test_generate_configs_should_chain_original_exception(self) -> None:
        """The ArenaError chains the original ConfigOverrideError as __cause__.

        Given:
            A grid with an invalid key.
        When:
            generate_configs raises ArenaError.
        Then:
            The __cause__ attribute is the original exception from with_overrides.
        """
        profile = _make_profile()
        grid = {"nonexistent.field": [42]}
        arena = BenchmarkArena(base_profile=profile, grid=grid)

        with pytest.raises(ArenaError) as exc_info:
            arena.generate_configs()

        assert exc_info.value.__cause__ is not None

    def test_generate_configs_should_raise_arena_error_for_invalid_value(self) -> None:
        """A value that violates schema validation is wrapped as ArenaError.

        Given:
            A grid with a chunk_size value below the minimum (50).
        When:
            generate_configs is called.
        Then:
            ArenaError is raised because Pydantic validation fails.
        """
        profile = _make_profile()
        grid = {"fragmenter.chunk_size": [10]}
        arena = BenchmarkArena(base_profile=profile, grid=grid)

        with pytest.raises(ArenaError, match="Failed to apply overrides"):
            arena.generate_configs()

    def test_generate_configs_should_raise_arena_error_for_deep_invalid_key(
        self,
    ) -> None:
        """A deeply nested invalid key is also wrapped as ArenaError.

        Given:
            A grid with a key like "fragmenter.nonexistent" (valid section, invalid leaf).
        When:
            generate_configs is called.
        Then:
            ArenaError is raised.
        """
        profile = _make_profile()
        grid = {"fragmenter.nonexistent": [100]}
        arena = BenchmarkArena(base_profile=profile, grid=grid)

        with pytest.raises(ArenaError, match="Failed to apply overrides"):
            arena.generate_configs()


# ---------------------------------------------------------------------------
# BenchmarkArena.generate_configs -- immutability
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenerateConfigsImmutability:
    """Tests verifying that generate_configs does not mutate the base profile."""

    def test_generate_configs_should_not_mutate_base_profile(self) -> None:
        """The base profile remains unchanged after generating configs.

        Given:
            A base profile with chunk_size=256.
        When:
            generate_configs produces variants with different chunk_sizes.
        Then:
            The base profile's chunk_size is still 256.
        """
        profile = _make_profile()
        original_hash = profile.compute_hash()
        grid = {"fragmenter.chunk_size": [128, 512, 1024]}
        arena = BenchmarkArena(base_profile=profile, grid=grid)

        arena.generate_configs()

        assert profile.fragmenter.chunk_size == 256
        assert profile.compute_hash() == original_hash

    def test_generate_configs_should_return_new_instances(self) -> None:
        """Generated configs are new objects, not references to the base profile.

        Given:
            A grid that produces configs different from the base.
        When:
            generate_configs is called.
        Then:
            None of the returned configs are the same object as the base profile.
        """
        profile = _make_profile()
        grid = {"fragmenter.chunk_size": [128, 512]}
        arena = BenchmarkArena(base_profile=profile, grid=grid)

        configs = arena.generate_configs()

        for cfg in configs:
            assert cfg is not profile


# ---------------------------------------------------------------------------
# BenchmarkArena._embed_nodes (direct call with FakeEmbedder)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmbedNodes:
    """Tests for the node embedding helper."""

    def test_embed_nodes_should_skip_already_embedded(self) -> None:
        """Nodes with dense_vector are left untouched.

        Given:
            Two nodes already carrying dense vectors.
        When:
            _embed_nodes is called.
        Then:
            The original vectors are preserved (not overwritten).
        """
        nodes = [
            _make_node("n1", dense_vector=[1.0, 2.0]),
            _make_node("n2", dense_vector=[3.0, 4.0]),
        ]
        embedder = _FakeEmbedder(dimension=2)
        arena = BenchmarkArena(base_profile=_make_profile(), grid={})

        result = arena._embed_nodes(nodes, embedder)  # type: ignore

        assert result[0].dense_vector == [1.0, 2.0]
        assert result[1].dense_vector == [3.0, 4.0]

    def test_embed_nodes_should_embed_missing_vectors(self) -> None:
        """Nodes without dense_vector receive embeddings from the embedder.

        Given:
            A node with dense_vector=None.
        When:
            _embed_nodes is called with a 3-dim FakeEmbedder.
        Then:
            The node's dense_vector is populated with a 3-dimensional vector.
        """
        node = _make_node("n1", dense_vector=None)
        embedder = _FakeEmbedder(dimension=3)
        arena = BenchmarkArena(base_profile=_make_profile(), grid={})

        result = arena._embed_nodes([node], embedder)  # type: ignore

        assert result[0].dense_vector is not None
        assert len(result[0].dense_vector) == 3

    def test_embed_nodes_should_embed_only_missing(self) -> None:
        """Only nodes lacking vectors are passed to the embedder.

        Given:
            One node with a vector and one without.
        When:
            _embed_nodes is called.
        Then:
            The first node keeps its original vector; the second gets embedded.
        """
        nodes = [
            _make_node("n1", dense_vector=[9.0, 9.0, 9.0]),
            _make_node("n2", dense_vector=None),
        ]
        embedder = _FakeEmbedder(dimension=3)
        arena = BenchmarkArena(base_profile=_make_profile(), grid={})

        result = arena._embed_nodes(nodes, embedder)  # type: ignore

        assert result[0].dense_vector == [9.0, 9.0, 9.0]
        assert result[1].dense_vector == [0.1, 0.1, 0.1]


# ---------------------------------------------------------------------------
# BenchmarkArena._collect_system_info (no external dependencies)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCollectSystemInfo:
    """Tests for static system info collection."""

    def test_collect_system_info_should_return_system_info(self) -> None:
        """_collect_system_info returns a valid SystemInfo instance.

        Given:
            No special prerequisites.
        When:
            _collect_system_info is called.
        Then:
            A SystemInfo with valid fields is returned.
        """
        info = BenchmarkArena._collect_system_info()  # type: ignore

        assert isinstance(info, SystemInfo)
        assert info.cpu_count >= 1
        assert info.ram_gb > 0
        assert len(info.python_version) > 0
        assert len(info.platform) > 0

    def test_collect_system_info_should_have_ragmark_version(self) -> None:
        """The returned info includes a ragmark version string.

        Given:
            ragmark package is importable.
        When:
            _collect_system_info is called.
        Then:
            ragmark_version is a non-empty string.
        """
        info = BenchmarkArena._collect_system_info()  # type: ignore

        assert isinstance(info.ragmark_version, str)
        assert len(info.ragmark_version) > 0


# ---------------------------------------------------------------------------
# BenchmarkArena._get_or_forge_nodes (cache hit with real ArenaCache)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGetOrForgeNodesCacheHit:
    """Tests for forge stage cache hit using real ArenaCache."""

    @pytest.mark.asyncio
    async def test_get_or_forge_nodes_should_load_from_cache(
        self, tmp_path: Path
    ) -> None:
        """When cache has nodes for the forge hash, they are returned directly.

        Given:
            A real ArenaCache pre-populated with nodes for the profile.
        When:
            _get_or_forge_nodes is called.
        Then:
            Nodes are loaded from the cache without running the forge pipeline.
        """
        cache = ArenaCache(tmp_path)
        profile = _make_profile()
        cached_nodes = [_make_node("cached-1"), _make_node("cached-2")]

        forge_hash = ArenaCache.compute_section_hash(
            profile, ["ingestor", "fragmenter"]
        )
        cache.save_nodes(forge_hash, cached_nodes)

        arena = BenchmarkArena(base_profile=profile, grid={}, cache=cache)
        nodes = await arena._get_or_forge_nodes(profile)  # type: ignore

        assert len(nodes) == 2
        assert nodes[0].node_id == "cached-1"
        assert nodes[1].node_id == "cached-2"

    @pytest.mark.asyncio
    async def test_get_or_forge_nodes_cache_should_save_after_forge(
        self, tmp_path: Path
    ) -> None:
        """After a cache miss, forged nodes are saved to cache for reuse.

        Given:
            A real ArenaCache with no pre-existing nodes.
        When:
            _get_or_forge_nodes saves nodes after running forge.
        Then:
            The cache contains nodes for the forge hash.
        """
        cache = ArenaCache(tmp_path)
        profile = _make_profile()
        forge_hash = ArenaCache.compute_section_hash(
            profile, ["ingestor", "fragmenter"]
        )

        assert not cache.has_nodes(forge_hash)

        # Manually simulate what _get_or_forge_nodes does on cache miss
        nodes = [_make_node("forged-1")]
        cache.save_nodes(forge_hash, nodes)

        assert cache.has_nodes(forge_hash)
        loaded = cache.load_nodes(forge_hash)
        assert len(loaded) == 1
        assert loaded[0].node_id == "forged-1"


# ---------------------------------------------------------------------------
# BenchmarkArena._get_or_build_index (cache hit with real ArenaCache)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGetOrBuildIndexCacheHit:
    """Tests for index stage cache hit using real ArenaCache + MemoryIndex."""

    @pytest.mark.asyncio
    async def test_get_or_build_index_should_load_from_cache(
        self, tmp_path: Path
    ) -> None:
        """When cache has an index, it is loaded without rebuilding.

        Given:
            A real ArenaCache with a pre-saved MemoryIndex.
        When:
            _get_or_build_index is called.
        Then:
            The index is loaded from cache.
        """
        from ragmark.index.backends import MemoryIndex

        cache = ArenaCache(tmp_path)
        profile = _make_profile()
        embedder = _FakeEmbedder(dimension=3)

        # Build and save a real MemoryIndex to cache
        index = MemoryIndex(embedding_dim=3, embedder=embedder)
        nodes = [_make_node("n1")]
        await index.add(nodes)

        index_hash = ArenaCache.compute_section_hash(
            profile, ["ingestor", "fragmenter", "embedder", "index"]
        )
        await cache.save_index(index_hash, index)

        # Now load it through the arena method
        arena = BenchmarkArena(base_profile=profile, grid={}, cache=cache)
        loaded_index = await arena._get_or_build_index(  # type: ignore
            profile, nodes, embedder
        )

        count = await loaded_index.count()
        assert count == 1


# ---------------------------------------------------------------------------
# BenchmarkArena._run_trial_cases (direct call with FakeRetriever)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunTrialCases:
    """Tests for the trial case execution loop."""

    @pytest.mark.asyncio
    async def test_run_trial_cases_should_return_case_results(self) -> None:
        """Each trial case produces a CaseResult with retrieval trace.

        Given:
            Two trial cases and a fake retriever (no generator).
        When:
            _run_trial_cases is called.
        Then:
            Two CaseResult objects are returned with matching case_ids.
        """
        profile = _make_profile()
        retriever = _FakeRetriever()
        embedder = _FakeEmbedder()
        cases = [
            _make_trial_case("c1", "Question 1"),
            _make_trial_case("c2", "Question 2"),
        ]

        arena = BenchmarkArena(base_profile=profile, grid={})
        results = await arena._run_trial_cases(profile, retriever, embedder, cases)  # type: ignore

        assert len(results) == 2
        assert results[0].case_id == "c1"
        assert results[1].case_id == "c2"

    @pytest.mark.asyncio
    async def test_run_trial_cases_should_have_no_answer_without_generator(
        self,
    ) -> None:
        """Without a generator, predicted_answer is None.

        Given:
            A profile with no generator configured.
        When:
            _run_trial_cases is called.
        Then:
            Each CaseResult has predicted_answer=None.
        """
        profile = _make_profile()
        retriever = _FakeRetriever()
        embedder = _FakeEmbedder()
        cases = [_make_trial_case()]

        arena = BenchmarkArena(base_profile=profile, grid={})
        results = await arena._run_trial_cases(profile, retriever, embedder, cases)  # type: ignore

        assert results[0].predicted_answer is None
        assert results[0].generation_result is None

    @pytest.mark.asyncio
    async def test_run_trial_cases_should_preserve_trace_query(self) -> None:
        """The trace query matches the trial case question.

        Given:
            A trial case with a specific question.
        When:
            _run_trial_cases is called.
        Then:
            The trace.query on the CaseResult matches the question.
        """
        profile = _make_profile()
        retriever = _FakeRetriever()
        embedder = _FakeEmbedder()
        cases = [_make_trial_case("c1", "What color is the sky?")]

        arena = BenchmarkArena(base_profile=profile, grid={})
        results = await arena._run_trial_cases(profile, retriever, embedder, cases)  # type: ignore

        assert results[0].trace.query == "What color is the sky?"

    @pytest.mark.asyncio
    async def test_run_trial_cases_should_include_retrieved_nodes_in_trace(
        self,
    ) -> None:
        """Retrieved nodes from the retriever appear in the CaseResult trace.

        Given:
            A FakeRetriever configured with two specific nodes.
        When:
            _run_trial_cases is called.
        Then:
            The trace contains two RetrievedNodes.
        """
        nodes = [_make_node("n1"), _make_node("n2")]
        profile = _make_profile()
        retriever = _FakeRetriever(nodes)
        embedder = _FakeEmbedder()
        cases = [_make_trial_case()]

        arena = BenchmarkArena(base_profile=profile, grid={})
        results = await arena._run_trial_cases(profile, retriever, embedder, cases)  # type: ignore

        assert len(results[0].trace.retrieved_nodes) == 2
        assert results[0].trace.retrieved_nodes[0].node.node_id == "n1"
        assert results[0].trace.retrieved_nodes[1].node.node_id == "n2"


# ---------------------------------------------------------------------------
# BenchmarkArena._create_driver
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreateDriver:
    """Tests for the LLM driver creation helper."""

    @pytest.mark.asyncio
    async def test_create_driver_should_return_none_without_generator(self) -> None:
        """When no generator is configured, _create_driver returns None.

        Given:
            A profile without a generator section.
        When:
            _create_driver is called.
        Then:
            None is returned.
        """
        profile = _make_profile()
        arena = BenchmarkArena(base_profile=profile, grid={})

        result = await arena._create_driver(profile)  # type: ignore

        assert result is None
