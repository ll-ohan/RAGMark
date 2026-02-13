"""Benchmark arena orchestrator for comparative RAG evaluation.

Runs the full RAG pipeline across multiple configuration variants
generated from a parameter grid, producing per-variant AuditReports.
"""

from __future__ import annotations

import asyncio
import itertools
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ragmark.arena.cache import ArenaCache
from ragmark.config.profile import ExperimentProfile
from ragmark.exceptions import ArenaError
from ragmark.forge.runner import ForgeRunner
from ragmark.generation.context import ContextManager
from ragmark.index.embedders import BaseEmbedder, EmbedderFactory
from ragmark.index.factory import IndexFactory
from ragmark.logger import get_logger
from ragmark.metrics.evaluation.generation import (
    JudgePromptConfig,
    compute_generation_metrics,
)
from ragmark.metrics.evaluation.retrieval import compute_retrieval_batch
from ragmark.retrieval.factory import RetrieverFactory
from ragmark.schemas.documents import KnowledgeNode
from ragmark.schemas.evaluation import (
    AuditReport,
    CaseResult,
    SystemInfo,
    TrialCase,
)
from ragmark.schemas.generation import GenerationResult

logger = get_logger(__name__)


class BenchmarkArena:
    """Orchestrate comparative benchmarking across config variants.

    Takes a base profile and a parameter grid, generates all variant
    configurations via cartesian product, and runs each through the
    full RAG pipeline with optional caching of intermediate results.

    Attributes:
        base_profile: The base experiment configuration.
        grid: Parameter grid mapping dotted keys to value lists.
        cache: Optional arena cache for intermediate results.
    """

    def __init__(
        self,
        base_profile: ExperimentProfile,
        grid: dict[str, list[Any]],
        sources: list[Path] | None = None,
        cache: ArenaCache | None = None,
    ) -> None:
        """Initialize the arena.

        Args:
            base_profile: Base configuration to apply overrides onto.
            grid: Parameter grid with dotted keys mapping to value lists.
            sources: Source document paths for forge pipeline. Required on
                cache miss; may be omitted when all forge results are cached.
            cache: Optional cache for reusing forge/index results.
        """
        self.base_profile = base_profile
        self.grid = grid
        self.sources = sources or []
        self.cache = cache

    def generate_configs(self) -> list[ExperimentProfile]:
        """Generate all configuration variants from the parameter grid.

        Computes the cartesian product of all grid values and applies
        each combination as overrides to the base profile. Returned
        profiles are immutable Pydantic models.

        Returns:
            One ExperimentProfile per grid combination.

        Raises:
            ArenaError: If override application fails.
        """
        if not self.grid:
            logger.debug("Empty grid, returning base profile only")
            return [self.base_profile]

        keys = list(self.grid.keys())
        value_lists = [self.grid[k] for k in keys]
        configs: list[ExperimentProfile] = []

        for combo in itertools.product(*value_lists):
            overrides = dict(zip(keys, combo, strict=True))
            try:
                variant = self.base_profile.with_overrides(overrides)
                configs.append(variant)
            except Exception as exc:
                raise ArenaError(f"Failed to apply overrides {overrides}") from exc

        logger.info(
            "Generated %d config variants from grid with %d parameters",
            len(configs),
            len(keys),
        )
        return configs

    async def run(
        self,
        trial_cases: list[TrialCase],
        parallel: int = 1,
        on_config_complete: Callable[[int, AuditReport], None] | None = None,
    ) -> list[AuditReport]:
        """Execute the full benchmark pipeline for all config variants.

        Args:
            trial_cases: Evaluation trial cases.
            parallel: Maximum concurrent config executions.
            on_config_complete: Optional callback invoked after each config.

        Returns:
            AuditReport per configuration variant, in generation order.

        Raises:
            ArenaError: If any pipeline stage fails.
        """
        configs = self.generate_configs()
        semaphore = asyncio.Semaphore(parallel)

        logger.info(
            "Arena run started: configs=%d, parallel=%d, cases=%d",
            len(configs),
            parallel,
            len(trial_cases),
        )

        async def _run_guarded(index: int, profile: ExperimentProfile) -> AuditReport:
            async with semaphore:
                return await self._run_single_config(profile, trial_cases, index)

        tasks = [_run_guarded(idx, cfg) for idx, cfg in enumerate(configs)]
        reports: list[AuditReport] = []

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.error("Config %d/%d failed: %s", idx + 1, len(configs), result)
                logger.debug("Config failure details: index=%d", idx, exc_info=result)
                raise ArenaError(f"Config {idx + 1}/{len(configs)} failed") from result

            reports.append(result)
            if on_config_complete is not None:
                on_config_complete(idx, result)

        logger.info("Arena run completed: reports=%d", len(reports))
        return reports

    async def _run_single_config(
        self,
        profile: ExperimentProfile,
        trial_cases: list[TrialCase],
        config_index: int,
    ) -> AuditReport:
        """Execute the pipeline for a single configuration variant.

        Args:
            profile: Configuration for this variant.
            trial_cases: Evaluation trial cases.
            config_index: Ordinal index of this config in the grid.

        Returns:
            AuditReport for this configuration variant.

        Raises:
            ArenaError: If any pipeline stage fails.
        """
        config_hash = profile.compute_hash()
        start = time.perf_counter()

        logger.info("Config %d started: hash=%s", config_index + 1, config_hash[:12])

        try:
            # ---- Stage 1: Forge (ingest + fragment) ----
            nodes = await self._get_or_forge_nodes(profile)

            # ---- Stage 2: Embed ----
            embedder = EmbedderFactory.create(profile.embedder)
            nodes = self._embed_nodes(nodes, embedder)

            # ---- Stage 3: Index ----
            index = await self._get_or_build_index(profile, nodes, embedder)

            # ---- Stage 4: Retriever ----
            retriever = RetrieverFactory.create(profile.retrieval, index)

            # ---- Stage 5-6: Generate answers for each trial case ----
            case_results = await self._run_trial_cases(
                profile, retriever, embedder, trial_cases
            )

            # ---- Stage 7: Retrieval metrics ----
            ground_truth_ids: dict[str, list[str]] = {}
            for tc in trial_cases:
                if tc.ground_truth_node_ids:
                    ground_truth_ids[tc.case_id] = tc.ground_truth_node_ids

            retrieval_metrics: dict[str, float] = {}
            if ground_truth_ids:
                retrieval_metrics = compute_retrieval_batch(
                    case_results, ground_truth_ids
                )

            # ---- Stage 8: Generation metrics ----
            generation_metrics: dict[str, float] = {}
            if profile.generator is not None:
                gt_map = {tc.case_id: tc for tc in trial_cases}

                judge = await self._create_driver(profile)
                if judge is not None:
                    try:
                        generation_metrics = await compute_generation_metrics(
                            results=case_results,
                            ground_truth=gt_map,
                            judge=judge,
                            config=JudgePromptConfig(),
                            embedder=embedder,
                        )
                    finally:
                        await judge.__aexit__(None, None, None)

            # ---- Stage 9: Assemble report ----
            duration = time.perf_counter() - start
            all_metrics = {**retrieval_metrics, **generation_metrics}

            report = AuditReport(
                experiment_profile_hash=config_hash,
                duration_seconds=duration,
                metrics=all_metrics,
                per_case_results=case_results,
                system_info=self._collect_system_info(),
            )

            logger.info(
                "Config %d completed: hash=%s, duration=%.2fs, metrics=%d",
                config_index + 1,
                config_hash[:12],
                duration,
                len(all_metrics),
            )
            return report

        except ArenaError:
            raise
        except Exception as exc:
            raise ArenaError(f"Pipeline failed for config {config_hash[:12]}") from exc

    # ------------------------------------------------------------------
    # Pipeline stage helpers
    # ------------------------------------------------------------------

    async def _get_or_forge_nodes(
        self, profile: ExperimentProfile
    ) -> list[KnowledgeNode]:
        """Load nodes from cache or run the forge pipeline."""
        forge_hash = ArenaCache.compute_section_hash(
            profile, ["ingestor", "fragmenter"]
        )

        if self.cache and self.cache.has_nodes(forge_hash):
            return self.cache.load_nodes(forge_hash)

        logger.debug("Forge cache miss: forge_hash=%s", forge_hash[:12])
        runner = ForgeRunner.from_profile(profile)
        nodes = runner.process(self.sources)

        if self.cache:
            self.cache.save_nodes(forge_hash, nodes)

        return nodes

    def _embed_nodes(
        self,
        nodes: list[KnowledgeNode],
        embedder: BaseEmbedder,
    ) -> list[KnowledgeNode]:
        """Embed nodes that lack dense vectors."""
        texts_to_embed: list[str] = []
        indices: list[int] = []

        for i, node in enumerate(nodes):
            if node.dense_vector is None:
                texts_to_embed.append(node.content)
                indices.append(i)

        if not texts_to_embed:
            logger.debug("All nodes already have embeddings")
            return nodes

        logger.debug("Embedding %d/%d nodes", len(texts_to_embed), len(nodes))
        vectors = embedder.embed(texts_to_embed)

        for idx, vec in zip(indices, vectors, strict=True):
            nodes[idx].dense_vector = vec

        return nodes

    async def _get_or_build_index(
        self,
        profile: ExperimentProfile,
        nodes: list[KnowledgeNode],
        embedder: BaseEmbedder,
    ) -> Any:
        """Load index from cache or build a new one."""
        index_hash = ArenaCache.compute_section_hash(
            profile, ["ingestor", "fragmenter", "embedder", "index"]
        )

        if self.cache and self.cache.has_index(index_hash):
            return await self.cache.load_index(index_hash, embedder=embedder)

        logger.debug("Index cache miss: index_hash=%s", index_hash[:12])
        index = IndexFactory.create(profile.index, embedder=embedder)
        await index.add(nodes)

        if self.cache:
            await self.cache.save_index(index_hash, index)

        return index

    async def _run_trial_cases(
        self,
        profile: ExperimentProfile,
        retriever: Any,
        embedder: BaseEmbedder,
        trial_cases: list[TrialCase],
    ) -> list[CaseResult]:
        """Run all trial cases through retrieval (and optional generation)."""
        case_results: list[CaseResult] = []
        driver = None

        if profile.generator is not None:
            driver = await self._create_driver(profile)

        try:
            for tc in trial_cases:
                trace = await retriever.retrieve(tc.question)

                predicted_answer: str | None = None
                gen_result: GenerationResult | None = None

                if driver is not None:
                    context_mgr = ContextManager(driver)
                    context_chunks = [rn.node.content for rn in trace.retrieved_nodes]
                    system_msg = (
                        "You are a helpful AI assistant. "
                        "Answer the question based on the provided context."
                    )
                    prompt = context_mgr.fit_context(
                        system=system_msg,
                        context_chunks=context_chunks,
                        user_query=tc.question,
                        max_completion=512,
                    )
                    gen_result = await driver.generate(
                        prompt=prompt, max_tokens=512, temperature=0.7
                    )
                    if gen_result is not None:
                        predicted_answer = gen_result.text

                case_results.append(
                    CaseResult(
                        case_id=tc.case_id,
                        predicted_answer=predicted_answer,
                        trace=trace,
                        generation_result=gen_result,
                    )
                )
        finally:
            if driver is not None:
                await driver.__aexit__(None, None, None)

        return case_results

    async def _create_driver(self, profile: ExperimentProfile) -> Any:
        """Create an LLM driver from the profile's generator config."""
        if profile.generator is None:
            return None

        from ragmark.generation.drivers import LlamaCppDriver

        driver = LlamaCppDriver(
            model_path=profile.generator.model_path,
            n_ctx=profile.generator.context_window,
        )
        await driver.__aenter__()
        return driver

    @staticmethod
    def _collect_system_info() -> SystemInfo:
        """Gather system information for reproducibility."""
        import os
        import sys

        try:
            import ragmark

            version = getattr(ragmark, "__version__", "0.1.0")
        except Exception:
            version = "0.1.0"

        ram_gb = 0.0
        try:
            import psutil

            ram_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            ram_gb = 8.0

        return SystemInfo(
            python_version=sys.version,
            ragmark_version=version,
            platform=platform.platform(),
            cpu_count=os.cpu_count() or 1,
            ram_gb=round(ram_gb, 2),
            gpu_info=None,
        )
