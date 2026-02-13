"""Generation quality evaluation metrics using LLM-as-judge patterns.

Provides three metrics for evaluating RAG generation quality:
- Faithfulness: proportion of answer claims supported by context.
- AnswerRelevancy: semantic relevance between question and answer.
- Correctness: factual alignment of answer vs ground truth.

Each metric exposes an async ``compute_async`` method that delegates
evaluation to a judge LLM via ``BaseLLMDriver``.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ragmark.exceptions import EvaluationError
from ragmark.generation.prompts import PromptTemplate
from ragmark.logger import get_logger
from ragmark.metrics.base import EvaluationMetric
from ragmark.metrics.evaluation.judge_cache import JudgeCache
from ragmark.metrics.registry import MetricRegistry
from ragmark.schemas.evaluation import CaseResult, TrialCase

logger = get_logger(__name__)

if TYPE_CHECKING:
    from ragmark.generation.drivers import BaseLLMDriver
    from ragmark.index.embedders import BaseEmbedder


def _clamp(value: float) -> float:
    """Clamp a score to the [0.0, 1.0] interval."""
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# Default prompt templates
# ---------------------------------------------------------------------------

_DEFAULT_FAITHFULNESS_DECOMPOSE_TPL = PromptTemplate(
    template=(
        "You are a precise claim extractor. Given the following answer, "
        "decompose it into a list of independent atomic factual claims.\n\n"
        "Answer:\n{{ answer }}\n\n"
        'Respond with a JSON object: {"claims": ["claim1", "claim2", ...]}'
    ),
    input_variables=["answer"],
)

_DEFAULT_FAITHFULNESS_VERIFY_TPL = PromptTemplate(
    template=(
        "You are a faithful evaluator. Given a claim and supporting context, "
        "determine whether the claim is fully supported by the context.\n\n"
        "Claim: {{ claim }}\n\n"
        "Context:\n{{ context }}\n\n"
        "Respond with a JSON object: "
        '{"verdict": "supported"} or {"verdict": "unsupported"}'
    ),
    input_variables=["claim", "context"],
)

_DEFAULT_RELEVANCY_TPL = PromptTemplate(
    template=(
        "You are a relevancy evaluator. Your job is to assess whether the "
        "answer directly and specifically addresses the question asked.\n\n"
        "A highly relevant answer focuses on what was asked, provides the "
        "requested information, and does not drift into unrelated topics.\n\n"
        "Question: {{ question }}\n\n"
        "Answer:\n{{ answer }}\n\n"
        "Rate the relevancy on a scale from 0.0 to 1.0 where:\n"
        "- 1.0 = the answer directly and completely addresses the question\n"
        "- 0.5 = the answer partially addresses the question but includes "
        "irrelevant content or misses key aspects\n"
        "- 0.0 = the answer does not address the question at all\n\n"
        'Respond with a JSON object: {"score": <float between 0.0 and 1.0>}'
    ),
    input_variables=["question", "answer"],
)

_DEFAULT_CORRECTNESS_TPL = PromptTemplate(
    template=(
        "You are a correctness evaluator. Compare the predicted answer against "
        "the ground truth and assign a score from 0.0 (completely wrong) to "
        "1.0 (fully correct).\n\n"
        "Question: {{ question }}\n\n"
        "Ground Truth:\n{{ ground_truth }}\n\n"
        "Predicted Answer:\n{{ answer }}\n\n"
        'Respond with a JSON object: {"score": <float between 0.0 and 1.0>}'
    ),
    input_variables=["question", "ground_truth", "answer"],
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class JudgePromptConfig(BaseModel):
    """Configuration for LLM judge prompts and batching.

    Attributes:
        faithfulness_decompose_template: Template to decompose answer into claims.
        faithfulness_verify_template: Template to verify a claim against context.
        relevancy_template: Template for direct relevancy scoring.
        correctness_template: Template for answer correctness evaluation.
        batch_size: Maximum number of cases processed concurrently.
        relevancy_embedder_weight: Weight for embedder signal in relevancy (0-1).
    """

    model_config = ConfigDict(strict=True, extra="forbid")

    faithfulness_decompose_template: str = Field(
        default=_DEFAULT_FAITHFULNESS_DECOMPOSE_TPL.render(answer="{{ answer }}"),
        description="Jinja2 template for claim decomposition",
    )
    faithfulness_verify_template: str = Field(
        default=_DEFAULT_FAITHFULNESS_VERIFY_TPL.render(
            claim="{{ claim }}", context="{{ context }}"
        ),
        description="Jinja2 template for claim verification",
    )
    relevancy_template: str = Field(
        default=_DEFAULT_RELEVANCY_TPL.render(
            question="{{ question }}", answer="{{ answer }}"
        ),
        description="Jinja2 template for direct relevancy scoring",
    )
    correctness_template: str = Field(
        default=_DEFAULT_CORRECTNESS_TPL.render(
            question="{{ question }}",
            ground_truth="{{ ground_truth }}",
            answer="{{ answer }}",
        ),
        description="Jinja2 template for correctness evaluation",
    )
    batch_size: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Maximum cases per concurrent batch",
    )
    relevancy_embedder_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight of embedder cosine signal when combined with judge score",
    )


# ---------------------------------------------------------------------------
# Helper to call judge and parse JSON safely
# ---------------------------------------------------------------------------


async def _judge_call(
    judge: BaseLLMDriver,
    prompt: str,
    *,
    max_tokens: int = 512,
) -> dict[str, Any]:
    """Call the judge LLM and parse a JSON response.

    Args:
        judge: LLM driver to use for evaluation.
        prompt: Rendered prompt string.
        max_tokens: Maximum tokens for the response.

    Returns:
        Parsed JSON dictionary.

    Raises:
        EvaluationError: If generation or JSON parsing fails.
    """
    try:
        result = await judge.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        logger.error("Judge LLM call failed")
        logger.debug("Judge call error details: %s", exc, exc_info=True)
        raise EvaluationError(f"Judge LLM call failed: {exc}") from exc

    try:
        parsed: dict[str, Any] = json.loads(result.text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Judge response not valid JSON: response=%s",
            result.text[:200],
        )
        logger.debug("JSON parse error details: %s", exc, exc_info=True)
        raise EvaluationError(
            f"Judge returned invalid JSON: {result.text[:200]}"
        ) from exc

    return parsed


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------


@MetricRegistry.register
class Faithfulness(EvaluationMetric[Any, float]):
    """Measure if generated answer is grounded in provided context.

    Decomposes the answer into individual atomic claims via the judge LLM,
    then verifies each claim against the context passages. The final score
    is the proportion of supported claims, clamped to [0.0, 1.0].
    """

    @property
    def name(self) -> str:
        return "faithfulness"

    @property
    def description(self) -> str:
        return "Proportion of claims in answer supported by context"

    def compute(self, **inputs: Any) -> float:
        """Synchronous entry point for registry compatibility.

        Raises:
            EvaluationError: Always raised to direct callers to compute_async.
        """
        raise EvaluationError(
            "Faithfulness requires an LLM judge. Use compute_async() instead."
        )

    async def compute_async(
        self,
        answer: str,
        context: list[str],
        judge: BaseLLMDriver,
        config: JudgePromptConfig | None = None,
        cache: JudgeCache | None = None,
    ) -> float:
        """Compute faithfulness score asynchronously.

        Args:
            answer: Generated answer text.
            context: Retrieved context passages.
            judge: LLM driver for claim decomposition and verification.
            config: Optional prompt configuration overrides.
            cache: Optional result cache.

        Returns:
            Score between 0.0 and 1.0.

        Raises:
            EvaluationError: If judge calls fail.
        """
        if not answer or not answer.strip():
            logger.debug("Faithfulness: empty answer, returning 0.0")
            return 0.0

        if not context:
            logger.debug("Faithfulness: no context provided, returning 0.0")
            return 0.0

        cache_key = JudgeCache.make_key(answer, *context) if cache else None
        if cache and cache_key:
            cached = await cache.get(cache_key)
            if cached is not None:
                return cached

        cfg = config or JudgePromptConfig()

        # Step 1: Decompose answer into claims
        decompose_tpl = PromptTemplate(
            template=cfg.faithfulness_decompose_template,
            input_variables=["answer"],
        )
        decompose_prompt = decompose_tpl.render(answer=answer)
        decompose_result = await _judge_call(judge, decompose_prompt)

        claims: list[str] = decompose_result.get("claims", [])
        if not claims:
            logger.debug("Faithfulness: no claims extracted, returning 0.0")
            score = 0.0
            if cache and cache_key:
                await cache.put(cache_key, score)
            return score

        logger.debug("Faithfulness: extracted %d claims", len(claims))

        # Step 2: Verify each claim against context
        joined_context = "\n---\n".join(context)
        verify_tpl = PromptTemplate(
            template=cfg.faithfulness_verify_template,
            input_variables=["claim", "context"],
        )

        async def _verify_claim(claim: str) -> int:
            prompt = verify_tpl.render(claim=claim, context=joined_context)
            result = await _judge_call(judge, prompt, max_tokens=64)
            verdict = result.get("verdict", "unsupported").lower().strip()
            return 1 if verdict == "supported" else 0

        verdicts = await asyncio.gather(*[_verify_claim(c) for c in claims])

        score = _clamp(sum(verdicts) / len(claims))
        logger.debug(
            "Faithfulness computed: claims=%d, supported=%d, score=%.4f",
            len(claims),
            sum(verdicts),
            score,
        )

        if cache and cache_key:
            await cache.put(cache_key, score)

        return score


# ---------------------------------------------------------------------------
# AnswerRelevancy
# ---------------------------------------------------------------------------


@MetricRegistry.register
class AnswerRelevancy(EvaluationMetric[Any, float]):
    """Measure how well the generated answer addresses the question.

    Uses direct LLM-as-judge scoring as the primary signal: the judge
    evaluates whether the answer specifically addresses what was asked.
    When an embedder is available, a weighted cosine similarity between
    the question and answer embeddings is blended in as a supplementary
    signal (controlled by ``relevancy_embedder_weight``).
    """

    @property
    def name(self) -> str:
        return "answer_relevancy"

    @property
    def description(self) -> str:
        return "Semantic relevance between question and answer"

    def compute(self, **inputs: Any) -> float:
        """Synchronous entry point for registry compatibility.

        Raises:
            EvaluationError: Always raised to direct callers to compute_async.
        """
        raise EvaluationError(
            "AnswerRelevancy requires an LLM judge. " "Use compute_async() instead."
        )

    async def compute_async(
        self,
        question: str,
        answer: str,
        judge: BaseLLMDriver,
        embedder: BaseEmbedder | None = None,
        config: JudgePromptConfig | None = None,
        cache: JudgeCache | None = None,
    ) -> float:
        """Compute answer relevancy score asynchronously.

        The judge LLM directly rates how well the answer addresses the
        question. When an embedder is available, the judge score is blended
        with cosine similarity between question and answer embeddings.

        Args:
            question: Original question.
            answer: Generated answer text.
            judge: LLM driver for relevancy evaluation.
            embedder: Optional embedder for supplementary cosine signal.
            config: Optional prompt configuration overrides.
            cache: Optional result cache.

        Returns:
            Score between 0.0 and 1.0.

        Raises:
            EvaluationError: If judge call fails.
        """
        if not answer or not answer.strip():
            logger.debug("AnswerRelevancy: empty answer, returning 0.0")
            return 0.0

        cache_key = JudgeCache.make_key(question, answer) if cache else None
        if cache and cache_key:
            cached = await cache.get(cache_key)
            if cached is not None:
                return cached

        cfg = config or JudgePromptConfig()

        # Primary: direct LLM judge scoring
        judge_score = await self._score_with_judge(question, answer, judge, cfg)

        # Supplementary: embedder cosine similarity
        if embedder is not None and cfg.relevancy_embedder_weight > 0:
            embed_score = await self._score_with_embedder(question, answer, embedder)
            if embed_score is not None:
                weight = cfg.relevancy_embedder_weight
                score = _clamp((1.0 - weight) * judge_score + weight * embed_score)
                logger.debug(
                    "AnswerRelevancy (blended): judge=%.4f, embed=%.4f, "
                    "weight=%.2f, final=%.4f",
                    judge_score,
                    embed_score,
                    weight,
                    score,
                )
            else:
                score = _clamp(judge_score)
        else:
            score = _clamp(judge_score)

        if cache and cache_key:
            await cache.put(cache_key, score)

        return score

    async def _score_with_judge(
        self,
        question: str,
        answer: str,
        judge: BaseLLMDriver,
        config: JudgePromptConfig,
    ) -> float:
        """Ask the judge LLM to directly rate relevancy."""
        relevancy_tpl = PromptTemplate(
            template=config.relevancy_template,
            input_variables=["question", "answer"],
        )
        prompt = relevancy_tpl.render(question=question, answer=answer)
        result = await _judge_call(judge, prompt, max_tokens=64)

        raw_score = result.get("score", 0.0)
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "AnswerRelevancy: could not parse score=%s, defaulting to 0.0",
                raw_score,
            )
            logger.debug("Score parse error: %s", exc, exc_info=True)
            score = 0.0

        logger.debug("AnswerRelevancy (judge): score=%.4f", score)
        return score

    async def _score_with_embedder(
        self,
        question: str,
        answer: str,
        embedder: BaseEmbedder,
    ) -> float | None:
        """Compute cosine similarity between question and answer embeddings."""
        try:
            loop = asyncio.get_running_loop()
            embeddings = await loop.run_in_executor(
                None, embedder.embed, [question, answer]
            )
        except Exception as exc:
            logger.warning(
                "AnswerRelevancy: embedding failed, using judge-only: %s", exc
            )
            logger.debug("Embedding error details: %s", exc, exc_info=True)
            return None

        q_vec = np.array(embeddings[0], dtype=np.float32)
        a_vec = np.array(embeddings[1], dtype=np.float32)

        q_norm = float(np.linalg.norm(q_vec))
        a_norm = float(np.linalg.norm(a_vec))

        if q_norm == 0 or a_norm == 0:
            logger.warning("AnswerRelevancy: zero-norm embedding vector")
            return None

        cos_sim = float(np.dot(q_vec, a_vec) / (q_norm * a_norm))
        logger.debug("AnswerRelevancy (embedder): cosine_sim=%.4f", cos_sim)
        return _clamp(cos_sim)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


@MetricRegistry.register
class Correctness(EvaluationMetric[Any, float]):
    """Evaluate semantic correctness of answer versus ground truth.

    Uses the judge LLM to compare the predicted answer against the
    ground truth, producing a score reflecting factual alignment on
    a [0.0, 1.0] scale.
    """

    @property
    def name(self) -> str:
        return "correctness"

    @property
    def description(self) -> str:
        return "Factual alignment of answer vs ground truth"

    def compute(self, **inputs: Any) -> float:
        """Synchronous entry point for registry compatibility.

        Raises:
            EvaluationError: Always raised to direct callers to compute_async.
        """
        raise EvaluationError(
            "Correctness requires an LLM judge. Use compute_async() instead."
        )

    async def compute_async(
        self,
        answer: str,
        ground_truth: str,
        judge: BaseLLMDriver,
        question: str = "",
        config: JudgePromptConfig | None = None,
        cache: JudgeCache | None = None,
    ) -> float:
        """Compute correctness score asynchronously.

        Args:
            answer: Predicted answer.
            ground_truth: Expected answer.
            judge: LLM driver for evaluation.
            question: Original question for additional context.
            config: Optional prompt configuration overrides.
            cache: Optional result cache.

        Returns:
            Score between 0.0 and 1.0.

        Raises:
            EvaluationError: If judge call fails.
        """
        if not answer or not answer.strip():
            logger.debug("Correctness: empty answer, returning 0.0")
            return 0.0

        cache_key = (
            JudgeCache.make_key(answer, ground_truth, question) if cache else None
        )
        if cache and cache_key:
            cached = await cache.get(cache_key)
            if cached is not None:
                return cached

        cfg = config or JudgePromptConfig()

        correctness_tpl = PromptTemplate(
            template=cfg.correctness_template,
            input_variables=["question", "ground_truth", "answer"],
        )
        prompt = correctness_tpl.render(
            question=question, ground_truth=ground_truth, answer=answer
        )
        result = await _judge_call(judge, prompt, max_tokens=64)

        raw_score = result.get("score", 0.0)
        try:
            score = _clamp(float(raw_score))
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Correctness: could not parse score=%s, defaulting to 0.0",
                raw_score,
            )
            logger.debug("Score parse error: %s", exc, exc_info=True)
            score = 0.0

        logger.debug("Correctness computed: score=%.4f", score)

        if cache and cache_key:
            await cache.put(cache_key, score)

        return score


# ---------------------------------------------------------------------------
# Batch computation
# ---------------------------------------------------------------------------


async def compute_generation_metrics(
    results: list[CaseResult],
    ground_truth: dict[str, TrialCase],
    judge: BaseLLMDriver,
    config: JudgePromptConfig | None = None,
    embedder: BaseEmbedder | None = None,
    cache: JudgeCache | None = None,
) -> dict[str, float]:
    """Compute generation quality metrics across multiple cases.

    Updates each CaseResult.case_metrics in-place with per-case generation
    scores, then returns aggregated (mean) metrics across all valid cases.
    Batch ordering of input results is preserved.

    Args:
        results: Case results with predicted answers and retrieval traces.
        ground_truth: Mapping from case_id to TrialCase (for ground_truth_answer).
        judge: LLM driver for judge evaluations.
        config: Optional prompt configuration.
        embedder: Optional embedder for answer relevancy cosine supplement.
        cache: Optional judge result cache.

    Returns:
        Aggregated metrics dict (e.g., {"faithfulness": 0.82, ...}).

    Raises:
        EvaluationError: If metric computation fails.
    """
    cfg = config or JudgePromptConfig()

    if not results:
        logger.warning(
            "Generation metrics computation skipped: no case results provided"
        )
        return {}

    logger.info(
        "Generation metrics computation started: cases=%d, batch_size=%d",
        len(results),
        cfg.batch_size,
    )

    faithfulness_metric = Faithfulness()
    relevancy_metric = AnswerRelevancy()
    correctness_metric = Correctness()

    metric_sums: dict[str, float] = {}
    valid_cases = 0
    skipped = 0

    for batch_start in range(0, len(results), cfg.batch_size):
        batch = results[batch_start : batch_start + cfg.batch_size]
        tasks: list[asyncio.Task[dict[str, float]]] = []

        for case_result in batch:
            tasks.append(
                asyncio.create_task(
                    _compute_single_case(
                        case_result=case_result,
                        ground_truth=ground_truth,
                        judge=judge,
                        faithfulness=faithfulness_metric,
                        relevancy=relevancy_metric,
                        correctness=correctness_metric,
                        embedder=embedder,
                        config=cfg,
                        cache=cache,
                    )
                )
            )

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for case_result, batch_result in zip(batch, batch_results, strict=False):
            if isinstance(batch_result, BaseException):
                skipped += 1
                logger.error(
                    "Generation metrics failed: case_id=%s, reason=%s",
                    case_result.case_id,
                    batch_result,
                )
                logger.debug(
                    "Case failure details: case_id=%s",
                    case_result.case_id,
                    exc_info=batch_result,
                )
                continue

            case_result.case_metrics.update(batch_result)
            for metric_name, value in batch_result.items():
                metric_sums[metric_name] = metric_sums.get(metric_name, 0.0) + value
            valid_cases += 1

    if valid_cases == 0:
        logger.warning(
            "Generation metrics computation failed: valid_cases=0, skipped=%d",
            skipped,
        )
        return {}

    aggregated = {metric: total / valid_cases for metric, total in metric_sums.items()}

    logger.info(
        "Generation metrics computation completed: valid_cases=%d, "
        "skipped=%d, metrics=%s",
        valid_cases,
        skipped,
        list(aggregated.keys()),
    )

    return aggregated


async def _compute_single_case(
    case_result: CaseResult,
    ground_truth: dict[str, TrialCase],
    judge: BaseLLMDriver,
    faithfulness: Faithfulness,
    relevancy: AnswerRelevancy,
    correctness: Correctness,
    embedder: BaseEmbedder | None,
    config: JudgePromptConfig,
    cache: JudgeCache | None,
) -> dict[str, float]:
    """Compute all generation metrics for a single case result."""
    metrics: dict[str, float] = {}

    predicted = case_result.predicted_answer
    if not predicted:
        logger.debug(
            "Case skipped: case_id=%s, reason=no_predicted_answer",
            case_result.case_id,
        )
        return metrics

    trial = ground_truth.get(case_result.case_id)
    question = case_result.trace.query

    # Extract context from retrieved nodes
    context = [node.node.content for node in case_result.trace.retrieved_nodes]

    # Faithfulness
    try:
        metrics["faithfulness"] = await faithfulness.compute_async(
            answer=predicted,
            context=context,
            judge=judge,
            config=config,
            cache=cache,
        )
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError(
            f"Faithfulness computation failed for case {case_result.case_id}"
        ) from exc

    # Answer Relevancy
    try:
        metrics["answer_relevancy"] = await relevancy.compute_async(
            question=question,
            answer=predicted,
            judge=judge,
            embedder=embedder,
            config=config,
            cache=cache,
        )
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError(
            f"AnswerRelevancy computation failed for case {case_result.case_id}"
        ) from exc

    # Correctness (only if ground truth answer is available)
    if trial and trial.ground_truth_answer:
        try:
            metrics["correctness"] = await correctness.compute_async(
                answer=predicted,
                ground_truth=trial.ground_truth_answer,
                judge=judge,
                question=question,
                config=config,
                cache=cache,
            )
        except EvaluationError:
            raise
        except Exception as exc:
            raise EvaluationError(
                f"Correctness computation failed for case {case_result.case_id}"
            ) from exc

    logger.debug(
        "Case generation metrics computed: case_id=%s, metrics=%s",
        case_result.case_id,
        metrics,
    )

    return metrics
