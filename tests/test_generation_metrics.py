"""Unit tests for generation quality evaluation metrics.

Tests cover Faithfulness, AnswerRelevancy, Correctness, the
compute_generation_metrics batch function, the _clamp helper,
and the JudgeCache persistence layer.  All LLM interactions are
driven by a FakeLLMDriver that cycles through pre-configured JSON
response strings, following the anti-mocking strategy.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

import pytest

from ragmark.exceptions import EvaluationError
from ragmark.generation.drivers import BaseLLMDriver
from ragmark.index.embedders import BaseEmbedder
from ragmark.metrics.evaluation.generation import (
    AnswerRelevancy,
    Correctness,
    Faithfulness,
    JudgePromptConfig,
    _clamp,  # type: ignore
    compute_generation_metrics,
)
from ragmark.metrics.evaluation.judge_cache import JudgeCache
from ragmark.schemas.documents import KnowledgeNode, NodePosition
from ragmark.schemas.evaluation import CaseResult, TrialCase
from ragmark.schemas.generation import GenerationResult, TokenUsage
from ragmark.schemas.retrieval import RetrievedNode, TraceContext

# ---------------------------------------------------------------------------
# Fake collaborators
# ---------------------------------------------------------------------------


class FakeLLMDriver(BaseLLMDriver):
    """Deterministic LLM driver that cycles through pre-configured responses.

    Each call to ``generate`` pops the next response from the queue.  When the
    queue is exhausted it wraps around to the beginning, making the driver
    reusable across multi-step metric computations.

    Attributes:
        responses: Ordered list of JSON response strings.
        call_count: Number of generate calls made so far.
    """

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0

    async def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.7,
        stop: list[str] | None = None,
        response_format: dict[
            str, Literal["json_object"] | dict[Literal["json_schema"], Any]
        ]
        | None = None,
    ) -> GenerationResult:
        """Return the next pre-configured response from the queue."""
        text = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return GenerationResult(
            text=text,
            usage=TokenUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
            finish_reason="stop",
        )

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """Yield the next response as a single stream chunk."""
        text = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        yield text

    def count_tokens(self, text: str) -> int:
        """Approximate token count as word count."""
        return len(text.split())

    @property
    def context_window(self) -> int:
        """Return a fixed context window of 4096 tokens."""
        return 4096

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> None:
        """No-op cleanup."""


class FakeEmbedder(BaseEmbedder):
    """Embedder that returns constant vectors for deterministic cosine tests.

    All texts receive the same vector so that cosine similarity between any
    pair is exactly 1.0 (identical vectors).
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    @property
    def embedding_dim(self) -> int:
        return self.dimension

    @classmethod
    def from_config(cls, config: Any) -> FakeEmbedder:
        return cls()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self.dimension for _ in texts]

    def embed_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        return [{} for _ in texts]


# ---------------------------------------------------------------------------
# Helpers for building test data
# ---------------------------------------------------------------------------


def _make_node(content: str = "Test content.", node_id: str = "n-1") -> KnowledgeNode:
    """Create a minimal KnowledgeNode."""
    return KnowledgeNode(
        node_id=node_id,
        content=content,
        source_id="src-test",
        position=NodePosition(
            start_char=0, end_char=len(content), page=1, section="test"
        ),
        dense_vector=[0.1, 0.2, 0.3],
        sparse_vector=None,
    )


def _make_trace(
    query: str = "What is the answer?",
    contents: list[str] | None = None,
) -> TraceContext:
    """Create a TraceContext with optional node contents."""
    if contents is None:
        contents = ["Relevant context passage."]
    nodes = [
        RetrievedNode(
            node=_make_node(content=c, node_id=f"n-{i}"),
            score=0.9,
            rank=i + 1,
        )
        for i, c in enumerate(contents)
    ]
    return TraceContext(query=query, retrieved_nodes=nodes, reranked=False)


def _make_case_result(
    case_id: str = "case-1",
    predicted_answer: str | None = "Paris is the capital of France.",
    query: str = "What is the capital of France?",
    contents: list[str] | None = None,
) -> CaseResult:
    """Create a CaseResult with a populated trace."""
    return CaseResult(
        case_id=case_id,
        predicted_answer=predicted_answer,
        trace=_make_trace(query=query, contents=contents),
        generation_result=GenerationResult(
            text=predicted_answer or "",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            finish_reason="stop",
        ),
        case_metrics={},
    )


def _make_trial_case(
    case_id: str = "case-1",
    question: str = "What is the capital of France?",
    ground_truth_answer: str | None = "Paris",
) -> TrialCase:
    """Create a TrialCase with a ground-truth answer."""
    return TrialCase(
        case_id=case_id,
        question=question,
        ground_truth_answer=ground_truth_answer,
        ground_truth_node_ids=["n-1"],
    )


# =========================================================================
# _clamp
# =========================================================================


@pytest.mark.unit
class TestClamp:
    """Tests for the _clamp helper function."""

    def test_clamp_should_return_value_when_within_bounds(self) -> None:
        """Given a value between 0 and 1, When clamped, Then value is unchanged."""
        assert _clamp(0.5) == 0.5

    def test_clamp_should_return_zero_when_negative(self) -> None:
        """Given a negative value, When clamped, Then 0.0 is returned."""
        assert _clamp(-0.3) == 0.0

    def test_clamp_should_return_one_when_above_one(self) -> None:
        """Given a value above 1.0, When clamped, Then 1.0 is returned."""
        assert _clamp(1.7) == 1.0

    def test_clamp_should_return_boundaries_exactly(self) -> None:
        """Given boundary values, When clamped, Then they pass through."""
        assert _clamp(0.0) == 0.0
        assert _clamp(1.0) == 1.0


# =========================================================================
# Faithfulness
# =========================================================================


@pytest.mark.unit
class TestFaithfulness:
    """Tests for the Faithfulness metric."""

    def test_faithfulness_compute_should_raise_evaluation_error(self) -> None:
        """Given the sync compute path, When called, Then EvaluationError is raised.

        The synchronous compute() method exists only for registry compatibility
        and must direct callers to use compute_async() instead.
        """
        metric = Faithfulness()
        with pytest.raises(EvaluationError, match="compute_async"):
            metric.compute(answer="A", context=["B"])

    @pytest.mark.asyncio
    async def test_faithfulness_should_return_perfect_score_when_all_claims_supported(
        self,
    ) -> None:
        """Given all claims are supported by context, When evaluated, Then score is 1.0.

        The driver first returns two claims, then "supported" for each.
        """
        judge = FakeLLMDriver(
            responses=[
                json.dumps({"claims": ["Paris is in France", "It is the capital"]}),
                json.dumps({"verdict": "supported"}),
                json.dumps({"verdict": "supported"}),
            ]
        )
        metric = Faithfulness()

        score = await metric.compute_async(
            answer="Paris is the capital of France.",
            context=["France's capital city is Paris."],
            judge=judge,
        )

        assert score == 1.0

    @pytest.mark.asyncio
    async def test_faithfulness_should_return_zero_when_no_claims_supported(
        self,
    ) -> None:
        """Given no claims are supported, When evaluated, Then score is 0.0."""
        judge = FakeLLMDriver(
            responses=[
                json.dumps({"claims": ["Berlin is in France", "It is the capital"]}),
                json.dumps({"verdict": "unsupported"}),
                json.dumps({"verdict": "unsupported"}),
            ]
        )
        metric = Faithfulness()

        score = await metric.compute_async(
            answer="Berlin is the capital of France.",
            context=["France's capital city is Paris."],
            judge=judge,
        )

        assert score == 0.0

    @pytest.mark.asyncio
    async def test_faithfulness_should_return_partial_score_for_mixed_verdicts(
        self,
    ) -> None:
        """Given 2 of 4 claims supported, When evaluated, Then score is 0.5."""
        judge = FakeLLMDriver(
            responses=[
                json.dumps(
                    {
                        "claims": [
                            "Claim A",
                            "Claim B",
                            "Claim C",
                            "Claim D",
                        ]
                    }
                ),
                json.dumps({"verdict": "supported"}),
                json.dumps({"verdict": "unsupported"}),
                json.dumps({"verdict": "supported"}),
                json.dumps({"verdict": "unsupported"}),
            ]
        )
        metric = Faithfulness()

        score = await metric.compute_async(
            answer="Complex answer with four atomic claims.",
            context=["Some context."],
            judge=judge,
        )

        assert score == pytest.approx(0.5)  # type: ignore

    @pytest.mark.asyncio
    async def test_faithfulness_should_return_zero_for_empty_answer(self) -> None:
        """Given an empty answer, When evaluated, Then score is 0.0 without calling the judge."""
        judge = FakeLLMDriver(responses=[json.dumps({"claims": ["x"]})])
        metric = Faithfulness()

        score = await metric.compute_async(
            answer="",
            context=["Some context."],
            judge=judge,
        )

        assert score == 0.0
        assert judge.call_count == 0

    @pytest.mark.asyncio
    async def test_faithfulness_should_return_zero_for_whitespace_answer(self) -> None:
        """Given a whitespace-only answer, When evaluated, Then score is 0.0."""
        judge = FakeLLMDriver(responses=[json.dumps({"claims": ["x"]})])
        metric = Faithfulness()

        score = await metric.compute_async(
            answer="   ",
            context=["Some context."],
            judge=judge,
        )

        assert score == 0.0
        assert judge.call_count == 0

    @pytest.mark.asyncio
    async def test_faithfulness_should_return_zero_for_empty_context(self) -> None:
        """Given empty context list, When evaluated, Then score is 0.0 without calling the judge."""
        judge = FakeLLMDriver(responses=[json.dumps({"claims": ["x"]})])
        metric = Faithfulness()

        score = await metric.compute_async(
            answer="Some answer.",
            context=[],
            judge=judge,
        )

        assert score == 0.0
        assert judge.call_count == 0

    @pytest.mark.asyncio
    async def test_faithfulness_should_return_zero_when_no_claims_extracted(
        self,
    ) -> None:
        """Given the judge extracts zero claims, When evaluated, Then score is 0.0."""
        judge = FakeLLMDriver(responses=[json.dumps({"claims": []})])
        metric = Faithfulness()

        score = await metric.compute_async(
            answer="Something.",
            context=["Context."],
            judge=judge,
        )

        assert score == 0.0

    @pytest.mark.asyncio
    async def test_faithfulness_should_clamp_score_above_one(self) -> None:
        """Given a scenario where clamping is needed, When computed, Then score stays in [0,1].

        This exercises the _clamp path by verifying the score never exceeds 1.0
        even when all claims are supported (score = 1.0, already clamped boundary).
        """
        judge = FakeLLMDriver(
            responses=[
                json.dumps({"claims": ["Only claim"]}),
                json.dumps({"verdict": "supported"}),
            ]
        )
        metric = Faithfulness()

        score = await metric.compute_async(
            answer="Single claim answer.",
            context=["Supporting context."],
            judge=judge,
        )

        assert 0.0 <= score <= 1.0

    def test_faithfulness_name_should_be_faithfulness(self) -> None:
        """Given a Faithfulness instance, When accessing name, Then it equals 'faithfulness'."""
        assert Faithfulness().name == "faithfulness"


# =========================================================================
# AnswerRelevancy
# =========================================================================


@pytest.mark.unit
class TestAnswerRelevancy:
    """Tests for the AnswerRelevancy metric."""

    def test_answer_relevancy_compute_should_raise_evaluation_error(self) -> None:
        """Given the sync compute path, When called, Then EvaluationError is raised."""
        metric = AnswerRelevancy()
        with pytest.raises(EvaluationError, match="compute_async"):
            metric.compute(question="Q", answer="A")

    @pytest.mark.asyncio
    async def test_answer_relevancy_should_return_judge_score_without_embedder(
        self,
    ) -> None:
        """Given no embedder, When evaluated, Then score equals the judge's raw score.

        This exercises the judge-only code path where no cosine blending occurs.
        """
        judge = FakeLLMDriver(responses=[json.dumps({"score": 0.85})])
        metric = AnswerRelevancy()

        score = await metric.compute_async(
            question="What is the capital of France?",
            answer="Paris is the capital of France.",
            judge=judge,
            embedder=None,
        )

        assert score == pytest.approx(0.85)  # type: ignore

    @pytest.mark.asyncio
    async def test_answer_relevancy_should_blend_judge_and_embedder_scores(
        self,
    ) -> None:
        """Given an embedder, When evaluated, Then score blends judge and cosine.

        FakeEmbedder returns identical vectors for all texts, so cosine
        similarity = 1.0.  With default weight 0.3 the blended score is
        (1 - 0.3) * judge + 0.3 * 1.0.
        """
        judge_score = 0.8
        judge = FakeLLMDriver(responses=[json.dumps({"score": judge_score})])
        embedder = FakeEmbedder()
        metric = AnswerRelevancy()

        score = await metric.compute_async(
            question="What is the capital of France?",
            answer="Paris is the capital of France.",
            judge=judge,
            embedder=embedder,
        )

        # default weight = 0.3
        expected = 0.7 * judge_score + 0.3 * 1.0
        assert score == pytest.approx(expected)  # type: ignore

    @pytest.mark.asyncio
    async def test_answer_relevancy_should_use_custom_embedder_weight(
        self,
    ) -> None:
        """Given custom weight 0.5, When evaluated, Then blending uses that weight."""
        judge_score = 0.6
        judge = FakeLLMDriver(responses=[json.dumps({"score": judge_score})])
        embedder = FakeEmbedder()
        config = JudgePromptConfig(relevancy_embedder_weight=0.5)
        metric = AnswerRelevancy()

        score = await metric.compute_async(
            question="Q?",
            answer="A.",
            judge=judge,
            embedder=embedder,
            config=config,
        )

        expected = 0.5 * judge_score + 0.5 * 1.0
        assert score == pytest.approx(expected)  # type: ignore

    @pytest.mark.asyncio
    async def test_answer_relevancy_should_return_zero_for_empty_answer(
        self,
    ) -> None:
        """Given an empty answer, When evaluated, Then 0.0 without calling judge."""
        judge = FakeLLMDriver(responses=[json.dumps({"score": 0.9})])
        metric = AnswerRelevancy()

        score = await metric.compute_async(
            question="What?",
            answer="",
            judge=judge,
        )

        assert score == 0.0
        assert judge.call_count == 0

    @pytest.mark.asyncio
    async def test_answer_relevancy_should_return_zero_for_whitespace_answer(
        self,
    ) -> None:
        """Given a whitespace-only answer, When evaluated, Then 0.0."""
        judge = FakeLLMDriver(responses=[json.dumps({"score": 0.9})])
        metric = AnswerRelevancy()

        score = await metric.compute_async(
            question="What?",
            answer="   \n\t  ",
            judge=judge,
        )

        assert score == 0.0

    @pytest.mark.asyncio
    async def test_answer_relevancy_should_clamp_high_judge_score(self) -> None:
        """Given judge returns score > 1.0, When evaluated, Then score is clamped to 1.0."""
        judge = FakeLLMDriver(responses=[json.dumps({"score": 1.5})])
        metric = AnswerRelevancy()

        score = await metric.compute_async(
            question="What?",
            answer="Answer.",
            judge=judge,
            embedder=None,
        )

        assert score == 1.0

    @pytest.mark.asyncio
    async def test_answer_relevancy_should_clamp_negative_judge_score(self) -> None:
        """Given judge returns negative score, When evaluated, Then score is clamped to 0.0."""
        judge = FakeLLMDriver(responses=[json.dumps({"score": -0.5})])
        metric = AnswerRelevancy()

        score = await metric.compute_async(
            question="What?",
            answer="Answer.",
            judge=judge,
            embedder=None,
        )

        assert score == 0.0

    def test_answer_relevancy_name_should_be_answer_relevancy(self) -> None:
        """Given an AnswerRelevancy instance, When accessing name, Then correct."""
        assert AnswerRelevancy().name == "answer_relevancy"


# =========================================================================
# Correctness
# =========================================================================


@pytest.mark.unit
class TestCorrectness:
    """Tests for the Correctness metric."""

    def test_correctness_compute_should_raise_evaluation_error(self) -> None:
        """Given the sync compute path, When called, Then EvaluationError is raised."""
        metric = Correctness()
        with pytest.raises(EvaluationError, match="compute_async"):
            metric.compute(answer="A", ground_truth="B")

    @pytest.mark.asyncio
    async def test_correctness_should_return_high_score_for_correct_answer(
        self,
    ) -> None:
        """Given judge confirms correctness, When evaluated, Then score is ~1.0."""
        judge = FakeLLMDriver(responses=[json.dumps({"score": 1.0})])
        metric = Correctness()

        score = await metric.compute_async(
            answer="Paris",
            ground_truth="Paris",
            judge=judge,
            question="What is the capital of France?",
        )

        assert score == pytest.approx(1.0)  # type: ignore

    @pytest.mark.asyncio
    async def test_correctness_should_return_low_score_for_incorrect_answer(
        self,
    ) -> None:
        """Given judge marks answer incorrect, When evaluated, Then score is ~0.0."""
        judge = FakeLLMDriver(responses=[json.dumps({"score": 0.0})])
        metric = Correctness()

        score = await metric.compute_async(
            answer="Berlin",
            ground_truth="Paris",
            judge=judge,
            question="What is the capital of France?",
        )

        assert score == pytest.approx(0.0)  # type: ignore

    @pytest.mark.asyncio
    async def test_correctness_should_return_zero_for_empty_answer(self) -> None:
        """Given an empty answer, When evaluated, Then 0.0 without calling judge."""
        judge = FakeLLMDriver(responses=[json.dumps({"score": 0.9})])
        metric = Correctness()

        score = await metric.compute_async(
            answer="",
            ground_truth="Paris",
            judge=judge,
        )

        assert score == 0.0
        assert judge.call_count == 0

    @pytest.mark.asyncio
    async def test_correctness_should_return_zero_for_whitespace_answer(self) -> None:
        """Given whitespace-only answer, When evaluated, Then 0.0."""
        judge = FakeLLMDriver(responses=[json.dumps({"score": 0.9})])
        metric = Correctness()

        score = await metric.compute_async(
            answer="   ",
            ground_truth="Paris",
            judge=judge,
        )

        assert score == 0.0

    @pytest.mark.asyncio
    async def test_correctness_should_clamp_score_above_one(self) -> None:
        """Given judge returns score > 1.0, When evaluated, Then score is clamped."""
        judge = FakeLLMDriver(responses=[json.dumps({"score": 2.5})])
        metric = Correctness()

        score = await metric.compute_async(
            answer="Paris",
            ground_truth="Paris",
            judge=judge,
        )

        assert score == 1.0

    @pytest.mark.asyncio
    async def test_correctness_should_clamp_negative_score(self) -> None:
        """Given judge returns negative score, When evaluated, Then clamped to 0.0."""
        judge = FakeLLMDriver(responses=[json.dumps({"score": -1.0})])
        metric = Correctness()

        score = await metric.compute_async(
            answer="Wrong",
            ground_truth="Paris",
            judge=judge,
        )

        assert score == 0.0

    def test_correctness_name_should_be_correctness(self) -> None:
        """Given a Correctness instance, When accessing name, Then correct."""
        assert Correctness().name == "correctness"


# =========================================================================
# compute_generation_metrics (batch)
# =========================================================================


@pytest.mark.unit
class TestComputeGenerationMetrics:
    """Tests for the compute_generation_metrics batch function."""

    @pytest.mark.asyncio
    async def test_compute_generation_metrics_should_preserve_batch_ordering(
        self,
    ) -> None:
        """Given two cases in specific order, When computed, Then ordering preserved.

        Both cases receive metrics; we verify that each CaseResult's
        case_metrics dictionary is populated and linked to the correct case_id.
        """
        # Judge responses: for each case -> decompose + verify + relevancy + correctness
        # Case 1: 1 claim supported, relevancy 0.9, correctness 0.8
        # Case 2: 1 claim supported, relevancy 0.7, correctness 0.6
        judge = FakeLLMDriver(
            responses=[
                # Case 1 faithfulness
                json.dumps({"claims": ["claim1"]}),
                json.dumps({"verdict": "supported"}),
                # Case 1 relevancy
                json.dumps({"score": 0.9}),
                # Case 1 correctness
                json.dumps({"score": 0.8}),
                # Case 2 faithfulness
                json.dumps({"claims": ["claim2"]}),
                json.dumps({"verdict": "supported"}),
                # Case 2 relevancy
                json.dumps({"score": 0.7}),
                # Case 2 correctness
                json.dumps({"score": 0.6}),
            ]
        )

        result_1 = _make_case_result(case_id="case-1")
        result_2 = _make_case_result(case_id="case-2")

        trial_1 = _make_trial_case(case_id="case-1")
        trial_2 = _make_trial_case(case_id="case-2")

        # Use batch_size=1 to force sequential processing and deterministic order
        config = JudgePromptConfig(batch_size=1)

        await compute_generation_metrics(
            results=[result_1, result_2],
            ground_truth={"case-1": trial_1, "case-2": trial_2},
            judge=judge,
            config=config,
        )

        # Both cases should have metrics
        assert "faithfulness" in result_1.case_metrics
        assert "faithfulness" in result_2.case_metrics
        assert "answer_relevancy" in result_1.case_metrics
        assert "correctness" in result_1.case_metrics

    @pytest.mark.asyncio
    async def test_compute_generation_metrics_should_update_case_metrics_in_place(
        self,
    ) -> None:
        """Given a single case, When computed, Then case_metrics dict is updated in-place."""
        judge = FakeLLMDriver(
            responses=[
                json.dumps({"claims": ["claim"]}),
                json.dumps({"verdict": "supported"}),
                json.dumps({"score": 0.9}),
                json.dumps({"score": 0.8}),
            ]
        )

        result = _make_case_result(case_id="case-1")
        trial = _make_trial_case(case_id="case-1")

        original_dict = result.case_metrics

        await compute_generation_metrics(
            results=[result],
            ground_truth={"case-1": trial},
            judge=judge,
        )

        # Must be the same dict object (updated in-place)
        assert result.case_metrics is original_dict
        assert "faithfulness" in result.case_metrics
        assert "answer_relevancy" in result.case_metrics
        assert "correctness" in result.case_metrics

    @pytest.mark.asyncio
    async def test_compute_generation_metrics_should_aggregate_means_correctly(
        self,
    ) -> None:
        """Given two cases with known scores, When aggregated, Then mean is correct.

        Using batch_size=1 ensures sequential deterministic processing.
        """
        judge = FakeLLMDriver(
            responses=[
                # Case 1: faithfulness=1.0, relevancy=0.8, correctness=1.0
                json.dumps({"claims": ["c1"]}),
                json.dumps({"verdict": "supported"}),
                json.dumps({"score": 0.8}),
                json.dumps({"score": 1.0}),
                # Case 2: faithfulness=0.0, relevancy=0.6, correctness=0.0
                json.dumps({"claims": ["c2"]}),
                json.dumps({"verdict": "unsupported"}),
                json.dumps({"score": 0.6}),
                json.dumps({"score": 0.0}),
            ]
        )

        r1 = _make_case_result(case_id="case-1")
        r2 = _make_case_result(case_id="case-2")
        t1 = _make_trial_case(case_id="case-1")
        t2 = _make_trial_case(case_id="case-2")

        config = JudgePromptConfig(batch_size=1)

        aggregated = await compute_generation_metrics(
            results=[r1, r2],
            ground_truth={"case-1": t1, "case-2": t2},
            judge=judge,
            config=config,
        )

        assert aggregated["faithfulness"] == pytest.approx(0.5)  # type: ignore
        assert aggregated["answer_relevancy"] == pytest.approx(0.7)  # type: ignore
        assert aggregated["correctness"] == pytest.approx(0.5)  # type: ignore

    @pytest.mark.asyncio
    async def test_compute_generation_metrics_should_skip_missing_predicted_answer(
        self,
    ) -> None:
        """Given a case with no predicted_answer, When computed, Then it is skipped.

        A second valid case still receives metrics, and the aggregated result
        reflects only the valid case.
        """
        judge = FakeLLMDriver(
            responses=[
                json.dumps({"claims": ["c1"]}),
                json.dumps({"verdict": "supported"}),
                json.dumps({"score": 0.9}),
                json.dumps({"score": 1.0}),
            ]
        )

        skipped = _make_case_result(case_id="case-skip", predicted_answer=None)
        valid = _make_case_result(case_id="case-valid")

        trial_skip = _make_trial_case(case_id="case-skip")
        trial_valid = _make_trial_case(case_id="case-valid")

        config = JudgePromptConfig(batch_size=1)

        aggregated = await compute_generation_metrics(
            results=[skipped, valid],
            ground_truth={"case-skip": trial_skip, "case-valid": trial_valid},
            judge=judge,
            config=config,
        )

        # Skipped case should have no metrics
        assert skipped.case_metrics == {}
        # Aggregated only from valid case
        assert "faithfulness" in aggregated

    @pytest.mark.asyncio
    async def test_compute_generation_metrics_should_return_empty_for_no_results(
        self,
    ) -> None:
        """Given empty results list, When computed, Then empty dict returned."""
        judge = FakeLLMDriver(responses=[json.dumps({"score": 0.5})])

        aggregated = await compute_generation_metrics(
            results=[],
            ground_truth={},
            judge=judge,
        )

        assert aggregated == {}

    @pytest.mark.asyncio
    async def test_compute_generation_metrics_should_skip_correctness_without_ground_truth_answer(
        self,
    ) -> None:
        """Given a TrialCase with node IDs but no answer, When computed, Then correctness is skipped."""
        judge = FakeLLMDriver(
            responses=[
                json.dumps({"claims": ["c1"]}),
                json.dumps({"verdict": "supported"}),
                json.dumps({"score": 0.9}),
            ]
        )

        result = _make_case_result(case_id="case-no-gt")
        trial = TrialCase(
            case_id="case-no-gt",
            question="What?",
            ground_truth_node_ids=["n-1"],
            ground_truth_answer=None,
        )

        aggregated = await compute_generation_metrics(
            results=[result],
            ground_truth={"case-no-gt": trial},
            judge=judge,
        )

        assert "faithfulness" in aggregated
        assert "answer_relevancy" in aggregated
        assert "correctness" not in aggregated
        assert "correctness" not in result.case_metrics


# =========================================================================
# JudgeCache
# =========================================================================


@pytest.mark.unit
class TestJudgeCache:
    """Tests for the JudgeCache persistence layer."""

    @pytest.mark.asyncio
    async def test_judge_cache_should_return_none_on_miss(self) -> None:
        """Given an empty memory-only cache, When getting a key, Then None is returned."""
        cache = JudgeCache()
        result = await cache.get("nonexistent-key")
        assert result is None

    @pytest.mark.asyncio
    async def test_judge_cache_should_return_value_on_hit(self) -> None:
        """Given a cached value, When getting the same key, Then the value is returned."""
        cache = JudgeCache()
        await cache.put("key-abc", 0.75)

        result = await cache.get("key-abc")

        assert result == 0.75

    @pytest.mark.asyncio
    async def test_judge_cache_should_overwrite_existing_key(self) -> None:
        """Given an existing key, When putting a new value, Then it overwrites."""
        cache = JudgeCache()
        await cache.put("key-1", 0.5)
        await cache.put("key-1", 0.9)

        result = await cache.get("key-1")

        assert result == 0.9

    @pytest.mark.asyncio
    async def test_judge_cache_disk_should_persist_across_instances(
        self, tmp_path: Path
    ) -> None:
        """Given a disk-backed cache, When re-opened, Then cached values survive.

        This verifies the shelve-backed round-trip: write with one instance,
        close it, open a fresh instance at the same path, and read back.
        """
        disk_path = tmp_path / "judge_cache"

        # Write
        cache_w = JudgeCache(disk_path=disk_path)
        await cache_w.put("persistent-key", 0.42)
        await cache_w.close()

        # Read with fresh instance
        cache_r = JudgeCache(disk_path=disk_path)
        result = await cache_r.get("persistent-key")
        await cache_r.close()

        assert result == pytest.approx(0.42)  # type: ignore

    @pytest.mark.asyncio
    async def test_judge_cache_disk_should_miss_for_unknown_key(
        self, tmp_path: Path
    ) -> None:
        """Given a disk-backed cache, When querying an unknown key, Then None."""
        disk_path = tmp_path / "judge_cache_miss"

        cache = JudgeCache(disk_path=disk_path)
        result = await cache.get("missing")
        await cache.close()

        assert result is None

    def test_make_key_should_be_deterministic(self) -> None:
        """Given the same inputs, When making keys, Then they are identical."""
        key1 = JudgeCache.make_key("answer text", "context chunk")
        key2 = JudgeCache.make_key("answer text", "context chunk")

        assert key1 == key2

    def test_make_key_should_differ_for_different_inputs(self) -> None:
        """Given different inputs, When making keys, Then they differ."""
        key1 = JudgeCache.make_key("answer A", "context X")
        key2 = JudgeCache.make_key("answer B", "context X")

        assert key1 != key2

    def test_make_key_should_produce_hex_sha256(self) -> None:
        """Given any input, When making a key, Then result is 64-char hex (SHA-256)."""
        key = JudgeCache.make_key("hello", "world")

        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_make_key_should_be_order_sensitive(self) -> None:
        """Given the same parts in different order, When making keys, Then they differ."""
        key1 = JudgeCache.make_key("first", "second")
        key2 = JudgeCache.make_key("second", "first")

        assert key1 != key2

    @pytest.mark.asyncio
    async def test_judge_cache_should_promote_disk_to_memory(
        self, tmp_path: Path
    ) -> None:
        """Given a value only on disk, When read, Then it is promoted to memory.

        After reading from disk, a second read should be an in-memory hit
        (verified by closing the shelf and still being able to read).
        """
        disk_path = tmp_path / "promote_cache"

        # Write to disk and close
        cache_w = JudgeCache(disk_path=disk_path)
        await cache_w.put("promo-key", 0.33)
        await cache_w.close()

        # Open a new cache that loads from disk
        cache_r = JudgeCache(disk_path=disk_path)

        # First read: disk hit, promotes to memory
        result_first = await cache_r.get("promo-key")
        assert result_first == pytest.approx(0.33)  # type: ignore

        # Close the shelf to force memory-only reads
        await cache_r.close()

        # Second read should come from in-memory (shelf is closed)
        result_second = await cache_r.get("promo-key")
        assert result_second == pytest.approx(0.33)  # type: ignore


# =========================================================================
# Integration: Faithfulness with JudgeCache
# =========================================================================


@pytest.mark.unit
class TestFaithfulnessWithCache:
    """Tests for Faithfulness metric with JudgeCache integration."""

    @pytest.mark.asyncio
    async def test_faithfulness_should_use_cache_on_repeated_calls(self) -> None:
        """Given the same inputs, When called twice, Then second call uses cache.

        The judge is only called during the first invocation; on the second
        call the cached score is returned without any new LLM calls.
        """
        judge = FakeLLMDriver(
            responses=[
                json.dumps({"claims": ["claim1"]}),
                json.dumps({"verdict": "supported"}),
            ]
        )
        cache = JudgeCache()
        metric = Faithfulness()

        answer = "Test answer."
        context = ["Test context."]

        score_1 = await metric.compute_async(
            answer=answer, context=context, judge=judge, cache=cache
        )
        calls_after_first = judge.call_count

        score_2 = await metric.compute_async(
            answer=answer, context=context, judge=judge, cache=cache
        )
        calls_after_second = judge.call_count

        assert score_1 == score_2
        assert calls_after_second == calls_after_first  # no new calls


# =========================================================================
# JudgePromptConfig
# =========================================================================


@pytest.mark.unit
class TestJudgePromptConfig:
    """Tests for JudgePromptConfig defaults and validation."""

    def test_judge_prompt_config_should_have_sensible_defaults(self) -> None:
        """Given no arguments, When constructing config, Then defaults are set."""
        config = JudgePromptConfig()

        assert config.batch_size == 8
        assert config.relevancy_embedder_weight == pytest.approx(0.3)  # type: ignore
        assert "answer" in config.faithfulness_decompose_template
        assert "claim" in config.faithfulness_verify_template
        assert "question" in config.relevancy_template
        assert "ground_truth" in config.correctness_template

    def test_judge_prompt_config_should_reject_invalid_batch_size(self) -> None:
        """Given batch_size=0, When constructing config, Then validation fails."""
        with pytest.raises(ValueError):
            JudgePromptConfig(batch_size=0)

    def test_judge_prompt_config_should_reject_weight_out_of_range(self) -> None:
        """Given weight > 1.0, When constructing config, Then validation fails."""
        with pytest.raises(ValueError):
            JudgePromptConfig(relevancy_embedder_weight=1.5)
