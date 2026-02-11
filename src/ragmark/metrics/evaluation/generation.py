"""Generation quality evaluation metrics.

Placeholder module for future generation metrics like faithfulness
and answer relevancy. These will be implemented with LLM-as-judge patterns.
"""

from typing import Any

from ragmark.logger import get_logger
from ragmark.metrics.base import EvaluationMetric
from ragmark.metrics.registry import MetricRegistry

logger = get_logger(__name__)


@MetricRegistry.register
class Faithfulness(EvaluationMetric[Any, float]):
    """Measure if generated answer is grounded in context.

    Placeholder for future implementation using LLM-as-judge or
    claim verification against retrieved context.
    """

    @property
    def name(self) -> str:
        return "faithfulness"

    @property
    def description(self) -> str:
        return "Proportion of claims in answer supported by context"

    def compute(self, **inputs: Any) -> float:
        """Compute faithfulness score.

        Args:
            answer: Generated answer text.
            context: Retrieved context passages.
            **kwargs: Additional arguments for future implementation.

        Returns:
            Faithfulness score between 0.0 and 1.0.

        Raises:
            NotImplementedError: Currently not implemented.
        """
        answer = inputs.get("answer", "")
        context = inputs.get("context", [])

        logger.debug(
            "Faithfulness metric not yet implemented: answer_len=%d, context_count=%d",
            len(answer),
            len(context),
        )
        raise NotImplementedError(
            "Faithfulness metric not yet implemented. "
            "Use LLM-as-judge or claim verification approaches."
        )


@MetricRegistry.register
class AnswerRelevancy(EvaluationMetric[Any, float]):
    """Measure if generated answer addresses the question.

    Placeholder for future implementation using semantic similarity
    or LLM-based relevancy scoring.
    """

    @property
    def name(self) -> str:
        return "answer_relevancy"

    @property
    def description(self) -> str:
        return "Semantic relevance between question and answer"

    def compute(self, **inputs: Any) -> float:
        """Compute answer relevancy score.

        Args:
            question: Original question.
            answer: Generated answer text.
            **kwargs: Additional arguments for future implementation.

        Returns:
            Relevancy score between 0.0 and 1.0.

        Raises:
            NotImplementedError: Currently not implemented.
        """
        question = inputs.get("question", "")
        answer = inputs.get("answer", "")

        logger.debug(
            "AnswerRelevancy metric not yet implemented: "
            "question_len=%d, answer_len=%d",
            len(question),
            len(answer),
        )
        raise NotImplementedError(
            "AnswerRelevancy metric not yet implemented. "
            "Use semantic similarity or LLM-based scoring approaches."
        )
