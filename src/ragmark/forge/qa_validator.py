"""Quality validation for synthetic question-answer pairs.

This module provides interfaces and implementations for validating the quality
of synthetically generated QA pairs.
"""

from abc import ABC, abstractmethod

from ragmark.logger import get_logger
from ragmark.schemas.qa import SyntheticQA

logger = get_logger(__name__)


class QAValidator(ABC):
    """Abstract validator for QA pairs quality."""

    @abstractmethod
    def validate(self, qa_pairs: list[SyntheticQA]) -> list[SyntheticQA]:
        """Filter and validate QA pairs.

        Args:
            qa_pairs: List of QA pairs to validate.

        Returns:
            Filtered list of valid QA pairs.
        """
        pass


class BasicQAValidator(QAValidator):
    """Basic validation for question format, length, and duplicates.

    Validates QA pairs against simple heuristics including length constraints,
    format requirements (question mark), and duplicate detection.

    Attributes:
        min_question_length: Minimum question length.
        max_question_length: Maximum question length.
        min_answer_length: Minimum answer length.
        require_question_mark: Questions must end with '?'.
    """

    def __init__(
        self,
        min_question_length: int = 10,
        max_question_length: int = 500,
        min_answer_length: int = 5,
        require_question_mark: bool = True,
    ):
        """Initialize the basic QA validator.

        Args:
            min_question_length: Minimum question length in characters.
            max_question_length: Maximum question length in characters.
            min_answer_length: Minimum answer length in characters.
            require_question_mark: Whether questions must end with '?'.
        """
        self.min_question_length = min_question_length
        self.max_question_length = max_question_length
        self.min_answer_length = min_answer_length
        self.require_question_mark = require_question_mark

    def validate(self, qa_pairs: list[SyntheticQA]) -> list[SyntheticQA]:
        """Validate QA pairs with basic rules.

        Filters out QA pairs that fail validation rules:
        - Question too short or too long
        - Answer too short
        - Question missing '?' (if required)
        - Duplicate questions

        Args:
            qa_pairs: List of QA pairs to validate.

        Returns:
            Filtered list of valid QA pairs.
        """
        valid_pairs = []
        seen_questions = set()

        for qa in qa_pairs:
            if len(qa.question) < self.min_question_length:
                logger.debug("Question too short: %d chars", len(qa.question))
                continue

            if len(qa.question) > self.max_question_length:
                logger.debug("Question too long: %d chars", len(qa.question))
                continue

            if len(qa.answer) < self.min_answer_length:
                logger.debug("Answer too short: %d chars", len(qa.answer))
                continue

            if self.require_question_mark and not qa.question.rstrip().endswith("?"):
                logger.debug("Question missing '?': %s", qa.question[:50])
                continue

            normalized_q = qa.question.lower().strip()
            if normalized_q in seen_questions:
                logger.debug("Duplicate question: %s", qa.question[:50])
                continue

            seen_questions.add(normalized_q)
            valid_pairs.append(qa)

        logger.debug(
            "Validation: %d/%d QA pairs passed", len(valid_pairs), len(qa_pairs)
        )

        return valid_pairs
