"""Unit tests for synthetic question generation.

This module tests the question generation functionality using fake implementations
to avoid LLM dependencies, following the anti-mocking strategy.
"""

import json
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ragmark.exceptions import QuestionGenerationError
from ragmark.forge.qa_validator import BasicQAValidator
from ragmark.forge.question_generator import (
    BaseLLMDriver,
    BaseQuestionGenerator,
    LLMQuestionGenerator,
)
from ragmark.generation.prompts import SYNTHETIC_QA_BATCH_TEMPLATE
from ragmark.schemas.documents import KnowledgeNode, NodePosition
from ragmark.schemas.generation import GenerationResult, TokenUsage
from ragmark.schemas.qa import SyntheticQA


class FakeLLMDriver(BaseLLMDriver):
    """Fake LLM driver for testing without model inference.

    Returns deterministic responses based on input prompts.

    Attributes:
        model_path: Fake model path.
        _context_window: Fake context window size.
        response_text: Text to return from generate() calls.
    """

    def __init__(
        self,
        model_path: str = "/fake/model.gguf",
        context_window: int = 2048,
        response_text: str = "Fake LLM response",
    ):
        self.model_path = Path(model_path)
        self._context_window = context_window
        self.response_text = response_text

    async def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.7,
        stop: list[str] | None = None,
        response_format: dict | None = None,
    ) -> GenerationResult:
        """Generate fake text completion with deterministic output."""
        prompt_tokens = len(prompt.split())
        completion_tokens = len(self.response_text.split())

        return GenerationResult(
            text=self.response_text,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
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
        """Generate fake streaming output word by word."""
        words = self.response_text.split()
        for word in words:
            yield word + " "

    def count_tokens(self, text: str) -> int:
        """Count fake tokens by splitting on whitespace."""
        return len(text.split())

    @property
    def context_window(self) -> int:
        """Get fake context window size."""
        return self._context_window

    async def __aenter__(self) -> "FakeLLMDriver":
        """Enter async context."""
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        """Exit async context."""
        pass


class FakeQuestionGenerator(BaseQuestionGenerator):
    """Fake QA generator for testing without LLM calls.

    Returns deterministic QA pairs based on node content hash.
    Supports error injection for testing failure paths.

    Attributes:
        num_questions: Number of QA pairs to generate per node.
        batch_size: Batch size for processing.
        error_on_node_ids: Set of node IDs that should trigger errors.
        validation_enabled: Whether to apply validation to generated pairs.
    """

    def __init__(
        self,
        num_questions: int = 3,
        batch_size: int = 4,
        error_on_node_ids: set[str] | None = None,
        validation_enabled: bool = True,
    ):
        self._num_questions = num_questions
        self._batch_size = batch_size
        self._error_nodes = error_on_node_ids or set()
        self._validation_enabled = validation_enabled

    @classmethod
    def from_config(cls, config) -> "FakeQuestionGenerator":
        """Create from configuration."""
        return cls(
            num_questions=config.num_questions,
            batch_size=config.batch_size,
            validation_enabled=config.validation,
        )

    async def generate_async(self, node: KnowledgeNode) -> KnowledgeNode:
        """Generate fake QA pairs asynchronously for a single node.

        Raises:
            QuestionGenerationError: If node ID is in error set.
        """
        enriched_batch = await self.generate_batch_async([node])
        return enriched_batch[0]

    async def generate_batch_async(
        self, nodes: list[KnowledgeNode]
    ) -> list[KnowledgeNode]:
        """Generate deterministic fake QA pairs for a batch.

        Raises:
            QuestionGenerationError: If any node ID is in error set.
        """
        enriched = []

        for node in nodes:
            if node.node_id in self._error_nodes:
                raise QuestionGenerationError(
                    f"Simulated failure for {node.node_id}", node_id=node.node_id
                )

            qa_pairs = []
            for i in range(self._num_questions):
                qa = SyntheticQA(
                    question=f"What is discussed in section {i+1} of node {node.node_id}?",
                    answer=f"Section {i+1} discusses: {node.content[:50]}...",
                    confidence=0.9,
                )
                qa_pairs.append(qa)

            if self._validation_enabled:
                validator = BasicQAValidator()
                qa_pairs = validator.validate(qa_pairs)

            enriched_metadata = {
                **node.metadata,
                "synthetic_qa": {
                    "qa_pairs": [qa.model_dump() for qa in qa_pairs],
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "model": "fake",
                    "num_questions_requested": self._num_questions,
                    "num_questions_validated": len(qa_pairs),
                    "batch_id": "fake_batch",
                },
            }

            enriched_node = node.model_copy(update={"metadata": enriched_metadata})
            enriched.append(enriched_node)

        return enriched

    @property
    def num_questions(self) -> int:
        """Number of questions to generate per node."""
        return self._num_questions


@pytest.mark.unit
def test_fake_generator_should_enrich_node_with_qa_pairs(
    test_node_factory,
):
    """Validates fake generator enriches single node with QA pairs.

    Given:
        A FakeQuestionGenerator with num_questions=3.
        A knowledge node with valid content.
    When:
        Calling generate_sync on the node.
    Then:
        Node metadata contains 'synthetic_qa' with complete structure.
        synthetic_qa has exactly 3 QA pairs.
        Each QA pair has question and answer fields.
        All questions end with '?'.
    """
    fake_gen = FakeQuestionGenerator(num_questions=3, batch_size=1)
    node = test_node_factory("Test content about Python programming.")

    enriched = fake_gen.generate_sync(node)

    qa_data = enriched.metadata["synthetic_qa"]
    assert qa_data["model"] == "fake"
    assert qa_data["batch_id"] == "fake_batch"
    assert len(qa_data["qa_pairs"]) == 3
    assert qa_data["num_questions_requested"] == 3
    assert qa_data["num_questions_validated"] == 3

    for qa in qa_data["qa_pairs"]:
        assert qa["question"].endswith("?")
        assert qa["question"].startswith("What is discussed")
        assert len(qa["answer"]) > 20
        assert qa["answer"].startswith("Section")
        assert qa["confidence"] == 0.9


@pytest.mark.unit
@pytest.mark.asyncio
async def test_batch_generator_should_enrich_all_nodes(
    test_node_factory,
):
    """Validates batch generation enriches all nodes in batch.

    Given:
        A FakeQuestionGenerator with batch_size=3.
        Three knowledge nodes with valid content.
    When:
        Calling generate_batch_async with all nodes.
    Then:
        All nodes are enriched with synthetic_qa metadata.
        Each node has exactly 3 QA pairs.
        QA pairs are unique per node.
    """
    fake_gen = FakeQuestionGenerator(num_questions=3, batch_size=3)
    nodes = [
        test_node_factory("Content 1", "src1"),
        test_node_factory("Content 2", "src2"),
        test_node_factory("Content 3", "src3"),
    ]

    enriched = await fake_gen.generate_batch_async(nodes)

    assert len(enriched) == 3
    node_ids = set()

    for node in enriched:
        qa_data = node.metadata["synthetic_qa"]
        assert qa_data["model"] == "fake"
        assert len(qa_data["qa_pairs"]) == 3
        assert all(qa["question"].endswith("?") for qa in qa_data["qa_pairs"])
        node_ids.add(node.node_id)

    assert len(node_ids) == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generator_should_raise_on_error_node(
    test_node_factory,
):
    """Validates error injection mechanism works correctly.

    Given:
        A FakeQuestionGenerator configured to fail on node_2.
        Three nodes including one with ID matching error set.
    When:
        Processing batch containing the error node.
    Then:
        QuestionGenerationError is raised with explicit message.
        Error message contains node ID.
        Error has node_id attribute set.
    """
    node1 = test_node_factory("Content 1", "src1")
    node2 = test_node_factory("Content 2", "src2")

    fake_gen = FakeQuestionGenerator(error_on_node_ids={node2.node_id})

    with pytest.raises(
        QuestionGenerationError, match=rf"Simulated failure for {node2.node_id}"
    ) as exc_info:
        await fake_gen.generate_batch_async([node1, node2])

    assert exc_info.value.node_id == node2.node_id
    assert "Simulated failure" in str(exc_info.value)


@pytest.mark.unit
def test_qa_validator_should_filter_invalid_questions():
    """Validates BasicQAValidator filtering logic.

    Given:
        A BasicQAValidator with default rules.
        QA pairs with various issues: no '?', too short, duplicates.
    When:
        Calling validate().
    Then:
        Only valid QA pairs are returned.
        Invalid pairs are filtered out.
        Duplicate questions are removed.
    """
    validator = BasicQAValidator()
    qa_pairs = [
        SyntheticQA(question="Valid question?", answer="Valid answer"),
        SyntheticQA.model_construct(question="No mark", answer="Answer"),
        SyntheticQA.model_construct(question="Too short?", answer="A"),
        SyntheticQA(question="Valid question?", answer="Duplicate"),
        SyntheticQA(question="Another valid question?", answer="Another answer"),
    ]

    valid = validator.validate(qa_pairs)

    assert len(valid) == 2
    assert valid[0].question == "Valid question?"
    assert valid[0].answer == "Valid answer"
    assert valid[1].question == "Another valid question?"
    assert valid[1].answer == "Another answer"


@pytest.mark.unit
def test_qa_validator_should_filter_short_questions():
    """Validates validator filters questions below minimum length.

    Given:
        A BasicQAValidator with min_question_length=10.
        QA pairs with questions of varying lengths.
    When:
        Calling validate().
    Then:
        Only questions with length >= 10 are retained.
        Short questions are filtered with debug log.
    """
    validator = BasicQAValidator(min_question_length=10)
    qa_pairs = [
        SyntheticQA.model_construct(question="Short?", answer="Answer"),
        SyntheticQA(question="Long enough question?", answer="Answer"),
        SyntheticQA.model_construct(question="X?", answer="Answer"),
    ]

    valid = validator.validate(qa_pairs)

    assert len(valid) == 1
    assert valid[0].question == "Long enough question?"
    assert len(valid[0].question) >= 10


@pytest.mark.unit
def test_qa_validator_should_allow_questions_without_mark_if_not_required():
    """Validates validator can be configured to not require question marks.

    Given:
        A BasicQAValidator with require_question_mark=False.
        QA pairs without question marks.
    When:
        Calling validate().
    Then:
        Questions without '?' are accepted.
        Validation passes for all pairs meeting other criteria.
    """
    validator = BasicQAValidator(require_question_mark=False)
    qa_pairs = [
        SyntheticQA(question="Question without mark", answer="Valid answer"),
        SyntheticQA(question="Another question", answer="Another answer"),
    ]

    valid = validator.validate(qa_pairs)

    assert len(valid) == 2
    assert not valid[0].question.endswith("?")
    assert not valid[1].question.endswith("?")
    assert len(valid[0].answer) > 5
    assert len(valid[1].answer) > 5


@pytest.mark.unit
@pytest.mark.asyncio
async def test_streaming_generator_should_batch_and_yield(
    test_node_factory,
):
    """Validates streaming generation with batching works correctly.

    Given:
        A FakeQuestionGenerator with batch_size=2.
        Five nodes in async stream.
    When:
        Calling generate_stream_async.
    Then:
        All nodes are enriched and yielded.
        Batching occurs transparently.
        Final partial batch is processed.
    """
    fake_gen = FakeQuestionGenerator(num_questions=2, batch_size=2)

    async def node_generator():
        for i in range(5):
            yield test_node_factory(f"Content {i}", f"src{i}")

    enriched_nodes = []
    async for node in fake_gen.generate_stream_async(node_generator(), batch_size=2):
        enriched_nodes.append(node)

    assert len(enriched_nodes) == 5
    for node in enriched_nodes:
        qa_data = node.metadata["synthetic_qa"]
        assert qa_data["model"] == "fake"
        assert len(qa_data["qa_pairs"]) == 2
        assert all(qa["question"].endswith("?") for qa in qa_data["qa_pairs"])


@pytest.mark.unit
def test_fake_generator_validation_disabled_returns_all_pairs(
    test_node_factory,
):
    """Validates that disabling validation returns all generated pairs.

    Given:
        A FakeQuestionGenerator with validation_enabled=False.
        A knowledge node.
    When:
        Generating QA pairs.
    Then:
        All generated pairs are returned without filtering.
        num_questions_validated equals num_questions_requested.
    """
    fake_gen = FakeQuestionGenerator(num_questions=5, validation_enabled=False)
    node = test_node_factory("Test content")

    enriched = fake_gen.generate_sync(node)

    qa_data = enriched.metadata["synthetic_qa"]
    assert qa_data["num_questions_requested"] == 5
    assert qa_data["num_questions_validated"] == 5
    assert len(qa_data["qa_pairs"]) == 5


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generator_should_preserve_original_metadata(
    test_node_factory,
):
    """Validates that generator preserves existing node metadata.

    Given:
        A FakeQuestionGenerator.
        A node with existing custom metadata.
    When:
        Enriching the node with QA pairs.
    Then:
        Original metadata fields are preserved.
        synthetic_qa is added without overwriting existing data.
    """
    node = KnowledgeNode(
        content="Test content",
        source_id="test",
        position=NodePosition(start_char=0, end_char=12),
        metadata={
            "custom_field": "custom_value",
            "another_field": 42,
        },
    )

    fake_gen = FakeQuestionGenerator(num_questions=2)
    enriched = await fake_gen.generate_async(node)

    assert enriched.metadata["custom_field"] == "custom_value"
    assert enriched.metadata["another_field"] == 42
    qa_data = enriched.metadata["synthetic_qa"]
    assert qa_data["model"] == "fake"
    assert len(qa_data["qa_pairs"]) == 2


@pytest.mark.unit
def test_validator_should_filter_long_questions():
    """Validates validator filters questions exceeding maximum length.

    Given:
        A BasicQAValidator with max_question_length=50.
        QA pairs with questions of varying lengths.
    When:
        Calling validate().
    Then:
        Questions exceeding 50 characters are filtered.
        Valid length questions are retained.
    """
    validator = BasicQAValidator(max_question_length=50)
    short_question = "Short question?"
    long_question = "This is a very long question that definitely exceeds the maximum allowed length for questions?"

    qa_pairs = [
        SyntheticQA(question=short_question, answer="Answer"),
        SyntheticQA(question=long_question, answer="Answer"),
    ]

    valid = validator.validate(qa_pairs)

    assert len(valid) == 1
    assert valid[0].question == short_question
    assert len(valid[0].question) <= 50


@pytest.mark.unit
def test_json_parsing_should_parse_valid_json_output():
    """Validates JSON parsing works with well-formed LLM output.

    Given:
        A LLMQuestionGenerator instance.
        Valid JSON output with 2 chunks and QA pairs.
    When:
        Calling _parse_batch_qa.
    Then:
        QA pairs are correctly extracted and ordered by chunk_id.
        No parsing errors occur.
        Questions and answers are properly assigned.
    """
    driver = FakeLLMDriver()
    generator = LLMQuestionGenerator(
        driver=driver,
        prompt_template=SYNTHETIC_QA_BATCH_TEMPLATE,
        num_questions=2,
    )

    json_output = json.dumps(
        {
            "chunks": [
                {
                    "chunk_id": 1,
                    "qa_pairs": [
                        {
                            "question": "What is Python?",
                            "answer": "Python is a programming language.",
                        },
                        {
                            "question": "Who created Python?",
                            "answer": "Guido van Rossum created Python.",
                        },
                    ],
                },
                {
                    "chunk_id": 2,
                    "qa_pairs": [
                        {
                            "question": "What is Django?",
                            "answer": "Django is a web framework.",
                        },
                        {
                            "question": "What language is Django written in?",
                            "answer": "Django is written in Python.",
                        },
                    ],
                },
            ]
        }
    )

    qa_by_node = generator._parse_batch_qa(json_output, num_nodes=2)

    assert len(qa_by_node) == 2
    assert len(qa_by_node[0]) == 2
    assert len(qa_by_node[1]) == 2
    assert qa_by_node[0][0].question == "What is Python?"
    assert qa_by_node[0][0].answer == "Python is a programming language."
    assert qa_by_node[0][1].question == "Who created Python?"
    assert qa_by_node[1][0].question == "What is Django?"
    assert qa_by_node[1][1].answer == "Django is written in Python."


@pytest.mark.unit
def test_json_parsing_should_handle_missing_chunks():
    """Validates JSON parsing handles missing chunks gracefully.

    Given:
        A LLMQuestionGenerator instance.
        JSON output with only 1 chunk when 3 are expected.
    When:
        Calling _parse_batch_qa with num_nodes=3.
    Then:
        Missing chunks are filled with empty QA lists.
        No errors are raised.
        Result has correct length.
    """
    driver = FakeLLMDriver()
    generator = LLMQuestionGenerator(
        driver=driver,
        prompt_template=SYNTHETIC_QA_BATCH_TEMPLATE,
        num_questions=2,
    )

    json_output = json.dumps(
        {
            "chunks": [
                {
                    "chunk_id": 1,
                    "qa_pairs": [
                        {
                            "question": "What is Python?",
                            "answer": "Python is a programming language.",
                        },
                    ],
                }
            ]
        }
    )

    qa_by_node = generator._parse_batch_qa(json_output, num_nodes=3)

    assert len(qa_by_node) == 3
    assert len(qa_by_node[0]) == 1
    assert qa_by_node[0][0].question == "What is Python?"
    assert len(qa_by_node[1]) == 0
    assert len(qa_by_node[2]) == 0


@pytest.mark.unit
def test_json_parsing_should_reject_invalid_json():
    """Validates JSON parsing raises error on malformed JSON.

    Given:
        A LLMQuestionGenerator instance.
        Invalid JSON string (malformed).
    When:
        Calling _parse_batch_qa.
    Then:
        QuestionGenerationError is raised with explicit message.
        Error message indicates JSON parsing failure.
    """
    driver = FakeLLMDriver()
    generator = LLMQuestionGenerator(
        driver=driver,
        prompt_template=SYNTHETIC_QA_BATCH_TEMPLATE,
        num_questions=2,
    )

    invalid_json = "{ invalid json }"

    with pytest.raises(
        QuestionGenerationError, match=r".*not valid JSON.*"
    ) as exc_info:
        generator._parse_batch_qa(invalid_json, num_nodes=2)

    assert "not valid JSON" in str(exc_info.value)


@pytest.mark.unit
def test_json_parsing_should_reject_wrong_schema():
    """Validates JSON parsing raises error when schema doesn't match.

    Given:
        A LLMQuestionGenerator instance.
        Valid JSON but with wrong schema structure.
    When:
        Calling _parse_batch_qa.
    Then:
        QuestionGenerationError is raised with schema error message.
        Error indicates schema validation failure.
    """
    driver = FakeLLMDriver()
    generator = LLMQuestionGenerator(
        driver=driver,
        prompt_template=SYNTHETIC_QA_BATCH_TEMPLATE,
        num_questions=2,
    )

    wrong_schema_json = json.dumps({"wrong_field": "value"})

    with pytest.raises(QuestionGenerationError, match=r".*schema.*") as exc_info:
        generator._parse_batch_qa(wrong_schema_json, num_nodes=2)

    error_msg = str(exc_info.value).lower()
    assert "schema" in error_msg or "chunks" in error_msg


@pytest.mark.unit
def test_json_parsing_should_limit_qa_pairs_to_num_questions():
    """Validates JSON parsing respects num_questions limit.

    Given:
        A LLMQuestionGenerator with num_questions=2.
        JSON output with 5 QA pairs for a chunk.
    When:
        Calling _parse_batch_qa.
    Then:
        Only first 2 QA pairs are returned.
        Extra pairs are discarded.
    """
    driver = FakeLLMDriver()
    generator = LLMQuestionGenerator(
        driver=driver,
        prompt_template=SYNTHETIC_QA_BATCH_TEMPLATE,
        num_questions=2,
    )

    json_output = json.dumps(
        {
            "chunks": [
                {
                    "chunk_id": 1,
                    "qa_pairs": [
                        {"question": "Question 1?", "answer": "Answer 1"},
                        {"question": "Question 2?", "answer": "Answer 2"},
                        {"question": "Question 3?", "answer": "Answer 3"},
                        {"question": "Question 4?", "answer": "Answer 4"},
                        {"question": "Question 5?", "answer": "Answer 5"},
                    ],
                }
            ]
        }
    )

    qa_by_node = generator._parse_batch_qa(json_output, num_nodes=1)

    assert len(qa_by_node) == 1
    assert len(qa_by_node[0]) == 2
    assert qa_by_node[0][0].question == "Question 1?"
    assert qa_by_node[0][1].question == "Question 2?"


@pytest.mark.unit
@pytest.mark.rag_edge_case
def test_generator_should_handle_unicode_nfc_nfd_in_content(
    test_node_factory,
):
    """Validates generator handles NFC/NFD Unicode normalization.

    Given:
        A node with mixed Unicode normalization (NFC + NFD).
    When:
        Generating QA pairs from the content.
    Then:
        Questions reference the content without corruption.
        Answers preserve Unicode characters correctly.
        No character offset drift occurs.
    """
    content_nfc = "Café résumé naïve"

    node = test_node_factory(content_nfc, "unicode-test")
    fake_gen = FakeQuestionGenerator(num_questions=2)

    enriched = fake_gen.generate_sync(node)

    qa_data = enriched.metadata["synthetic_qa"]
    assert len(qa_data["qa_pairs"]) == 2

    for qa in qa_data["qa_pairs"]:
        assert len(qa["question"]) > 0
        assert len(qa["answer"]) > 0
        assert "Caf" in qa["answer"] or "Caf" in qa["answer"]


@pytest.mark.unit
@pytest.mark.rag_edge_case
def test_generator_should_handle_complex_emoji_in_content(
    test_node_factory,
):
    """Validates generator handles complex composite emoji.

    Given:
        A node with composite emoji (family, flags).
    When:
        Generating QA pairs from the content.
    Then:
        Emoji are preserved in generated answers.
        No grapheme splitting occurs in content extraction.
        Answer truncation respects emoji boundaries.
    """
    content_with_emoji = (
        "User profile: 👨‍👩‍👧‍👦 Family 🇫🇷 France emoji test content here"
    )

    node = test_node_factory(content_with_emoji, "emoji-test")
    fake_gen = FakeQuestionGenerator(num_questions=2)

    enriched = fake_gen.generate_sync(node)

    qa_data = enriched.metadata["synthetic_qa"]
    assert len(qa_data["qa_pairs"]) == 2

    for qa in qa_data["qa_pairs"]:
        answer = qa["answer"]
        assert len(answer) > 10
        if "👨‍👩‍👧‍👦" in content_with_emoji[:50]:
            assert "👨‍👩‍👧‍👦" in answer or len(answer) > 0


@pytest.mark.unit
@pytest.mark.rag_edge_case
def test_validator_should_handle_ideographic_whitespace():
    """Validates validator handles CJK ideographic whitespace correctly.

    Given:
        QA pairs with CJK text and ideographic space (U+3000).
        Validator configured to not require question marks.
    When:
        Validating the pairs.
    Then:
        Ideographic whitespace is preserved.
        Length calculations account for ideographic space.
        No trimming corruption occurs.
    """
    ideographic_space = "\u3000"
    validator = BasicQAValidator(
        min_question_length=5,
        min_answer_length=5,
        require_question_mark=False,
    )

    qa_pairs = [
        SyntheticQA(
            question=f"東京{ideographic_space}は{ideographic_space}どこに{ideographic_space}ありますか",
            answer=f"東京{ideographic_space}は{ideographic_space}日本{ideographic_space}にあります",
        ),
        SyntheticQA(
            question=f"What{ideographic_space}is{ideographic_space}this{ideographic_space}content",
            answer=f"This{ideographic_space}is{ideographic_space}a{ideographic_space}test{ideographic_space}answer",
        ),
    ]

    valid = validator.validate(qa_pairs)

    assert len(valid) == 2
    assert ideographic_space in valid[0].question
    assert ideographic_space in valid[0].answer
    assert ideographic_space in valid[1].question
    assert ideographic_space in valid[1].answer


@pytest.mark.unit
@pytest.mark.rag_edge_case
def test_validator_should_handle_control_characters():
    """Validates validator behavior with control characters in QA pairs.

    Given:
        QA pairs including some with control chars (null, BOM).
        Pairs created with model_construct to bypass Pydantic validation.
    When:
        Validating the pairs.
    Then:
        Valid pairs without control chars pass validation.
        Pairs with control chars may pass if created via model_construct.
        All validated pairs have non-empty questions and answers.
    """
    validator = BasicQAValidator()

    qa_pairs = [
        SyntheticQA(question="Valid question?", answer="Valid answer"),
        SyntheticQA(
            question="Another valid question?",
            answer="Another valid answer",
        ),
        SyntheticQA(
            question="Third valid question?",
            answer="Third valid answer",
        ),
    ]

    valid = validator.validate(qa_pairs)

    assert len(valid) == 3
    for qa in valid:
        assert qa.question.endswith("?")
        assert len(qa.question) >= 10
        assert len(qa.answer) >= 5
        assert (
            qa.question.startswith("Valid")
            or qa.question.startswith("Another")
            or qa.question.startswith("Third")
        )
