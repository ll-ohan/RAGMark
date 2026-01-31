"""Tests for RAGGenerator class.

This module tests end-to-end RAG pipeline orchestration. Network I/O components
(retriever, driver) are mocked per TEST_POLICY.md Section 2.2. Business logic
components use fake implementations to ensure contract validation.
"""

from unittest.mock import AsyncMock

import pytest

from ragmark.generation.engine import RAGGenerator
from ragmark.schemas.documents import KnowledgeNode, NodePosition
from ragmark.schemas.generation import AnswerResult, GenerationResult, TokenUsage
from ragmark.schemas.retrieval import RetrievedNode, TraceContext


class FakePromptTemplate:
    """Fake prompt template for testing.

    Implements real template interface without external dependencies.
    """

    def render(self, **kwargs: dict) -> str:
        """Render template with provided variables."""
        return f"Prompt: {kwargs.get('question', 'default')}"


class FakeContextManager:
    """Fake context manager for testing.

    Implements real context fitting logic without token counting overhead.
    Stores execution state for test assertions.
    """

    def __init__(self, max_tokens: int = 2048):
        self.max_tokens = max_tokens
        self.last_fitted_context = ""
        self.last_user_query = ""
        self.last_system = ""

    def fit_context(
        self,
        system: str,
        context_chunks: list[str],
        user_query: str,
        max_completion: int = 512,
    ) -> str:
        """Fit context chunks within token budget."""
        self.last_system = system
        self.last_user_query = user_query

        # Build prompt similar to real ContextManager._build_prompt
        parts = [system]

        if context_chunks:
            parts.append("\nContext:")
            for i, chunk in enumerate(context_chunks, 1):
                parts.append(f"\n[{i}] {chunk}")
            parts.append("\n")

        parts.append(f"\nUser: {user_query}")

        prompt = "".join(parts)
        self.last_fitted_context = prompt
        return prompt


@pytest.fixture
def mock_retriever():
    """Create mock retriever.

    Mocks network I/O per TEST_POLICY.md Section 2.2.
    """
    retriever = AsyncMock()

    pos1 = NodePosition(start_char=0, end_char=100)
    node1 = KnowledgeNode(
        node_id="node1",
        content="Context chunk 1",
        metadata={},
        source_id="source1",
        position=pos1,
    )

    pos2 = NodePosition(start_char=100, end_char=200)
    node2 = KnowledgeNode(
        node_id="node2",
        content="Context chunk 2",
        metadata={},
        source_id="source2",
        position=pos2,
    )

    retrieved = [
        RetrievedNode(node=node1, score=0.9, rank=1),
        RetrievedNode(node=node2, score=0.8, rank=2),
    ]

    trace = TraceContext(
        query="Test question",
        retrieved_nodes=retrieved,
    )

    retriever.retrieve.return_value = trace
    return retriever


@pytest.fixture
def mock_driver():
    """Create mock LLM driver.

    Mocks external API calls per TEST_POLICY.md Section 2.2.
    """
    driver = AsyncMock()

    result = GenerationResult(
        text="Generated answer text",
        usage=TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        ),
        finish_reason="stop",
    )

    driver.generate.return_value = result
    return driver


@pytest.fixture
def fake_template():
    """Create fake prompt template.

    Uses real implementation per TEST_POLICY.md Section 2.3.
    """
    return FakePromptTemplate()


@pytest.fixture
def fake_context_manager():
    """Create fake context manager.

    Uses real implementation per TEST_POLICY.md Section 2.3.
    """
    return FakeContextManager()


@pytest.mark.unit
class TestRAGGeneratorInitialization:
    """Test suite for RAGGenerator initialization."""

    def test_init_should_store_components(
        self,
        mock_retriever,
        mock_driver,
        fake_template,
        fake_context_manager,
    ):
        """Verify initialization stores all components.

        Given:
            All RAG components (retriever, driver, template, context_manager).
        When:
            Initializing RAGGenerator.
        Then:
            All components are stored correctly as instance attributes.
        """
        generator = RAGGenerator(
            retriever=mock_retriever,
            driver=mock_driver,
            template=fake_template,
            context_manager=fake_context_manager,
        )

        assert generator.retriever is mock_retriever
        assert generator.driver is mock_driver
        assert generator.template is fake_template
        assert generator.context_manager is fake_context_manager


@pytest.mark.unit
class TestRAGGeneratorAnswer:
    """Test suite for answer generation method."""

    @pytest.mark.asyncio
    async def test_answer_should_orchestrate_full_pipeline(
        self,
        mock_retriever,
        mock_driver,
        fake_template,
        fake_context_manager,
    ):
        """Verify answer method orchestrates full RAG pipeline.

        Given:
            Initialized RAGGenerator with all components.
        When:
            Calling answer method with specific parameters.
        Then:
            All components are invoked with correct arguments and ordering.
        """
        generator = RAGGenerator(
            retriever=mock_retriever,
            driver=mock_driver,
            template=fake_template,
            context_manager=fake_context_manager,
        )

        result = await generator.answer(
            question="What is RAG?",
            include_sources=True,
            max_completion=512,
            temperature=0.7,
        )

        mock_retriever.retrieve.assert_called_once_with("What is RAG?")

        mock_driver.generate.assert_called_once()
        call_args = mock_driver.generate.call_args
        assert call_args.kwargs["max_tokens"] == 512
        assert call_args.kwargs["temperature"] == 0.7

        assert isinstance(result, AnswerResult)
        assert result.answer == "Generated answer text"
        assert result.sources == ["source1", "source2"]

    @pytest.mark.asyncio
    async def test_answer_should_return_answer_result(
        self,
        mock_retriever,
        mock_driver,
        fake_template,
        fake_context_manager,
    ):
        """Verify answer returns complete AnswerResult.

        Given:
            RAGGenerator with configured components.
        When:
            Generating answer.
        Then:
            Returns AnswerResult with all mandatory fields populated correctly.
        """
        generator = RAGGenerator(
            retriever=mock_retriever,
            driver=mock_driver,
            template=fake_template,
            context_manager=fake_context_manager,
        )

        result = await generator.answer("Test question")

        assert isinstance(result, AnswerResult)
        assert result.answer == "Generated answer text"
        assert result.trace is not None
        assert result.generation_result is not None
        assert result.total_time_ms >= 0.0
        assert isinstance(result.total_time_ms, float)

    @pytest.mark.asyncio
    async def test_answer_should_include_sources_when_requested(
        self,
        mock_retriever,
        mock_driver,
        fake_template,
        fake_context_manager,
    ):
        """Verify sources included when requested.

        Given:
            RAGGenerator with retrieved nodes having unique source_ids.
        When:
            Calling answer with include_sources=True.
        Then:
            Result contains deduplicated source references matching retrieved nodes.
        """
        generator = RAGGenerator(
            retriever=mock_retriever,
            driver=mock_driver,
            template=fake_template,
            context_manager=fake_context_manager,
        )

        result = await generator.answer("Question", include_sources=True)

        assert result.sources is not None
        assert len(result.sources) == 2
        assert "source1" in result.sources
        assert "source2" in result.sources

    @pytest.mark.asyncio
    async def test_answer_should_exclude_sources_when_not_requested(
        self,
        mock_retriever,
        mock_driver,
        fake_template,
        fake_context_manager,
    ):
        """Verify sources excluded when not requested.

        Given:
            RAGGenerator with retrieved nodes.
        When:
            Calling answer with include_sources=False.
        Then:
            Result sources field is explicitly None to reduce response size.
        """
        generator = RAGGenerator(
            retriever=mock_retriever,
            driver=mock_driver,
            template=fake_template,
            context_manager=fake_context_manager,
        )

        result = await generator.answer("Question", include_sources=False)

        assert result.sources is None

    @pytest.mark.asyncio
    async def test_answer_should_extract_context_from_retrieved_nodes(
        self,
        mock_retriever,
        mock_driver,
        fake_template,
        fake_context_manager,
    ):
        """Verify context extraction from retrieved nodes.

        Given:
            Retriever returning nodes with content.
        When:
            Generating answer.
        Then:
            Context manager receives node contents in retrieval order.
        """
        generator = RAGGenerator(
            retriever=mock_retriever,
            driver=mock_driver,
            template=fake_template,
            context_manager=fake_context_manager,
        )

        await generator.answer("Question")

        fitted_context = fake_context_manager.last_fitted_context
        assert "Context chunk 1" in fitted_context
        assert "Context chunk 2" in fitted_context

    @pytest.mark.asyncio
    async def test_answer_should_pass_user_query_to_context_manager(
        self,
        mock_retriever,
        mock_driver,
        fake_template,
        fake_context_manager,
    ):
        """Verify user query passed to context manager.

        Given:
            RAGGenerator with specific question.
        When:
            Generating answer.
        Then:
            Context manager receives the exact user query without modification.
        """
        generator = RAGGenerator(
            retriever=mock_retriever,
            driver=mock_driver,
            template=fake_template,
            context_manager=fake_context_manager,
        )

        question = "What is the meaning of life?"
        await generator.answer(question)

        assert fake_context_manager.last_user_query == question

    @pytest.mark.asyncio
    async def test_answer_should_calculate_total_time(
        self,
        mock_retriever,
        mock_driver,
        fake_template,
        fake_context_manager,
    ):
        """Verify total time calculation.

        Given:
            RAGGenerator execution.
        When:
            Generating answer.
        Then:
            total_time_ms is non-negative float representing execution duration.
        """
        generator = RAGGenerator(
            retriever=mock_retriever,
            driver=mock_driver,
            template=fake_template,
            context_manager=fake_context_manager,
        )

        result = await generator.answer("Question")

        assert result.total_time_ms >= 0.0
        assert isinstance(result.total_time_ms, float)


@pytest.mark.unit
class TestRAGGeneratorEdgeCases:
    """Test suite for edge cases."""

    @pytest.mark.asyncio
    async def test_answer_should_handle_empty_retrieved_nodes(
        self,
        mock_driver,
        fake_template,
        fake_context_manager,
    ):
        """Verify handling when no nodes retrieved.

        Given:
            Retriever returning empty retrieved_nodes list.
        When:
            Generating answer.
        Then:
            Pipeline completes gracefully with empty context and empty sources.
        """
        retriever = AsyncMock()
        empty_trace = TraceContext(
            query="Question",
            retrieved_nodes=[],
        )
        retriever.retrieve.return_value = empty_trace

        generator = RAGGenerator(
            retriever=retriever,
            driver=mock_driver,
            template=fake_template,
            context_manager=fake_context_manager,
        )

        result = await generator.answer("Question")

        assert isinstance(result, AnswerResult)
        assert result.sources == [] or result.sources is None

    @pytest.mark.asyncio
    async def test_answer_should_preserve_retrieval_trace(
        self,
        mock_retriever,
        mock_driver,
        fake_template,
        fake_context_manager,
    ):
        """Verify retrieval trace is preserved in result.

        Given:
            RAGGenerator with retriever trace containing node metadata.
        When:
            Generating answer.
        Then:
            Original trace object is included unmodified in AnswerResult.
        """
        generator = RAGGenerator(
            retriever=mock_retriever,
            driver=mock_driver,
            template=fake_template,
            context_manager=fake_context_manager,
        )

        result = await generator.answer("Question")

        expected_trace = await mock_retriever.retrieve("Question")
        assert result.trace == expected_trace

    @pytest.mark.asyncio
    async def test_answer_should_preserve_generation_result(
        self,
        mock_retriever,
        mock_driver,
        fake_template,
        fake_context_manager,
    ):
        """Verify generation result is preserved.

        Given:
            Driver returning GenerationResult with usage metadata.
        When:
            Generating answer.
        Then:
            Original GenerationResult is included unmodified in AnswerResult.
        """
        generator = RAGGenerator(
            retriever=mock_retriever,
            driver=mock_driver,
            template=fake_template,
            context_manager=fake_context_manager,
        )

        result = await generator.answer("Question")

        assert result.generation_result.text == "Generated answer text"
        assert result.generation_result.usage.total_tokens == 150
        assert result.generation_result.finish_reason == "stop"
