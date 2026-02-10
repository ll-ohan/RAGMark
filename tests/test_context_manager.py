"""Test suite for context window management with truncation strategies.

Validates ContextManager behavior across all truncation strategies (priority,
truncate_end, truncate_middle). Uses a fake driver implementation per
TEST_POLICY.md Section 2.3 to avoid heavy tokenizer dependencies.
"""

import time
from collections.abc import AsyncIterator
from typing import Literal

import pytest
from typing_extensions import Any

from ragmark.generation.context import ContextManager
from ragmark.generation.drivers import BaseLLMDriver
from ragmark.schemas.generation import GenerationResult


class FakeLLMDriver(BaseLLMDriver):
    """Fake LLM driver for testing context management.

    Implements simplified token counting without external dependencies.
    Uses 1 token per 5 characters rule for deterministic testing.
    """

    def __init__(self, context_window: int = 100):
        self._context_window = context_window

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
    ) -> "GenerationResult":
        raise NotImplementedError("FakeLLMDriver does not implement generate")

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> "AsyncIterator[str]":
        raise NotImplementedError("FakeLLMDriver does not implement generate_stream")

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> None:
        return None

    def count_tokens(self, text: str) -> int:
        """Count tokens using simplified character-based approximation."""
        return max(1, len(text) // 5)

    @property
    def context_window(self) -> int:
        return self._context_window

    @context_window.setter
    def context_window(self, value: int) -> None:
        self._context_window = value


@pytest.fixture
def fake_driver() -> FakeLLMDriver:
    """Create fake LLM driver with deterministic token counting.

    Uses real implementation per TEST_POLICY.md Section 2.3.
    """
    return FakeLLMDriver(context_window=100)


@pytest.mark.unit
class TestContextManagerInitialization:
    """Test suite for ContextManager initialization."""

    def test_init_should_store_driver_and_strategy(self, fake_driver: FakeLLMDriver):
        """Verify initialization stores configuration.

        Given:
            Driver and explicit truncate_end strategy.
        When:
            Initializing ContextManager.
        Then:
            Both driver and strategy are stored as instance attributes.
        """
        manager = ContextManager(fake_driver, strategy="truncate_end")

        assert manager.driver is fake_driver
        assert manager.strategy == "truncate_end"

    def test_init_should_default_to_priority_strategy(self, fake_driver: FakeLLMDriver):
        """Verify default strategy is priority.

        Given:
            Driver initialization without explicit strategy parameter.
        When:
            Initializing ContextManager.
        Then:
            Strategy defaults to 'priority' for relevance-based ordering.
        """
        manager = ContextManager(fake_driver)

        assert manager.strategy == "priority"


@pytest.mark.unit
class TestContextManagerFitContext:
    """Test suite for fit_context method."""

    def test_fit_context_should_include_all_chunks_when_space_available(
        self, fake_driver: FakeLLMDriver
    ):
        """Verify all chunks included when they fit.

        Given:
            Two small chunks that fit within the 100-token context window.
        When:
            Calling fit_context with priority strategy.
        Then:
            All chunks plus system prompt and query are present in output.
        """
        manager = ContextManager(fake_driver, strategy="priority")

        prompt = manager.fit_context(
            system="System",
            context_chunks=["Chunk 1", "Chunk 2"],
            user_query="Query?",
            max_completion=20,
        )

        assert "Chunk 1" in prompt
        assert "Chunk 2" in prompt
        assert "System" in prompt
        assert "Query?" in prompt

    def test_fit_context_should_truncate_when_exceeds_window(
        self, fake_driver: FakeLLMDriver
    ):
        """Verify truncation when chunks exceed window.

        Given:
            10 large chunks totaling more than 100 tokens (context window).
        When:
            Calling fit_context with priority strategy.
        Then:
            Only highest priority chunks that fit are included.
        """
        manager = ContextManager(fake_driver, strategy="priority")

        chunks = [
            f"Chunk {i} with lots of text content here to make it long"
            for i in range(10)
        ]

        prompt = manager.fit_context(
            system="Sys",
            context_chunks=chunks,
            user_query="Q?",
            max_completion=20,
        )

        included_count = sum(1 for chunk in chunks if chunk in prompt)
        assert 0 < included_count < len(chunks)
        assert "Chunk 0" in prompt

    def test_fit_context_should_handle_empty_chunks_list(
        self, fake_driver: FakeLLMDriver
    ):
        """Verify handling of empty chunks list.

        Given:
            Empty context_chunks list (no retrieval results).
        When:
            Calling fit_context.
        Then:
            Prompt contains system and query without context section.
        """
        manager = ContextManager(fake_driver, strategy="priority")

        prompt = manager.fit_context(
            system="System",
            context_chunks=[],
            user_query="Query?",
            max_completion=20,
        )

        assert "System" in prompt
        assert "Query?" in prompt
        assert "Context:" not in prompt

    def test_fit_context_should_handle_system_query_exceeding_window(
        self, fake_driver: FakeLLMDriver
    ):
        """Verify handling when system + query exceed window.

        Given:
            System prompt and query that together exceed the context window.
        When:
            Calling fit_context.
        Then:
            System prompt is preserved and function handles overflow gracefully.
        """
        manager = ContextManager(fake_driver, strategy="priority")

        long_system = "S" * 200
        long_query = "Q" * 300

        prompt = manager.fit_context(
            system=long_system,
            context_chunks=["Chunk"],
            user_query=long_query,
            max_completion=20,
        )

        assert prompt
        assert len(prompt) > 0
        assert long_system in prompt


@pytest.mark.unit
class TestContextManagerTruncateEnd:
    """Test suite for truncate_end strategy."""

    def test_truncate_end_should_keep_first_chunks(self, fake_driver: FakeLLMDriver):
        """Verify truncate_end keeps first chunks that fit.

        Given:
            10 large chunks exceeding the 100-token context window.
        When:
            Using truncate_end strategy.
        Then:
            First chunks that fit are kept; later chunks beyond budget are dropped.
        """
        manager = ContextManager(fake_driver, strategy="truncate_end")

        chunks = [
            f"Chunk {i} with lots of text content here to make it long"
            for i in range(10)
        ]

        prompt = manager.fit_context(
            system="S",
            context_chunks=chunks,
            user_query="Q",
            max_completion=20,
        )

        assert "Chunk 0" in prompt
        assert "Chunk 1" in prompt
        assert "Chunk 9" not in prompt

    def test_truncate_end_should_include_all_when_fit(self, fake_driver: FakeLLMDriver):
        """Verify all chunks included when they fit.

        Given:
            Two tiny chunks ("A", "B") that easily fit in window.
        When:
            Using truncate_end strategy.
        Then:
            Both chunks are present in the generated prompt.
        """
        manager = ContextManager(fake_driver, strategy="truncate_end")

        chunks = ["A", "B"]

        prompt = manager.fit_context(
            system="S",
            context_chunks=chunks,
            user_query="Q",
            max_completion=10,
        )

        assert "A" in prompt
        assert "B" in prompt


@pytest.mark.unit
class TestContextManagerTruncateMiddle:
    """Test suite for truncate_middle strategy."""

    def test_truncate_middle_should_keep_first_and_last(
        self, fake_driver: FakeLLMDriver
    ):
        """Verify truncate_middle keeps first and last chunks.

        Given:
            10 chunks exceeding available context space.
        When:
            Using truncate_middle strategy.
        Then:
            First chunk (Chunk 0) and last chunk (Chunk 9) are both preserved.
        """
        manager = ContextManager(fake_driver, strategy="truncate_middle")

        chunks = [f"Chunk {i} with text" for i in range(10)]

        prompt = manager.fit_context(
            system="S",
            context_chunks=chunks,
            user_query="Q",
            max_completion=20,
        )

        assert "Chunk 0" in prompt
        assert "Chunk 9" in prompt

    def test_truncate_middle_should_handle_single_chunk(
        self, fake_driver: FakeLLMDriver
    ):
        """Verify single chunk handling.

        Given:
            Single chunk in context_chunks list.
        When:
            Using truncate_middle strategy.
        Then:
            The single chunk is included (no truncation needed).
        """
        manager = ContextManager(fake_driver, strategy="truncate_middle")

        chunks = ["Only chunk"]

        prompt = manager.fit_context(
            system="S",
            context_chunks=chunks,
            user_query="Q",
            max_completion=20,
        )

        assert "Only chunk" in prompt

    def test_truncate_middle_should_handle_empty_chunks(
        self, fake_driver: FakeLLMDriver
    ):
        """Verify empty chunks handling.

        Given:
            Empty chunks list.
        When:
            Using truncate_middle strategy.
        Then:
            Prompt omits context section entirely.
        """
        manager = ContextManager(fake_driver, strategy="truncate_middle")

        prompt = manager.fit_context(
            system="S",
            context_chunks=[],
            user_query="Q",
            max_completion=20,
        )

        assert "Context:" not in prompt


@pytest.mark.unit
class TestContextManagerTruncatePriority:
    """Test suite for truncate_priority strategy."""

    def test_truncate_priority_should_keep_most_relevant(
        self, fake_driver: FakeLLMDriver
    ):
        """Verify priority strategy keeps most relevant chunks.

        Given:
            10 large chunks ordered by relevance (most relevant first).
        When:
            Using priority strategy with limited context budget.
        Then:
            Highest priority chunks (Relevant 0, 1) are kept; lowest (9) is dropped.
        """
        manager = ContextManager(fake_driver, strategy="priority")

        chunks = [
            f"Relevant {i} with lots of text content here to make it long"
            for i in range(10)
        ]

        prompt = manager.fit_context(
            system="S",
            context_chunks=chunks,
            user_query="Q",
            max_completion=20,
        )

        assert "Relevant 0" in prompt
        assert "Relevant 1" in prompt
        assert "Relevant 9" not in prompt

    def test_truncate_priority_should_respect_chunk_order(
        self, fake_driver: FakeLLMDriver
    ):
        """Verify priority respects input order.

        Given:
            Chunks ordered by decreasing relevance: High, Med, Low.
        When:
            Using priority strategy.
        Then:
            Output preserves this order with High appearing before Med.
        """
        manager = ContextManager(fake_driver, strategy="priority")

        chunks = ["High", "Med", "Low"]

        prompt = manager.fit_context(
            system="S",
            context_chunks=chunks,
            user_query="Q",
            max_completion=20,
        )

        high_idx = prompt.find("High")
        med_idx = prompt.find("Med")
        assert high_idx != -1
        assert med_idx != -1
        assert high_idx < med_idx


@pytest.mark.unit
class TestContextManagerPromptBuilding:
    """Test suite for prompt assembly."""

    def test_build_prompt_should_format_correctly(self, fake_driver: FakeLLMDriver):
        """Verify prompt formatting.

        Given:
            System prompt, two context chunks, and user query.
        When:
            Building prompt via fit_context.
        Then:
            Prompt contains all components in correct format with numbered chunks.
        """
        manager = ContextManager(fake_driver, strategy="priority")

        prompt = manager.fit_context(
            system="You are a helpful assistant.",
            context_chunks=["Context 1", "Context 2"],
            user_query="What is the answer?",
            max_completion=20,
        )

        assert prompt.startswith("You are a helpful assistant.")
        assert "Context:" in prompt
        assert "[1] Context 1" in prompt
        assert "[2] Context 2" in prompt
        assert "User: What is the answer?" in prompt

    def test_build_prompt_should_number_chunks(self, fake_driver: FakeLLMDriver):
        """Verify chunks are numbered.

        Given:
            Three context chunks (A, B, C).
        When:
            Building prompt.
        Then:
            Chunks are numbered with [1], [2], [3] prefixes for citation.
        """
        manager = ContextManager(fake_driver, strategy="priority")

        prompt = manager.fit_context(
            system="System",
            context_chunks=["A", "B", "C"],
            user_query="Query",
            max_completion=10,
        )

        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "[3]" in prompt


@pytest.mark.unit
class TestContextManagerEdgeCases:
    """Test suite for edge cases."""

    def test_fit_context_with_very_small_window(self, fake_driver: FakeLLMDriver):
        """Verify handling of very small context window.

        Given:
            Severely constrained context window of only 10 tokens.
        When:
            Attempting to fit normal content.
        Then:
            Produces valid prompt without crashing despite tight constraints.
        """
        fake_driver.context_window = 10

        manager = ContextManager(fake_driver, strategy="priority")

        prompt = manager.fit_context(
            system="S",
            context_chunks=["Chunk"],
            user_query="Q",
            max_completion=5,
        )

        assert len(prompt) > 0
        assert "S" in prompt

    def test_fit_context_with_unicode_text(self, fake_driver: FakeLLMDriver):
        """Verify handling of Unicode text.

        Given:
            Context with French accents and Chinese characters (multi-byte UTF-8).
        When:
            Fitting context.
        Then:
            Unicode is preserved correctly without corruption or truncation issues.
        """
        manager = ContextManager(fake_driver, strategy="priority")

        prompt = manager.fit_context(
            system="Système",
            context_chunks=["Données françaises", "中文内容"],
            user_query="Quelle est la réponse?",
            max_completion=20,
        )

        assert "Système" in prompt
        assert "Données françaises" in prompt
        assert "中文内容" in prompt

    def test_fit_context_with_zero_max_completion(self, fake_driver: FakeLLMDriver):
        """Verify handling of zero max_completion.

        Given:
            max_completion explicitly set to 0 tokens.
        When:
            Fitting context.
        Then:
            All available window space is allocated to context chunks.
        """
        manager = ContextManager(fake_driver, strategy="priority")

        prompt = manager.fit_context(
            system="S",
            context_chunks=["Chunk 1", "Chunk 2"],
            user_query="Q",
            max_completion=0,
        )

        assert len(prompt) > 0
        assert "Chunk 1" in prompt or "Chunk 2" in prompt


@pytest.mark.unit
@pytest.mark.benchmark
@pytest.mark.performance
class TestContextTruncateMiddlePerformance:
    """Benchmark tests for Context._truncate_middle() deque optimization."""

    def test_truncate_middle_1000_chunks_should_complete_under_5ms(self) -> None:
        """Verifies _truncate_middle() achieves O(n) performance via deque optimization.

        Given:
            A context manager with 1000 chunks to truncate.
        When:
            Applying truncate_middle strategy with tight token budget.
        Then:
            Operation completes in under 5ms (40x faster than O(n²)), and
            first/last chunks are preserved per truncate_middle semantics.
        """
        driver = FakeLLMDriver(context_window=1000)
        manager = ContextManager(driver, strategy="truncate_middle")
        chunks = [f"Chunk {i} with some content here" for i in range(1000)]

        start = time.perf_counter()
        selected = manager.fit_context(
            system="System",
            context_chunks=chunks,
            user_query="Query",
            max_completion=100,
        )
        duration_ms = (time.perf_counter() - start) * 1000

        assert "Chunk 0" in selected
        assert "Chunk 999" in selected
        assert duration_ms < 5, f"Truncate took {duration_ms:.2f}ms, expected <5ms"

    def test_truncate_middle_should_preserve_ordering_of_first_and_last_chunks(
        self,
    ) -> None:
        """Verifies truncate_middle preserves chunk ordering for first and last halves.

        Given:
            Context manager with 20 identifiable chunks exceeding budget.
        When:
            Applying truncate_middle strategy.
        Then:
            First chunks appear in original order, last chunks appear in
            original order, and middle chunks are omitted to fit budget.
        """
        driver = FakeLLMDriver(context_window=100)
        manager = ContextManager(driver, strategy="truncate_middle")
        chunks = [f"Item_{i:02d}" for i in range(20)]

        selected = manager.fit_context(
            system="S",
            context_chunks=chunks,
            user_query="Q",
            max_completion=20,
        )

        first_idx = selected.find("Item_00")
        last_idx = selected.find("Item_19")
        assert first_idx != -1
        assert last_idx != -1
        assert first_idx < last_idx
