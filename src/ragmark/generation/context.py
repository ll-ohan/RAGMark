"""Context window management for LLM prompts.

This module provides utilities for fitting retrieved context into LLM context windows
while maximizing information retention through various truncation strategies.
"""

from typing import Literal

from ragmark.generation.drivers import BaseLLMDriver
from ragmark.logger import get_logger

logger = get_logger(__name__)


class ContextManager:
    """Manage context window truncation for LLM prompts.

    Ensures total prompt fits within model's context window while
    maximizing relevant information retention through configurable
    truncation strategies.

    Attributes:
        driver: LLM driver providing token counting and context window.
        strategy: Truncation strategy to apply when context exceeds window.
    """

    def __init__(
        self,
        driver: BaseLLMDriver,
        strategy: Literal["truncate_end", "truncate_middle", "priority"] = "priority",
    ):
        """Initialize context manager.

        Args:
            driver: Provides token counting and context window size.
            strategy: Truncation approach when context exceeds window.
                - truncate_end: Keep first N chunks that fit.
                - truncate_middle: Keep first and last chunks, remove middle.
                - priority: Keep highest priority (most relevant) chunks.
        """
        self.driver = driver
        self.strategy = strategy

    def fit_context(
        self,
        system: str,
        context_chunks: list[str],
        user_query: str,
        max_completion: int = 512,
    ) -> str:
        """Fit context within window using configured strategy.

        Args:
            system: System message text.
            context_chunks: Retrieved context ordered by relevance.
            user_query: User's question.
            max_completion: Tokens reserved for model completion.

        Returns:
            Assembled prompt fitting within context window.
        """
        available_tokens = self.driver.context_window - max_completion

        system_tokens = self.driver.count_tokens(system)
        query_tokens = self.driver.count_tokens(user_query)
        fixed_tokens = system_tokens + query_tokens

        logger.debug(
            "Context fitting started: window=%d, reserved_completion=%d, "
            "available=%d, system_tokens=%d, query_tokens=%d",
            self.driver.context_window,
            max_completion,
            available_tokens,
            system_tokens,
            query_tokens,
        )

        if fixed_tokens >= available_tokens:
            logger.warning(
                "Fixed prompt exceeds window: system=%d, query=%d, available=%d",
                system_tokens,
                query_tokens,
                available_tokens,
            )
            remaining = available_tokens - system_tokens
            if remaining < 10:
                logger.error(
                    "Context window exhausted: cannot fit minimal query, "
                    "system=%d, window=%d",
                    system_tokens,
                    self.driver.context_window,
                )
                return self._build_prompt(system, [], user_query[:50])

            truncated_query = self._truncate_text_to_tokens(user_query, remaining - 5)
            logger.warning(
                "Query truncated to fit: original_tokens=%d, truncated_to=%d",
                query_tokens,
                remaining - 5,
            )
            return self._build_prompt(system, [], truncated_query)

        available_for_context = available_tokens - fixed_tokens

        if self.strategy == "truncate_end":
            selected_chunks = self._truncate_end(context_chunks, available_for_context)
        elif self.strategy == "truncate_middle":
            selected_chunks = self._truncate_middle(
                context_chunks, available_for_context
            )
        else:
            selected_chunks = self._truncate_priority(
                context_chunks, available_for_context
            )

        if len(selected_chunks) < len(context_chunks):
            removed = len(context_chunks) - len(selected_chunks)
            logger.warning(
                "Context chunks truncated: strategy=%s, removed=%d, kept=%d",
                self.strategy,
                removed,
                len(selected_chunks),
            )
        else:
            logger.debug(
                "All context chunks fit: total=%d, strategy=%s",
                len(context_chunks),
                self.strategy,
            )

        return self._build_prompt(system, selected_chunks, user_query)

    def _truncate_end(self, chunks: list[str], available: int) -> list[str]:
        """Keep first N chunks that fit.

        Args:
            chunks: Context chunks ordered by relevance.
            available: Token budget for context.

        Returns:
            Prefix of chunks fitting within token budget.
        """
        selected = []
        used_tokens = 0

        for chunk in chunks:
            chunk_tokens = self.driver.count_tokens(chunk)
            if used_tokens + chunk_tokens <= available:
                selected.append(chunk)
                used_tokens += chunk_tokens
            else:
                break

        logger.debug(
            "Truncate end strategy: selected=%d/%d, used_tokens=%d/%d",
            len(selected),
            len(chunks),
            used_tokens,
            available,
        )
        return selected

    def _truncate_middle(self, chunks: list[str], available: int) -> list[str]:
        """Keep first and last chunks, remove middle.

        Preserves context from beginning and end of retrieved set,
        alternating between adding from start and end until budget exhausted.

        Args:
            chunks: Context chunks ordered by relevance.
            available: Token budget for context.

        Returns:
            Chunks from start and end fitting within token budget.
        """
        if not chunks:
            return []

        first_half = []
        last_half = []
        used_tokens = 0

        if chunks:
            first_chunk = chunks[0]
            first_tokens = self.driver.count_tokens(first_chunk)
            if first_tokens <= available:
                first_half.append(first_chunk)
                used_tokens += first_tokens

        if len(chunks) > 1:
            last_chunk = chunks[-1]
            last_tokens = self.driver.count_tokens(last_chunk)
            if used_tokens + last_tokens <= available:
                last_half.append(last_chunk)
                used_tokens += last_tokens

        left_idx = 1
        right_idx = len(chunks) - 2
        add_left = True

        while left_idx <= right_idx and used_tokens < available:
            if add_left:
                chunk = chunks[left_idx]
                tokens = self.driver.count_tokens(chunk)
                if used_tokens + tokens <= available:
                    first_half.append(chunk)
                    used_tokens += tokens
                    left_idx += 1
                add_left = False
            else:
                chunk = chunks[right_idx]
                tokens = self.driver.count_tokens(chunk)
                if used_tokens + tokens <= available:
                    last_half.insert(0, chunk)
                    used_tokens += tokens
                    right_idx -= 1
                add_left = True

        selected = first_half + last_half
        logger.debug(
            "Truncate middle strategy: selected=%d/%d, used_tokens=%d/%d",
            len(selected),
            len(chunks),
            used_tokens,
            available,
        )
        return selected

    def _truncate_priority(self, chunks: list[str], available: int) -> list[str]:
        """Keep highest priority (most relevant) chunks.

        Since chunks are pre-sorted by retrieval score, this strategy
        is equivalent to keeping the first N chunks that fit.

        Args:
            chunks: Context chunks ordered by relevance (most relevant first).
            available: Token budget for context.

        Returns:
            Highest priority chunks fitting within token budget.
        """
        selected = []
        used_tokens = 0

        for chunk in chunks:
            chunk_tokens = self.driver.count_tokens(chunk)
            if used_tokens + chunk_tokens <= available:
                selected.append(chunk)
                used_tokens += chunk_tokens
            else:
                break

        logger.debug(
            "Truncate priority strategy: selected=%d/%d, used_tokens=%d/%d",
            len(selected),
            len(chunks),
            used_tokens,
            available,
        )
        return selected

    def _truncate_text_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit in maximum tokens.

        Uses character-based estimation rather than actual tokenization
        for performance. Assumes ~4 characters per token for English.

        Args:
            text: Text to truncate.
            max_tokens: Maximum token budget.

        Returns:
            Truncated text with ellipsis if shortened.
        """
        estimated_chars = max_tokens * 4
        if len(text) <= estimated_chars:
            return text

        return text[:estimated_chars] + "..."

    def _build_prompt(self, system: str, chunks: list[str], query: str) -> str:
        """Assemble final prompt from components.

        Args:
            system: System message.
            chunks: Selected context chunks.
            query: User query.

        Returns:
            Formatted prompt with numbered context chunks.
        """
        parts = [system]

        if chunks:
            parts.append("\nContext:")
            for i, chunk in enumerate(chunks, 1):
                parts.append(f"\n[{i}] {chunk}")
            parts.append("\n")

        parts.append(f"\nUser: {query}")

        prompt = "".join(parts)
        logger.debug(
            "Prompt assembled: total_chars=%d, chunks=%d", len(prompt), len(chunks)
        )
        return prompt
