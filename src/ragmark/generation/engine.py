"""RAG pipeline orchestration.

This module provides the RAGGenerator class that combines retrieval, context
management, prompting, and generation into a single cohesive interface.
"""

import time

from ragmark.generation.context import ContextManager
from ragmark.generation.drivers import BaseLLMDriver
from ragmark.generation.prompts import PromptTemplate
from ragmark.logger import get_logger
from ragmark.retrieval.base import BaseRetriever
from ragmark.schemas.generation import AnswerResult

logger = get_logger(__name__)


class RAGGenerator:
    """End-to-end RAG pipeline orchestrator.

    Combines retrieval, context management, prompting, and generation
    into a single cohesive interface for question answering.

    Attributes:
        retriever: Retrieval component.
        driver: LLM driver for generation.
        template: Prompt template.
        context_manager: Context window manager.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        driver: BaseLLMDriver,
        template: PromptTemplate,
        context_manager: ContextManager,
    ):
        """Initialize RAG generator.

        Args:
            retriever: Retrieval component for finding relevant context.
            driver: LLM driver for text generation.
            template: Prompt template for formatting context and query.
            context_manager: Manager for fitting context in window.
        """
        self.retriever = retriever
        self.driver = driver
        self.template = template
        self.context_manager = context_manager

        logger.info(
            "RAGGenerator initialized: retriever=%s, driver=%s",
            type(retriever).__name__,
            type(driver).__name__,
        )

    async def answer(
        self,
        question: str,
        include_sources: bool = True,
        max_completion: int = 512,
        temperature: float = 0.7,
    ) -> AnswerResult:
        """Generate answer for a question using RAG pipeline.

        Args:
            question: User's question.
            include_sources: Include source references in result.
            max_completion: Tokens reserved for answer generation.
            temperature: Sampling temperature for generation.

        Returns:
            Complete answer with retrieval trace and metadata.
        """
        start_time = time.perf_counter()
        logger.info("RAG pipeline started: question_preview=%s...", question[:80])

        logger.debug("Initiating retrieval for query")
        trace = await self.retriever.retrieve(question)
        logger.debug("Retrieval completed: nodes=%d", len(trace.retrieved_nodes))

        context_chunks = [node.node.content for node in trace.retrieved_nodes]
        logger.debug("Extracted context chunks: count=%d", len(context_chunks))

        # Using direct system message instead of template.render() for simplicity
        # Complex prompt engineering should use PromptTemplate.render() instead
        system_message = (
            "You are a helpful AI assistant. "
            "Answer the question based on the provided context."
        )

        logger.debug(
            "Fitting context to model window: max_completion=%d", max_completion
        )
        prompt = self.context_manager.fit_context(
            system=system_message,
            context_chunks=context_chunks,
            user_query=question,
            max_completion=max_completion,
        )
        logger.debug("Context fitting completed: final_prompt_len=%d", len(prompt))

        logger.debug("Starting generation: temperature=%.2f", temperature)
        generation_result = await self.driver.generate(
            prompt=prompt,
            max_tokens=max_completion,
            temperature=temperature,
        )

        total_time = (time.perf_counter() - start_time) * 1000
        logger.info(
            "RAG pipeline completed: answer_len=%d, total_time_ms=%.2f",
            len(generation_result.text),
            total_time,
        )

        sources = None
        if include_sources:
            sources = [node.node.source_id for node in trace.retrieved_nodes]
            logger.debug("Source extraction completed: sources=%d", len(sources))

        return AnswerResult(
            answer=generation_result.text,
            trace=trace,
            generation_result=generation_result,
            total_time_ms=total_time,
            sources=sources,
        )
