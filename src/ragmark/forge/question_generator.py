"""Synthetic question-answer generation for knowledge nodes.

This module provides interfaces and implementations for generating synthetic
QA pairs from knowledge node content using language models.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ragmark.exceptions import QuestionGenerationError
from ragmark.generation.drivers import BaseLLMDriver, LlamaCppDriver
from ragmark.generation.prompts import PromptTemplate
from ragmark.logger import get_logger
from ragmark.schemas.documents import KnowledgeNode
from ragmark.schemas.qa import BatchQAOutput, SyntheticQA

if TYPE_CHECKING:
    from ragmark.config.profile import QuestionGeneratorConfig
    from ragmark.forge.qa_validator import QAValidator

logger = get_logger(__name__)


class BaseQuestionGenerator(ABC):
    """Abstract base class for synthetic question-answer generation.

    Generates synthetic QA pairs for knowledge nodes to support
    evaluation and test case creation.
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: "QuestionGeneratorConfig") -> "BaseQuestionGenerator":
        """Create generator from configuration.

        Args:
            config: Question generator configuration.

        Returns:
            Configured question generator instance.
        """
        pass

    @abstractmethod
    async def generate_async(self, node: KnowledgeNode) -> KnowledgeNode:
        """Generate QA pairs asynchronously for a single node.

        Args:
            node: Knowledge node to enrich with QA pairs.

        Returns:
            Enriched node with synthetic_qa metadata.

        Raises:
            QuestionGenerationError: If generation fails.
        """
        pass

    @abstractmethod
    async def generate_batch_async(
        self, nodes: list[KnowledgeNode]
    ) -> list[KnowledgeNode]:
        """Generate QA pairs for multiple nodes in batch (optimized).

        Args:
            nodes: List of knowledge nodes to process.

        Returns:
            List of enriched nodes with synthetic_qa metadata.

        Raises:
            QuestionGenerationError: If batch generation fails.
        """
        pass

    def generate_sync(self, node: KnowledgeNode) -> KnowledgeNode:
        """Synchronous wrapper for single node generation.

        Args:
            node: Knowledge node to enrich.

        Returns:
            Enriched node with synthetic_qa metadata.

        Raises:
            QuestionGenerationError: If generation fails.
        """
        return asyncio.run(self.generate_async(node))

    async def generate_stream_async(
        self, nodes: AsyncIterator[KnowledgeNode], batch_size: int = 4
    ) -> AsyncIterator[KnowledgeNode]:
        """Generate QA pairs with streaming input/output and batching.

        Args:
            nodes: Async iterator of knowledge nodes.
            batch_size: Number of nodes to batch per LLM call.

        Yields:
            Enriched nodes with synthetic_qa metadata.

        Raises:
            QuestionGenerationError: If generation fails for a batch.
        """
        buffer: list[KnowledgeNode] = []

        async for node in nodes:
            buffer.append(node)

            if len(buffer) >= batch_size:
                logger.debug("Processing QA batch: size=%d", len(buffer))
                try:
                    enriched_batch = await self.generate_batch_async(buffer)
                    for enriched_node in enriched_batch:
                        yield enriched_node
                except QuestionGenerationError as e:
                    logger.warning("QA batch failed: %s", e.message)
                    logger.debug("Batch failure details: %s", e, exc_info=True)
                    raise
                buffer.clear()

        if buffer:
            logger.debug("Processing final QA batch: size=%d", len(buffer))
            try:
                enriched_batch = await self.generate_batch_async(buffer)
                for enriched_node in enriched_batch:
                    yield enriched_node
            except QuestionGenerationError as e:
                logger.warning("Final QA batch failed: %s", e.message)
                logger.debug("Batch failure details: %s", e, exc_info=True)
                raise

    @property
    @abstractmethod
    def num_questions(self) -> int:
        """Number of questions to generate per node.

        Returns:
            Configured number of QA pairs.
        """
        pass


class LLMQuestionGenerator(BaseQuestionGenerator):
    """LLM-based synthetic QA generator with batching and validation.

    Uses a language model to generate relevant question-answer pairs that can
    be answered from the content of each knowledge node. Supports batch
    processing for performance and optional validation for quality control.

    Attributes:
        driver: LLM driver for generation.
        prompt_template: Template for QA generation prompts.
        num_questions: Number of QA pairs per node.
        batch_size: Nodes to process per LLM call.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens per generation.
        validator: QA validation strategy.
    """

    def __init__(
        self,
        driver: BaseLLMDriver,
        prompt_template: PromptTemplate,
        num_questions: int = 3,
        batch_size: int = 4,
        temperature: float = 0.7,
        max_tokens: int = 512,
        validator: "QAValidator | None" = None,
    ):
        """Initialize the LLM question generator.

        Args:
            driver: LLM driver instance.
            prompt_template: Prompt template for generation.
            num_questions: QA pairs per node.
            batch_size: Nodes per LLM batch.
            temperature: Sampling temperature.
            max_tokens: Maximum generation tokens.
            validator: Optional QA validator.
        """
        self._driver = driver
        self._template = prompt_template
        self._num_questions = num_questions
        self._batch_size = batch_size
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._validator = validator
        self._batch_counter = 0

    @classmethod
    def from_config(cls, config: "QuestionGeneratorConfig") -> "LLMQuestionGenerator":
        """Create from configuration.

        Args:
            config: Question generator configuration.

        Returns:
            Configured LLM question generator instance.
        """
        from ragmark.forge.qa_validator import BasicQAValidator
        from ragmark.generation.prompts import SYNTHETIC_QA_BATCH_TEMPLATE

        driver = LlamaCppDriver(
            model_path=config.model_path,
            n_ctx=config.context_window,
            n_gpu_layers=config.n_gpu_layers,
        )

        template = SYNTHETIC_QA_BATCH_TEMPLATE

        validator = BasicQAValidator() if config.validation else None

        return cls(
            driver=driver,
            prompt_template=template,
            num_questions=config.num_questions,
            batch_size=config.batch_size,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            validator=validator,
        )

    async def generate_async(self, node: KnowledgeNode) -> KnowledgeNode:
        """Generate QA pairs asynchronously for a single node.

        Args:
            node: Knowledge node to enrich.

        Returns:
            Enriched node with synthetic_qa metadata.

        Raises:
            QuestionGenerationError: If generation fails.
        """
        try:
            enriched_batch = await self.generate_batch_async([node])
            return enriched_batch[0]
        except QuestionGenerationError:
            raise
        except Exception as e:
            logger.error("QA generation failed: node=%s", node.node_id)
            logger.debug("Generation error details: %s", e, exc_info=True)
            raise QuestionGenerationError(
                f"Failed to generate QA for node {node.node_id}", node_id=node.node_id
            ) from e

    async def generate_batch_async(
        self, nodes: list[KnowledgeNode]
    ) -> list[KnowledgeNode]:
        """Generate QA pairs for multiple nodes in single LLM call.

        Combines node contents with separators, generates all QA pairs
        in one call, then parses and distributes back to nodes.

        Args:
            nodes: List of knowledge nodes.

        Returns:
            List of enriched nodes.

        Raises:
            QuestionGenerationError: If batch generation fails.
        """
        if not nodes:
            return []

        self._batch_counter += 1
        batch_id = f"batch_{self._batch_counter}"

        start_time = time.perf_counter()

        logger.debug(
            "Starting QA batch: batch_id=%s, nodes=%d, num_questions=%d",
            batch_id,
            len(nodes),
            self._num_questions,
        )

        try:
            prompt = self._template.render(
                nodes=[{"content": n.content, "node_id": n.node_id} for n in nodes],
                num_questions=self._num_questions,
            )

            logger.debug(
                "Generating QA batch: batch_id=%s, prompt_chars=%d",
                batch_id,
                len(prompt),
            )

            # Use JSON mode for structured output (requires compatible LLM)
            result = await self._driver.generate(
                prompt=prompt,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                response_format={"type": "json_object"},
            )

            logger.debug(
                "QA batch generated: batch_id=%s, output_chars=%d",
                batch_id,
                len(result.text),
            )

            qa_pairs_by_node = self._parse_batch_qa(result.text, len(nodes))

            enriched = []
            total_parsed = 0
            total_validated = 0

            for node, qa_pairs in zip(nodes, qa_pairs_by_node, strict=True):
                total_parsed += len(qa_pairs)

                if self._validator and qa_pairs:
                    validated_pairs = self._validator.validate(qa_pairs)
                    total_validated += len(validated_pairs)

                    if len(validated_pairs) < len(qa_pairs):
                        logger.debug(
                            "QA validation filtered: node=%s, parsed=%d, valid=%d",
                            node.node_id,
                            len(qa_pairs),
                            len(validated_pairs),
                        )
                else:
                    validated_pairs = qa_pairs
                    total_validated += len(qa_pairs)

                enriched_node = self._enrich_node(node, validated_pairs, batch_id)
                enriched.append(enriched_node)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            logger.debug(
                "QA batch completed: batch_id=%s, nodes=%d, parsed=%d, validated=%d, "
                "avg_qa=%.1f, elapsed_ms=%.2f",
                batch_id,
                len(nodes),
                total_parsed,
                total_validated,
                total_validated / max(len(nodes), 1),
                elapsed_ms,
            )

            return enriched

        except Exception as e:
            logger.error(
                "QA batch generation failed: batch_id=%s, nodes=%d",
                batch_id,
                len(nodes),
            )
            logger.debug("Batch generation error details: %s", e, exc_info=True)
            raise QuestionGenerationError(
                f"Failed to generate QA batch {batch_id}"
            ) from e

    def _parse_batch_qa(
        self, generated_text: str, num_nodes: int
    ) -> list[list[SyntheticQA]]:
        """Parse QA pairs from batch generation JSON output.

        Uses structured JSON parsing with Pydantic validation instead of
        fragile regex patterns. This approach is robust to LLM output variations.

        Expected JSON format:
        {
          "chunks": [
            {
              "chunk_id": 1,
              "qa_pairs": [
                {"question": "...", "answer": "..."},
                ...
              ]
            },
            ...
          ]
        }

        Args:
            generated_text: LLM JSON output text.
            num_nodes: Expected number of nodes.

        Returns:
            List of QA pairs per node, ordered by chunk_id.

        Raises:
            QuestionGenerationError: If JSON parsing or validation fails.
        """
        try:
            # Parse JSON with error handling
            json_data = json.loads(generated_text)

            # Validate with Pydantic schema
            batch_output = BatchQAOutput.model_validate(json_data)

            # Create a mapping from chunk_id to qa_pairs
            qa_by_chunk_id: dict[int, list[SyntheticQA]] = {}

            for chunk in batch_output.chunks:
                # Limit to requested number of questions
                limited_pairs = chunk.qa_pairs[: self._num_questions]
                qa_by_chunk_id[chunk.chunk_id] = limited_pairs

            # Build output list ordered by chunk_id (1-indexed)
            qa_by_node: list[list[SyntheticQA]] = []
            for i in range(1, num_nodes + 1):
                qa_pairs = qa_by_chunk_id.get(i, [])
                qa_by_node.append(qa_pairs)

            logger.debug(
                "Parsed QA pairs (JSON): %d nodes, avg %.1f pairs/node",
                len(qa_by_node),
                sum(len(pairs) for pairs in qa_by_node) / max(len(qa_by_node), 1),
            )

            return qa_by_node

        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON output: %s", e)
            logger.debug("Invalid JSON text: %s", generated_text[:500])
            raise QuestionGenerationError(f"LLM output is not valid JSON: {e}") from e

        except ValidationError as e:
            logger.error("JSON schema validation failed: %s", e)
            logger.debug("Invalid JSON data: %s", json_data)
            raise QuestionGenerationError(
                f"LLM JSON output doesn't match expected schema: {e}"
            ) from e

        except Exception as e:
            logger.error("Unexpected error parsing QA batch: %s", e)
            logger.debug("Parsing error details", exc_info=True)
            raise QuestionGenerationError(
                f"Failed to parse QA batch output: {e}"
            ) from e

    def _enrich_node(
        self, node: KnowledgeNode, qa_pairs: list[SyntheticQA], batch_id: str
    ) -> KnowledgeNode:
        """Enrich node with synthetic QA metadata.

        Args:
            node: Original knowledge node.
            qa_pairs: Generated and validated QA pairs.
            batch_id: Batch identifier for tracking.

        Returns:
            Node with synthetic_qa metadata.
        """
        enriched_metadata = {
            **node.metadata,
            "synthetic_qa": {
                "qa_pairs": [qa.model_dump() for qa in qa_pairs],
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "num_questions_requested": self._num_questions,
                "num_questions_validated": len(qa_pairs),
                "batch_id": batch_id,
            },
        }

        return node.model_copy(update={"metadata": enriched_metadata})

    @property
    def num_questions(self) -> int:
        """Number of questions to generate per node.

        Returns:
            Configured number of QA pairs.
        """
        return self._num_questions
