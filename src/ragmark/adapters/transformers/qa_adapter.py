"""Adapter for transforming knowledge nodes with synthetic QA to trial cases."""

from typing import Any, TypeGuard, cast

from ragmark.adapters.base import Adapter
from ragmark.logger import get_logger
from ragmark.schemas.documents import KnowledgeNode, MetadataDict, MetadataValue
from ragmark.schemas.evaluation import TrialCase
from ragmark.schemas.qa import SyntheticQAMetadata, SyntheticQAPairData

logger = get_logger(__name__)


class NodeToTrialCaseAdapter(Adapter[KnowledgeNode, TrialCase]):
    """Convert knowledge nodes with synthetic QA to trial cases.

    Transforms nodes containing synthetic_qa metadata into TrialCase format
    suitable for RAG evaluation benchmarks. Replaces the logic of QAExporter
    with a generic adapter pattern.
    """

    def __init__(
        self,
        include_ground_truth_nodes: bool = True,
        metadata_filter: set[str] | None = None,
    ):
        """Initialize QA adapter.

        Args:
            include_ground_truth_nodes: Whether to include ground truth node IDs.
            metadata_filter: Metadata keys to exclude from output.
        """
        self.include_ground_truth_nodes = include_ground_truth_nodes
        self.metadata_filter = metadata_filter or {"synthetic_qa"}

    def validate(self, source: KnowledgeNode) -> bool:
        """Check if node has required synthetic QA metadata.

        Args:
            source: Knowledge node to validate.

        Returns:
            True if node has synthetic_qa metadata, False otherwise.
        """
        return "synthetic_qa" in source.metadata

    def adapt(self, source: KnowledgeNode) -> list[TrialCase]:
        """Transform node to trial cases (one-to-many).

        Extracts synthetic QA pairs from node metadata and creates one
        TrialCase per pair. Handles validation, metadata filtering, and
        error cases gracefully.

        Args:
            source: Knowledge node with synthetic_qa metadata.

        Returns:
            List of trial cases, one per QA pair. Empty list if node
            cannot be adapted or has no valid QA pairs.
        """
        if not self.validate(source):
            logger.debug(
                "Node validation failed: node_id=%s, reason=missing_synthetic_qa",
                source.node_id,
            )
            return []

        qa_data = self._get_synthetic_qa(source.metadata)
        if qa_data is None:
            logger.debug(
                "Node synthetic_qa metadata invalid: node_id=%s", source.node_id
            )
            return []

        qa_pairs = qa_data.get("qa_pairs", [])

        if not qa_pairs:
            logger.debug("Node has empty qa_pairs: node_id=%s", source.node_id)
            return []

        trial_cases: list[TrialCase] = []

        for i, qa_pair in enumerate(qa_pairs):
            try:
                if not self._is_qa_pair(qa_pair):
                    raise ValueError(
                        f"Invalid QA pair type at index {i}: expected dict"
                    )
                trial_case = self._create_trial_case(
                    source,
                    qa_pair,
                    i,
                )
                trial_cases.append(trial_case)
            except ValueError as exc:
                logger.error(
                    "Trial case creation failed: node_id=%s, qa_index=%d",
                    source.node_id,
                    i,
                )
                logger.debug("Creation failure details: %s", exc, exc_info=True)
                continue

        return trial_cases

    def _create_trial_case(
        self,
        node: KnowledgeNode,
        qa_pair: SyntheticQAPairData,
        index: int,
    ) -> TrialCase:
        """Create single trial case from QA pair.

        Args:
            node: Source knowledge node.
            qa_pair: QA pair dictionary.
            index: Index of QA pair in node.

        Returns:
            Constructed trial case.

        Raises:
            ValueError: If QA pair is invalid or missing required fields.
        """
        question = qa_pair.get("question")
        answer = qa_pair.get("answer")
        if not (isinstance(question, str) and isinstance(answer, str)):
            raise ValueError(
                f"Invalid QA pair at index {index}: question and answer must be non-empty strings"
            )

        qa_data = self._get_synthetic_qa(node.metadata) or {}
        qa_model = qa_data.get("model")
        qa_bactch_id = qa_data.get("batch_id")
        qa_generated_at = qa_pair.get("generated_at") or qa_data.get("generated_at")

        metadata: dict[str, str | int | float | dict[str, Any] | None] = {
            "source_node_id": node.node_id,
            "source_id": node.source_id,
            "qa_index": index,
            "generated_at": qa_generated_at,
            "model": qa_model,
            "batch_id": qa_bactch_id,
        }

        metadata["confidence"] = qa_pair.get("confidence", None)

        # Filter source metadata (exclude synthetic_qa to avoid bloat)
        if node.metadata:
            filtered = {
                k: v for k, v in node.metadata.items() if k not in self.metadata_filter
            }
            if filtered:
                metadata["source_metadata"] = filtered

        return TrialCase(
            question=question,
            ground_truth_answer=answer,
            ground_truth_node_ids=(
                [node.node_id] if self.include_ground_truth_nodes else None
            ),
            metadata=metadata,
        )

    def _get_synthetic_qa(self, metadata: MetadataDict) -> SyntheticQAMetadata | None:
        qa_data = metadata.get("synthetic_qa")
        if not isinstance(qa_data, dict):
            return None
        return cast(SyntheticQAMetadata, qa_data)

    def _is_qa_pair(self, value: MetadataValue) -> TypeGuard[SyntheticQAPairData]:
        if not isinstance(value, dict):
            return False
        question = value.get("question")
        answer = value.get("answer")
        return isinstance(question, str) and isinstance(answer, str)
