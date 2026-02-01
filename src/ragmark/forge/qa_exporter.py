"""Export synthetic QA pairs to evaluation trial cases.

This module provides utilities to convert enriched knowledge nodes with
synthetic QA metadata into TrialCase format for benchmark evaluation.
"""

import json
from pathlib import Path

from ragmark.logger import get_logger
from ragmark.schemas.documents import KnowledgeNode
from ragmark.schemas.evaluation import TrialCase

logger = get_logger(__name__)


class QAExporter:
    """Export synthetic QA pairs from enriched nodes to trial cases.

    Converts knowledge nodes with synthetic_qa metadata into TrialCase
    format suitable for RAG evaluation benchmarks.
    """

    @staticmethod
    def nodes_to_trial_cases(
        nodes: list[KnowledgeNode],
        include_ground_truth_nodes: bool = True,
    ) -> list[TrialCase]:
        """Convert enriched nodes to trial cases.

        Extracts all synthetic QA pairs from nodes and creates TrialCase
        instances with proper ground truth references.

        Args:
            nodes: Knowledge nodes with synthetic_qa metadata.
            include_ground_truth_nodes: If True, sets ground_truth_node_ids
                to the source node ID. If False, only includes answer.

        Returns:
            List of trial cases ready for evaluation.

        Raises:
            ValueError: If nodes don't have synthetic_qa metadata.
        """
        trial_cases: list[TrialCase] = []
        skipped_nodes = 0

        for node in nodes:
            if "synthetic_qa" not in node.metadata:
                logger.debug(
                    "Node missing synthetic_qa metadata: node_id=%s", node.node_id
                )
                skipped_nodes += 1
                continue

            qa_data = node.metadata["synthetic_qa"]
            qa_pairs = qa_data.get("qa_pairs", [])

            if not qa_pairs:
                logger.debug("Node has empty qa_pairs: node_id=%s", node.node_id)
                skipped_nodes += 1
                continue

            for i, qa_pair in enumerate(qa_pairs):
                question = qa_pair.get("question")
                answer = qa_pair.get("answer")

                if not question or not answer:
                    logger.debug(
                        "Invalid QA pair (missing question or answer): node=%s, index=%d",
                        node.node_id,
                        i,
                    )
                    continue

                # Build metadata with source information
                metadata = {
                    "source_node_id": node.node_id,
                    "source_id": node.source_id,
                    "qa_index": i,
                    "generated_at": qa_data.get("generated_at"),
                    "model": qa_data.get("model"),
                    "batch_id": qa_data.get("batch_id"),
                }

                # Add confidence score if available
                if "confidence" in qa_pair and qa_pair["confidence"] is not None:
                    metadata["confidence"] = qa_pair["confidence"]

                # Add source node metadata (if relevant)
                if node.metadata:
                    # Filter out synthetic_qa to avoid bloat
                    source_metadata = {
                        k: v for k, v in node.metadata.items() if k != "synthetic_qa"
                    }
                    if source_metadata:
                        metadata["source_metadata"] = source_metadata

                trial_case = TrialCase(
                    question=question,
                    ground_truth_answer=answer,
                    ground_truth_node_ids=(
                        [node.node_id] if include_ground_truth_nodes else None
                    ),
                    metadata=metadata,
                )

                trial_cases.append(trial_case)

        logger.info(
            "Converted %d nodes to %d trial cases (%d nodes skipped)",
            len(nodes),
            len(trial_cases),
            skipped_nodes,
        )

        return trial_cases

    @staticmethod
    def export_to_jsonl(
        nodes: list[KnowledgeNode],
        output_path: Path,
        include_ground_truth_nodes: bool = True,
    ) -> int:
        """Export enriched nodes to JSONL trial cases file.

        Args:
            nodes: Knowledge nodes with synthetic_qa metadata.
            output_path: Destination file path (.jsonl extension).
            include_ground_truth_nodes: Whether to include ground_truth_node_ids.

        Returns:
            Number of trial cases exported.

        Raises:
            ValueError: If output_path doesn't have .jsonl extension.
        """
        if output_path.suffix != ".jsonl":
            raise ValueError(f"Output path must have .jsonl extension: {output_path}")

        trial_cases = QAExporter.nodes_to_trial_cases(
            nodes=nodes,
            include_ground_truth_nodes=include_ground_truth_nodes,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for case in trial_cases:
                json_line = case.model_dump_json()
                f.write(json_line + "\n")

        logger.info(
            "Exported %d trial cases to JSONL: path=%s",
            len(trial_cases),
            output_path,
        )

        return len(trial_cases)

    @staticmethod
    def export_to_json(
        nodes: list[KnowledgeNode],
        output_path: Path,
        include_ground_truth_nodes: bool = True,
        indent: int = 2,
    ) -> int:
        """Export enriched nodes to JSON trial cases file.

        Args:
            nodes: Knowledge nodes with synthetic_qa metadata.
            output_path: Destination file path (.json extension).
            include_ground_truth_nodes: Whether to include ground_truth_node_ids.
            indent: JSON indentation level (default: 2).

        Returns:
            Number of trial cases exported.

        Raises:
            ValueError: If output_path doesn't have .json extension.
        """
        if output_path.suffix != ".json":
            raise ValueError(f"Output path must have .json extension: {output_path}")

        trial_cases = QAExporter.nodes_to_trial_cases(
            nodes=nodes,
            include_ground_truth_nodes=include_ground_truth_nodes,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cases_data = [case.model_dump() for case in trial_cases]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cases_data, f, indent=indent, ensure_ascii=False)

        logger.info(
            "Exported %d trial cases to JSON: path=%s",
            len(trial_cases),
            output_path,
        )

        return len(trial_cases)
