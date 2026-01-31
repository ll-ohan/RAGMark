"""Retrieval evaluation metrics for RAG systems.

This module provides standard information retrieval metrics for evaluating
the quality of retrieval components in RAG pipelines.

Metrics included:
    - Recall@k: Proportion of relevant items found in top-k results.
    - Precision@k: Proportion of top-k items that are relevant.
    - MRR: Mean Reciprocal Rank (position of first relevant item).
    - NDCG@k: Normalized Discounted Cumulative Gain (ranking quality).
    - MAP@k: Mean Average Precision (precision across relevant items).

Usage:
    Compute individual metrics for a single query, or use
    compute_retrieval_metrics() for batch evaluation across multiple cases.
"""

import math

from ragmark.logger import get_logger
from ragmark.schemas.evaluation import CaseResult

logger = get_logger(__name__)


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Calculate recall@k metric.

    Measures the proportion of relevant items found in the top-k results.
    A score of 1.0 indicates all relevant items were retrieved.

    Args:
        retrieved_ids: Retrieved item IDs in ranked order.
        relevant_ids: Ground truth relevant item IDs.
        k: Number of top results to consider.

    Returns:
        Recall score between 0.0 and 1.0.
    """
    if not relevant_ids:
        logger.warning("Recall@%d computation skipped: no ground truth provided", k)
        return 0.0

    if not retrieved_ids:
        return 0.0

    top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)

    hits = len(top_k & relevant_set)
    recall = hits / len(relevant_set)

    logger.debug(
        "Recall@%d computed: hits=%d/%d, score=%.4f",
        k,
        hits,
        len(relevant_set),
        recall,
    )

    return recall


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Calculate precision@k metric.

    Measures the proportion of retrieved items in top-k that are relevant.
    A score of 1.0 indicates all retrieved items are relevant.

    Args:
        retrieved_ids: Retrieved item IDs in ranked order.
        relevant_ids: Ground truth relevant item IDs.
        k: Number of top results to consider.

    Returns:
        Precision score between 0.0 and 1.0.
    """
    if not retrieved_ids:
        return 0.0

    if not relevant_ids:
        logger.warning("Precision@%d computation skipped: no ground truth provided", k)
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)

    hits = sum(1 for item_id in top_k if item_id in relevant_set)
    precision = hits / len(top_k) if top_k else 0.0

    logger.debug(
        "Precision@%d computed: hits=%d/%d, score=%.4f",
        k,
        hits,
        len(top_k),
        precision,
    )

    return precision


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Calculate Mean Reciprocal Rank metric.

    Computes the reciprocal of the rank position of the first relevant item.
    A score of 1.0 indicates the first retrieved item is relevant.

    Args:
        retrieved_ids: Retrieved item IDs in ranked order.
        relevant_ids: Ground truth relevant item IDs.

    Returns:
        MRR score between 0.0 and 1.0.
    """
    if not retrieved_ids:
        return 0.0

    if not relevant_ids:
        logger.warning("MRR computation skipped: no ground truth provided")
        return 0.0

    relevant_set = set(relevant_ids)

    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant_set:
            mrr_score = 1.0 / rank
            logger.debug(
                "MRR computed: first_relevant_rank=%d, score=%.4f", rank, mrr_score
            )
            return mrr_score

    logger.debug("MRR computed: no relevant items found, score=0.0")
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k.

    Measures ranking quality with position-based logarithmic discount.
    Uses binary relevance and log2 discount factor. A score of 1.0
    indicates perfect ranking of all relevant items.

    Args:
        retrieved_ids: Retrieved item IDs in ranked order.
        relevant_ids: Ground truth relevant item IDs.
        k: Number of top results to consider.

    Returns:
        NDCG score between 0.0 and 1.0.
    """
    if not retrieved_ids:
        return 0.0

    if not relevant_ids:
        logger.warning("NDCG@%d computation skipped: no ground truth provided", k)
        return 0.0

    relevant_set = set(relevant_ids)

    dcg = 0.0
    for rank, item_id in enumerate(retrieved_ids[:k], start=1):
        if item_id in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)

    idcg = sum(
        1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant_ids)) + 1)
    )

    if idcg == 0.0:
        return 0.0

    ndcg_score = dcg / idcg

    logger.debug(
        "NDCG@%d computed: dcg=%.4f, idcg=%.4f, score=%.4f",
        k,
        dcg,
        idcg,
        ndcg_score,
    )

    return ndcg_score


def map_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Calculate Mean Average Precision at k.

    Measures precision across all relevant items up to rank k,
    rewarding early retrieval of relevant items. A score of 1.0
    indicates perfect retrieval with all relevant items ranked first.

    Args:
        retrieved_ids: Retrieved item IDs in ranked order.
        relevant_ids: Ground truth relevant item IDs.
        k: Number of top results to consider.

    Returns:
        MAP score between 0.0 and 1.0.
    """
    if not retrieved_ids:
        return 0.0

    if not relevant_ids:
        logger.warning("MAP@%d computation skipped: no ground truth provided", k)
        return 0.0

    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]

    precision_sum = 0.0
    hits = 0

    for rank, item_id in enumerate(top_k, start=1):
        if item_id in relevant_set:
            hits += 1
            precision_sum += hits / rank

    if hits == 0:
        return 0.0

    map_score = precision_sum / len(relevant_set)

    logger.debug(
        "MAP@%d computed: hits=%d, precision_sum=%.4f, score=%.4f",
        k,
        hits,
        precision_sum,
        map_score,
    )

    return map_score


def compute_retrieval_metrics(
    results: list[CaseResult],
    ground_truth: dict[str, list[str]],
    ks: list[int] | None = None,
) -> dict[str, float]:
    """Compute aggregated retrieval metrics across all cases.

    Calculates metrics for each case and aggregates by averaging.
    Updates case_metrics attribute for each CaseResult in-place.

    Args:
        results: Case results with retrieval traces to evaluate.
        ground_truth: Mapping from case_id to relevant node IDs.
        ks: K values for @k metrics (recall@k, precision@k, etc.).

    Returns:
        Aggregated metrics with mean values across all valid cases.
    """
    if ks is None:
        ks = [5, 10, 20]

    if not results:
        logger.warning(
            "Retrieval metrics computation skipped: no case results provided"
        )
        return {}

    logger.info(
        "Retrieval metrics computation started: cases=%d, k_values=%s",
        len(results),
        ks,
    )

    metric_sums: dict[str, float] = {}
    valid_cases = 0
    skipped_no_trace = 0
    skipped_no_ground_truth = 0

    for case_result in results:
        if not case_result.trace or not case_result.trace.retrieved_nodes:
            skipped_no_trace += 1
            logger.debug(
                "Case skipped: case_id=%s, reason=no_retrieval_trace",
                case_result.case_id,
            )
            continue

        retrieved_ids = [
            node.node.node_id for node in case_result.trace.retrieved_nodes
        ]

        relevant_ids = ground_truth.get(case_result.case_id, [])
        if not relevant_ids:
            skipped_no_ground_truth += 1
            logger.debug(
                "Case skipped: case_id=%s, reason=no_ground_truth",
                case_result.case_id,
            )
            continue

        logger.debug(
            "Evaluating case: case_id=%s, retrieved=%d, relevant=%d",
            case_result.case_id,
            len(retrieved_ids),
            len(relevant_ids),
        )

        case_metrics: dict[str, float] = {}

        for k in ks:
            case_metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
            case_metrics[f"precision@{k}"] = precision_at_k(
                retrieved_ids, relevant_ids, k
            )
            case_metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)
            case_metrics[f"map@{k}"] = map_at_k(retrieved_ids, relevant_ids, k)

        case_metrics["mrr"] = mrr(retrieved_ids, relevant_ids)

        case_result.case_metrics = case_metrics

        for metric_name, value in case_metrics.items():
            metric_sums[metric_name] = metric_sums.get(metric_name, 0.0) + value

        valid_cases += 1

    if valid_cases == 0:
        logger.warning(
            "Retrieval metrics computation failed: valid_cases=0, "
            "skipped_no_trace=%d, skipped_no_ground_truth=%d",
            skipped_no_trace,
            skipped_no_ground_truth,
        )
        return {}

    aggregated_metrics = {
        metric: total / valid_cases for metric, total in metric_sums.items()
    }

    logger.info(
        "Retrieval metrics computation completed: valid_cases=%d, "
        "skipped_no_trace=%d, skipped_no_ground_truth=%d, metrics=%d",
        valid_cases,
        skipped_no_trace,
        skipped_no_ground_truth,
        len(aggregated_metrics),
    )

    return aggregated_metrics
