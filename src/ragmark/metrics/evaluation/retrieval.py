"""Retrieval evaluation metrics for RAG systems.

Provides standard information retrieval metrics for evaluating the quality
of retrieval components in RAG pipelines.
"""

import math
from typing import Any, cast

from ragmark.logger import get_logger
from ragmark.metrics.base import EvaluationMetric, MetricValidationError
from ragmark.metrics.registry import MetricRegistry
from ragmark.schemas.evaluation import CaseResult

logger = get_logger(__name__)


@MetricRegistry.register
class RecallAtK(EvaluationMetric[Any, float]):
    """Calculate recall@k metric for retrieval evaluation.

    Measures the proportion of relevant items found in the top-k results.
    A score of 1.0 indicates all relevant items were retrieved.
    """

    def __init__(self, k: int = 5):
        """Initialize recall@k metric.

        Args:
            k: Number of top results to consider.
        """
        self.k = k

    @property
    def name(self) -> str:
        return f"recall@{self.k}"

    @property
    def description(self) -> str:
        return f"Proportion of relevant items found in top-{self.k} results"

    def validate_inputs(self, **inputs: Any) -> None:
        """Validate metric inputs.

        Raises:
            MetricValidationError: If inputs are invalid.
        """
        relevant_ids = inputs.get("relevant_ids", [])
        if not relevant_ids:
            logger.warning(
                "Recall@%d validation failed: no ground truth provided", self.k
            )
            raise MetricValidationError("relevant_ids cannot be empty")

    def compute(self, **inputs: Any) -> float:
        """Compute recall@k score.

        Args:
            retrieved_ids: Retrieved item IDs in ranked order.
            relevant_ids: Ground truth relevant item IDs.

        Returns:
            Recall score between 0.0 and 1.0.
        """
        retrieved_ids = inputs.get("retrieved_ids", [])
        relevant_ids = inputs.get("relevant_ids", [])

        if not retrieved_ids or not relevant_ids:
            return 0.0

        top_k = set(retrieved_ids[: self.k])
        relevant_set = set(relevant_ids)

        hits = len(top_k & relevant_set)
        recall = hits / len(relevant_set)

        logger.debug(
            "Recall@%d computed: hits=%d/%d, score=%.4f",
            self.k,
            hits,
            len(relevant_set),
            recall,
        )

        return recall


@MetricRegistry.register
class PrecisionAtK(EvaluationMetric[Any, float]):
    """Calculate precision@k metric for retrieval evaluation.

    Measures the proportion of retrieved items in top-k that are relevant.
    A score of 1.0 indicates all retrieved items are relevant.
    """

    def __init__(self, k: int = 5):
        """Initialize precision@k metric.

        Args:
            k: Number of top results to consider.
        """
        self.k = k

    @property
    def name(self) -> str:
        return f"precision@{self.k}"

    @property
    def description(self) -> str:
        return f"Proportion of top-{self.k} items that are relevant"

    def validate_inputs(self, **inputs: Any) -> None:
        """Validate metric inputs.

        Raises:
            MetricValidationError: If inputs are invalid.
        """
        relevant_ids = inputs.get("relevant_ids", [])
        if not relevant_ids:
            logger.warning(
                "Precision@%d validation failed: no ground truth provided", self.k
            )
            raise MetricValidationError("relevant_ids cannot be empty")

    def compute(self, **inputs: Any) -> float:
        """Compute precision@k score.

        Args:
            retrieved_ids: Retrieved item IDs in ranked order.
            relevant_ids: Ground truth relevant item IDs.

        Returns:
            Precision score between 0.0 and 1.0.
        """
        retrieved_ids = inputs.get("retrieved_ids", [])
        relevant_ids = inputs.get("relevant_ids", [])

        if not retrieved_ids:
            return 0.0

        top_k = retrieved_ids[: self.k]
        relevant_set = set(relevant_ids)

        hits = sum(1 for item_id in top_k if item_id in relevant_set)
        precision = hits / len(top_k) if top_k else 0.0

        logger.debug(
            "Precision@%d computed: hits=%d/%d, score=%.4f",
            self.k,
            hits,
            len(top_k),
            precision,
        )

        return precision


@MetricRegistry.register
class MRR(EvaluationMetric[Any, float]):
    """Calculate Mean Reciprocal Rank metric for retrieval evaluation.

    Computes the reciprocal of the rank position of the first relevant item.
    A score of 1.0 indicates the first retrieved item is relevant.
    """

    @property
    def name(self) -> str:
        return "mrr"

    @property
    def description(self) -> str:
        return "Reciprocal rank of first relevant item"

    def validate_inputs(self, **inputs: Any) -> None:
        """Validate metric inputs.

        Raises:
            MetricValidationError: If inputs are invalid.
        """
        relevant_ids = inputs.get("relevant_ids", [])
        if not relevant_ids:
            logger.warning("MRR validation failed: no ground truth provided")
            raise MetricValidationError("relevant_ids cannot be empty")

    def compute(self, **inputs: Any) -> float:
        """Compute MRR score.

        Args:
            retrieved_ids: Retrieved item IDs in ranked order.
            relevant_ids: Ground truth relevant item IDs.

        Returns:
            MRR score between 0.0 and 1.0.
        """
        retrieved_ids = inputs.get("retrieved_ids", [])
        relevant_ids = inputs.get("relevant_ids", [])

        if not retrieved_ids:
            return 0.0

        relevant_set = set(relevant_ids)

        for rank, item_id in enumerate(retrieved_ids, start=1):
            if item_id in relevant_set:
                mrr_score = 1.0 / rank
                logger.debug(
                    "MRR computed: first_relevant_rank=%d, score=%.4f",
                    rank,
                    mrr_score,
                )
                return mrr_score

        logger.debug("MRR computed: no relevant items found, score=0.0")
        return 0.0


@MetricRegistry.register
class NDCG(EvaluationMetric[Any, float]):
    """Calculate Normalized Discounted Cumulative Gain at k.

    Measures ranking quality with position-based logarithmic discount.
    Uses binary relevance and log2 discount factor. A score of 1.0
    indicates perfect ranking of all relevant items.
    """

    def __init__(self, k: int = 5):
        """Initialize NDCG@k metric.

        Args:
            k: Number of top results to consider.
        """
        self.k = k

    @property
    def name(self) -> str:
        return f"ndcg@{self.k}"

    @property
    def description(self) -> str:
        return f"Normalized discounted cumulative gain at {self.k}"

    def validate_inputs(self, **inputs: Any) -> None:
        """Validate metric inputs.

        Raises:
            MetricValidationError: If inputs are invalid.
        """
        relevant_ids = inputs.get("relevant_ids", [])
        if not relevant_ids:
            logger.warning(
                "NDCG@%d validation failed: no ground truth provided", self.k
            )
            raise MetricValidationError("relevant_ids cannot be empty")

    def compute(self, **inputs: Any) -> float:
        """Compute NDCG@k score.

        Args:
            retrieved_ids: Retrieved item IDs in ranked order.
            relevant_ids: Ground truth relevant item IDs.

        Returns:
            NDCG score between 0.0 and 1.0.
        """
        retrieved_ids = inputs.get("retrieved_ids", [])
        relevant_ids = inputs.get("relevant_ids", [])

        if not retrieved_ids:
            return 0.0

        relevant_set = set(relevant_ids)

        dcg = 0.0
        for rank, item_id in enumerate(retrieved_ids[: self.k], start=1):
            if item_id in relevant_set:
                dcg += 1.0 / math.log2(rank + 1)

        idcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, min(self.k, len(relevant_ids)) + 1)
        )

        if idcg == 0.0:
            return 0.0

        ndcg_score = dcg / idcg

        logger.debug(
            "NDCG@%d computed: dcg=%.4f, idcg=%.4f, score=%.4f",
            self.k,
            dcg,
            idcg,
            ndcg_score,
        )

        return ndcg_score


@MetricRegistry.register
class MAP(EvaluationMetric[Any, float]):
    """Calculate Mean Average Precision at k for retrieval evaluation.

    Measures precision across all relevant items up to rank k,
    rewarding early retrieval of relevant items. A score of 1.0
    indicates perfect retrieval with all relevant items ranked first.
    """

    def __init__(self, k: int = 5):
        """Initialize MAP@k metric.

        Args:
            k: Number of top results to consider.
        """
        self.k = k

    @property
    def name(self) -> str:
        return f"map@{self.k}"

    @property
    def description(self) -> str:
        return f"Mean average precision at {self.k}"

    def validate_inputs(self, **inputs: Any) -> None:
        """Validate metric inputs.

        Raises:
            MetricValidationError: If inputs are invalid.
        """
        relevant_ids = inputs.get("relevant_ids", [])
        if not relevant_ids:
            logger.warning("MAP@%d validation failed: no ground truth provided", self.k)
            raise MetricValidationError("relevant_ids cannot be empty")

    def compute(self, **inputs: Any) -> float:
        """Compute MAP@k score.

        Args:
            retrieved_ids: Retrieved item IDs in ranked order.
            relevant_ids: Ground truth relevant item IDs.

        Returns:
            MAP score between 0.0 and 1.0.
        """
        retrieved_ids = inputs.get("retrieved_ids", [])
        relevant_ids = inputs.get("relevant_ids", [])

        if not retrieved_ids:
            return 0.0

        relevant_set = set(relevant_ids)
        top_k = retrieved_ids[: self.k]

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
            self.k,
            hits,
            precision_sum,
            map_score,
        )

        return map_score


def compute_retrieval_batch(
    results: list[CaseResult],
    ground_truth: dict[str, list[str]],
    ks: list[int] | None = None,
    metric_parameters: dict[str, dict[str, Any]] | None = None,
) -> dict[str, float]:
    """Compute aggregated retrieval metrics across multiple cases.

    Calculates metrics for each case using registered metric implementations
    and aggregates by averaging. Updates case_metrics attribute for each
    CaseResult in-place.

    Args:
        results: Case results with retrieval traces to evaluate.
        ground_truth: Mapping from case_id to relevant node IDs.
        ks: K values for @k metrics (recall@k, precision@k, etc.).
            Defaults to [5, 10, 20].
        metric_parameters: Optional metric parameters keyed by base name.

    Returns:
        Aggregated metrics with mean values across all valid cases.
    """
    metric_k_values = {
        "recall": _resolve_metric_k_values("recall", ks, metric_parameters),
        "precision": _resolve_metric_k_values("precision", ks, metric_parameters),
        "ndcg": _resolve_metric_k_values("ndcg", ks, metric_parameters),
        "map": _resolve_metric_k_values("map", ks, metric_parameters),
    }

    if not results:
        logger.warning(
            "Retrieval metrics computation skipped: no case results provided"
        )
        return {}

    logger.info(
        "Retrieval metrics computation started: cases=%d, k_values=%s",
        len(results),
        metric_k_values,
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

        for k in metric_k_values["recall"]:
            try:
                metric = MetricRegistry.create(
                    f"recall@{k}", **_metric_kwargs(metric_parameters, "recall")
                )
                if not isinstance(metric, EvaluationMetric):
                    raise TypeError(
                        f"Metric {metric.__class__.__name__} is not an EvaluationMetric"
                    )
                metric = cast(EvaluationMetric[Any, float], metric)
                case_metrics[f"recall@{k}"] = metric.compute(
                    retrieved_ids=retrieved_ids, relevant_ids=relevant_ids
                )
            except Exception as exc:
                logger.error("Recall@%d computation failed: %s", k, exc)
                logger.debug("Recall@%d error details: %s", k, exc, exc_info=True)
                raise

        for k in metric_k_values["precision"]:
            try:
                metric = MetricRegistry.create(
                    f"precision@{k}", **_metric_kwargs(metric_parameters, "precision")
                )
                if not isinstance(metric, EvaluationMetric):
                    raise TypeError(
                        f"Metric {metric.__class__.__name__} is not an EvaluationMetric"
                    )
                metric = cast(EvaluationMetric[Any, float], metric)
                case_metrics[f"precision@{k}"] = metric.compute(
                    retrieved_ids=retrieved_ids, relevant_ids=relevant_ids
                )
            except Exception as exc:
                logger.error("Precision@%d computation failed: %s", k, exc)
                logger.debug("Precision@%d error details: %s", k, exc, exc_info=True)
                raise

        for k in metric_k_values["ndcg"]:
            try:
                metric = MetricRegistry.create(
                    f"ndcg@{k}", **_metric_kwargs(metric_parameters, "ndcg")
                )
                if not isinstance(metric, EvaluationMetric):
                    raise TypeError(
                        f"Metric {metric.__class__.__name__} is not an EvaluationMetric"
                    )
                metric = cast(EvaluationMetric[Any, float], metric)
                case_metrics[f"ndcg@{k}"] = metric.compute(
                    retrieved_ids=retrieved_ids, relevant_ids=relevant_ids
                )
            except Exception as exc:
                logger.error("NDCG@%d computation failed: %s", k, exc)
                logger.debug("NDCG@%d error details: %s", k, exc, exc_info=True)
                raise

        for k in metric_k_values["map"]:
            try:
                metric = MetricRegistry.create(
                    f"map@{k}", **_metric_kwargs(metric_parameters, "map")
                )
                if not isinstance(metric, EvaluationMetric):
                    raise TypeError(
                        f"Metric {metric.__class__.__name__} is not an EvaluationMetric"
                    )
                metric = cast(EvaluationMetric[Any, float], metric)
                case_metrics[f"map@{k}"] = metric.compute(
                    retrieved_ids=retrieved_ids, relevant_ids=relevant_ids
                )
            except Exception as exc:
                logger.error("MAP@%d computation failed: %s", k, exc)
                logger.debug("MAP@%d error details: %s", k, exc, exc_info=True)
                raise

        try:
            metric = MetricRegistry.create(
                "mrr", **_metric_kwargs(metric_parameters, "mrr")
            )
            if not isinstance(metric, EvaluationMetric):
                raise TypeError(
                    f"Metric {metric.__class__.__name__} is not an EvaluationMetric"
                )
            metric = cast(EvaluationMetric[Any, float], metric)
            case_metrics["mrr"] = metric.compute(
                retrieved_ids=retrieved_ids, relevant_ids=relevant_ids
            )
        except Exception as exc:
            logger.error("MRR computation failed: %s", exc)
            logger.debug("MRR error details: %s", exc, exc_info=True)
            raise

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


def _resolve_metric_k_values(
    base_name: str,
    ks: list[int] | None,
    metric_parameters: dict[str, dict[str, Any]] | None,
) -> list[int]:
    if ks is not None:
        return ks

    if not metric_parameters:
        return [5, 10, 20]

    params = metric_parameters.get(base_name, {})
    param_ks = params.get("ks")
    if isinstance(param_ks, list):
        values: list[int] = []
        for value in cast(list[int | str], param_ks):
            values.append(int(value))
        if values:
            return sorted(set(values))

    param_k = params.get("k")
    if param_k is not None:
        return [int(param_k)]

    return [5, 10, 20]


def _metric_kwargs(
    metric_parameters: dict[str, dict[str, Any]] | None,
    base_name: str,
) -> dict[str, Any]:
    if not metric_parameters:
        return {}

    params = metric_parameters.get(base_name, {})
    return {key: value for key, value in params.items() if key not in {"k", "ks"}}
