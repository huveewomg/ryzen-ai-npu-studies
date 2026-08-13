"""Numerical-drift and retrieval metrics used by the fidelity study."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def embedding_drift(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {candidate.shape}")
    reference_norm = normalize_rows(reference.astype(np.float64, copy=False))
    candidate_norm = normalize_rows(candidate.astype(np.float64, copy=False))
    cosines = np.sum(reference_norm * candidate_norm, axis=1)
    max_abs_by_item = np.max(np.abs(reference - candidate), axis=1)
    return {
        "per_item_cosine": cosines.tolist(),
        "mean_cosine": float(np.mean(cosines)),
        "min_cosine": float(np.min(cosines)),
        "max_absolute_error": float(np.max(max_abs_by_item)),
        "worst_cosine_index": int(np.argmin(cosines)),
        "worst_absolute_error_index": int(np.argmax(max_abs_by_item)),
    }


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    gains = [1.0 if item in relevant_ids else 0.0 for item in ranked_ids[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal_count = min(len(relevant_ids), k)
    ideal = sum(1.0 / math.log2(index + 2) for index in range(ideal_count))
    return dcg / ideal if ideal else 0.0


def average_precision_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(ranked_ids[:k], start=1):
        if item in relevant_ids:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(relevant_ids), k)


def reciprocal_rank_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    for rank, item in enumerate(ranked_ids[:k], start=1):
        if item in relevant_ids:
            return 1.0 / rank
    return 0.0


def rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks with deterministic tie handling."""

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Spearman inputs must be same-shaped one-dimensional arrays")
    left_rank = rankdata(left)
    right_rank = rankdata(right)
    left_centered = left_rank - np.mean(left_rank)
    right_centered = right_rank - np.mean(right_rank)
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    if denominator == 0:
        return 1.0 if np.array_equal(left_rank, right_rank) else 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def retrieval_metrics(
    *,
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    query_ids: Sequence[str],
    document_ids: Sequence[str],
    qrels: Mapping[str, set[str]],
    reference_scores: np.ndarray | None = None,
) -> dict[str, Any]:
    queries = normalize_rows(query_embeddings)
    documents = normalize_rows(document_embeddings)
    scores = queries @ documents.T
    recall_1: list[float] = []
    recall_5: list[float] = []
    recall_10: list[float] = []
    recall_100: list[float] = []
    ndcg_10: list[float] = []
    average_precision_10: list[float] = []
    reciprocal_rank_10: list[float] = []
    precision_10: list[float] = []
    correlations: list[float] = []

    for index, query_id in enumerate(query_ids):
        order = np.argsort(-scores[index], kind="mergesort")
        ranked = [document_ids[position] for position in order]
        relevant = qrels.get(query_id, set())
        recall_1.append(recall_at_k(ranked, relevant, 1))
        recall_5.append(recall_at_k(ranked, relevant, 5))
        recall_10.append(recall_at_k(ranked, relevant, 10))
        recall_100.append(recall_at_k(ranked, relevant, 100))
        ndcg_10.append(ndcg_at_k(ranked, relevant, 10))
        average_precision_10.append(average_precision_at_k(ranked, relevant, 10))
        reciprocal_rank_10.append(reciprocal_rank_at_k(ranked, relevant, 10))
        precision_10.append(len(set(ranked[:10]) & relevant) / 10.0)
        if reference_scores is not None:
            correlations.append(spearman_correlation(reference_scores[index], scores[index]))

    result: dict[str, Any] = {
        "recall_at_1": float(np.mean(recall_1)),
        "recall_at_5": float(np.mean(recall_5)),
        "recall_at_10": float(np.mean(recall_10)),
        "recall_at_100": float(np.mean(recall_100)),
        "ndcg_at_10": float(np.mean(ndcg_10)),
        "map_at_10": float(np.mean(average_precision_10)),
        "mrr_at_10": float(np.mean(reciprocal_rank_10)),
        "precision_at_10": float(np.mean(precision_10)),
    }
    if correlations:
        result["mean_ranking_spearman"] = float(np.mean(correlations))
    return result
