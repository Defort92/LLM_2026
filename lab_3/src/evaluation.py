from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer


def _encode(
    model: SentenceTransformer,
    texts: list[str],
    prefix: str,
    batch_size: int,
    description: str,
) -> np.ndarray:
    return model.encode(
        [f"{prefix}{text}" for text in texts],
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def evaluate_retrieval(
    model: SentenceTransformer,
    queries: list[str],
    documents: list[str],
    batch_size: int,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    query_embeddings = _encode(model, queries, "query: ", batch_size, "queries")
    document_embeddings = _encode(model, documents, "passage: ", batch_size, "documents")
    # Векторы нормализованы, поэтому скалярное произведение равно cosine similarity.
    # Получаем оценку каждой пары «вопрос — документ» одной матричной операцией.
    scores = query_embeddings @ document_embeddings.T

    top_k = min(10, len(documents))
    # argpartition быстро выбирает десять кандидатов без полной сортировки корпуса;
    # затем сортируем только их, чтобы получить точный порядок внутри top-10.
    candidate_ids = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
    candidate_scores = np.take_along_axis(scores, candidate_ids, axis=1)
    order = np.argsort(-candidate_scores, axis=1)
    ranked_ids = np.take_along_axis(candidate_ids, order, axis=1)

    ranks = np.full(len(queries), np.inf, dtype=np.float64)
    for query_id, ids in enumerate(ranked_ids):
        positions = np.flatnonzero(ids == query_id)
        if positions.size:
            ranks[query_id] = float(positions[0] + 1)

    recall_at_5 = float(np.mean(ranks <= 5))
    recall_at_10 = float(np.mean(ranks <= 10))
    mrr_at_10 = float(np.mean(np.where(ranks <= 10, 1.0 / ranks, 0.0)))
    # Релевантный документ только один, поэтому AP@10 совпадает с reciprocal rank,
    # если этот документ находится в первой десятке.
    map_at_10 = mrr_at_10
    ndcg_at_10 = float(np.mean(np.where(ranks <= 10, 1.0 / np.log2(ranks + 1.0), 0.0)))

    metrics: dict[str, float | int] = {
        "queries": len(queries),
        "corpus_documents": len(documents),
        "accuracy_at_1": round(float(np.mean(ranks == 1)), 6),
        "recall_at_5": round(recall_at_5, 6),
        "recall_at_10": round(recall_at_10, 6),
        "mrr_at_10": round(mrr_at_10, 6),
        "map_at_10": round(map_at_10, 6),
        "ndcg_at_10": round(ndcg_at_10, 6),
    }
    return metrics, ranks, ranked_ids


def evaluate_pairwise(
    model: SentenceTransformer,
    queries: list[str],
    documents: list[str],
    labels: list[int],
    batch_size: int,
) -> dict[str, Any]:
    query_embeddings = _encode(model, queries, "query: ", batch_size, "pair queries")
    document_embeddings = _encode(model, documents, "passage: ", batch_size, "pair documents")
    similarities = np.sum(query_embeddings * document_embeddings, axis=1)
    labels_array = np.asarray(labels)
    positive_scores = similarities[labels_array == 1]
    negative_scores = similarities[labels_array == 0]
    correlation = spearmanr(labels_array, similarities).statistic

    return {
        "pairs": len(labels),
        "roc_auc": round(float(roc_auc_score(labels_array, similarities)), 6),
        "spearman": round(float(correlation), 6),
        "positive_cosine_mean": round(float(np.mean(positive_scores)), 6),
        "negative_cosine_mean": round(float(np.mean(negative_scores)), 6),
        "cosine_margin": round(float(np.mean(positive_scores) - np.mean(negative_scores)), 6),
    }


def ranking_examples(
    queries: list[str],
    documents: list[str],
    baseline_ranks: np.ndarray,
    baseline_ids: np.ndarray,
    fine_tuned_ranks: np.ndarray,
    fine_tuned_ids: np.ndarray,
    count_each: int = 5,
) -> list[dict[str, Any]]:
    # Ранг inf означает, что документ не попал в top-10; для сравнения считаем
    # его условной позицией 11, следующей сразу после наблюдаемой выдачи.
    baseline_numeric = np.where(np.isfinite(baseline_ranks), baseline_ranks, 11.0)
    fine_tuned_numeric = np.where(np.isfinite(fine_tuned_ranks), fine_tuned_ranks, 11.0)
    improvement = baseline_numeric - fine_tuned_numeric
    improved_ids = np.argsort(-improvement)[:count_each]
    degraded_ids = np.argsort(improvement)[:count_each]

    rows: list[dict[str, Any]] = []
    for change_type, selected_ids in [("improved", improved_ids), ("degraded", degraded_ids)]:
        for query_id in selected_ids:
            rows.append(
                {
                    "change": change_type,
                    "query": queries[query_id],
                    "relevant_document": documents[query_id],
                    "baseline_rank": int(baseline_ranks[query_id]) if np.isfinite(baseline_ranks[query_id]) else ">10",
                    "fine_tuned_rank": int(fine_tuned_ranks[query_id]) if np.isfinite(fine_tuned_ranks[query_id]) else ">10",
                    "baseline_top1": documents[int(baseline_ids[query_id, 0])],
                    "fine_tuned_top1": documents[int(fine_tuned_ids[query_id, 0])],
                }
            )
    return rows
