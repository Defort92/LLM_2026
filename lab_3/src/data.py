from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset


@dataclass(frozen=True)
class SearchExample:
    query: str
    document: str
    label: float


def load_pickle_examples(path: Path) -> list[SearchExample]:
    with path.open("rb") as file:
        rows = pickle.load(file)
    return [SearchExample(query=row[0], document=row[1], label=float(row[2])) for row in rows]


def positive_examples(rows: list[SearchExample]) -> list[SearchExample]:
    return [row for row in rows if row.label == 1.0]


def build_training_dataset(
    rows: list[SearchExample],
    limit: int,
    seed: int,
) -> tuple[Dataset, dict[str, int]]:
    positives = positive_examples(rows)
    rng = random.Random(seed)
    rng.shuffle(positives)
    selected = positives[:limit]

    dataset = Dataset.from_dict(
        {
            "anchor": [f"query: {row.query}" for row in selected],
            "positive": [f"passage: {row.document}" for row in selected],
        }
    )
    metadata = {
        "all_rows": len(rows),
        "all_positive_rows": len(positives),
        "selected_positive_rows": len(selected),
        "seed": seed,
    }
    return dataset, metadata


def build_retrieval_test_set(
    rows: list[SearchExample],
    positive_limit: int,
) -> tuple[list[str], list[str], dict[str, int]]:
    """Keep one question for each unique positive document.

    The provided notebook uses the first 1000 positive SberQuAD examples and
    removes repeated contexts.  Keeping the first query for every context gives
    one known relevant document per query and makes exact ranking unambiguous.
    """
    positives = positive_examples(rows)[:positive_limit]
    seen_documents: set[str] = set()
    queries: list[str] = []
    documents: list[str] = []

    for row in positives:
        if row.document in seen_documents:
            continue
        seen_documents.add(row.document)
        queries.append(row.query)
        documents.append(row.document)

    metadata = {
        "requested_positive_rows": positive_limit,
        "unique_queries": len(queries),
        "unique_documents": len(documents),
        "duplicates_removed": len(positives) - len(documents),
    }
    return queries, documents, metadata


def build_pairwise_test_set(
    rows: list[SearchExample],
    per_class_limit: int,
) -> tuple[list[str], list[str], list[int]]:
    positives = [row for row in rows if row.label == 1.0][:per_class_limit]
    negatives = [row for row in rows if row.label == 0.0][:per_class_limit]
    selected = positives + negatives
    return (
        [row.query for row in selected],
        [row.document for row in selected],
        [int(row.label) for row in selected],
    )

