from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import TensorDataset
from transformers import PreTrainedTokenizerBase


LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


@dataclass(frozen=True)
class DatasetSizes:
    train_per_class: int = 3_000
    validation_per_class: int = 1_000


def _indices_by_label(dataset: Dataset) -> dict[int, list[int]]:
    result: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(dataset["label"]):
        result[int(label)].append(index)
    return result


def build_stratified_splits(seed: int, sizes: DatasetSizes) -> DatasetDict:
    """Load AG News and create balanced train/validation subsets.

    The official test split is kept intact. Validation examples are taken only
    from the official training split, so the test set remains unseen.
    """

    raw = load_dataset("fancyzhx/ag_news")
    train_source = raw["train"]
    grouped = _indices_by_label(train_source)
    rng = random.Random(seed)

    train_indices: list[int] = []
    validation_indices: list[int] = []
    requested = sizes.train_per_class + sizes.validation_per_class

    for label in sorted(grouped):
        label_indices = grouped[label]
        if len(label_indices) < requested:
            raise ValueError(
                f"Class {label} has {len(label_indices)} examples, "
                f"but {requested} were requested."
            )
        rng.shuffle(label_indices)
        train_indices.extend(label_indices[: sizes.train_per_class])
        validation_indices.extend(
            label_indices[sizes.train_per_class : requested]
        )

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)

    return DatasetDict(
        train=train_source.select(train_indices),
        validation=train_source.select(validation_indices),
        test=raw["test"],
    )


def tokenize_split(
    split: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> TensorDataset:
    encoded = tokenizer(
        list(split["text"]),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = torch.tensor(list(split["label"]), dtype=torch.long)
    return TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels)
