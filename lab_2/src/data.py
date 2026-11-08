from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import Dataset


DATASET_NAME = "OpenRL/daily_dialog"


@dataclass(frozen=True)
class DialoguePair:
    message: str
    response: str


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def load_dialogue_pairs(split: str, limit: int | None, seed: int) -> tuple[list[DialoguePair], dict[str, Any]]:
    """Convert each dialogue into adjacent message-response pairs.

    DailyDialog contains ordered utterances but no explicit speaker names.  Because
    speakers alternate, every adjacent transition can be treated as one training
    example: utterance[i] is the incoming message and utterance[i + 1] is the reply.
    """
    raw_split = load_dataset(DATASET_NAME, split=split)
    pairs: list[DialoguePair] = []

    for row in raw_split:
        # Parquet versions use ``dialog``; older DailyDialog scripts used
        # ``utterances``. Supporting both keeps preprocessing transparent.
        dialogue = row.get("dialog", row.get("utterances"))
        if dialogue is None:
            raise KeyError("Expected a 'dialog' or 'utterances' column in DailyDialog")
        utterances = [_clean(value) for value in dialogue]
        for index in range(len(utterances) - 1):
            message, response = utterances[index], utterances[index + 1]
            if message and response:
                pairs.append(DialoguePair(message=message, response=response))

    all_pairs = len(pairs)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    if limit is not None:
        pairs = pairs[:limit]

    metadata = {
        "split": split,
        "dialogues": len(raw_split),
        "all_adjacent_pairs": all_pairs,
        "selected_pairs": len(pairs),
        "seed": seed,
    }
    return pairs, metadata


def decoder_prompt(message: str) -> str:
    return f"User: {message}\nAssistant:"


def seq2seq_prompt(message: str) -> str:
    return f"Generate a natural reply to this dialogue message: {message}"


class DecoderOnlyDataset(Dataset):
    def __init__(
        self,
        pairs: list[DialoguePair],
        tokenizer: Any,
        max_input_length: int,
        max_target_length: int,
    ) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.max_length = max_input_length + max_target_length
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        pair = self.pairs[index]
        prompt_ids = self.tokenizer.encode(
            decoder_prompt(pair.message),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_input_length,
        )
        response_ids = self.tokenizer.encode(
            " " + pair.response,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_target_length - 1,
        )
        response_ids.append(self.eos_token_id)

        input_ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids.copy()
        attention_mask = [1] * len(input_ids)
        padding_length = self.max_length - len(input_ids)

        input_ids.extend([self.pad_token_id] * padding_length)
        attention_mask.extend([0] * padding_length)
        labels.extend([-100] * padding_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class EncoderDecoderDataset(Dataset):
    def __init__(
        self,
        pairs: list[DialoguePair],
        tokenizer: Any,
        max_input_length: int,
        max_target_length: int,
    ) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        pair = self.pairs[index]
        source = self.tokenizer(
            seq2seq_prompt(pair.message),
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer(
            text_target=pair.response,
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = target["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source["input_ids"].squeeze(0),
            "attention_mask": source["attention_mask"].squeeze(0),
            "labels": labels,
        }
