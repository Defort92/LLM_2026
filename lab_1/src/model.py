from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from transformers import AutoModel


PoolingType = Literal["mean", "cls"]


class TextClassifier(nn.Module):
    """Encoder model with a selectable token-pooling strategy."""

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pooling: PoolingType,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "cls"}:
            raise ValueError(f"Unsupported pooling: {pooling}")

        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def _pool(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.pooling == "cls":
            return token_embeddings[:, 0]

        expanded_mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        embedding_sum = (token_embeddings * expanded_mask).sum(dim=1)
        valid_token_count = expanded_mask.sum(dim=1).clamp(min=1e-9)
        return embedding_sum / valid_token_count

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_vector = self._pool(
            token_embeddings=encoder_output.last_hidden_state,
            attention_mask=attention_mask,
        )
        return self.classifier(self.dropout(text_vector))

