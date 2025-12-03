import abc
from typing import Dict

import torch
import torch.nn as nn


class RecModelBase(nn.Module, metaclass=abc.ABCMeta):
    """
    Base interface for news recommendation models.
    """

    @abc.abstractmethod
    def encode_news(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of news token sequences.
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
        Returns:
            news_vec: (batch, embed_dim)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: dict produced by collator. Must include:
                history_input_ids: (B, H, L)
                history_attention_mask: (B, H, L)
                history_mask: (B, H) bool
                candidate_input_ids: (B, K, L)
                candidate_attention_mask: (B, K, L)
        Returns:
            scores: (B, K)
        """
        raise NotImplementedError
