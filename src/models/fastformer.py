import math
from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoModel

from src.models.base import RecModelBase
from src.models.registry import register_model


class NewsEncoder(nn.Module):
    """
    Encodes a single news item using a pretrained language model (e.g., BERT)
    and projects the [CLS] token to a configurable embedding dimension.
    """

    def __init__(self, pretrained_model_name: str, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        hidden = self.encoder.config.hidden_size
        self.proj = nn.Linear(hidden, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
        Returns:
            news_vec: (batch, embed_dim)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_state = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
        news_vec = self.proj(cls_state)  # (batch, embed_dim)
        return self.dropout(news_vec)


class FastformerAggregator(nn.Module):
    """
    Lightweight additive attention inspired by Fastformer to aggregate
    a sequence of news embeddings into a single user embedding.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            mask: (batch, seq_len) bool mask for valid history items
        Returns:
            user_vec: (batch, embed_dim)
        """
        mask = mask.unsqueeze(-1)  # (batch, seq_len, 1)
        q = self.query(x)  # (batch, seq_len, embed_dim)
        k = self.key(x)  # (batch, seq_len, embed_dim)
        v = self.value(x)  # (batch, seq_len, embed_dim)

        q_logits = (q / self.scale).sum(dim=-1)  # (batch, seq_len)
        q_logits = q_logits.masked_fill(~mask.squeeze(-1), -1e4)
        q_weights = torch.softmax(q_logits, dim=-1).unsqueeze(-1)  # (batch, seq_len, 1)
        q_context = torch.sum(q_weights * k, dim=1)  # (batch, embed_dim)

        k_logits = torch.sum(v * q_context.unsqueeze(1), dim=-1)  # (batch, seq_len)
        k_logits = k_logits.masked_fill(~mask.squeeze(-1), -1e4)
        k_weights = torch.softmax(k_logits, dim=-1).unsqueeze(-1)  # (batch, seq_len, 1)
        v_context = torch.sum(k_weights * v, dim=1)  # (batch, embed_dim)

        fused = torch.cat([q_context, v_context], dim=-1)  # (batch, 2 * embed_dim)
        fused = self.dropout(self.out(fused))
        return self.norm(fused + q_context)  # residual with query summary


@register_model("fastformer")
class FastformerRecModel(RecModelBase):
    """
    Full recommendation model: news encoder + Fastformer-style user encoder.
    """

    def __init__(self, model_cfg: Dict) -> None:
        super().__init__()
        pretrained_model_name = model_cfg["pretrained_model_name"]
        embed_dim = model_cfg["embed_dim"]
        dropout = model_cfg.get("dropout", 0.0)
        self.news_encoder = NewsEncoder(pretrained_model_name, embed_dim, dropout)
        self.user_encoder = FastformerAggregator(embed_dim, dropout)

    def encode_news(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.news_encoder(input_ids=input_ids, attention_mask=attention_mask)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: dict with keys:
                history_input_ids: (B, H, L)
                history_attention_mask: (B, H, L)
                history_mask: (B, H) bool for valid history rows
                candidate_input_ids: (B, K, L)
                candidate_attention_mask: (B, K, L)
        Returns:
            scores: (B, K)
        """
        history_input_ids = batch["history_input_ids"]
        history_attention_mask = batch["history_attention_mask"]
        history_mask = batch.get("history_mask")
        candidate_input_ids = batch["candidate_input_ids"]
        candidate_attention_mask = batch["candidate_attention_mask"]

        bsz, hist_len, seq_len = history_input_ids.shape
        history_flat = history_input_ids.view(-1, seq_len)
        history_mask_flat = history_attention_mask.view(-1, seq_len)
        history_emb = self.news_encoder(history_flat, history_mask_flat)  # (B*H, E)
        history_emb = history_emb.view(bsz, hist_len, -1)  # (B, H, E)
        if history_mask is None:
            history_mask = history_attention_mask.sum(dim=-1) > 0  # (B, H)
        user_vec = self.user_encoder(history_emb, history_mask)  # (B, E)

        _, num_cand, _ = candidate_input_ids.shape
        cand_flat = candidate_input_ids.view(-1, seq_len)
        cand_mask_flat = candidate_attention_mask.view(-1, seq_len)
        cand_emb = self.news_encoder(cand_flat, cand_mask_flat).view(bsz, num_cand, -1)  # (B, K, E)

        scores = torch.bmm(cand_emb, user_vec.unsqueeze(-1)).squeeze(-1)  # (B, K)
        return scores
