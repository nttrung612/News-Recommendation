import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel

from src.models.base import RecModelBase
from src.models.registry import register_model


class NewsEncoder(nn.Module):
    """
    Encodes a single news item using a pretrained language model (e.g., BERT/Roberta).
    Token-level hidden states are aggregated with a Fastformer-style pooling instead of plain [CLS].
    Optionally adds category/subcategory embeddings before projecting to embed_dim.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        embed_dim: int,
        dropout: float,
        use_token_fastformer: bool = True,
        category_dim: int = 0,
        subcategory_dim: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        hidden = self.encoder.config.hidden_size
        self.use_token_fastformer = use_token_fastformer
        if use_token_fastformer:
            self.token_pool = FastformerAggregator(hidden, dropout)  # pool tokens -> hidden
        self.proj_in = hidden

        self.category_embed = nn.Embedding(1, 1) if category_dim <= 0 else nn.Embedding(500, category_dim, padding_idx=0)
        self.subcategory_embed = nn.Embedding(1, 1) if subcategory_dim <= 0 else nn.Embedding(500, subcategory_dim, padding_idx=0)
        cat_added_dim = (category_dim if category_dim > 0 else 0) + (subcategory_dim if subcategory_dim > 0 else 0)
        self.proj = nn.Linear(self.proj_in + cat_added_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        category_ids: Optional[torch.Tensor] = None,
        subcategory_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            category_ids: (batch,) optional
            subcategory_ids: (batch,) optional
        Returns:
            news_vec: (batch, embed_dim)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_states = outputs.last_hidden_state  # (batch, seq_len, hidden)
        if self.use_token_fastformer:
            # history_attention_mask: 1 for real tokens; convert to bool mask
            token_mask = attention_mask.bool()
            pooled = self.token_pool(token_states, token_mask)  # (batch, hidden)
        else:
            pooled = token_states[:, 0, :]  # fallback to [CLS]

        feats = [pooled]
        if category_ids is not None and self.category_embed.num_embeddings > 1:
            feats.append(self.category_embed(category_ids))
        if subcategory_ids is not None and self.subcategory_embed.num_embeddings > 1:
            feats.append(self.subcategory_embed(subcategory_ids))
        combined = torch.cat(feats, dim=-1) if len(feats) > 1 else pooled

        news_vec = self.proj(combined)  # (batch, embed_dim)
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
            mask: (batch, seq_len) bool mask for valid items
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
        use_token_fastformer = model_cfg.get("use_token_fastformer", True)
        category_dim = model_cfg.get("category_dim", 0)
        subcategory_dim = model_cfg.get("subcategory_dim", 0)
        self.news_encoder = NewsEncoder(
            pretrained_model_name,
            embed_dim,
            dropout,
            use_token_fastformer=use_token_fastformer,
            category_dim=category_dim,
            subcategory_dim=subcategory_dim,
        )
        self.user_encoder = FastformerAggregator(embed_dim, dropout)

    def encode_news(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        category_ids: Optional[torch.Tensor] = None,
        subcategory_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.news_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            category_ids=category_ids,
            subcategory_ids=subcategory_ids,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: dict with keys:
                history_input_ids: (B, H, L)
                history_attention_mask: (B, H, L)
                history_mask: (B, H) bool for valid history rows
                candidate_input_ids: (B, K, L)
                candidate_attention_mask: (B, K, L)
                history_category / candidate_category (optional): (B, H)/(B, K)
                history_subcategory / candidate_subcategory (optional): (B, H)/(B, K)
        Returns:
            scores: (B, K)
        """
        history_input_ids = batch["history_input_ids"]
        history_attention_mask = batch["history_attention_mask"]
        history_mask = batch.get("history_mask")
        candidate_input_ids = batch["candidate_input_ids"]
        candidate_attention_mask = batch["candidate_attention_mask"]
        history_category = batch.get("history_category")
        candidate_category = batch.get("candidate_category")
        history_subcategory = batch.get("history_subcategory")
        candidate_subcategory = batch.get("candidate_subcategory")

        bsz, hist_len, seq_len = history_input_ids.shape
        history_flat = history_input_ids.view(-1, seq_len)
        history_mask_flat = history_attention_mask.view(-1, seq_len)
        hist_cat_flat = history_category.view(-1) if history_category is not None else None
        hist_subcat_flat = history_subcategory.view(-1) if history_subcategory is not None else None
        history_emb = self.news_encoder(
            history_flat,
            history_mask_flat,
            hist_cat_flat,
            hist_subcat_flat,
        )  # (B*H, E)
        history_emb = history_emb.view(bsz, hist_len, -1)  # (B, H, E)
        if history_mask is None:
            history_mask = history_attention_mask.sum(dim=-1) > 0  # (B, H)
        user_vec = self.user_encoder(history_emb, history_mask)  # (B, E)

        _, num_cand, _ = candidate_input_ids.shape
        cand_flat = candidate_input_ids.view(-1, seq_len)
        cand_mask_flat = candidate_attention_mask.view(-1, seq_len)
        cand_cat_flat = candidate_category.view(-1) if candidate_category is not None else None
        cand_subcat_flat = candidate_subcategory.view(-1) if candidate_subcategory is not None else None
        cand_emb = self.news_encoder(
            cand_flat,
            cand_mask_flat,
            cand_cat_flat,
            cand_subcat_flat,
        ).view(bsz, num_cand, -1)  # (B, K, E)

        scores = torch.bmm(cand_emb, user_vec.unsqueeze(-1)).squeeze(-1)  # (B, K)
        return scores
