"""
UNBERT-style model: shared pretrained LM for news/user, contrastive alignment with in-batch negatives,
and self-attention pooling over user history.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel

from src.models.base import RecModelBase
from src.models.registry import register_model


def _bool_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.bool:
        mask = mask != 0
    return mask


class AttentionPooling(nn.Module):
    """
    Additive attention pooling.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Parameter(torch.randn(embed_dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = _bool_mask(mask)
        scores = torch.matmul(torch.tanh(self.proj(x)), self.query)  # (B, L)
        scores = scores.masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, L, 1)
        return torch.sum(weights * x, dim=1)  # (B, D)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, ffn_mult: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * ffn_mult)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * ffn_mult), embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # attn_mask expects True for masked positions; invert
        key_padding_mask = ~_bool_mask(mask)  # (B, L)
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + attn_out
        ffn_out = self.ffn(self.norm2(x))
        return x + ffn_out


class StackedBlocks(nn.Module):
    def __init__(self, num_layers: int, embed_dim: int, num_heads: int, dropout: float, ffn_mult: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, dropout, ffn_mult) for _ in range(max(0, num_layers))]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class NewsEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        embed_dim: int,
        dropout: float,
        category_dim: int = 0,
        subcategory_dim: int = 0,
        category_vocab_size: int = 0,
        subcategory_vocab_size: int = 0,
        token_num_layers: int = 1,
        token_num_heads: int = 8,
        token_ffn_mult: float = 4.0,
        shared_lm: Optional[AutoModel] = None,
    ) -> None:
        super().__init__()
        self.encoder = shared_lm if shared_lm is not None else AutoModel.from_pretrained(pretrained_model_name)
        hidden = self.encoder.config.hidden_size
        self.token_blocks = StackedBlocks(token_num_layers, hidden, token_num_heads, dropout, token_ffn_mult)
        self.token_pool = AttentionPooling(hidden)
        self.token_norm = nn.LayerNorm(hidden)

        cat_vocab = category_vocab_size if category_vocab_size > 0 else 500
        subcat_vocab = subcategory_vocab_size if subcategory_vocab_size > 0 else 500
        self.category_embed = nn.Embedding(1, 1) if category_dim <= 0 else nn.Embedding(cat_vocab, category_dim, padding_idx=0)
        self.subcategory_embed = nn.Embedding(1, 1) if subcategory_dim <= 0 else nn.Embedding(subcat_vocab, subcategory_dim, padding_idx=0)
        cat_added_dim = (category_dim if category_dim > 0 else 0) + (subcategory_dim if subcategory_dim > 0 else 0)

        self.proj = nn.Linear(hidden + cat_added_dim, embed_dim)
        self.out_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        category_ids: Optional[torch.Tensor] = None,
        subcategory_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_states = outputs.last_hidden_state  # (B, L, H)
        token_states = self.token_blocks(token_states, attention_mask)
        pooled = self.token_pool(token_states, attention_mask)
        pooled = self.token_norm(pooled)

        feats = [pooled]
        if category_ids is not None and self.category_embed.num_embeddings > 1:
            feats.append(self.category_embed(category_ids))
        if subcategory_ids is not None and self.subcategory_embed.num_embeddings > 1:
            feats.append(self.subcategory_embed(subcategory_ids))
        combined = torch.cat(feats, dim=-1) if len(feats) > 1 else pooled

        news_vec = self.proj(combined)
        news_vec = self.out_norm(self.dropout(news_vec))
        return news_vec


@register_model("unbert")
class UNBERTRecModel(RecModelBase):
    """
    UNBERT-style recommendation: shared LM, self-attention pooling, dot-product scoring + optional contrastive loss.
    """

    def __init__(self, model_cfg: Dict) -> None:
        super().__init__()
        pretrained_model_name = model_cfg["pretrained_model_name"]
        embed_dim = model_cfg["embed_dim"]
        dropout = model_cfg.get("dropout", 0.1)
        category_dim = model_cfg.get("category_dim", 0)
        subcategory_dim = model_cfg.get("subcategory_dim", 0)
        category_vocab_size = model_cfg.get("category_vocab_size", 0)
        subcategory_vocab_size = model_cfg.get("subcategory_vocab_size", 0)
        token_num_layers = model_cfg.get("token_num_layers", 1)
        token_num_heads = model_cfg.get("token_num_heads", 8)
        token_ffn_mult = model_cfg.get("token_ffn_mult", 4.0)

        user_num_layers = model_cfg.get("user_num_layers", 1)
        user_num_heads = model_cfg.get("user_num_heads", 8)
        user_ffn_mult = model_cfg.get("user_ffn_mult", 4.0)
        self.use_user_positional = model_cfg.get("use_user_positional", True)
        user_max_history = model_cfg.get("user_max_history", 512)

        self.contrastive_weight = model_cfg.get("contrastive_weight", 0.0)
        self.contrastive_temp = model_cfg.get("contrastive_temp", 0.07)

        shared_lm = AutoModel.from_pretrained(pretrained_model_name)
        self.news_encoder = NewsEncoder(
            pretrained_model_name=pretrained_model_name,
            embed_dim=embed_dim,
            dropout=dropout,
            category_dim=category_dim,
            subcategory_dim=subcategory_dim,
            category_vocab_size=category_vocab_size,
            subcategory_vocab_size=subcategory_vocab_size,
            token_num_layers=token_num_layers,
            token_num_heads=token_num_heads,
            token_ffn_mult=token_ffn_mult,
            shared_lm=shared_lm,
        )
        # user encoder only needs self-attention + pooling; reuse news_encoder for per-news encoding
        self.user_blocks = StackedBlocks(user_num_layers, embed_dim, user_num_heads, dropout, user_ffn_mult)
        self.user_pool = AttentionPooling(embed_dim)
        self.user_norm = nn.LayerNorm(embed_dim)
        self.user_pos_embed = (
            nn.Embedding(user_max_history, embed_dim) if self.use_user_positional else None
        )

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

    def _contrastive_loss(self, user_vec: torch.Tensor, cand_emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE with in-batch negatives using candidate embeddings.
        user_vec: (B, E)
        cand_emb: (B, K, E)
        labels: (B, K) one-hot (train batch)
        """
        if self.contrastive_weight <= 0 or not self.training:
            return user_vec.new_tensor(0.0)
        bsz, num_cand, _ = cand_emb.shape
        pos_indices = labels.argmax(dim=1)  # (B,)
        pos_emb = cand_emb[torch.arange(bsz, device=cand_emb.device), pos_indices]  # (B, E)

        pool = cand_emb.view(-1, cand_emb.size(-1))  # (B*K, E)
        logits = torch.matmul(user_vec, pool.t()) / self.contrastive_temp  # (B, B*K)
        target = pos_indices + torch.arange(bsz, device=cand_emb.device) * num_cand  # index in flattened pool
        loss = nn.functional.cross_entropy(logits, target)
        return loss

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor] | torch.Tensor:
        history_input_ids = batch["history_input_ids"]
        history_attention_mask = batch["history_attention_mask"]
        history_mask = batch.get("history_mask")
        candidate_input_ids = batch["candidate_input_ids"]
        candidate_attention_mask = batch["candidate_attention_mask"]
        history_category = batch.get("history_category")
        candidate_category = batch.get("candidate_category")
        history_subcategory = batch.get("history_subcategory")
        candidate_subcategory = batch.get("candidate_subcategory")
        labels = batch.get("labels")  # may be None during inference

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

        if self.user_pos_embed is not None:
            pos_ids = torch.arange(history_emb.size(1), device=history_emb.device).unsqueeze(0).expand(bsz, -1)
            pos_ids = pos_ids.clamp_max(self.user_pos_embed.num_embeddings - 1)
            history_emb = history_emb + self.user_pos_embed(pos_ids)

        user_seq = self.user_blocks(history_emb, history_mask)
        user_vec = self.user_pool(user_seq, history_mask)
        user_vec = self.user_norm(user_vec)  # (B, E)

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
        out: Dict[str, torch.Tensor] = {"scores": scores}
        if labels is not None:
            out["contrastive_loss"] = self._contrastive_loss(user_vec, cand_emb, labels)
        return out
