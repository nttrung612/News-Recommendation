import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel

from src.models.base import RecModelBase
from src.models.registry import register_model


def _ensure_bool_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.bool:
        mask = mask != 0
    if mask.dim() == 2:
        return mask
    if mask.dim() == 3:
        return mask.any(dim=-1)
    raise ValueError(f"Unsupported mask shape {mask.shape}")


class FastformerAttention(nn.Module):
    """
    Multi-head additive attention from Fastformer.
    Keeps sequence length; uses global query/key to reweight values.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            mask: (batch, seq_len) bool for valid tokens/items
        Returns:
            out: (batch, seq_len, embed_dim)
        """
        bsz, seq_len, _ = x.shape
        attn_mask = _ensure_bool_mask(mask)  # (B, L)
        mask_exp = attn_mask.unsqueeze(1)  # (B, 1, L)

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, hd)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q_logits = (q.sum(dim=-1) / self.scale)  # (B, H, L)
        q_logits = q_logits.masked_fill(~mask_exp, -1e4)
        q_weights = torch.softmax(q_logits, dim=-1)  # (B, H, L)
        q_global = torch.sum(q_weights.unsqueeze(-1) * q, dim=2)  # (B, H, hd)

        k_logits = torch.sum(k * q_global.unsqueeze(2), dim=-1)  # (B, H, L)
        k_logits = k_logits.masked_fill(~mask_exp, -1e4)
        k_weights = torch.softmax(k_logits, dim=-1)
        k_global = torch.sum(k_weights.unsqueeze(-1) * k, dim=2)  # (B, H, hd)

        v_logits = torch.sum(q * k_global.unsqueeze(2), dim=-1)  # (B, H, L)
        v_logits = v_logits.masked_fill(~mask_exp, -1e4)
        v_weights = torch.softmax(v_logits, dim=-1)
        v_weights = self.dropout(v_weights)
        attn_out = v_weights.unsqueeze(-1) * v  # (B, H, L, hd)

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)  # (B, L, D)
        return self.out_proj(attn_out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_mult: float, dropout: float) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * hidden_mult)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FastformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, ffn_mult: float) -> None:
        super().__init__()
        self.attn = FastformerAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_mult, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x), mask)
        x = x + attn_out
        ffn_out = self.ffn(self.norm2(x))
        return x + ffn_out


class FastformerEncoder(nn.Module):
    """
    Stack of Fastformer blocks (sequence-in, sequence-out).
    """

    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, dropout: float, ffn_mult: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [FastformerBlock(embed_dim, num_heads, dropout, ffn_mult) for _ in range(max(0, num_layers))]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not self.layers:
            return x
        for layer in self.layers:
            x = layer(x, mask)
        return x


class FastformerPooling(nn.Module):
    """
    Global pooling using Fastformer additive attention (multi-head).
    Returns a single vector per sequence.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            mask: (batch, seq_len) bool
        Returns:
            pooled: (batch, embed_dim)
        """
        bsz, seq_len, _ = x.shape
        attn_mask = _ensure_bool_mask(mask)  # (B, L)
        mask_exp = attn_mask.unsqueeze(1)  # (B,1,L)

        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,L,hd)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q_logits = (q.sum(dim=-1) / self.scale)  # (B,H,L)
        q_logits = q_logits.masked_fill(~mask_exp, -1e4)
        q_weights = torch.softmax(q_logits, dim=-1)
        q_global = torch.sum(q_weights.unsqueeze(-1) * q, dim=2)  # (B,H,hd)

        k_logits = torch.sum(k * q_global.unsqueeze(2), dim=-1)  # (B,H,L)
        k_logits = k_logits.masked_fill(~mask_exp, -1e4)
        k_weights = torch.softmax(k_logits, dim=-1)
        k_global = torch.sum(k_weights.unsqueeze(-1) * k, dim=2)  # (B,H,hd)

        v_logits = torch.sum(q * k_global.unsqueeze(2), dim=-1)  # (B,H,L)
        v_logits = v_logits.masked_fill(~mask_exp, -1e4)
        v_weights = torch.softmax(v_logits, dim=-1)
        v_weights = self.dropout(v_weights)
        v_global = torch.sum(v_weights.unsqueeze(-1) * v, dim=2)  # (B,H,hd)

        fused = torch.cat([q_global, v_global], dim=-1)  # (B,H,2hd)
        fused = fused.transpose(1, 2).contiguous().view(bsz, self.embed_dim * 2)  # (B,2D)
        out = self.dropout(self.out_proj(fused))  # (B,D)

        # Residual from global query signal (flattened to D)
        residual = q_global.transpose(1, 2).contiguous().view(bsz, self.embed_dim)
        return self.norm(out + residual)


class NewsEncoder(nn.Module):
    """
    Encodes a single news item using a pretrained language model (e.g., DeBERTa).
    Token states are refined with stacked Fastformer blocks + pooling, then projected to embed_dim.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        embed_dim: int,
        dropout: float,
        use_token_fastformer: bool = True,
        category_dim: int = 0,
        subcategory_dim: int = 0,
        category_vocab_size: int = 0,
        subcategory_vocab_size: int = 0,
        token_num_layers: int = 1,
        token_num_heads: int = 8,
        token_ffn_mult: float = 4.0,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        hidden = self.encoder.config.hidden_size
        self.use_token_fastformer = use_token_fastformer
        if use_token_fastformer:
            self.token_encoder = FastformerEncoder(hidden, token_num_heads, token_num_layers, dropout, token_ffn_mult)
            self.token_pool = FastformerPooling(hidden, token_num_heads, dropout)
            self.token_norm = nn.LayerNorm(hidden)
        else:
            self.token_encoder = None
            self.token_pool = None
            self.token_norm = None

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

        if self.use_token_fastformer:
            token_mask = attention_mask.bool()
            token_states = self.token_encoder(token_states, token_mask)
            pooled = self.token_pool(token_states, token_mask)
            pooled = self.token_norm(pooled)
        else:
            pooled = token_states[:, 0, :]  # [CLS]

        feats = [pooled]
        if category_ids is not None and self.category_embed.num_embeddings > 1:
            feats.append(self.category_embed(category_ids))
        if subcategory_ids is not None and self.subcategory_embed.num_embeddings > 1:
            feats.append(self.subcategory_embed(subcategory_ids))
        combined = torch.cat(feats, dim=-1) if len(feats) > 1 else pooled

        news_vec = self.proj(combined)
        news_vec = self.out_norm(self.dropout(news_vec))
        return news_vec


@register_model("fastformer")
class FastformerRecModel(RecModelBase):
    """
    Full recommendation model: news encoder + stacked Fastformer user encoder + pooling.
    """

    def __init__(self, model_cfg: Dict) -> None:
        super().__init__()
        pretrained_model_name = model_cfg["pretrained_model_name"]
        embed_dim = model_cfg["embed_dim"]
        dropout = model_cfg.get("dropout", 0.0)
        use_token_fastformer = model_cfg.get("use_token_fastformer", True)
        category_dim = model_cfg.get("category_dim", 0)
        subcategory_dim = model_cfg.get("subcategory_dim", 0)
        category_vocab_size = model_cfg.get("category_vocab_size", 0)
        subcategory_vocab_size = model_cfg.get("subcategory_vocab_size", 0)
        token_num_layers = model_cfg.get("token_num_layers", 1)
        token_num_heads = model_cfg.get("token_num_heads", 8)
        token_ffn_mult = model_cfg.get("token_ffn_mult", 4.0)

        user_num_layers = model_cfg.get("user_num_layers", 2)
        user_num_heads = model_cfg.get("user_num_heads", 8)
        user_ffn_mult = model_cfg.get("user_ffn_mult", 4.0)
        self.use_user_positional = model_cfg.get("use_user_positional", True)
        user_max_history = model_cfg.get("user_max_history", 512)

        self.news_encoder = NewsEncoder(
            pretrained_model_name,
            embed_dim,
            dropout,
            use_token_fastformer=use_token_fastformer,
            category_dim=category_dim,
            subcategory_dim=subcategory_dim,
            category_vocab_size=category_vocab_size,
            subcategory_vocab_size=subcategory_vocab_size,
            token_num_layers=token_num_layers,
            token_num_heads=token_num_heads,
            token_ffn_mult=token_ffn_mult,
        )
        self.user_encoder = FastformerEncoder(embed_dim, user_num_heads, user_num_layers, dropout, user_ffn_mult)
        self.user_pool = FastformerPooling(embed_dim, user_num_heads, dropout)
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

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
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

        if self.user_pos_embed is not None:
            pos_ids = torch.arange(history_emb.size(1), device=history_emb.device).unsqueeze(0).expand(bsz, -1)
            pos_ids = pos_ids.clamp_max(self.user_pos_embed.num_embeddings - 1)
            history_emb = history_emb + self.user_pos_embed(pos_ids)

        user_seq = self.user_encoder(history_emb, history_mask)
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
        return scores
