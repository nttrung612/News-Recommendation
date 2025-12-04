import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


@dataclass
class NewsTensorStore:
    """
    Container for tokenized news and metadata features.
    """

    input_ids: torch.Tensor  # (num_news + 1, seq_len); row 0 is padding
    attention_mask: torch.Tensor  # (num_news + 1, seq_len); row 0 is zeros
    news2idx: Dict[str, int]
    idx2news: List[str]
    category_ids: torch.Tensor | None = None  # (num_news + 1,)
    subcategory_ids: torch.Tensor | None = None  # (num_news + 1,)
    category_vocab: Dict[str, int] | None = None
    subcategory_vocab: Dict[str, int] | None = None


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Load a Hugging Face tokenizer. Requires the model to be available locally
    or downloadable (network).
    """
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception:
        # fallback for models whose fast tokenizer is not bundled correctly
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)


def _vocab_compatible(stored_vocab: Optional[Dict[str, int]], reference_vocab: Optional[Dict[str, int]]) -> bool:
    """
    Check that overlapping category/subcategory ids match between stored cache and provided vocab.
    """
    if reference_vocab is None:
        return True
    if stored_vocab is None:
        return False
    for key, ref_val in reference_vocab.items():
        stored_val = stored_vocab.get(key)
        if stored_val is not None and stored_val != ref_val:
            return False
    return True


def load_news_tensors(
    news_path: str,
    tokenizer: AutoTokenizer,
    max_len: int,
    cache_dir: str | None = None,
    category_vocab: Dict[str, int] | None = None,
    subcategory_vocab: Dict[str, int] | None = None,
) -> NewsTensorStore:
    """
    Tokenize news file and return tensors suitable for indexing.
    Adds a padding row at index 0 so history/candidate index 0 can be used
    as the padded news item.
    """
    news_path = Path(news_path)
    cache_file = None
    cache_version = "v3"  # bump to invalidate older cache formats
    if cache_dir:
        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir_path / f"{news_path.stem}_{tokenizer.name_or_path.replace('/', '_')}_{max_len}_{cache_version}.pt"
        if cache_file.exists():
            # torch>=2.6 defaults to weights_only=True; allow NewsTensorStore class
            try:
                torch.serialization.add_safe_globals([NewsTensorStore])
                cached = torch.load(cache_file, weights_only=False)
            except TypeError:
                # fallback for older torch versions without weights_only arg
                cached = torch.load(cache_file)
            if _vocab_compatible(getattr(cached, "category_vocab", None), category_vocab) and _vocab_compatible(
                getattr(cached, "subcategory_vocab", None), subcategory_vocab
            ):
                return cached

    columns = [
        "news_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]
    df = pd.read_table(news_path, header=None, names=columns)
    texts = (df["title"].fillna("") + ". " + df["abstract"].fillna("")).tolist()

    encodings = tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    input_ids = torch.tensor(encodings["input_ids"], dtype=torch.long)  # (num_news, seq_len)
    attention_mask = torch.tensor(encodings["attention_mask"], dtype=torch.long)  # (num_news, seq_len)

    # categorical features for NAML-style models
    cat_series = df["category"].fillna("uncat")
    subcat_series = df["subcategory"].fillna("unsubcat")
    cat_vocab = dict(category_vocab) if category_vocab else {}
    subcat_vocab = dict(subcategory_vocab) if subcategory_vocab else {}
    cat_ids_list = []
    for c in cat_series:
        if c not in cat_vocab:
            cat_vocab[c] = len(cat_vocab) + 1
        cat_ids_list.append(cat_vocab[c])
    subcat_ids_list = []
    for c in subcat_series:
        if c not in subcat_vocab:
            subcat_vocab[c] = len(subcat_vocab) + 1
        subcat_ids_list.append(subcat_vocab[c])
    cat_ids = torch.tensor(cat_ids_list, dtype=torch.long)  # (num_news,)
    subcat_ids = torch.tensor(subcat_ids_list, dtype=torch.long)  # (num_news,)

    # prepend padding row to simplify masking later
    pad_row = torch.zeros(1, input_ids.size(1), dtype=torch.long)
    input_ids = torch.cat([pad_row, input_ids], dim=0)  # (num_news + 1, seq_len)
    attention_mask = torch.cat([pad_row, attention_mask], dim=0)  # (num_news + 1, seq_len)
    pad_cat = torch.zeros(1, dtype=torch.long)
    cat_ids = torch.cat([pad_cat, cat_ids], dim=0)  # (num_news + 1,)
    subcat_ids = torch.cat([pad_cat, subcat_ids], dim=0)  # (num_news + 1,)

    news2idx = {nid: i + 1 for i, nid in enumerate(df["news_id"].tolist())}
    idx2news = ["[PAD]"] + df["news_id"].tolist()

    store = NewsTensorStore(
        input_ids=input_ids,
        attention_mask=attention_mask,
        news2idx=news2idx,
        idx2news=idx2news,
        category_ids=cat_ids,
        subcategory_ids=subcat_ids,
        category_vocab=cat_vocab,
        subcategory_vocab=subcat_vocab,
    )
    if cache_file:
        torch.save(store, cache_file)
    return store


class MindTrainDataset(Dataset):
    """
    Generates training samples with one positive and neg_k negatives for each impression.
    """

    def __init__(
        self,
        behaviors_path: str,
        news2idx: Dict[str, int],
        max_history: int,
        neg_k: int,
    ) -> None:
        self.news2idx = news2idx
        self.max_history = max_history
        self.neg_k = neg_k
        self.samples: List[Dict] = []
        self._prepare(behaviors_path)

    def _prepare(self, behaviors_path: str) -> None:
        with open(behaviors_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 5:
                    continue
                history_raw = row[3].split() if row[3] else []
                history_idx = [self.news2idx.get(nid, 0) for nid in history_raw][-self.max_history :]
                impressions = row[4].split()
                pos, neg = [], []
                for imp in impressions:
                    if "-" not in imp:
                        continue
                    nid, label = imp.rsplit("-", 1)
                    idx = self.news2idx.get(nid, 0)
                    if label == "1":
                        pos.append(idx)
                    else:
                        neg.append(idx)

                for pos_idx in pos:
                    if not neg:
                        continue  # skip if no negatives to sample
                    self.samples.append(
                        {
                            "history": history_idx,
                            "positive": pos_idx,
                            "neg_pool": neg,
                        }
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        neg_pool = sample["neg_pool"]
        negatives = random.sample(neg_pool, self.neg_k) if len(neg_pool) >= self.neg_k else random.choices(neg_pool, k=self.neg_k)
        return {
            "history": sample["history"],
            "positive": sample["positive"],
            "negatives": negatives,
        }


class MindEvalDataset(Dataset):
    """
    Provides evaluation samples; keeps full impression list.
    """

    def __init__(
        self,
        behaviors_path: str,
        news2idx: Dict[str, int],
        max_history: int,
    ) -> None:
        self.news2idx = news2idx
        self.max_history = max_history
        self.samples: List[Dict] = []
        self._prepare(behaviors_path)

    def _prepare(self, behaviors_path: str) -> None:
        with open(behaviors_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 5:
                    continue
                impression_id = row[0]
                history_raw = row[3].split() if row[3] else []
                history_idx = [self.news2idx.get(nid, 0) for nid in history_raw][-self.max_history :]
                impressions = row[4].split()
                candidates, labels = [], []
                for imp in impressions:
                    if "-" in imp:
                        nid, label = imp.rsplit("-", 1)
                    else:  # test/inference set may omit labels
                        nid, label = imp, "0"
                    candidates.append(self.news2idx.get(nid, 0))
                    labels.append(int(label))
                if candidates:
                    self.samples.append(
                        {
                            "impression_id": impression_id,
                            "history": history_idx,
                            "candidates": candidates,
                            "labels": labels,
                        }
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


class MindTrainCollator:
    """
    Collate for training with fixed candidate count (1 positive + neg_k negatives).
    """

    def __init__(
        self,
        news_store: NewsTensorStore,
        max_history: int,
        neg_k: int,
    ) -> None:
        self.news_store = news_store
        self.max_history = max_history
        self.neg_k = neg_k

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        device = self.news_store.input_ids.device
        seq_len = self.news_store.input_ids.size(1)

        histories = torch.zeros(batch_size, self.max_history, dtype=torch.long, device=device)
        for i, item in enumerate(batch):
            hist = item["history"][-self.max_history :]
            if hist:
                histories[i, -len(hist) :] = torch.tensor(hist, dtype=torch.long, device=device)

        candidates = torch.zeros(batch_size, self.neg_k + 1, dtype=torch.long, device=device)
        labels = torch.zeros(batch_size, self.neg_k + 1, dtype=torch.long, device=device)
        for i, item in enumerate(batch):
            cand_tensor = torch.tensor([item["positive"]] + item["negatives"], dtype=torch.long, device=device)
            perm = torch.randperm(cand_tensor.size(0), device=device)
            shuffled = cand_tensor[perm]
            candidates[i] = shuffled
            pos_idx = int((perm == 0).nonzero(as_tuple=False)[0].item())  # positive was at index 0 before shuffle
            labels[i, pos_idx] = 1

        cand_flat = candidates.view(-1)  # (batch_size * (neg_k + 1))
        cand_input_ids = self.news_store.input_ids[cand_flat].view(batch_size, self.neg_k + 1, seq_len)
        cand_attention = self.news_store.attention_mask[cand_flat].view(batch_size, self.neg_k + 1, seq_len)

        hist_flat = histories.view(-1)  # (batch_size * max_history)
        hist_input_ids = self.news_store.input_ids[hist_flat].view(batch_size, self.max_history, seq_len)
        hist_attention = self.news_store.attention_mask[hist_flat].view(batch_size, self.max_history, seq_len)
        history_mask = hist_attention.sum(dim=-1) > 0  # (batch_size, max_history) bool mask

        candidate_mask = torch.ones(batch_size, self.neg_k + 1, dtype=torch.bool, device=device)

        out: Dict[str, torch.Tensor] = {
            # history_input_ids: (batch_size, max_history, seq_len)
            "history_input_ids": hist_input_ids,
            # history_attention_mask: (batch_size, max_history, seq_len)
            "history_attention_mask": hist_attention,
            "history_mask": history_mask,  # (batch_size, max_history)
            # candidate_input_ids: (batch_size, neg_k + 1, seq_len)
            "candidate_input_ids": cand_input_ids,
            # candidate_attention_mask: (batch_size, neg_k + 1, seq_len)
            "candidate_attention_mask": cand_attention,
            "candidate_mask": candidate_mask,  # (batch_size, neg_k + 1) all True
            # candidate_indices: (batch_size, neg_k + 1) news indices before token lookup
            "candidate_indices": candidates,
            # labels: (batch_size, neg_k + 1) with one positive
            "labels": labels,
        }

        if self.news_store.category_ids is not None:
            hist_cat = self.news_store.category_ids[hist_flat].view(batch_size, self.max_history)
            cand_cat = self.news_store.category_ids[cand_flat].view(batch_size, self.neg_k + 1)
            out["history_category"] = hist_cat  # (batch_size, max_history)
            out["candidate_category"] = cand_cat  # (batch_size, neg_k + 1)
        if self.news_store.subcategory_ids is not None:
            hist_subcat = self.news_store.subcategory_ids[hist_flat].view(batch_size, self.max_history)
            cand_subcat = self.news_store.subcategory_ids[cand_flat].view(batch_size, self.neg_k + 1)
            out["history_subcategory"] = hist_subcat  # (batch_size, max_history)
            out["candidate_subcategory"] = cand_subcat  # (batch_size, neg_k + 1)

        return out


class MindEvalCollator:
    """
    Collate for evaluation with variable candidate counts per impression.
    Pads candidates to the maximum length in the batch.
    """

    def __init__(
        self,
        news_store: NewsTensorStore,
        max_history: int,
    ) -> None:
        self.news_store = news_store
        self.max_history = max_history

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor | List[str]]:
        batch_size = len(batch)
        device = self.news_store.input_ids.device
        seq_len = self.news_store.input_ids.size(1)
        max_cand = max(len(item["candidates"]) for item in batch)

        histories = torch.zeros(batch_size, self.max_history, dtype=torch.long, device=device)
        for i, item in enumerate(batch):
            hist = item["history"][-self.max_history :]
            if hist:
                histories[i, -len(hist) :] = torch.tensor(hist, dtype=torch.long, device=device)

        candidates = torch.zeros(batch_size, max_cand, dtype=torch.long, device=device)
        candidate_mask = torch.zeros(batch_size, max_cand, dtype=torch.bool, device=device)
        labels = torch.zeros(batch_size, max_cand, dtype=torch.long, device=device)
        impression_ids = []
        for i, item in enumerate(batch):
            cand = item["candidates"]
            lbls = item["labels"]
            length = len(cand)
            candidates[i, :length] = torch.tensor(cand, dtype=torch.long, device=device)
            labels[i, :length] = torch.tensor(lbls, dtype=torch.long, device=device)
            candidate_mask[i, :length] = 1
            impression_ids.append(item["impression_id"])

        cand_flat = candidates.view(-1)
        cand_input_ids = self.news_store.input_ids[cand_flat].view(batch_size, max_cand, seq_len)
        cand_attention = self.news_store.attention_mask[cand_flat].view(batch_size, max_cand, seq_len)

        hist_flat = histories.view(-1)
        hist_input_ids = self.news_store.input_ids[hist_flat].view(batch_size, self.max_history, seq_len)
        hist_attention = self.news_store.attention_mask[hist_flat].view(batch_size, self.max_history, seq_len)
        history_mask = hist_attention.sum(dim=-1) > 0  # (batch_size, max_history) bool mask

        out: Dict[str, torch.Tensor | List[str]] = {
            "history_input_ids": hist_input_ids,  # (batch_size, max_history, seq_len)
            "history_attention_mask": hist_attention,  # (batch_size, max_history, seq_len)
            "history_mask": history_mask,  # (batch_size, max_history) bool mask
            "candidate_input_ids": cand_input_ids,  # (batch_size, max_cand, seq_len)
            "candidate_attention_mask": cand_attention,  # (batch_size, max_cand, seq_len)
            "candidate_mask": candidate_mask,  # (batch_size, max_cand) bool mask for valid candidates
            "candidate_indices": candidates,  # (batch_size, max_cand) news indices before token lookup
            "labels": labels,  # (batch_size, max_cand)
            "impression_ids": impression_ids,
        }

        if self.news_store.category_ids is not None:
            hist_cat = self.news_store.category_ids[hist_flat].view(batch_size, self.max_history)
            cand_cat = self.news_store.category_ids[cand_flat].view(batch_size, max_cand)
            out["history_category"] = hist_cat  # (batch_size, max_history)
            out["candidate_category"] = cand_cat  # (batch_size, max_cand)
        if self.news_store.subcategory_ids is not None:
            hist_subcat = self.news_store.subcategory_ids[hist_flat].view(batch_size, self.max_history)
            cand_subcat = self.news_store.subcategory_ids[cand_flat].view(batch_size, max_cand)
            out["history_subcategory"] = hist_subcat  # (batch_size, max_history)
            out["candidate_subcategory"] = cand_subcat  # (batch_size, max_cand)

        return out
