import argparse
import json
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.config import load_config
from src.data.mind import MindEvalCollator, MindEvalDataset, load_news_tensors, load_tokenizer
import src.models.fastformer  # register fastformer
from src.models.registry import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained checkpoint path (.pt).")
    parser.add_argument("--behaviors", type=str, required=True, help="Behaviors file for inference.")
    parser.add_argument("--news", type=str, required=True, help="News file corresponding to behaviors.")
    parser.add_argument("--output", type=str, default="predictions.jsonl", help="Output JSONL file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_cfg = checkpoint.get("model_config", cfg["model"])
    tokenizer_name = checkpoint.get("tokenizer_name", model_cfg.get("tokenizer_name", model_cfg.get("pretrained_model_name")))
    category_vocab = checkpoint.get("category_vocab")
    subcategory_vocab = checkpoint.get("subcategory_vocab")
    if "category_vocab_size" not in model_cfg and category_vocab:
        model_cfg["category_vocab_size"] = len(category_vocab) + 1
    if "subcategory_vocab_size" not in model_cfg and subcategory_vocab:
        model_cfg["subcategory_vocab_size"] = len(subcategory_vocab) + 1
    tokenizer = load_tokenizer(tokenizer_name)

    data_cfg = checkpoint.get("config", cfg)["data"]
    news_store = load_news_tensors(
        args.news,
        tokenizer,
        data_cfg["max_seq_len"],
        data_cfg["cache_dir"],
        category_vocab=category_vocab,
        subcategory_vocab=subcategory_vocab,
    )
    news_store.input_ids = news_store.input_ids.to(device)
    news_store.attention_mask = news_store.attention_mask.to(device)
    if news_store.category_ids is not None:
        news_store.category_ids = news_store.category_ids.to(device)
    if news_store.subcategory_ids is not None:
        news_store.subcategory_ids = news_store.subcategory_ids.to(device)

    dataset = MindEvalDataset(
        behaviors_path=args.behaviors,
        news2idx=news_store.news2idx,
        max_history=data_cfg["max_history"],
    )
    collate = MindEvalCollator(news_store, data_cfg["max_history"])
    dataloader = DataLoader(dataset, batch_size=cfg["eval"]["batch_size"], shuffle=False, collate_fn=collate)

    model = build_model(model_cfg)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    model.eval()
    results = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Infer"):
            # Move tensor parts to device
            tensor_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            scores = model(tensor_batch)  # (batch, max_cand)
            scores = scores.masked_fill(~tensor_batch["candidate_mask"], float("-inf"))
            for i in range(scores.size(0)):
                valid_len = int(tensor_batch["candidate_mask"][i].sum().item())
                preds = scores[i, :valid_len].tolist()
                cand_indices = batch["candidate_indices"][i, :valid_len].cpu().tolist()
                news_ids = [news_store.idx2news[idx] for idx in cand_indices]
                ranking = sorted(zip(news_ids, preds), key=lambda x: x[1], reverse=True)
                results.append(
                    {
                        "impression_id": batch["impression_ids"][i],
                        "ranking": ranking,
                    }
                )

    # results are list of dicts; write to JSONL
    with open(args.output, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
