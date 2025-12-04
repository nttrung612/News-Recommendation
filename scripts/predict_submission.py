import argparse
import zipfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.config import load_config
from src.data.mind import MindEvalCollator, MindEvalDataset, load_news_tensors, load_tokenizer
import src.models.fastformer  # register fastformer
import src.models.nrms  # register nrms
import src.models.unbert  # register unbert
from src.models.registry import build_model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Generate MIND submission predictions and zip them.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint (.pt) path.")
    parser.add_argument("--behaviors", type=str, required=True, help="behaviors.tsv for test set.")
    parser.add_argument("--news", type=str, required=True, help="news.tsv for test set.")
    parser.add_argument("--output_txt", type=str, default="prediction.txt", help="Output prediction txt.")
    parser.add_argument("--output_zip", type=str, default="prediction.zip", help="Output zip file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_cfg = checkpoint.get("model_config", cfg["model"])
    data_cfg = checkpoint.get("config", cfg)["data"]
    tokenizer_name = checkpoint.get("tokenizer_name", model_cfg.get("tokenizer_name", model_cfg.get("pretrained_model_name")))
    category_vocab = checkpoint.get("category_vocab")
    subcategory_vocab = checkpoint.get("subcategory_vocab")
    if "category_vocab_size" not in model_cfg and category_vocab:
        model_cfg["category_vocab_size"] = len(category_vocab) + 1
    if "subcategory_vocab_size" not in model_cfg and subcategory_vocab:
        model_cfg["subcategory_vocab_size"] = len(subcategory_vocab) + 1
    tokenizer = load_tokenizer(tokenizer_name)

    news_store = load_news_tensors(
        args.news,
        tokenizer,
        data_cfg["max_seq_len"],
        data_cfg["cache_dir"],
        category_vocab=category_vocab,
        subcategory_vocab=subcategory_vocab,
    )
    # move cached tensors to device
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
    num_workers = 0 if device.type == "cuda" else cfg["train"]["num_workers"]
    pin_memory = False if news_store.input_ids.is_cuda else (device.type == "cuda")
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=pin_memory,
    )

    model = build_model(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    lines = []
    for batch in tqdm(dataloader, desc="Predict"):
        tensor_batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        output = model(tensor_batch)
        scores = output["scores"] if isinstance(output, dict) else output  # (batch, max_cand)
        scores = scores.masked_fill(~tensor_batch["candidate_mask"], float("-inf"))
        for i in range(scores.size(0)):
            valid_len = int(tensor_batch["candidate_mask"][i].sum().item())
            cand_scores = scores[i, :valid_len]
            # compute rank per candidate in the original order (1 = highest score)
            order = torch.argsort(cand_scores, descending=True)
            ranks = torch.zeros_like(order)  # (valid_len,)
            ranks[order] = torch.arange(1, valid_len + 1, device=order.device)
            ranking_str = "[" + ",".join(str(int(r)) for r in ranks.tolist()) + "]"
            impression_id = batch["impression_ids"][i]
            lines.append(f"{impression_id} {ranking_str}")

    output_txt = Path(args.output_txt)
    with output_txt.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"Wrote predictions to {output_txt}")

    output_zip = Path(args.output_zip)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_txt, arcname=output_txt.name)
    print(f"Zipped predictions to {output_zip}")


if __name__ == "__main__":
    main()
