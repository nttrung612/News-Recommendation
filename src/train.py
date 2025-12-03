import argparse
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler

from src.config.config import load_config
from src.data.mind import (
    MindEvalCollator,
    MindEvalDataset,
    MindTrainCollator,
    MindTrainDataset,
    load_news_tensors,
    load_tokenizer,
)
import src.models.fastformer  # register fastformer
from src.models.base import RecModelBase
from src.models.registry import build_model
from src.utils.metrics import impression_auc


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_epoch(
    model: RecModelBase,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    scaler: GradScaler,
    grad_accum_steps: int,
    log_interval: int,
    save_every_steps: int,
    start_step: int,
    save_fn,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    running_loss = 0.0
    optimizer.zero_grad()
    global_step = start_step
    for step, batch in enumerate(tqdm(dataloader, desc="Train", leave=False)):
        batch = move_batch_to_device(batch, device)
        targets = batch["labels"].argmax(dim=1)  # (batch,)

        with autocast(enabled=scaler.is_enabled()):
            scores = model(batch)  # (batch, neg_k + 1)
            loss = F.cross_entropy(scores, targets)
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()
        if (step + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * grad_accum_steps
        running_loss += loss.item() * grad_accum_steps
        if (step + 1) % log_interval == 0:
            avg = running_loss / log_interval
            tqdm.write(f"Step {step+1}/{len(dataloader)} - loss: {avg:.4f}")
            running_loss = 0.0

        global_step += 1
        if save_every_steps and global_step % save_every_steps == 0:
            save_fn(global_step)
    return total_loss / len(dataloader), global_step


@torch.no_grad()
def evaluate(model: RecModelBase, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(dataloader, desc="Eval", leave=False):
        batch = move_batch_to_device(batch, device)
        scores = model(batch)  # (batch, max_cand)
        # mask out padded candidates
        scores = scores.masked_fill(~batch["candidate_mask"], float("-inf"))
        for i in range(scores.size(0)):
            valid_len = int(batch["candidate_mask"][i].sum().item())
            all_preds.append(scores[i, :valid_len].tolist())
            all_labels.append(batch["labels"][i, :valid_len].tolist())
    return impression_auc(all_preds, all_labels)


def subsample_dataset(dataset, ratio: float):
    if ratio >= 1.0:
        return dataset
    subset_size = max(1, int(len(dataset) * ratio))
    indices = random.sample(range(len(dataset)), subset_size)
    if hasattr(dataset, "samples"):
        dataset.samples = [dataset.samples[i] for i in indices]
    return dataset


def get_dataloaders(cfg: Dict, tokenizer, device: torch.device):
    train_news = load_news_tensors(
        cfg["data"]["train_news"],
        tokenizer,
        cfg["data"]["max_seq_len"],
        cfg["data"]["cache_dir"],
    )
    dev_news = load_news_tensors(
        cfg["data"]["dev_news"],
        tokenizer,
        cfg["data"]["max_seq_len"],
        cfg["data"]["cache_dir"],
    )

    train_ds = MindTrainDataset(
        cfg["data"]["train_behaviors"],
        train_news.news2idx,
        cfg["data"]["max_history"],
        cfg["train"]["neg_k"],
    )
    dev_ds = MindEvalDataset(
        cfg["data"]["dev_behaviors"],
        dev_news.news2idx,
        cfg["data"]["max_history"],
    )

    train_ds = subsample_dataset(train_ds, cfg["train"].get("sample_ratio", 1.0))
    dev_ds = subsample_dataset(dev_ds, cfg["eval"].get("sample_ratio", 1.0))

    train_collate = MindTrainCollator(train_news, cfg["data"]["max_history"], cfg["train"]["neg_k"])
    dev_collate = MindEvalCollator(dev_news, cfg["data"]["max_history"])

    use_pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=train_collate,
        pin_memory=use_pin_memory,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=dev_collate,
        pin_memory=use_pin_memory,
    )

    return train_loader, dev_loader


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # optimize conv kernels for fixed input sizes
    model_cfg = cfg["model"]
    tokenizer_name = model_cfg.get("tokenizer_name", model_cfg.get("pretrained_model_name"))
    tokenizer = load_tokenizer(tokenizer_name)

    train_loader, dev_loader = get_dataloaders(cfg, tokenizer, device)

    model = build_model(model_cfg).to(device)

    total_steps = (len(train_loader) * cfg["train"]["epochs"]) // max(cfg["train"]["grad_accum_steps"], 1)

    optimizer = AdamW(model.parameters(), lr=cfg["train"]["learning_rate"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg["train"]["warmup_steps"],
        num_training_steps=total_steps,
    )
    scaler = GradScaler(enabled=cfg["train"].get("fp16", False) and device.type == "cuda")

    best_auc = 0.0
    global_step = 0
    ckpt_path = Path("checkpoints")
    ckpt_path.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        def save_fn(step: int, current_epoch=epoch):
            save_path = ckpt_path / f"epoch{current_epoch}_step{step}.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "model_config": model_cfg,
                    "tokenizer_name": tokenizer_name,
                    "best_auc": best_auc,
                    "global_step": step,
                    "epoch": current_epoch,
                },
                save_path,
            )
            tqdm.write(f"Saved periodic checkpoint to {save_path}")

        train_loss, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            scaler,
            cfg["train"]["grad_accum_steps"],
            cfg["train"].get("log_interval", 100),
            cfg["train"].get("save_every_steps", 0),
            global_step,
            save_fn,
        )
        dev_auc = evaluate(model, dev_loader, device)
        print(f"Epoch {epoch} | train_loss={train_loss:.4f} | dev_auc={dev_auc:.4f}")
        if dev_auc > best_auc:
            best_auc = dev_auc
            save_path = ckpt_path / f"fastformer_epoch{epoch}_auc{dev_auc:.4f}.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "model_config": model_cfg,
                    "tokenizer_name": tokenizer_name,
                    "best_auc": best_auc,
                    "global_step": global_step,
                    "epoch": epoch,
                },
                save_path,
            )
            print(f"Saved new best checkpoint to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML.")
    args = parser.parse_args()
    main(args.config)
