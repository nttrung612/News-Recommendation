import argparse
import math
import random
from pathlib import Path
from typing import Dict, Callable, Optional

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
import src.models.nrms  # register nrms
from src.models.base import RecModelBase
from src.models.registry import build_model
from src.utils.metrics import impression_auc


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """
    Ensure Python and NumPy RNGs are seeded per worker for reproducible sampling/shuffling.
    """
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
    max_grad_norm: float,
    log_interval: int,
    save_every_steps: int,
    start_step: int,
    save_fn,
    log_fn: Optional[Callable[[Dict[str, float], int], None]] = None,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    global_step = start_step
    num_batches = len(dataloader)
    accum_counter = 0
    for step, batch in enumerate(tqdm(dataloader, desc="Train", leave=False)):
        batch = move_batch_to_device(batch, device)
        targets = batch["labels"].argmax(dim=1)  # (batch,)

        with autocast(enabled=scaler.is_enabled()):
            scores = model(batch)  # (batch, neg_k + 1)
            loss = F.cross_entropy(scores, targets)
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()
        accum_counter += 1
        update_now = (accum_counter % grad_accum_steps == 0) or (step + 1 == num_batches)
        if update_now:
            if max_grad_norm and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            accum_counter = 0

        total_loss += loss.item() * grad_accum_steps
        running_loss += loss.item() * grad_accum_steps
        if (step + 1) % log_interval == 0:
            avg = running_loss / log_interval
            tqdm.write(f"Step {step+1}/{len(dataloader)} - loss: {avg:.4f}")
            if log_fn is not None:
                log_fn({"train/loss": avg}, global_step)
            running_loss = 0.0

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


def build_optimizer(model: RecModelBase, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """
    Standard AdamW with no weight decay on bias/LayerNorm parameters.
    """
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "layernorm.weight"]
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )


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
        category_vocab=train_news.category_vocab,
        subcategory_vocab=train_news.subcategory_vocab,
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
    generator = torch.Generator()
    generator.manual_seed(cfg.get("seed", 42))
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=train_collate,
        pin_memory=use_pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=dev_collate,
        pin_memory=use_pin_memory,
    )

    return train_loader, dev_loader, train_news, dev_news


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # optimize conv kernels for fixed input sizes
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model_cfg = cfg["model"].copy()
    tokenizer_name = model_cfg.get("tokenizer_name", model_cfg.get("pretrained_model_name"))
    tokenizer = load_tokenizer(tokenizer_name)

    train_loader, dev_loader, train_news, dev_news = get_dataloaders(cfg, tokenizer, device)

    # tie vocab sizes to what was seen in training/dev to keep embeddings aligned for eval/inference
    default_cat_cap = 0 if model_cfg.get("category_dim", 0) <= 0 else model_cfg.get("category_vocab_size", 500)
    default_subcat_cap = 0 if model_cfg.get("subcategory_dim", 0) <= 0 else model_cfg.get("subcategory_vocab_size", 500)
    train_cat_size = (len(train_news.category_vocab) + 1) if train_news.category_vocab else 0
    dev_cat_size = (len(dev_news.category_vocab) + 1) if dev_news.category_vocab else 0
    train_subcat_size = (len(train_news.subcategory_vocab) + 1) if train_news.subcategory_vocab else 0
    dev_subcat_size = (len(dev_news.subcategory_vocab) + 1) if dev_news.subcategory_vocab else 0
    model_cfg["category_vocab_size"] = max(default_cat_cap, train_cat_size, dev_cat_size)
    model_cfg["subcategory_vocab_size"] = max(default_subcat_cap, train_subcat_size, dev_subcat_size)
    cfg["model"] = model_cfg

    model = build_model(model_cfg).to(device)

    grad_accum_steps = max(cfg["train"]["grad_accum_steps"], 1)
    total_steps = math.ceil(len(train_loader) * cfg["train"]["epochs"] / grad_accum_steps)
    warmup_cfg = cfg["train"].get("warmup_steps", 0)
    if isinstance(warmup_cfg, float) and warmup_cfg < 1:
        warmup_steps = max(1, int(total_steps * warmup_cfg)) if warmup_cfg > 0 else 0
    else:
        warmup_steps = int(warmup_cfg)
    warmup_steps = min(warmup_steps, total_steps)

    optimizer = build_optimizer(model, lr=cfg["train"]["learning_rate"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = GradScaler(enabled=cfg["train"].get("fp16", False) and device.type == "cuda")

    wandb_run = None
    wandb_log_fn = None
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enable", False):
        try:
            import wandb
        except ImportError as e:
            raise ImportError("wandb is not installed. Please `pip install wandb` or disable wandb in config.") from e
        wandb_run = wandb.init(
            project=wandb_cfg.get("project", "news-rec"),
            name=wandb_cfg.get("name"),
            config=cfg,
        )
        wandb_log_fn = lambda metrics, step=None: wandb.log(metrics, step=step)

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
                    "category_vocab": train_news.category_vocab,
                    "subcategory_vocab": train_news.subcategory_vocab,
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
            grad_accum_steps,
            cfg["train"].get("max_grad_norm", 0.0),
            cfg["train"].get("log_interval", 100),
            cfg["train"].get("save_every_steps", 0),
            global_step,
            save_fn,
            wandb_log_fn,
        )
        dev_auc = evaluate(model, dev_loader, device)
        print(f"Epoch {epoch} | train_loss={train_loss:.4f} | dev_auc={dev_auc:.4f}")
        if wandb_log_fn is not None:
            wandb_log_fn(
                {
                    "train/epoch_loss": train_loss,
                    "eval/auc": dev_auc,
                    "epoch": epoch,
                    "lr": scheduler.get_last_lr()[0],
                },
                global_step,
            )
        if dev_auc > best_auc:
            best_auc = dev_auc
            save_path = ckpt_path / f"fastformer_epoch{epoch}_auc{dev_auc:.4f}.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "model_config": model_cfg,
                    "tokenizer_name": tokenizer_name,
                    "category_vocab": train_news.category_vocab,
                    "subcategory_vocab": train_news.subcategory_vocab,
                    "best_auc": best_auc,
                    "global_step": global_step,
                    "epoch": epoch,
                },
                save_path,
            )
            print(f"Saved new best checkpoint to {save_path}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML.")
    args = parser.parse_args()
    main(args.config)
