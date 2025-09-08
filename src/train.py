# src/train.py

import os
import math
import random
from typing import Union
import time

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

import yaml

# ---- project imports (adjust if your paths/names differ) ----
from src.data.dataset import DeepfakeCsvDataset


# ---------------------------
# Utilities
# ---------------------------

def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def seed_everything(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # reasonable determinism defaults
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def normalize_device(dev: Union[str, torch.device]) -> torch.device:
    """
    Accepts: 'auto' | 'cuda' | 'cpu' | 'mps' | torch.device
    - 'auto' -> cuda if available, else mps (Apple) if available, else cpu
    - 'cuda' -> cuda if available, else cpu
    - 'mps'  -> Apple Metal (if available)
    """
    if isinstance(dev, torch.device):
        return dev

    s = str(dev).strip().lower()

    if s == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if s in {"cuda", "gpu"}:
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if s in {"cpu", "mps"}:
        return torch.device(s)

    # Fallback
    return torch.device("cpu")


# ---------------------------
# Data / Model
# ---------------------------

def build_dataloaders(cfg, device: Union[str, torch.device]):
    device = normalize_device(device)

    train_ds = DeepfakeCsvDataset(
        cfg["data"]["train_csv"],
        img_size=cfg["data"]["img_size"],
        train=True,
    )
    val_ds = DeepfakeCsvDataset(
        cfg["data"]["val_csv"],
        img_size=cfg["data"]["img_size"],
        train=False,
    )

    nw = cfg["train"].get("num_workers", 8)
    pin_mem = (device.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=nw,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=nw,
        pin_memory=pin_mem,
    )
    return train_loader, val_loader


class SimpleEffB0(nn.Module):
    def __init__(self, pretrained=True, in_chans=3, drop_rate=0.2, drop_path_rate=0.1):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=1,           # single logit
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

    @property
    def backbone_parameters(self):
        # for warmup-freeze logic if you want to address backbone only
        return self.backbone.parameters()

    def forward(self, x):
        return self.backbone(x)

def build_model(cfg, device):
    device = normalize_device(device)
    model = SimpleEffB0(
        pretrained=cfg["model"].get("pretrained", True),
        in_chans=cfg["model"].get("in_chans", 3),
        drop_rate=cfg["model"].get("drop_rate", 0.2),
        drop_path_rate=cfg["model"].get("drop_path_rate", 0.1),
    )
    model.to(device)
    # If somewhere else you iterate `model.backbone.parameters()`, keep:
    model.backbone = model.backbone                      # already set
    return model


# ---------------------------
# Train / Eval
# ---------------------------

def train_one_epoch(
    model,
    loader,
    loss_fn,
    optimizer,
    scaler,
    device,
    mixed_precision,
    freeze_backbone,
    epoch: int = 0,
    total_epochs: int = 0,
):
    model.train()

    # (Optional) freeze backbone logic you already have...
    for p in getattr(model, "backbone", model).parameters():
        p.requires_grad = not freeze_backbone

    losses = []

    # --- heartbeat setup ---
    beat_every = 10.0            # seconds
    last_beat = time.monotonic()
    total_batches = len(loader)
    # ------------------------

    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).unsqueeze(1)  # (B,) -> (B,1) if needed

        optimizer.zero_grad(set_to_none=True)

        # AMP context (works fine on CPU because 'enabled' will be False)
        with torch.cuda.amp.autocast(enabled=mixed_precision):
            logits = model(x)
            loss = loss_fn(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(float(loss.item()))

        # --- time-based heartbeat ---
        now = time.monotonic()
        if now - last_beat >= beat_every:
            avg_loss = sum(losses) / max(1, len(losses))
            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"epoch {epoch+1}/{total_epochs or '?'}  "
                f"batch {i+1}/{total_batches}  "
                f"avg_loss={avg_loss:.4f}",
                flush=True,
            )
            last_beat = now
        # ----------------------------

    return sum(losses) / max(1, len(losses))



@torch.no_grad()
def evaluate(model: torch.nn.Module,
             loader: DataLoader,
             device: Union[str, torch.device],
             mixed_precision: bool):
    device = normalize_device(device)
    model.eval()

    preds, gts = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(enabled=mixed_precision):
            logits = model(x)
            p = torch.sigmoid(logits).cpu().numpy().ravel().tolist()

        preds.extend(p)
        gts.extend(y.cpu().numpy().ravel().tolist())

    # Safe metrics (handle edge cases)
    try:
        auc = roc_auc_score(gts, preds)
    except Exception:
        auc = float("nan")

    thr = 0.5
    bin_preds = [1 if p > thr else 0 for p in preds]
    acc = accuracy_score(gts, bin_preds)
    f1 = f1_score(gts, bin_preds, zero_division=0)

    return {"AUC": auc, "ACC": acc, "F1": f1, "thr": thr}


# ---------------------------
# Main
# ---------------------------

def main():
    cfg = load_cfg("configs/effb0.yaml")
    seed_everything(cfg.get("seed", 1337))

    # resolve device from config; fallback to CPU if CUDA unavailable
    device_choice = cfg["train"].get("device", "cuda").lower()
    if device_choice == "cuda" and not torch.cuda.is_available():
        device_choice = "cpu"
    device = normalize_device(device_choice)
    print(f"[train] Using device: {device}")

    # data / model
    train_loader, val_loader = build_dataloaders(cfg, device)
    model = build_model(cfg, device)

    # optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    # AMP scaler (enabled only on CUDA if mixed_precision is True)
    use_amp = (device.type == "cuda") and bool(cfg["train"].get("mixed_precision", True))
    scaler = GradScaler(enabled=use_amp)

    # loss (pos_weight + optional label_smoothing)
    pos_w = cfg["model"].get("pos_weight", 1.0)
    if isinstance(pos_w, (int, float)):
        pos_w = torch.tensor([pos_w], device=device)
    label_smoothing = cfg["train"].get("label_smoothing", 0.0)

    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    loss_fn = lambda logits, targets: bce(logits, targets.float())

    # training setup
    epochs = cfg["train"]["epochs"]
    ckpt_dir = cfg["train"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    best_auc = -1.0
    best_path = None

    for epoch in range(epochs):
        freeze = epoch < cfg["model"].get("freeze_backbone_epochs", 0)

        train_loss = train_one_epoch(
            model,
            train_loader,
            bce,
            optimizer,
            scaler,
            device,
            mixed_precision=use_amp,
            freeze_backbone=freeze,
            epoch=epoch,
            total_epochs=epochs,
        )

        metrics = evaluate(
            model, val_loader, device, mixed_precision=use_amp
        )

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"loss: {train_loss:.4f} | "
            f"AUC: {metrics['AUC']:.4f} | ACC: {metrics['ACC']:.4f} | F1: {metrics['F1']:.4f}"
        )

        # Save best by AUC
        auc_value = metrics["AUC"]
        if math.isnan(auc_value):
            auc_value = -1.0

        if auc_value > best_auc:
            best_auc = auc_value
            best_path = os.path.join(ckpt_dir, f"best_epoch{epoch+1}_auc{best_auc:.4f}.pt")
            torch.save({"model": model.state_dict(), "cfg": cfg}, best_path)
            print(f"  ↳ New best AUC: {best_auc:.4f}. Saved: {best_path}")

    print("Training finished.")
    if best_path:
        print(f"Best checkpoint: {best_path} (AUC={best_auc:.4f})")


if __name__ == "__main__":
    # On Windows, multi-worker DataLoader needs the main-guard
    main()
