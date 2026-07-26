"""Training loop: per-epoch train/eval, checkpointing, and curve plotting."""

import csv
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from model.training.config import TrainingConfig
from model.training.metrics import confusion_counts, f1, iou, precision, recall

logger = logging.getLogger(__name__)

METRICS_FIELDNAMES = ["epoch", "train_loss", "val_loss", "val_precision", "val_recall", "val_f1", "val_iou"]


def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, loss_fn: nn.Module, device: torch.device
) -> float:
    model.train()
    total_loss = 0.0
    for patches, masks in loader:
        patches, masks = patches.to(device), masks.to(device).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(patches)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device, threshold: float) -> dict:
    model.eval()
    total_loss = 0.0
    tp_total = fp_total = fn_total = 0.0
    with torch.no_grad():
        for patches, masks in loader:
            patches, masks = patches.to(device), masks.to(device).unsqueeze(1)
            logits = model(patches)
            total_loss += loss_fn(logits, masks).item()
            tp, fp, fn, _ = confusion_counts(logits, masks, threshold)
            tp_total += tp.item()
            fp_total += fp.item()
            fn_total += fn.item()

    precision_value = precision(tp_total, fp_total)
    recall_value = recall(tp_total, fn_total)
    return {
        "loss": total_loss / len(loader),
        "precision": precision_value,
        "recall": recall_value,
        "f1": f1(precision_value, recall_value),
        "iou": iou(tp_total, fp_total, fn_total),
    }


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainingConfig,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> Path:
    """Runs cfg.epochs epochs, appending one metrics.csv row per epoch (crash-safe)
    and checkpointing model_best.pt whenever validation F1 improves. Returns the
    checkpoint path."""
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg.checkpoint_dir / "metrics.csv"
    best_model_path = cfg.checkpoint_dir / "model_best.pt"
    best_f1 = -1.0

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=METRICS_FIELDNAMES).writeheader()

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device, cfg.seg_threshold)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_iou": val_metrics["iou"],
        }
        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=METRICS_FIELDNAMES).writerow(row)

        logger.info(
            "Epoch %d/%d: train_loss=%.4f val_loss=%.4f val_f1=%.4f val_iou=%.4f",
            epoch, cfg.epochs, train_loss, val_metrics["loss"], val_metrics["f1"], val_metrics["iou"],
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_model_path)
            logger.info("New best val_f1=%.4f, saved %s", best_f1, best_model_path)

    return best_model_path


def plot_training_curves(metrics_csv_path: Path, output_png_path: Path) -> None:
    with metrics_csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    epochs = [int(row["epoch"]) for row in rows]

    fig, (ax_loss, ax_metrics) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(epochs, [float(row["train_loss"]) for row in rows], label="train_loss")
    ax_loss.plot(epochs, [float(row["val_loss"]) for row in rows], label="val_loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.legend()

    ax_metrics.plot(epochs, [float(row["val_f1"]) for row in rows], label="val_f1")
    ax_metrics.plot(epochs, [float(row["val_iou"]) for row in rows], label="val_iou")
    ax_metrics.set_xlabel("epoch")
    ax_metrics.legend()

    fig.tight_layout()
    fig.savefig(output_png_path)
    plt.close(fig)
