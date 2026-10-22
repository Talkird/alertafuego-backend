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

from model.training.config import THRESHOLD_CANDIDATES, TrainingConfig
from model.training.metrics import confusion_counts, f1, f_beta, fire_counts, iou, precision, recall, tolerant_counts

logger = logging.getLogger(__name__)

METRICS_FIELDNAMES = [
    "epoch",
    "lr",
    "train_loss",
    "val_loss",
    "val_threshold",
    "val_fire_recall",
    "val_f2",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_iou",
    "val_f1_tol",
]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
) -> float:
    """One pass over loader. On a GPU, in mixed precision (float16 where safe):
    about twice as fast, with the scaler keeping small gradients from rounding to 0."""
    model.train()
    total_loss = 0.0
    for patches, masks in loader:
        patches, masks = patches.to(device), masks.to(device).unsqueeze(1)
        optimizer.zero_grad()
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(patches)
            loss = loss_fn(logits.float(), masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate_thresholds(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device, thresholds: list[float]
) -> list[dict]:
    """Scores at every threshold in one pass over loader: loss, strict and tolerant
    (1 pixel, see metrics.tolerant_counts) precision/recall/F1, fire recall and false
    alarm rate (metrics.fire_counts), and F2 of tolerant precision and fire recall."""
    model.eval()
    total_loss = 0.0
    counts = torch.zeros(len(thresholds), 11, dtype=torch.float64)  # tp fp fn | tol: 4 | fire: 4
    with torch.no_grad():
        for patches, masks in loader:
            patches, masks = patches.to(device), masks.to(device).unsqueeze(1)
            logits = model(patches).float()
            total_loss += loss_fn(logits, masks).item()
            for i, threshold in enumerate(thresholds):
                tp, fp, fn, _ = confusion_counts(logits, masks, threshold)
                counts[i] += (
                    torch.stack(
                        [tp, fp, fn, *tolerant_counts(logits, masks, threshold), *fire_counts(logits, masks, threshold)]
                    )
                    .cpu()
                    .double()
                )

    results = []
    for threshold, row in zip(thresholds, counts.tolist()):
        tp, fp, fn, pred_near, pred_total, true_near, true_total, fires_hit, fires, alarms, no_fire = row
        precision_value, recall_value = precision(tp, fp), recall(tp, fn)
        precision_tol = pred_near / pred_total if pred_total else 0.0
        recall_tol = true_near / true_total if true_total else 0.0
        fire_recall = fires_hit / fires if fires else 0.0
        results.append(
            {
                "threshold": threshold,
                "loss": total_loss / len(loader),
                "precision": precision_value,
                "recall": recall_value,
                "f1": f1(precision_value, recall_value),
                "iou": iou(tp, fp, fn),
                "precision_tol": precision_tol,
                "recall_tol": recall_tol,
                "f1_tol": f1(precision_tol, recall_tol),
                "fire_recall": fire_recall,
                "false_alarm_rate": alarms / no_fire if no_fire else 0.0,
                "f2": f_beta(precision_tol, fire_recall, 2),
            }
        )
    return results


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device, threshold: float) -> dict:
    return evaluate_thresholds(model, loader, loss_fn, device, [threshold])[0]


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainingConfig,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> Path:
    """Runs cfg.epochs epochs with a cosine learning-rate schedule, appending one
    metrics.csv row per epoch (crash-safe) and checkpointing model_best.pt whenever
    the best validation F2 over THRESHOLD_CANDIDATES improves. F2 weighs fire recall
    twice as much as precision. Returns the checkpoint path."""
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cfg.checkpoint_dir / "metrics.csv"
    best_model_path = cfg.checkpoint_dir / "model_best.pt"
    best_f2 = -1.0
    # Decays the learning rate smoothly to ~0 by the last epoch: big steps early,
    # fine adjustments at the end.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=METRICS_FIELDNAMES).writeheader()

    for epoch in range(1, cfg.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        scheduler.step()
        val_sweep = evaluate_thresholds(model, val_loader, loss_fn, device, THRESHOLD_CANDIDATES)
        val_metrics = max(val_sweep, key=lambda m: m["f2"])

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_threshold": val_metrics["threshold"],
            "val_fire_recall": val_metrics["fire_recall"],
            "val_f2": val_metrics["f2"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_iou": val_metrics["iou"],
            "val_f1_tol": val_metrics["f1_tol"],
        }
        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=METRICS_FIELDNAMES).writerow(row)

        logger.info(
            "Epoch %d/%d: train_loss=%.4f val_loss=%.4f val_f2=%.4f (threshold %.3f, fire_recall=%.4f) val_f1_tol=%.4f",
            epoch, cfg.epochs, train_loss, val_metrics["loss"], val_metrics["f2"], val_metrics["threshold"],
            val_metrics["fire_recall"], val_metrics["f1_tol"],
        )

        if val_metrics["f2"] > best_f2:
            best_f2 = val_metrics["f2"]
            torch.save(model.state_dict(), best_model_path)
            logger.info("New best val_f2=%.4f, saved %s", best_f2, best_model_path)

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

    ax_metrics.plot(epochs, [float(row["val_fire_recall"]) for row in rows], label="val_fire_recall")
    ax_metrics.plot(epochs, [float(row["val_f2"]) for row in rows], label="val_f2")
    ax_metrics.plot(epochs, [float(row["val_f1"]) for row in rows], label="val_f1")
    ax_metrics.plot(epochs, [float(row["val_iou"]) for row in rows], label="val_iou")
    ax_metrics.plot(epochs, [float(row["val_f1_tol"]) for row in rows], label="val_f1_tol")
    ax_metrics.set_xlabel("epoch")
    ax_metrics.legend()

    fig.tight_layout()
    fig.savefig(output_png_path)
    plt.close(fig)
