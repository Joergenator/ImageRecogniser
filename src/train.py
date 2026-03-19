"""Training loop with AMP, early stopping, checkpointing, and logging."""

import os
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import torch_directml
from src.config import Config
from src.dataset import create_dataloaders
from src.models import create_model, freeze_backbone, unfreeze_backbone
from src.evaluate import predict, compute_metrics, plot_roc_curve, plot_confusion_matrix


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_score = -float("inf")
        self.counter = 0

    def step(self, score):
        if score > self.best_score:
            self.best_score = score
            self.counter = 0
            return False  # not stopping
        self.counter += 1
        return self.counter >= self.patience


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_probs = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits.float()).detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, auc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.float().to(device)

            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits.float()).detach().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, auc


def plot_training_curves(history, save_path):
    """Plot loss and AUC curves for train and validation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(history["train_auc"], label="Train")
    ax2.plot(history["val_auc"], label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC")
    ax2.set_title("AUC")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def train(cfg: Config, strategy="transfer", tag=None):
    """Full training pipeline.

    Args:
        cfg: Config dataclass.
        strategy: 'transfer' (freeze then finetune), 'scratch' (random init, full train).
        tag: Optional label for saving results (e.g., 'resnet50_transfer').
    """
    tag = tag or f"{cfg.model_name}_{strategy}"
    device = torch_directml.device()
    print(f"\n{'='*60}")
    print(f"Training: {tag} | Device: {device}")
    print(f"{'='*60}")

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(
        cfg.data_dir, cfg.batch_size, cfg.num_workers,
        cfg.val_ratio, cfg.test_ratio, cfg.seed,
        max_samples=cfg.max_samples,
    )

    # Model
    pretrained = strategy == "transfer"
    model = create_model(cfg.model_name, pretrained=pretrained, dropout=cfg.dropout)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    # Directories
    ckpt_dir = Path(cfg.checkpoint_dir) / tag
    plot_dir = Path(cfg.plot_dir) / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []}
    best_val_auc = -1
    total_epochs = 0
    start_time = time.time()

    def run_phase(epochs, lr, phase_name):
        nonlocal best_val_auc, total_epochs
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        early_stop = EarlyStopping(patience=cfg.early_stopping_patience)

        print(f"\n--- {phase_name} ({epochs} epochs, lr={lr}) ---")
        for epoch in range(1, epochs + 1):
            total_epochs += 1
            train_loss, train_auc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_auc = validate(model, val_loader, criterion, device)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_auc"].append(train_auc)
            history["val_auc"].append(val_auc)

            print(f"Epoch {total_epochs:3d} | "
                  f"Train Loss: {train_loss:.4f}  AUC: {train_auc:.4f} | "
                  f"Val Loss: {val_loss:.4f}  AUC: {val_auc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), ckpt_dir / "best.pt")

            if early_stop.step(val_auc):
                print(f"Early stopping at epoch {total_epochs}")
                break

    # Training phases
    if strategy == "transfer":
        freeze_backbone(model)
        run_phase(cfg.freeze_epochs, cfg.lr, "Phase 1: Frozen backbone")
        unfreeze_backbone(model)
        run_phase(cfg.epochs, cfg.lr_finetune, "Phase 2: Full fine-tune")
    else:
        run_phase(cfg.epochs, cfg.lr_finetune, "Training from scratch")

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.1f} min | Best val AUC: {best_val_auc:.4f}")

    # Save training curves
    plot_training_curves(history, plot_dir / "training_curves.png")

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", weights_only=False))
    probs, labels = predict(model, test_loader, device)
    metrics = compute_metrics(probs, labels)

    print(f"\nTest results for {tag}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save test plots and metrics
    plot_roc_curve(probs, labels, plot_dir / "roc_test.png", title=f"ROC — {tag}")
    plot_confusion_matrix(probs, labels, plot_dir / "confusion_test.png")

    results = {
        "tag": tag,
        "model": cfg.model_name,
        "strategy": strategy,
        "best_val_auc": best_val_auc,
        "test_metrics": metrics,
        "total_epochs": total_epochs,
        "training_time_min": round(elapsed / 60, 1),
    }
    with open(ckpt_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    cfg = Config()
    train(cfg, strategy="transfer")
