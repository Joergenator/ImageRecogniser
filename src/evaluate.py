"""Evaluation utilities: metrics, ROC curves, confusion matrices."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
)
from tqdm import tqdm


def predict(model, dataloader, device):
    """Run inference and collect predictions + labels."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


def compute_metrics(probs, labels, threshold=0.5):
    """Compute all classification metrics."""
    preds = (probs >= threshold).astype(int)
    return {
        "auc": roc_auc_score(labels, probs),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
    }


def plot_roc_curve(probs, labels, save_path, title="ROC Curve"):
    """Plot and save a ROC curve."""
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(probs, labels, save_path, threshold=0.5):
    """Plot and save a confusion matrix."""
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix")
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves_overlay(results_dict, save_path):
    """Plot multiple ROC curves overlaid.

    Args:
        results_dict: {model_name: (probs, labels)}
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    for name, (probs, labels) in results_dict.items():
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
