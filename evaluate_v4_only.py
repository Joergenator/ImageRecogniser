"""Evaluate resnet50_scratch_gelu_v4 on the fresh data/test/ folder."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import timm
import torch.nn as nn

from src.dataset import ImageDataset, collect_image_paths
from src.transforms import get_eval_transforms
from src.evaluate import predict, compute_metrics, plot_roc_curve, plot_confusion_matrix


TAG = "resnet50_scratch_gelu_v4"
CKPT = Path("results/checkpoints") / TAG / "best.pt"
OUT = Path("results/fresh_test_eval") / TAG
TEST_DIR = Path("data/test")
BATCH_SIZE = 64
NUM_WORKERS = 2


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    file_paths, labels = collect_image_paths(TEST_DIR)
    test_ds = ImageDataset(file_paths, labels, transform=get_eval_transforms())
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=False)
    print(f"Fresh test set: {len(test_ds)} images "
          f"({labels.count(0)} real, {labels.count(1)} fake)")

    # v4 was trained with GELU swap but plain Linear fc (pre-Sequential-head commit)
    model = timm.create_model("resnet50", pretrained=False, num_classes=1, drop_rate=0.3)

    def _swap_relu(m):
        for name, child in m.named_children():
            if isinstance(child, nn.ReLU):
                setattr(m, name, nn.GELU())
            else:
                _swap_relu(child)

    _swap_relu(model)
    model.load_state_dict(torch.load(CKPT, weights_only=False, map_location="cpu"))
    model = model.to(device)

    probs, true_labels = predict(model, test_loader, device)
    metrics = compute_metrics(probs, true_labels)

    print(f"\n=== {TAG} on data/test/ ===")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    plot_roc_curve(probs, true_labels, OUT / "roc.png", title=f"ROC — {TAG}")
    plot_confusion_matrix(probs, true_labels, OUT / "confusion.png")

    with open(OUT / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
