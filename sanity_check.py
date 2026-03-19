"""Sanity check: load a batch, verify shapes/labels, check for corrupt images."""

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from PIL import Image
from pathlib import Path

from src.dataset import create_dataloaders, collect_image_paths
from src.transforms import IMAGENET_MEAN, IMAGENET_STD


def check_corrupt_images(data_dir):
    """Scan for corrupt or unreadable images."""
    corrupt = []
    paths, labels = collect_image_paths(data_dir)
    print(f"Checking {len(paths)} images for corruption...")
    for p in tqdm(paths, desc="Scanning"):
        try:
            img = Image.open(p)
            img.verify()
        except Exception as e:
            corrupt.append((p, str(e)))
    if corrupt:
        print(f"\nFound {len(corrupt)} corrupt images:")
        for p, err in corrupt:
            print(f"  {p}: {err}")
    else:
        print("All images are valid.")
    return corrupt


def visualize_batch(train_loader):
    """Display a grid of sample images from one batch."""
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Label distribution in batch: real={int((labels == 0).sum())}, fake={int((labels == 1).sum())}")

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i >= len(images):
            break
        img = images[i] * std + mean
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        label_str = "Real" if labels[i] == 0 else "Fake"
        ax.imshow(img)
        ax.set_title(label_str)
        ax.axis("off")

    plt.tight_layout()
    out_path = "results/sample_batch.png"
    Path("results").mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=100)
    print(f"Sample batch saved to {out_path}")
    plt.close()


def main():
    data_dir = "data"

    # Check class balance
    paths, labels = collect_image_paths(data_dir)
    real_count = labels.count(0)
    fake_count = labels.count(1)
    print(f"Total images: {len(paths)} (real: {real_count}, fake: {fake_count})")

    # Check for corrupt images (optional, can be slow)
    if "--check-corrupt" in sys.argv:
        check_corrupt_images(data_dir)

    # Create dataloaders and visualize
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir, batch_size=32, num_workers=0
    )

    visualize_batch(train_loader)
    print("\nSanity check passed!")


if __name__ == "__main__":
    main()
