"""Evaluate all trained models on the fresh data/test/ folder and compare."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.models import create_model
from src.dataset import ImageDataset, collect_image_paths
from src.transforms import get_eval_transforms
from src.evaluate import predict, compute_metrics, plot_roc_curve, plot_confusion_matrix, plot_roc_curves_overlay


MODELS = [
    {"name": "resnet50", "tag": "resnet50_transfer"},
    {"name": "densenet121", "tag": "densenet121_transfer"},
    {"name": "vit_base_patch16_224", "tag": "vit_base_patch16_224_transfer"},
]

CHECKPOINT_DIR = Path("results/checkpoints")
OUTPUT_DIR = Path("results/fresh_test_eval")
TEST_DIR = Path("data/test")
BATCH_SIZE = 64
NUM_WORKERS = 2


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Build test dataloader from data/test/
    file_paths, labels = collect_image_paths(TEST_DIR)
    test_ds = ImageDataset(file_paths, labels, transform=get_eval_transforms())
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=False)
    print(f"Fresh test set: {len(test_ds)} images "
          f"({labels.count(0)} real, {labels.count(1)} fake)\n")

    all_results = {}
    roc_data = {}

    for entry in MODELS:
        model_name = entry["name"]
        tag = entry["tag"]
        ckpt_path = CHECKPOINT_DIR / tag / "best.pt"

        if not ckpt_path.exists():
            print(f"SKIP {tag}: no checkpoint at {ckpt_path}")
            continue

        print(f"{'='*50}")
        print(f"Evaluating: {tag}")
        print(f"{'='*50}")

        model = create_model(model_name, pretrained=False, dropout=0.3)
        model.load_state_dict(torch.load(ckpt_path, weights_only=False, map_location="cpu"))
        model = model.to(device)

        probs, true_labels = predict(model, test_loader, device)
        metrics = compute_metrics(probs, true_labels)

        for k, v in metrics.items():
            print(f"  {k:12s}: {v:.4f}")
        print()

        # Save per-model plots
        model_dir = OUTPUT_DIR / tag
        model_dir.mkdir(parents=True, exist_ok=True)
        plot_roc_curve(probs, true_labels, model_dir / "roc.png", title=f"ROC — {tag}")
        plot_confusion_matrix(probs, true_labels, model_dir / "confusion.png")

        all_results[tag] = metrics
        roc_data[tag] = (probs, true_labels)

    # Overlay ROC curves
    if roc_data:
        plot_roc_curves_overlay(roc_data, OUTPUT_DIR / "roc_all_models.png")
        print(f"Overlay ROC saved to {OUTPUT_DIR / 'roc_all_models.png'}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Model':<40} {'AUC':>7} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print(f"{'-'*70}")
    for tag, m in sorted(all_results.items(), key=lambda x: -x[1]["auc"]):
        print(f"{tag:<40} {m['auc']:>7.4f} {m['accuracy']:>7.4f} "
              f"{m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f}")

    best_tag = max(all_results, key=lambda t: all_results[t]["auc"])
    print(f"\nBest model: {best_tag} (AUC={all_results[best_tag]['auc']:.4f})")

    # Save results JSON
    with open(OUTPUT_DIR / "comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {OUTPUT_DIR / 'comparison.json'}")


if __name__ == "__main__":
    main()
