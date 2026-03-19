"""Batch 3: Transfer learning with fine-tuning for all three architectures."""

from multiprocessing import freeze_support
from src.config import Config
from src.train import train


def main():
    models = ["resnet50", "densenet121", "vit_base_patch16_224"]
    all_results = []

    for model_name in models:
        cfg = Config(model_name=model_name)
        result = train(cfg, strategy="transfer")
        all_results.append(result)

    print("\n" + "=" * 60)
    print("BATCH 3 SUMMARY — Transfer Learning")
    print("=" * 60)
    for r in all_results:
        print(f"{r['model']:30s} | Val AUC: {r['best_val_auc']:.4f} | "
              f"Test AUC: {r['test_metrics']['auc']:.4f} | "
              f"Time: {r['training_time_min']:.1f} min")


if __name__ == "__main__":
    freeze_support()
    main()
