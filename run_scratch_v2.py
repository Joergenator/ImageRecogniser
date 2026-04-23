"""Train ResNet-50 from scratch with GELU + aggressive augmentation and label smoothing.

Builds on v1 (resnet50_scratch_gelu) by adding stronger data augmentation
and label smoothing to combat the overfitting observed in the v1 training curves.
"""

from src.config import Config
from src.train import train


def main():
    cfg = Config(
        model_name="resnet50",
        pretrained=False,
        modified=True,
        epochs=80,
        lr=1e-3,
        weight_decay=1e-4,
        dropout=0.3,
        early_stopping_patience=8,
        augment_level="aggressive",
        label_smoothing=0.1,
    )
    train(cfg, strategy="scratch", tag="resnet50_scratch_gelu_v2")


if __name__ == "__main__":
    main()
