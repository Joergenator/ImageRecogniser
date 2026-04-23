"""Train ResNet-50 from scratch with GELU + mild augmentation and label smoothing.

Dials back from v2's aggressive augmentation to a gentler approach:
only colour jitter and random erasing on top of the standard pipeline,
keeping label smoothing at 0.1.
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
        augment_level="mild",
        label_smoothing=0.1,
    )
    train(cfg, strategy="scratch", tag="resnet50_scratch_gelu_v3")


if __name__ == "__main__":
    main()
