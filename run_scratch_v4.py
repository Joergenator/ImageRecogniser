"""Train ResNet-50 from scratch with GELU + mild augmentation, no label smoothing.

Keeps mild augmentation (colour jitter + random erasing) from v3 but drops
label smoothing, which capped discriminative sharpness in v2 and v3.
Patience increased to 10 to give the model more room to recover from dips.
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
        early_stopping_patience=10,
        augment_level="mild",
        label_smoothing=0.0,
    )
    train(cfg, strategy="scratch", tag="resnet50_scratch_gelu_v4")


if __name__ == "__main__":
    main()
