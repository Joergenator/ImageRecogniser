"""Train ResNet-50 from scratch with GELU activations (architectural modification).

This satisfies the project requirement of training at least one model from
randomly initialised weights, with at least one change to the reference
architecture (ReLU -> GELU throughout the residual blocks).
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
    )
    train(cfg, strategy="scratch", tag="resnet50_scratch_gelu")


if __name__ == "__main__":
    main()
