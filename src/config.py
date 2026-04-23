"""Hyperparameters and configuration."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # Data
    data_dir: str = "data"
    batch_size: int = 64
    num_workers: int = 2
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    max_samples: Optional[int] = None

    # Model
    model_name: str = "resnet50"  # resnet50, densenet121, vit_base_patch16_224
    pretrained: bool = True
    modified: bool = False  # apply architectural modification (GELU for ResNet-50)
    num_classes: int = 1  # binary with BCE

    # Training
    epochs: int = 30
    lr: float = 1e-3
    lr_finetune: float = 1e-5
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    scheduler: str = "cosine"
    early_stopping_patience: int = 5
    freeze_epochs: int = 5  # epochs to train with frozen backbone

    # Augmentation
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0
    dropout: float = 0.3
    augment_level: str = "standard"  # "standard" or "mild"
    label_smoothing: float = 0.0

    # Paths
    checkpoint_dir: str = "results/checkpoints"
    plot_dir: str = "results/plots"
