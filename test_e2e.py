"""Quick end-to-end test on a tiny subset to verify training pipeline."""

from src.config import Config
from src.train import train

cfg = Config(
    data_dir="data",
    batch_size=16,
    num_workers=0,
    epochs=2,
    freeze_epochs=1,
    early_stopping_patience=99,
    model_name="resnet50",
    checkpoint_dir="results/checkpoints_test",
    plot_dir="results/plots_test",
    max_samples=500,
)

train(cfg, strategy="transfer", tag="resnet50_e2e_test")
