"""Dataset and DataLoader utilities for real vs AI-generated image classification."""

import os
from pathlib import Path

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from src.transforms import get_train_transforms, get_eval_transforms


class ImageDataset(Dataset):
    """Binary classification dataset: real (0) vs fake/AI-generated (1)."""

    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def collect_image_paths(data_dir):
    """Collect all image paths and labels from data/real/ and data/fake/.

    Returns:
        file_paths: list of absolute path strings
        labels: list of ints (0=real, 1=fake)
    """
    data_dir = Path(data_dir)
    file_paths = []
    labels = []

    for label, folder in [(0, "real"), (1, "fake")]:
        folder_path = data_dir / folder
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                file_paths.append(str(folder_path / fname))
                labels.append(label)

    return file_paths, labels


def create_dataloaders(data_dir, batch_size=32, num_workers=4,
                       val_ratio=0.1, test_ratio=0.1, seed=42,
                       max_samples=None):
    """Create train, validation, and test DataLoaders.

    Splits: 80% train, 10% val, 10% test (stratified).
    max_samples: if set, subsample the dataset before splitting (useful for quick tests).
    """
    file_paths, labels = collect_image_paths(data_dir)

    if max_samples and max_samples < len(file_paths):
        file_paths, _, labels, _ = train_test_split(
            file_paths, labels,
            train_size=max_samples,
            stratify=labels,
            random_state=seed,
        )

    # First split: separate test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        file_paths, labels,
        test_size=test_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Second split: separate validation from training
    val_fraction = val_ratio / (1 - test_ratio)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=val_fraction,
        stratify=train_val_labels,
        random_state=seed,
    )

    train_ds = ImageDataset(train_paths, train_labels, transform=get_train_transforms())
    val_ds = ImageDataset(val_paths, val_labels, transform=get_eval_transforms())
    test_ds = ImageDataset(test_paths, test_labels, transform=get_eval_transforms())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"Dataset splits — Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
