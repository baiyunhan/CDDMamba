"""Leakage-free Vaihingen train/validation/test construction.

The repository's 755-image training directory contains the 150 validation
patches from areas 30, 32, and 34.  This helper removes those IDs from the
training dataset while keeping the existing files untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Tuple

from .vaihingen_dataset import VaihingenDataset


def build_strict_vaihingen_datasets(
    data_root: str,
    train_transform: Callable,
    val_transform: Callable,
    mosaic_ratio: float = 0.25,
) -> Tuple[VaihingenDataset, VaihingenDataset, VaihingenDataset]:
    root = Path(data_root)
    common = dict(
        img_dir="images_1024",
        mask_dir="masks_1024",
        img_suffix=".tif",
        mask_suffix=".png",
        img_size=(1024, 1024),
    )
    train_dataset = VaihingenDataset(
        data_root=str(root / "train"),
        mode="train",
        mosaic_ratio=mosaic_ratio,
        transform=train_transform,
        **common,
    )
    val_dataset = VaihingenDataset(
        data_root=str(root / "val"),
        mode="val",
        transform=val_transform,
        **common,
    )
    test_dataset = VaihingenDataset(
        data_root=str(root / "test"),
        mode="test",
        transform=val_transform,
        **common,
    )

    validation_ids = set(val_dataset.img_ids)
    train_dataset.img_ids = [
        image_id for image_id in train_dataset.img_ids if image_id not in validation_ids
    ]

    train_ids = set(train_dataset.img_ids)
    test_ids = set(test_dataset.img_ids)
    if train_ids & validation_ids:
        raise RuntimeError("Strict Vaihingen split still has train/validation overlap")
    if train_ids & test_ids or validation_ids & test_ids:
        raise RuntimeError("Strict Vaihingen split has test leakage")
    actual_counts = (len(train_dataset), len(val_dataset), len(test_dataset))
    expected_counts = (605, 150, 113)
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"Unexpected strict split counts: {actual_counts}, expected {expected_counts}"
        )
    return train_dataset, val_dataset, test_dataset

