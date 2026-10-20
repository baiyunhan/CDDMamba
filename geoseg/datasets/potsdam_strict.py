"""Leakage-free Potsdam patch datasets for the locked 21/3/14 tile split."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Tuple

from .potsdam_dataset import PotsdamDataset


def build_strict_potsdam_datasets(
    data_root: str,
    train_transform: Callable,
    val_transform: Callable,
    mosaic_ratio: float = 0.25,
) -> Tuple[PotsdamDataset, PotsdamDataset, PotsdamDataset]:
    root = Path(data_root)
    common = dict(
        img_dir="images_1024",
        mask_dir="masks_1024",
        img_suffix=".tif",
        mask_suffix=".png",
        img_size=(1024, 1024),
    )
    train_dataset = PotsdamDataset(
        data_root=str(root / "train"), mode="train", mosaic_ratio=mosaic_ratio,
        transform=train_transform, **common,
    )
    val_dataset = PotsdamDataset(
        data_root=str(root / "val"), mode="val", transform=val_transform, **common,
    )
    test_dataset = PotsdamDataset(
        data_root=str(root / "test"), mode="test", transform=val_transform, **common,
    )

    train_ids, val_ids, test_ids = map(
        set, (train_dataset.img_ids, val_dataset.img_ids, test_dataset.img_ids)
    )
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Strict Potsdam split has patch overlap")
    actual = (len(train_dataset), len(val_dataset), len(test_dataset))
    expected = (756, 108, 504)
    if actual != expected:
        raise RuntimeError(f"Unexpected strict Potsdam counts: {actual}, expected {expected}")
    return train_dataset, val_dataset, test_dataset
