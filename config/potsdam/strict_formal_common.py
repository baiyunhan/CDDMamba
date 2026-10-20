"""Apply the locked Potsdam data protocol to an imported Vaihingen config."""

import os
from torch.utils.data import DataLoader

from geoseg.datasets.potsdam_strict import build_strict_potsdam_datasets


def apply_potsdam_protocol(namespace: dict, weights_name: str) -> None:
    namespace["weights_name"] = weights_name
    namespace["weights_path"] = f"model_weights/potsdam/{weights_name}"
    namespace["test_weights_name"] = weights_name
    namespace["log_name"] = f"potsdam/{weights_name}"
    namespace["resume_ckpt_path"] = None
    namespace["seed"] = 42

    train_dataset, val_dataset, test_dataset = build_strict_potsdam_datasets(
        data_root=os.environ.get("POTSDAM_ROOT", "data/potsdam"),
        train_transform=namespace["train_aug"],
        val_transform=namespace["val_aug"],
        mosaic_ratio=0.25,
    )
    namespace.update(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=DataLoader(
            train_dataset,
            batch_size=namespace["train_batch_size"],
            num_workers=4,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        ),
        val_loader=DataLoader(
            val_dataset,
            batch_size=namespace["val_batch_size"],
            num_workers=4,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        ),
    )
