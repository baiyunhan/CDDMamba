"""Evaluate one locked validation or test split and export reproducible metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tools.cfg import py2cfg
from train import Supervision_Train, segmentation_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("-w", "--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def update_confusion(matrix: np.ndarray, truth: np.ndarray, pred: np.ndarray) -> None:
    class_count = matrix.shape[0]
    valid = (truth >= 0) & (truth < class_count)
    encoded = class_count * truth[valid].astype(np.int64) + pred[valid].astype(np.int64)
    matrix += np.bincount(encoded, minlength=class_count**2).reshape(class_count, class_count)


def summarize(matrix: np.ndarray, class_names: list[str]) -> dict[str, object]:
    true_positive = np.diag(matrix).astype(np.float64)
    union = matrix.sum(1) + matrix.sum(0) - true_positive
    f1_denominator = matrix.sum(1) + matrix.sum(0)
    iou = np.divide(true_positive, union, out=np.full_like(true_positive, np.nan), where=union > 0)
    f1 = np.divide(
        2 * true_positive,
        f1_denominator,
        out=np.full_like(true_positive, np.nan),
        where=f1_denominator > 0,
    )
    # The ISPRS "Clutter/Background" class is excluded from mIoU and mF1.
    scored = slice(None, -1)
    return {
        "OA": float(true_positive.sum() / matrix.sum()),
        "mF1": float(np.nanmean(f1[scored])),
        "mIoU": float(np.nanmean(iou[scored])),
        "class_IoU": {name: float(value) for name, value in zip(class_names, iou)},
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    cfg = py2cfg(args.config)
    dataset = cfg.val_dataset if args.split == "val" else cfg.test_dataset
    batch_size = args.batch_size or int(getattr(cfg, "val_batch_size", 1))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
        drop_last=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Supervision_Train.load_from_checkpoint(str(args.checkpoint), config=cfg)
    model.to(device).eval()

    class_names = list(cfg.classes)
    matrix = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    with torch.inference_mode():
        for batch in loader:
            image = batch["img"].to(device, non_blocking=True)
            truth = batch["gt_semantic_seg"].cpu().numpy()
            pred = segmentation_logits(model(image)).argmax(1).cpu().numpy()
            for target, output in zip(truth, pred):
                update_confusion(matrix, target, output)

    args.output.mkdir(parents=True)
    metrics = summarize(matrix, class_names)
    manifest = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "sample_count": len(dataset),
        "metrics": metrics,
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    np.savetxt(args.output / "confusion.csv", matrix, fmt="%d", delimiter=",")
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("OA", "mF1", "mIoU"))
        writer.writerow((metrics["OA"], metrics["mF1"], metrics["mIoU"]))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
