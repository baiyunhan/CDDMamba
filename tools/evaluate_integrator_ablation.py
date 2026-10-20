"""Locked validation-only comparison of Euler and RK4 CDFM solvers.

Both solvers reuse the same trained Potsdam checkpoint and validation split.
The Euler baseline follows the single linear update described in the method
section.  The test split is never accessed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import MethodType

import numpy as np
import torch

from tools.cfg import py2cfg
from train import Supervision_Train, segmentation_logits


CONFIG = Path("config/potsdam/proposed_method_v3_detail_potsdam_formal100.py")
CLASS_NAMES = ("ImSurf", "Building", "LowVeg", "Tree", "Car", "Clutter")


def update_confusion(matrix: np.ndarray, truth: np.ndarray, pred: np.ndarray) -> None:
    valid = (truth >= 0) & (truth < len(CLASS_NAMES))
    encoded = len(CLASS_NAMES) * truth[valid].astype(np.int64) + pred[valid].astype(np.int64)
    matrix += np.bincount(
        encoded, minlength=len(CLASS_NAMES) ** 2
    ).reshape(len(CLASS_NAMES), len(CLASS_NAMES))


def summarize(matrix: np.ndarray) -> dict[str, object]:
    true_positive = np.diag(matrix).astype(np.float64)
    union = matrix.sum(1) + matrix.sum(0) - true_positive
    f1_denominator = matrix.sum(1) + matrix.sum(0)
    iou = np.divide(
        true_positive,
        union,
        out=np.full_like(true_positive, np.nan),
        where=union > 0,
    )
    f1 = np.divide(
        2 * true_positive,
        f1_denominator,
        out=np.full_like(true_positive, np.nan),
        where=f1_denominator > 0,
    )
    return {
        "OA": float(true_positive.sum() / matrix.sum()),
        "mF1": float(np.nanmean(f1[:-1])),
        "mIoU": float(np.nanmean(iou[:-1])),
        "class_IoU": {
            name: float(value) for name, value in zip(CLASS_NAMES, iou)
        },
    }


def euler_single_step(self, source, condition, scale):
    """The ordinary one-step Euler baseline stated in the method section."""
    state = source
    steps = 1
    dt = 1.0 / steps
    batch = source.shape[0]
    for step in range(steps):
        t = source.new_full((batch,), step * dt)
        state = state + dt * self._velocity(state, condition, t, scale)
    return state


def evaluate(model: Supervision_Train, loader) -> tuple[np.ndarray, dict[str, object]]:
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    with torch.inference_mode():
        for batch in loader:
            image = batch["img"].cuda(non_blocking=True)
            truth = batch["gt_semantic_seg"].numpy()
            prediction = segmentation_logits(model(image)).argmax(1).cpu().numpy()
            for target, output in zip(truth, prediction):
                update_confusion(matrix, target, output)
    return matrix, summarize(matrix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("formal_val_results/potsdam_cdfm_integrator_locked"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    cfg = py2cfg(CONFIG)
    checkpoint = Path(cfg.weights_path) / f"{cfg.test_weights_name}.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    model = Supervision_Train.load_from_checkpoint(
        str(checkpoint), config=cfg
    ).cuda().eval()
    cdfm = model.net.cdfm
    if cdfm.rk_steps != 2:
        raise RuntimeError(f"Expected locked rk_steps=2, found {cdfm.rk_steps}")

    results: dict[str, object] = {}
    confusions: dict[str, np.ndarray] = {}

    confusions["Runge-Kutta (ours)"], results["Runge-Kutta (ours)"] = evaluate(
        model, cfg.val_loader
    )
    cdfm._rk4 = MethodType(euler_single_step, cdfm)
    confusions["Euler"], results["Euler"] = evaluate(model, cfg.val_loader)

    for name, matrix in confusions.items():
        safe_name = "rk4" if name.startswith("Runge") else "euler"
        np.savetxt(args.output / f"confusion_{safe_name}.csv", matrix, delimiter=",", fmt="%d")

    payload = {
        "dataset": "Potsdam",
        "split": "locked_validation_108_tiles",
        "checkpoint": str(checkpoint),
        "test_accessed": False,
        "comparison_control": "same checkpoint, split, and preprocessing; one-step Euler baseline from the method section",
        "results": results,
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    with (args.output / "paper_table.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("积分策略", "OA", "mF1", "mIoU"))
        writer.writeheader()
        for name in ("Euler", "Runge-Kutta (ours)"):
            row = results[name]
            writer.writerow(
                {
                    "积分策略": "欧拉积分" if name == "Euler" else "Runge-Kutta积分（我们的）",
                    "OA": f"{100 * row['OA']:.2f}",
                    "mF1": f"{100 * row['mF1']:.2f}",
                    "mIoU": f"{100 * row['mIoU']:.2f}",
                }
            )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
