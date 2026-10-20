"""Print best-validation metrics and class IoUs for a strict screening run."""

import ast
import csv
import json
import sys
from pathlib import Path


name = sys.argv[1]
root = Path(__file__).resolve().parents[1]
dataset = "potsdam" if "_potsdam_" in name else "vaihingen"
metric_root = root / f"lightning_logs/{dataset}/{name}"
versions = sorted(metric_root.glob("version_*/metrics.csv"), key=lambda path: int(path.parent.name.split("_")[-1]))
if not versions:
    raise FileNotFoundError(f"No metrics.csv below {metric_root}")
metric_path = versions[-1]
log_path = root / f"run_logs/{name}.log"
with metric_path.open(newline="") as handle:
    rows = [row for row in csv.DictReader(handle) if row.get("val_mIoU")]
assert rows, f"No validation metrics in {metric_path}"
best = max(rows, key=lambda row: float(row["val_mIoU"]))

val_blocks = []
lines = log_path.read_text(errors="replace").splitlines()
for index, line in enumerate(lines[:-1]):
    if not line.startswith("val: ") or not lines[index + 1].startswith("{"):
        continue
    try:
        score = ast.literal_eval(line.removeprefix("val: "))
        classes = ast.literal_eval(lines[index + 1])
    except (ValueError, SyntaxError):
        continue
    val_blocks.append((score, classes))

target = float(best["val_mIoU"])
match = min(val_blocks, key=lambda pair: abs(pair[0]["mIoU"] - target))
assert abs(match[0]["mIoU"] - target) < 1e-4, "Log and CSV best validation disagree"
print(json.dumps({
    "name": name,
    "best_epoch_zero_based": int(best["epoch"]),
    "mIoU": target,
    "F1": float(best["val_F1"]),
    "OA": float(best["val_OA"]),
    "class_IoU": match[1],
    "validation_epochs": len(rows),
}, ensure_ascii=False, indent=2))
