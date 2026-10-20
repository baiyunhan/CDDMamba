"""Audit a release tree for common publication and GitHub blockers."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLOCKED_SUFFIXES = {".ckpt", ".pth", ".pt", ".pem", ".key"}
TEXT_SUFFIXES = {".cff", ".csv", ".json", ".log", ".md", ".py", ".template", ".txt", ".yaml", ".yml"}
MAX_GITHUB_FILE_BYTES = 100 * 1024 * 1024
PLACEHOLDERS = ("TODO", "[FINAL PAPER TITLE]", "[GITHUB REPOSITORY URL]")
ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"/root/autodl(?:-tmp)?/", re.IGNORECASE),
    re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE),
)
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
)


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []
    self_path = Path(__file__).resolve()

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        size = path.stat().st_size
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            failures.append(f"blocked binary: {relative}")
        if size >= MAX_GITHUB_FILE_BYTES:
            failures.append(f"file is at least 100 MiB: {relative} ({size / 2**20:.1f} MiB)")
        if path.suffix.lower() not in TEXT_SUFFIXES or path.resolve() == self_path:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            failures.append(f"cannot read {relative}: {exc}")
            continue
        for pattern in ABSOLUTE_PATH_PATTERNS:
            if pattern.search(text):
                failures.append(f"private absolute path in {relative}: {pattern.pattern}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                failures.append(f"possible secret in {relative}: {pattern.pattern}")
        for marker in PLACEHOLDERS:
            if marker in text:
                warnings.append(f"placeholder '{marker}' in {relative}")

    for directory_name in ("__pycache__", ".pytest_cache", "data", "model_weights"):
        for directory in ROOT.rglob(directory_name):
            if directory.is_dir():
                failures.append(f"generated/private directory present: {directory.relative_to(ROOT)}")

    print(f"Audited: {ROOT}")
    for item in sorted(set(warnings)):
        print(f"WARNING: {item}")
    for item in sorted(set(failures)):
        print(f"ERROR: {item}")
    print(f"Summary: {len(set(failures))} error(s), {len(set(warnings))} warning(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
