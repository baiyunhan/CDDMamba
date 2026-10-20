# Repository publication contents

## Commit to the repository

- Final CDFM, DP-MPC, DGR-CSMamba, detail-decoder, and ablation source code.
- Locked Vaihingen and Potsdam configurations used in the manuscript.
- Strict split builders and patch-preparation scripts.
- Training, evaluation, metric, validation-summary, and release-audit scripts.
- Paper-facing CSV tables and their raw summaries, confusion matrices, and
  per-tile metrics.
- The three assembled qualitative figures used in the manuscript.
- Environment, dataset protocol, citation template, and third-party notices.

## Attach to GitHub Release v1.0.0

- `vaihingen_best.ckpt` (238,175,558 bytes)
- `potsdam_best.ckpt` (238,175,622 bytes)
- `SHA256SUMS.txt`

These files exceed GitHub's normal per-file Git limit and must not be committed
to repository history.

## Do not publish

- ISPRS source images, labels, or generated data patches.
- Redundant `last.ckpt` files, failed/interrupted checkpoints, caches, and logs
  unrelated to the reported experiments.
- Bulk prediction PNGs that can be regenerated from the released weights.
- SSH keys, tokens, AutoDL account information, or absolute local/server paths.
- Draft manuscripts containing personal metadata or reviewer correspondence.
