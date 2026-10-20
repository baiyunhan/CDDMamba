# Dataset protocol

The repository does not redistribute the ISPRS Vaihingen or Potsdam imagery or
reference labels. Users must obtain the datasets from the official ISPRS 2D
Semantic Labeling Benchmark and comply with its terms of use.

## Common preprocessing

- Input modality: RGB.
- Classes: Impervious Surface, Building, Low Vegetation, Tree, Car, Clutter.
- Patch size: 1024 x 1024 pixels.
- Patches used in the locked experiments do not overlap.
- Random seed: 42.

## Vaihingen

- Training: 605 patches.
- Validation: 150 patches from areas 30, 32, and 34.
- Test: 113 patches from areas 2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24,
  27, 29, 31, 33, 35, and 38.

`build_strict_vaihingen_datasets` removes every validation patch ID from the
training index and asserts the final 605/150/113 counts before training.

## Potsdam

The locked 21/3/14 large-tile split is:

- Training (21): `2_11`, `2_12`, `3_10`, `3_11`, `3_12`, `4_10`, `4_11`,
  `4_12`, `5_10`, `5_11`, `5_12`, `6_8`, `6_9`, `6_10`, `6_11`, `6_12`,
  `7_8`, `7_9`, `7_10`, `7_11`, `7_12`.
- Validation (3): `2_10`, `6_7`, `7_7`.
- Test (14): `2_13`, `2_14`, `3_13`, `3_14`, `4_13`, `4_14`, `4_15`,
  `5_13`, `5_14`, `5_15`, `6_13`, `6_14`, `6_15`, `7_13`.

This produces 756/108/504 patches. The strict builder checks both the counts
and the absence of patch-ID overlap among the three subsets.

