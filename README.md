# CDDMamba

This repository contains the implementation, locked experiment configurations,
fixed data protocol, and numerical results accompanying the submitted manuscript.
The proposed network contains three principal components: CDFM, DP-MPC, and
DGR-CSMamba, together with the high-resolution detail decoder used in the final
model.

## Repository contents

- `geoseg/models/proposed_method/`: final model and controlled core ablations.
- `geoseg/losses/`: the joint segmentation/flow-matching objective.
- `geoseg/datasets/`: strict dataset builders used to prevent split leakage.
- `config/`: the locked 100-epoch configurations and internal-design ablations.
- `tools/`: patch preparation, metric, validation-summary, and integrator scripts.
- `results/`: the numerical data supporting the tables in the manuscript.
- `figures/`: representative qualitative results already used in the manuscript.

The ISPRS Vaihingen and Potsdam source images are third-party datasets and are
not redistributed here. Download them from the official ISPRS benchmark page:
<https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/Default.aspx>.

## Environment

The experiments were conducted with Python 3.11, PyTorch 2.1.2, CUDA 12.1,
and an NVIDIA GeForce RTX 4090D. Install PyTorch for the CUDA version available
on your system, then install the remaining packages:

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`REGNET_WEIGHTS` may point to a local official RegNet-Y-1.6GF ImageNet
checkpoint. If it is omitted, `timm` downloads the official checkpoint into its
cache.

## Dataset preparation

Follow [`DATASETS.md`](DATASETS.md) to create the non-overlapping 1024 x 1024
RGB patches. The code expects the following structure:

```text
data/
  vaihingen/{train,val,test}/{images_1024,masks_1024}/
  potsdam/{train,val,test}/{images_1024,masks_1024}/
```

Set the dataset locations if they are outside the repository:

```bash
export VAIHINGEN_ROOT=/path/to/vaihingen
export POTSDAM_ROOT=/path/to/potsdam
```

## Training

Vaihingen:

```bash
python train.py -c config/vaihingen/proposed_method_v3_detail_formal100.py
```

Potsdam:

```bash
python train.py -c config/potsdam/proposed_method_v3_detail_potsdam_formal100.py
```

Both configurations use seed 42, 100 epochs, an effective batch size of 8,
AdamW, and cosine annealing. The checkpoint with the highest validation mIoU is
selected before the locked test evaluation.

Evaluate a selected checkpoint without changing the model or test protocol:

```bash
python evaluate.py \
  -c config/potsdam/proposed_method_v3_detail_potsdam_formal100.py \
  -w /path/to/best.ckpt \
  --split test \
  --output results/reproduced_potsdam_test
```

## Ablations

The `config/vaihingen/` and `config/potsdam/` directories contain the controlled
core-component ablations. The Potsdam directory also contains the CDFM grouping /
perturbation variants and the DGR-CSMamba scan-order variants. The Euler/RK4
comparison reuses the same trained checkpoint and is implemented in:

```bash
python tools/evaluate_integrator_ablation.py --output results/integrator_run
```

## Results

Paper-facing tables are stored in `results/main_comparison/` and
`results/ablations/`. Machine-readable confusion matrices, per-tile metrics,
and model summaries are retained under `results/raw/`. Test data were evaluated
only after model selection on the validation set.

## Checkpoints

Model checkpoints are intentionally not stored in the Git history. Add the
public download URL here after uploading the two final best checkpoints as
GitHub Release assets:

- Vaihingen: <https://github.com/baiyunhan/CDDMamba/releases/download/v1.0.0/vaihingen_best.ckpt>
- Potsdam: <https://github.com/baiyunhan/CDDMamba/releases/download/v1.0.0/potsdam_best.ckpt>

## Citation and license

Citation metadata are provided in `CITATION.cff`. The project was developed on
top of the public UrbanSSF codebase;
see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The authors should finish
the upstream-license review before adding a repository-wide software license.

