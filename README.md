# Alzheimer MRI Vision Mamba Benchmark

Languages: [English](README.md) | [繁體中文](README.zh-TW.md)

Reproducible Alzheimer MRI classification demo with OASIS data, subject-level
splits, CNN/ViT/Vision Mamba baselines, and a future explainable inference UI.

This repository is a portfolio and research-engineering project. It is not a
clinical diagnostic system.

## Why This Project Exists

Many Alzheimer MRI demos report high accuracy from pre-sliced 2D images. That is
easy to reproduce badly: the same subject or near-duplicate slices can leak
across train and test sets.

This repo is designed around a stricter workflow:

1. use official OASIS data, not Kaggle mirrors as the source of truth
2. split by subject before slice extraction
3. keep raw MRI data out of git
4. compare ResNet, ViT, and Vision Mamba under the same protocol
5. report per-class and subject-level metrics, not only accuracy

## Current Status

| Area | Status |
| --- | --- |
| Docker GPU training environment | Available |
| Vision Mamba style classifier | Available |
| ViT comparison utility | Available |
| OASIS-1 metadata parser | Available |
| OASIS-1 subject split generator | Available |
| OASIS preprocessing to 2D slices | Planned |
| OASIS baseline results | Planned |
| Demo UI | Planned |

## Repository Layout

```text
src/mamba2_vision/       model, checkpoint, and dataloader code
tools/                   benchmark, telemetry, OASIS metadata, split tools
scripts/                 PowerShell and shell wrappers
docs/                    dataset, GitHub, and repo design docs
metadata/                commit-safe schemas or generated metadata summaries
splits/                  commit-safe split schemas or generated split summaries
reports/                 final report artifacts
examples/                public-safe screenshots and demo images
```

Ignored local-only paths:

```text
data/
outputs/
checkpoints/
runs/
tmp/
*.pt
*.pth
*.onnx
*.pdf
```

## Dataset Plan

First target: OASIS-1.

Recommended first task:

```text
dataset: OASIS-1
filter: age >= 60
labels: CDR == 0 vs CDR > 0
split: subject-level, stratified, seed 1337
input: fixed middle brain slices or multi-slice aggregation
models: ResNet, ViT, Vision Mamba
main metric: subject-level macro F1
```

See [docs/DATASET_OASIS.md](docs/DATASET_OASIS.md).

## OASIS-1 Metadata and Split

After downloading OASIS-1 clinical CSV from the official OASIS page, put it under
`data/raw/oasis1/` locally. Do not commit it unless the data terms allow it.

Create normalized metadata:

```powershell
python tools/prepare_oasis1_metadata.py `
  --clinical-csv data/raw/oasis1/oasis_cross-sectional.csv `
  --output metadata/oasis1_subjects.csv `
  --min-age 60
```

Create subject-level splits:

```powershell
python tools/make_subject_splits.py `
  --metadata metadata/oasis1_subjects.csv `
  --output splits/oasis1_age60_binary_seed1337.csv `
  --seed 1337
```

The split tool fails if one subject has conflicting labels.

## Host Requirements

- Docker Desktop with Linux containers running
- NVIDIA GPU visible to Docker
- NVIDIA container runtime

Already validated on this machine:

- Docker Engine: `29.4.3`
- GPU: NVIDIA RTX 3500 Ada Generation Laptop GPU
- NVIDIA runtime: present
- CUDA container GPU smoke: passed with `nvidia/cuda:12.4.1-base-ubuntu22.04`

## Build

```powershell
.\scripts\build.ps1
```

## Smoke Test

Runs one synthetic image train step, saves checkpoints, reloads them, and runs
inference on a generated image.

```powershell
.\scripts\smoke.ps1
```

Expected artifacts:

- `outputs/smoke/smoke_mamba2.pt`
- `outputs/smoke/smoke_vim.pt`
- `outputs/smoke/sample.png`

## Train on ImageFolder Data

Put data in ImageFolder layout:

```text
data/imagefolder/train/class_a/*.jpg
data/imagefolder/train/class_b/*.jpg
data/imagefolder/val/class_a/*.jpg
data/imagefolder/val/class_b/*.jpg
```

Then run:

```powershell
.\scripts\train_imagefolder.ps1
```

Defaults are `--model-arch vim`, `--image-size 384`, and ImageNet-style
train/validation crops.

## Fair Vim/Mamba2 vs ViT Comparison

The comparison script trains Vim or Mamba2 first, then ViT with the same dataset,
batch size, epochs, optimizer, AMP mode, and seed. It writes metrics and plots:

- `metrics.csv`
- `summary.json`
- `loss_trend.png`
- `accuracy_trend.png`
- `throughput_gpu_params.png`
- `confusion_vim.png` or `confusion_mamba2.png`
- `confusion_<vit-model>.png`

```powershell
docker compose run --rm mamba2-vision python tools/fair_compare.py --dataset imagefolder --train-dir /workspace/data/imagefolder/train --val-dir /workspace/data/imagefolder/val --output-dir /workspace/outputs/fair-highres384 --mamba-arch vim --image-size 384 --batch-size 32 --epochs 20 --workers 2 --amp --vit-model vit_b_16 --vit-pretrained
```

## Direct Container Commands

```powershell
docker compose run --rm mamba2-vision python train.py --dataset cifar10 --data-dir /workspace/data --output-dir /workspace/outputs/cifar10
docker compose run --rm mamba2-vision python infer.py --checkpoint /workspace/outputs/cifar10/best.pt --image /workspace/data/sample.jpg
```

## Project Docs

- [docs/REPO_DESIGN.md](docs/REPO_DESIGN.md) - public repository structure and milestones
- [docs/DATASET_OASIS.md](docs/DATASET_OASIS.md) - OASIS-1/OASIS-2 dataset plan and leakage rules
- [docs/GITHUB_SETUP.md](docs/GITHUB_SETUP.md) - GitHub repository, project board, and CI setup
- [docs/GITHUB_PROJECT_TASKS.md](docs/GITHUB_PROJECT_TASKS.md) - issue list for GitHub Projects
- [docs/RESULTS.md](docs/RESULTS.md) - result report template

## Notes

- Do not claim Vision Mamba beats ViT/CNN unless the same split and preprocessing
  protocol prove it.
- CIFAR10 is only a pipeline smoke test.
- OASIS raw data, checkpoints, and generated outputs are intentionally ignored.
- `mamba-ssm` requires Linux, CUDA, PyTorch, and an NVIDIA GPU for practical use.
