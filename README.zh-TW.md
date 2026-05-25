# 阿茲海默症 MRI Vision Mamba Benchmark

語言：[English](README.md) | [繁體中文](README.zh-TW.md)

這是一個可重現的阿茲海默症腦部 MRI 分類作品集專案。目標是使用 OASIS
官方資料、subject-level split、CNN/ViT/Vision Mamba baseline，以及後續可解釋
推論 UI，建立一個能公開展示、能重跑、且不誇大結果的深度學習專案。

本專案是研究工程與作品集 demo，不是臨床診斷系統。

## 為什麼做這個專案

很多 Alzheimer MRI demo 只使用已切好的 2D 圖片，然後報很高的 accuracy。
這種做法很容易出問題：同一位受試者的不同掃描，或相近切片，可能同時出現在
train 和 test，造成資料外洩，讓結果看起來比實際更好。

這個 repo 採用比較嚴格的流程：

1. 使用 OASIS 官方資料，不把 Kaggle mirror 當主要來源
2. 先做 subject-level split，再做切片萃取
3. raw MRI、checkpoint、輸出結果不放進 git
4. 在同一套資料切分與前處理下比較 ResNet、ViT、Vision Mamba
5. 回報 per-class 與 subject-level metrics，不只看 accuracy

## 目前狀態

| 項目 | 狀態 |
| --- | --- |
| Docker GPU 訓練環境 | 已有 |
| Vision Mamba style classifier | 已有 |
| ViT 比較工具 | 已有 |
| OASIS-1 metadata parser | 已有 |
| OASIS-1 subject split generator | 已有 |
| OASIS 2D slice preprocessing | 規劃中 |
| OASIS baseline results | 規劃中 |
| Demo UI | 規劃中 |

## Repo 結構

```text
src/mamba2_vision/       model, checkpoint, dataloader 程式
tools/                   benchmark, telemetry, OASIS metadata, split 工具
scripts/                 PowerShell 與 shell wrapper
docs/                    dataset, GitHub, repo design 文件
metadata/                可提交的 schema 或 metadata 摘要
splits/                  可提交的 split schema 或 split 摘要
reports/                 最終報告產物
examples/                可公開展示的截圖與 demo 圖
```

本地專用、不提交：

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

## Dataset 計畫

第一個資料集：OASIS-1。

第一版建議任務：

```text
dataset: OASIS-1
filter: age >= 60
labels: CDR == 0 vs CDR > 0
split: subject-level, stratified, seed 1337
input: fixed middle brain slices or multi-slice aggregation
models: ResNet, ViT, Vision Mamba
main metric: subject-level macro F1
```

細節見 [docs/DATASET_OASIS.md](docs/DATASET_OASIS.md)。

## OASIS-1 Metadata 與 Split

下載 OASIS-1 官方 clinical CSV 後，放在本地 `data/raw/oasis1/`。除非資料使用條款
明確允許，否則不要提交原始資料或完整 metadata。

產生標準化 metadata：

```powershell
python tools/prepare_oasis1_metadata.py `
  --clinical-csv data/raw/oasis1/oasis_cross-sectional.csv `
  --output metadata/oasis1_subjects.csv `
  --min-age 60
```

產生 subject-level split：

```powershell
python tools/make_subject_splits.py `
  --metadata metadata/oasis1_subjects.csv `
  --output splits/oasis1_age60_binary_seed1337.csv `
  --seed 1337
```

如果同一個 subject 出現互相衝突的 label，split 工具會直接失敗。這是刻意設計，
避免資料切分時把錯誤默默帶進訓練。

## 主機需求

- Docker Desktop，並使用 Linux containers
- NVIDIA GPU 可被 Docker 使用
- NVIDIA container runtime

此機器已驗證：

- Docker Engine: `29.4.3`
- GPU: NVIDIA RTX 3500 Ada Generation Laptop GPU
- NVIDIA runtime: present
- CUDA container GPU smoke: passed with `nvidia/cuda:12.4.1-base-ubuntu22.04`

## Build

```powershell
.\scripts\build.ps1
```

## Smoke Test

Smoke test 會用合成圖片跑一次訓練 step，存 checkpoint，再重新載入 checkpoint
做推論。

```powershell
.\scripts\smoke.ps1
```

預期產物：

- `outputs/smoke/smoke_mamba2.pt`
- `outputs/smoke/smoke_vim.pt`
- `outputs/smoke/sample.png`

## 使用 ImageFolder 訓練

資料夾格式：

```text
data/imagefolder/train/class_a/*.jpg
data/imagefolder/train/class_b/*.jpg
data/imagefolder/val/class_a/*.jpg
data/imagefolder/val/class_b/*.jpg
```

執行：

```powershell
.\scripts\train_imagefolder.ps1
```

預設使用 `--model-arch vim`、`--image-size 384`，以及 ImageNet-style
train/validation transforms。

## Fair Vim/Mamba2 vs ViT 比較

比較工具會先訓練 Vim 或 Mamba2，再用相同 dataset、batch size、epochs、optimizer、
AMP 設定與 seed 訓練 ViT。輸出包括：

- `metrics.csv`
- `summary.json`
- `loss_trend.png`
- `accuracy_trend.png`
- `throughput_gpu_params.png`
- `confusion_vim.png` 或 `confusion_mamba2.png`
- `confusion_<vit-model>.png`

```powershell
docker compose run --rm mamba2-vision python tools/fair_compare.py --dataset imagefolder --train-dir /workspace/data/imagefolder/train --val-dir /workspace/data/imagefolder/val --output-dir /workspace/outputs/fair-highres384 --mamba-arch vim --image-size 384 --batch-size 32 --epochs 20 --workers 2 --amp --vit-model vit_b_16 --vit-pretrained
```

## 直接使用 Container 指令

```powershell
docker compose run --rm mamba2-vision python train.py --dataset cifar10 --data-dir /workspace/data --output-dir /workspace/outputs/cifar10
docker compose run --rm mamba2-vision python infer.py --checkpoint /workspace/outputs/cifar10/best.pt --image /workspace/data/sample.jpg
```

## 專案文件

- [docs/REPO_DESIGN.md](docs/REPO_DESIGN.md) - 公開 repo 結構與 milestones
- [docs/DATASET_OASIS.md](docs/DATASET_OASIS.md) - OASIS-1/OASIS-2 資料計畫與資料外洩規則
- [docs/GITHUB_SETUP.md](docs/GITHUB_SETUP.md) - GitHub repo、Project board、CI 設定
- [docs/GITHUB_PROJECT_TASKS.md](docs/GITHUB_PROJECT_TASKS.md) - 可轉成 GitHub Issues 的任務清單
- [docs/RESULTS.md](docs/RESULTS.md) - 結果報告模板

## 注意事項

- 不要宣稱 Vision Mamba 贏過 ViT/CNN，除非同一套 split 與 preprocessing protocol 的結果真的支持。
- CIFAR10 只用來做 pipeline smoke test。
- OASIS raw data、checkpoint、generated outputs 都刻意不提交。
- `mamba-ssm` 實務上需要 Linux、CUDA、PyTorch 和 NVIDIA GPU。
