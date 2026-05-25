# Vision Mamba Lab

Containerized image training and inference environment for Vision Mamba style
experiments using the `Mamba2` block from `mamba-ssm==2.2.6.post3`.

Default training now uses a Vim-style bidirectional classifier:

1. crop image with ImageNet-style transforms
2. split image into patches with a Conv2d patch embed
3. prepend a learned class token
4. add learned positional embeddings with interpolation for high resolution
5. pass tokens through bidirectional forward/backward `Mamba2` blocks
6. classify

The older alternating-direction `mamba2` classifier remains available with
`--model-arch mamba2`. Use `--model-arch vim` for the default vision path.

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

Runs one synthetic image train step, saves a checkpoint, reloads it, and runs
inference on a generated image.

```powershell
.\scripts\smoke.ps1
```

Expected artifacts:

- `outputs/smoke/smoke_mamba2.pt`
- `outputs/smoke/smoke_vim.pt`
- `outputs/smoke/sample.png`

## Train on CIFAR10

```powershell
.\scripts\train_cifar10.ps1
```

Output:

- `outputs/cifar10/last.pt`
- `outputs/cifar10/best.pt`

Training shows a live progress bar in interactive terminals. It reports ETA,
batch loss, running loss, running accuracy, images/second, learning rate, and
allocated GPU memory. Use `--no-progress` for plain log files.

## Train on Your Own Images

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

## Inference

```powershell
.\scripts\infer.ps1 -Checkpoint /workspace/outputs/cifar10/best.pt -Image /workspace/path/to/image.jpg
```

Use Linux container paths for files mounted under this repo. For example,
`C:\Users\yuhsu\OneDrive - NVIDIA Corporation\Documents\Mamba-2\data\sample.jpg`
is `/workspace/data/sample.jpg` inside the container.

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

## Notes

- This is not a claim that Vim/Mamba beats ViT/CNN for single-image tasks. It is
  a controlled environment for testing that hypothesis.
- CIFAR10 is only a pipeline smoke test. For real high-resolution vision work,
  use `imagefolder` with real high-resolution images.
- For temporal images or video frames, keep frame order and feed frame/patch
  tokens as a longer sequence in the next step.
- `mamba-ssm` requires Linux, CUDA, PyTorch, and an NVIDIA GPU for practical use.

## Portfolio Roadmap

This repository is being shaped into an Alzheimer MRI classification portfolio
project using OASIS data with subject-level splits and fair CNN/ViT/Vision Mamba
baselines.

Project docs:

- `docs/REPO_DESIGN.md` - public repository structure and milestones
- `docs/DATASET_OASIS.md` - OASIS-1/OASIS-2 dataset plan and leakage rules
- `docs/GITHUB_SETUP.md` - GitHub repository, project board, and CI setup
- `docs/GITHUB_PROJECT_TASKS.md` - issue list for GitHub Projects

Medical disclaimer: this repository is a reproducible research and portfolio
demo. It is not a clinical diagnostic system.
