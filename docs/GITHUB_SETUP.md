# GitHub Setup

## 1. Create Repository

Recommended repository name:

```text
alzheimer-mri-vim-benchmark
```

Good public description:

```text
Reproducible Alzheimer MRI classification demo with OASIS data, subject-level
splits, CNN/ViT/Vision Mamba baselines, and explainable inference UI.
```

Suggested topics:

```text
medical-imaging
alzheimer-disease
mri
vision-mamba
vit
pytorch
docker
reproducible-research
oasis
```

## 2. Local Git Setup

From this workspace:

```powershell
git status
git add README.md Dockerfile docker-compose.yml requirements.txt train.py infer.py src tools scripts docs .gitignore .dockerignore
git commit -m "Initialize Alzheimer MRI Vision Mamba portfolio repo"
git branch -M main
```

If the GitHub repository is empty:

```powershell
git remote add origin https://github.com/<your-user>/alzheimer-mri-vim-benchmark.git
git push -u origin main
```

If `origin` already exists:

```powershell
git remote -v
git remote set-url origin https://github.com/<your-user>/alzheimer-mri-vim-benchmark.git
git push -u origin main
```

## 3. GitHub Repository Settings

On GitHub:

1. Open repository `Settings`.
2. Set default branch to `main`.
3. Enable `Issues`.
4. Enable `Discussions` only if you want public Q&A.
5. In `Actions`, allow GitHub Actions.
6. Add branch protection later after CI is stable.

Do not upload:

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
*.tar.gz
*.zip
```

## 4. GitHub Project Board

Your screenshot shows an empty GitHub Project. Use it as a roadmap, not as the
source of truth for code.

Recommended fields:

```text
Status: Backlog / Ready / In Progress / Review / Done
Priority: P0 / P1 / P2
Milestone: Foundation / OASIS Pipeline / Baselines / Demo / Polish
Type: docs / data / model / eval / demo / infra
```

Add issues from:

```text
docs/GITHUB_PROJECT_TASKS.md
```

## 5. GitHub Actions

GitHub-hosted runners usually do not have an NVIDIA GPU. Do not make CI compile
`mamba-ssm` or run CUDA training.

Use CI for lightweight checks:

- required files exist
- forbidden large files are not committed
- Python files compile when dependencies are available
- docs links and Markdown can be checked later

Run GPU tests locally or on a GPU server:

```powershell
.\scripts\build.ps1
.\scripts\smoke.ps1
```

## 6. Release Strategy

First public release:

```text
v0.1.0 repo foundation
```

Include:

- README
- OASIS setup docs
- subject split plan
- model training scripts
- no data
- no checkpoints

Later release:

```text
v0.2.0 oasis1 binary baseline
```

Include:

- result report
- confusion matrix image
- trained checkpoint only if license and file size policy allow it
- model card

