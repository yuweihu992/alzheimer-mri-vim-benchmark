# Repository Design

## Positioning

This repository is an application-oriented portfolio project for Alzheimer MRI
classification. It should show a reproducible pipeline, not just a single high
accuracy number.

Target claim:

> Reproducible Alzheimer MRI classification pipeline using OASIS data with
> subject-level splits, leakage checks, CNN/ViT/Vision Mamba baselines, and an
> explainable demo UI.

Do not claim clinical diagnosis capability. Do not claim Vision Mamba is better
unless the benchmark proves it under the same split and preprocessing protocol.

## Recommended Public Structure

```text
.
├── README.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── train.py
├── infer.py
├── src/
│   └── mamba2_vision/
├── tools/
│   ├── fair_compare.py
│   ├── smoke_test.py
│   └── telemetry_training_loop.py
├── scripts/
│   ├── build.ps1
│   ├── smoke.ps1
│   ├── train_imagefolder.ps1
│   └── infer.ps1
├── docs/
│   ├── DATASET_OASIS.md
│   ├── GITHUB_SETUP.md
│   ├── GITHUB_PROJECT_TASKS.md
│   └── REPO_DESIGN.md
├── metadata/
│   └── .gitkeep
├── splits/
│   └── .gitkeep
├── reports/
│   └── .gitkeep
└── examples/
    └── .gitkeep
```

Local-only paths, never committed:

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

## Milestones

### Milestone 1 - Public Repo Foundation

- Clean README with project goal, warning, quick start, and result placeholder.
- Add data usage notes and OASIS download instructions.
- Add `.gitignore` and `.dockerignore` rules for datasets, checkpoints, PDFs,
  logs, archives, and temporary files.
- Add a GitHub Project board with issues from `docs/GITHUB_PROJECT_TASKS.md`.

### Milestone 2 - OASIS-1 Reproducible Dataset Pipeline

- Download OASIS-1 outside git.
- Parse demographic and clinical CSV.
- Build `metadata/oasis1_subjects.csv`.
- Filter to age 60+ for fair Alzheimer classification.
- Map labels first as binary: `CDR == 0` vs `CDR > 0`.
- Generate subject-level train/validation/test splits.
- Export selected 2D slices only after subject split.

### Milestone 3 - Baselines

- Train a compact CNN/ResNet baseline.
- Train ViT baseline under the same split.
- Train Vision Mamba baseline under the same split.
- Report accuracy, macro F1, per-class recall, confusion matrix, and runtime.

### Milestone 4 - Demo UI

- Add `demo_app.py` with Gradio or Streamlit.
- Upload one processed non-identifying brain slice.
- Show class probabilities and model confidence.
- Add saliency or Grad-CAM style visualization.
- Add a visible non-clinical-use disclaimer.

### Milestone 5 - Portfolio Polish

- Add `docs/RESULTS.md` with honest comparison.
- Add screenshots under `examples/`.
- Add GitHub Actions repo checks.
- Add license and citation section.
- Add model card for any released checkpoint.

## Evaluation Standard

Minimum report:

```text
accuracy
macro_precision
macro_recall
macro_f1
per_class_precision
per_class_recall
confusion_matrix
subject_count_by_split
scan_count_by_split
slice_count_by_split
runtime_seconds
peak_vram_gb
```

Leakage checks:

- Same subject must not appear in multiple splits.
- Same visit/session must not appear in multiple splits.
- Image hashes should not overlap across splits.
- Slice-level metrics should be secondary; subject-level aggregation should be
  the main result.

