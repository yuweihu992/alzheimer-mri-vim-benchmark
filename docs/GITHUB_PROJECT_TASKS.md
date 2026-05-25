# GitHub Project Tasks

Use these as GitHub Issues, then add them to your Project board.

## Foundation

### [P0] Clean public repository files

Remove local-only artifacts from git scope and confirm `.gitignore` blocks data,
outputs, checkpoints, logs, archives, temporary files, and PDFs.

Labels: `infra`, `docs`

### [P0] Rewrite README for portfolio review

Add project goal, architecture diagram, quick start, dataset disclaimer, planned
results table, and non-clinical-use warning.

Labels: `docs`

### [P1] Add repository license and citation section

Choose a code license and add OASIS acknowledgement/citation instructions.

Labels: `docs`, `legal`

## OASIS Pipeline

### [P0] Document OASIS-1 download process

Write steps for official OASIS access, local storage path, and what files are
expected after extraction.

Labels: `data`, `docs`

### [P0] Build OASIS-1 metadata parser

Parse subject ID, scan/session ID, age, sex, CDR, MMSE if available, and usable
image paths into a normalized metadata table.

Labels: `data`

### [P0] Add subject-level split generator

Create reproducible train/validation/test splits by subject, with no subject or
session overlap.

Labels: `data`, `eval`

### [P1] Add leakage checks

Verify no subject overlap, no image hash overlap, and no near-duplicate leakage
across splits.

Labels: `data`, `eval`

### [P1] Add 2D slice extraction

Extract fixed middle slices or a fixed multi-slice window after subject split.

Labels: `data`

## Baselines

### [P0] Train ResNet baseline

Use the same OASIS split and preprocessing as the other models.

Labels: `model`

### [P0] Train ViT baseline

Use the same OASIS split and preprocessing as ResNet and Vision Mamba.

Labels: `model`

### [P0] Train Vision Mamba baseline

Use the same OASIS split and preprocessing as ResNet and ViT.

Labels: `model`

### [P1] Add evaluation report generator

Generate metrics JSON, classification report CSV, confusion matrix PNG, and
subject-level aggregation report.

Labels: `eval`

## Demo

### [P1] Add Gradio or Streamlit demo

Upload one processed brain slice and show predicted class probabilities.

Labels: `demo`

### [P1] Add explainability visualization

Add Grad-CAM or saliency visualization for supported baseline models.

Labels: `demo`, `eval`

## Polish

### [P2] Add screenshots and architecture diagram

Add public-safe screenshots under `examples/`.

Labels: `docs`, `demo`

### [P2] Add lightweight GitHub Actions checks

Check repository hygiene without requiring GPU or OASIS data.

Labels: `infra`

