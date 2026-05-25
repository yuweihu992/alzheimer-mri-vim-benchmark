# OASIS Dataset Plan

## First Dataset

Use OASIS-1 first.

Reason: it is easier to explain, smaller than OASIS-3, and official enough for a
public portfolio project. It has cross-sectional T1-weighted MRI data with
clinical metadata.

Main risk: if all ages are used, the model may learn age instead of Alzheimer
features. For the first credible version, filter to older adults only.

## Backup Dataset

Use OASIS-2 second.

Reason: it is longitudinal and all subjects are older adults, so it is closer to
a real Alzheimer progression setting.

Main risk: visit leakage. The same subject has multiple visits, and no visit from
one subject can cross split boundaries.

## Access Rules

- Do not commit raw OASIS data.
- Do not commit full MRI volumes.
- Do not commit face/head reconstructions or any derivative that could support
  re-identification.
- Store raw data in `data/raw/oasis1/` locally.
- Store processed data in `data/processed/oasis1_2d/` locally.
- Commit only code, metadata templates, split-generation logic, and result
  summaries.

## Local Data Layout

```text
data/
├── raw/
│   └── oasis1/
│       ├── oasis_cross-sectional_disc1/
│       └── ...
├── interim/
│   └── oasis1/
│       └── extracted_subjects/
└── processed/
    └── oasis1_2d/
        ├── train/
        ├── val/
        └── test/
```

Commit-safe metadata:

```text
metadata/
├── oasis1_subjects.schema.csv
└── oasis1_label_mapping.md

splits/
├── oasis1_age60_binary_seed1337.schema.csv
└── README.md
```

Use schema files or tiny fake examples in git, not real protected records unless
the dataset terms permit it.

## First Label Mapping

Start with binary labels:

```text
CDR == 0  -> non_demented
CDR > 0   -> demented
```

Avoid 4-class severity in the first version. Moderate cases are usually too few,
and class metrics will be unstable.

## Split Rule

Split by subject before making 2D slices:

```text
subject metadata -> subject split -> scan selection -> slice extraction -> train
```

Never do:

```text
slice extraction -> random image split
```

That leaks near-duplicate slices and repeated scans into validation/test.

## First Baseline Protocol

Recommended first experiment:

```text
dataset: OASIS-1
filter: age >= 60
task: CDR 0 vs CDR > 0
split: subject-level, stratified, seed 1337
input: fixed middle brain slices or multi-slice bag
models: ResNet, ViT, Vision Mamba
main metric: subject-level macro F1
secondary metric: slice-level accuracy
```

## GitHub README Wording

Use this framing:

> This project is a reproducible research demo for Alzheimer MRI classification
> on OASIS data. It is not a clinical diagnostic system.

