# Results

No OASIS baseline result has been published yet.

This file should be updated only after the same split and preprocessing protocol
has been used for all compared models.

## Planned First Report

Dataset:

```text
OASIS-1
age filter: age >= 60
task: CDR 0 vs CDR > 0
split: subject-level stratified split
seed: 1337
```

Metrics:

```text
accuracy
macro_precision
macro_recall
macro_f1
per_class_recall
confusion_matrix
subject_count_by_split
scan_count_by_split
slice_count_by_split
runtime_seconds
peak_vram_gb
```

## Result Table Template

| Model | Accuracy | Macro F1 | Non-demented recall | Demented recall | Peak VRAM | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ResNet | TBD | TBD | TBD | TBD | TBD | Baseline |
| ViT | TBD | TBD | TBD | TBD | TBD | Same split |
| Vision Mamba | TBD | TBD | TBD | TBD | TBD | Same split |

## Required Interpretation

Do not write that Vision Mamba is better unless the numbers show it under the
same protocol.

If a model wins on accuracy but loses on minority-class recall, state that
clearly.

If reported slice-level performance is higher than subject-level performance,
lead with subject-level performance.
