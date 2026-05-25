# Splits

Generated split CSV files live here when they are safe to commit.

Rules:

- Split by subject before extracting 2D slices.
- Never create train/validation/test splits from already-extracted slices.
- Do not commit private or restricted metadata unless the dataset terms allow it.
- Commit schema examples and split-generation code by default.

Recommended first split:

```text
oasis1_age60_binary_seed1337.csv
```

Columns should include:

```text
subject_id,scan_id,age,sex,cdr,mmse,binary_label,severity_label,source_row,split
```
