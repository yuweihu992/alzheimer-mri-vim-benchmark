from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified subject-level splits.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", default="splits/oasis1_age60_binary_seed1337.csv")
    parser.add_argument("--subject-column", default="subject_id")
    parser.add_argument("--label-column", default="binary_label")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    return parser.parse_args()


def allocate_counts(n_items: int, val_fraction: float, test_fraction: float) -> tuple[int, int, int]:
    if n_items <= 0:
        return 0, 0, 0
    if n_items < 3:
        return n_items, 0, 0

    n_test = max(1, round(n_items * test_fraction))
    n_val = max(1, round(n_items * val_fraction))
    while n_val + n_test >= n_items:
        if n_test >= n_val and n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            break
    n_train = n_items - n_val - n_test
    return n_train, n_val, n_test


def main() -> None:
    args = parse_args()
    metadata = Path(args.metadata)
    output = Path(args.output)

    if not metadata.exists():
        raise SystemExit(f"metadata not found: {metadata}")

    total_fraction = args.train_fraction + args.val_fraction + args.test_fraction
    if abs(total_fraction - 1.0) > 1e-6:
        raise SystemExit("train/val/test fractions must sum to 1.0")

    with metadata.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if reader.fieldnames is None:
            raise SystemExit("metadata has no header")
        for column in (args.subject_column, args.label_column):
            if column not in reader.fieldnames:
                raise SystemExit(f"missing column {column}; got {reader.fieldnames}")

    labels_by_subject: dict[str, set[str]] = defaultdict(set)
    rows_by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        subject = row[args.subject_column].strip()
        label = row[args.label_column].strip()
        if not subject or not label:
            continue
        labels_by_subject[subject].add(label)
        rows_by_subject[subject].append(row)

    conflicts = {
        subject: labels
        for subject, labels in labels_by_subject.items()
        if len(labels) != 1
    }
    if conflicts:
        preview = "; ".join(
            f"{subject}: {sorted(labels)}" for subject, labels in sorted(conflicts.items())[:10]
        )
        raise SystemExit(f"subjects with conflicting labels: {preview}")

    subjects_by_label: dict[str, list[str]] = defaultdict(list)
    for subject, labels in labels_by_subject.items():
        label = next(iter(labels))
        subjects_by_label[label].append(subject)

    rng = random.Random(args.seed)
    split_by_subject: dict[str, str] = {}
    for label, subjects in sorted(subjects_by_label.items()):
        subjects = sorted(subjects)
        rng.shuffle(subjects)
        n_train, n_val, n_test = allocate_counts(
            len(subjects),
            args.val_fraction,
            args.test_fraction,
        )
        train_subjects = subjects[:n_train]
        val_subjects = subjects[n_train : n_train + n_val]
        test_subjects = subjects[n_train + n_val : n_train + n_val + n_test]
        for subject in train_subjects:
            split_by_subject[subject] = "train"
        for subject in val_subjects:
            split_by_subject[subject] = "val"
        for subject in test_subjects:
            split_by_subject[subject] = "test"

    output_rows: list[dict[str, str]] = []
    for row in rows:
        subject = row[args.subject_column].strip()
        if subject not in split_by_subject:
            continue
        output_row = dict(row)
        output_row["split"] = split_by_subject[subject]
        output_rows.append(output_row)

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) + ["split"] if rows else ["split"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    summary: dict[tuple[str, str], set[str]] = defaultdict(set)
    for subject, split in split_by_subject.items():
        label = next(iter(labels_by_subject[subject]))
        summary[(split, label)].add(subject)

    print(f"wrote={output}")
    print(f"subjects={len(split_by_subject)} rows={len(output_rows)}")
    for split in ("train", "val", "test"):
        parts = []
        for label in sorted(subjects_by_label):
            parts.append(f"{label}={len(summary[(split, label)])}")
        print(f"{split}: " + " ".join(parts))


if __name__ == "__main__":
    main()
