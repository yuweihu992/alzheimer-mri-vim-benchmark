from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize OASIS-1 demographic/clinical CSV into commit-safe metadata."
    )
    parser.add_argument("--clinical-csv", required=True, help="OASIS-1 demographic and clinical CSV.")
    parser.add_argument("--output", default="metadata/oasis1_subjects.csv")
    parser.add_argument("--min-age", type=float, default=60.0)
    parser.add_argument(
        "--allow-missing-cdr",
        action="store_true",
        help="Keep rows without CDR. Default skips them because labels cannot be built.",
    )
    return parser.parse_args()


def normalize_name(name: str) -> str:
    return name.strip().lstrip("\ufeff").lower().replace(" ", "").replace("_", "")


def find_column(fieldnames: Iterable[str], candidates: Iterable[str]) -> str | None:
    by_norm = {normalize_name(name): name for name in fieldnames}
    for candidate in candidates:
        match = by_norm.get(normalize_name(candidate))
        if match is not None:
            return match
    return None


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def subject_id_from_scan(scan_id: str) -> str:
    match = re.match(r"^(OAS1_\d+)", scan_id.strip(), flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return re.sub(r"[_-]?MR\d+$", "", scan_id.strip(), flags=re.IGNORECASE)


def severity_label(cdr: float | None) -> str:
    if cdr is None:
        return "unknown"
    value = ("%g" % cdr).replace(".", "p")
    return f"cdr_{value}"


def binary_label(cdr: float | None) -> str:
    if cdr is None:
        return "unknown"
    return "non_demented" if cdr == 0 else "demented"


def main() -> None:
    args = parse_args()
    clinical_csv = Path(args.clinical_csv)
    output = Path(args.output)

    if not clinical_csv.exists():
        raise SystemExit(f"clinical CSV not found: {clinical_csv}")

    with clinical_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit("clinical CSV has no header")

        id_col = find_column(reader.fieldnames, ["ID", "MRI ID", "Subject ID", "Subject"])
        age_col = find_column(reader.fieldnames, ["Age"])
        sex_col = find_column(reader.fieldnames, ["M/F", "Sex", "Gender"])
        cdr_col = find_column(reader.fieldnames, ["CDR"])
        mmse_col = find_column(reader.fieldnames, ["MMSE"])

        required = {"ID": id_col, "Age": age_col, "CDR": cdr_col}
        missing = [name for name, column in required.items() if column is None]
        if missing:
            raise SystemExit(f"missing required columns: {missing}; got {reader.fieldnames}")

        rows: list[dict[str, str]] = []
        skipped_age = 0
        skipped_cdr = 0
        for source_row, row in enumerate(reader, start=2):
            scan_id = (row.get(id_col) or "").strip()
            age = parse_float(row.get(age_col))
            cdr = parse_float(row.get(cdr_col))
            if not scan_id:
                continue
            if age is None or age < args.min_age:
                skipped_age += 1
                continue
            if cdr is None and not args.allow_missing_cdr:
                skipped_cdr += 1
                continue
            rows.append(
                {
                    "subject_id": subject_id_from_scan(scan_id),
                    "scan_id": scan_id,
                    "age": "" if age is None else "%g" % age,
                    "sex": (row.get(sex_col) or "").strip() if sex_col else "",
                    "cdr": "" if cdr is None else "%g" % cdr,
                    "mmse": (row.get(mmse_col) or "").strip() if mmse_col else "",
                    "binary_label": binary_label(cdr),
                    "severity_label": severity_label(cdr),
                    "source_row": str(source_row),
                }
            )

    labels_by_subject: dict[str, set[str]] = {}
    for row in rows:
        labels_by_subject.setdefault(row["subject_id"], set()).add(row["binary_label"])
    conflicts = {
        subject_id: labels
        for subject_id, labels in labels_by_subject.items()
        if len(labels) > 1 and "unknown" not in labels
    }
    if conflicts:
        for subject_id, labels in sorted(conflicts.items()):
            print(f"warning: subject {subject_id} has labels {sorted(labels)}", file=sys.stderr)

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject_id",
        "scan_id",
        "age",
        "sex",
        "cdr",
        "mmse",
        "binary_label",
        "severity_label",
        "source_row",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda item: (item["subject_id"], item["scan_id"])))

    unique_subjects = len({row["subject_id"] for row in rows})
    print(f"wrote={output}")
    print(f"rows={len(rows)} subjects={unique_subjects}")
    print(f"skipped_age={skipped_age} skipped_cdr={skipped_cdr}")


if __name__ == "__main__":
    main()
