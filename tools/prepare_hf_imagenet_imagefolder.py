from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize Hugging Face ImageNet-1K into torchvision ImageFolder layout."
    )
    parser.add_argument("--dataset-id", default="ILSVRC/imagenet-1k")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report-every", type=int, default=1000)
    return parser.parse_args()


def safe_class_name(label: int, names: list[str] | None) -> str:
    if names and 0 <= label < len(names):
        name = names[label]
    else:
        name = f"class_{label:04d}"
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return name or f"class_{label:04d}"


def class_names_from_features(dataset: Any) -> list[str] | None:
    try:
        feature = dataset.features["label"]
        return list(feature.names)
    except Exception:
        return None


def image_to_rgb(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError(f"unsupported image type: {type(image)!r}")


def materialize_split(
    *,
    dataset_id: str,
    split: str,
    output_root: Path,
    output_name: str,
    max_items: int | None,
    streaming: bool,
    overwrite: bool,
    report_every: int,
    token: str | None,
) -> dict[str, Any]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split=split, streaming=streaming, token=token)
    names = class_names_from_features(dataset)
    split_root = output_root / output_name
    split_root.mkdir(parents=True, exist_ok=True)

    count = 0
    class_counts: dict[str, int] = {}
    for index, row in enumerate(dataset):
        if max_items is not None and count >= max_items:
            break

        label = int(row["label"])
        class_name = safe_class_name(label, names)
        class_dir = split_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        target = class_dir / f"{index:08d}.jpg"
        if overwrite or not target.exists():
            image = image_to_rgb(row["image"])
            image.save(target, format="JPEG", quality=95)

        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        count += 1
        if report_every > 0 and count % report_every == 0:
            print(f"{output_name}: saved={count} classes={len(class_counts)}", flush=True)

    return {
        "split": split,
        "output_name": output_name,
        "images": count,
        "classes": len(class_counts),
        "output_root": str(split_root),
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")
    results = [
        materialize_split(
            dataset_id=args.dataset_id,
            split=args.train_split,
            output_root=output_root,
            output_name="train",
            max_items=args.max_train,
            streaming=args.streaming,
            overwrite=args.overwrite,
            report_every=args.report_every,
            token=token,
        ),
        materialize_split(
            dataset_id=args.dataset_id,
            split=args.val_split,
            output_root=output_root,
            output_name="val",
            max_items=args.max_val,
            streaming=args.streaming,
            overwrite=args.overwrite,
            report_every=args.report_every,
            token=token,
        ),
    ]

    manifest = {
        "dataset_id": args.dataset_id,
        "output_root": str(output_root),
        "streaming": args.streaming,
        "splits": results,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
