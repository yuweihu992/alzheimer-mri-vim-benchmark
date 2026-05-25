from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny ImageFolder demo dataset.")
    parser.add_argument("--output", default="/workspace/data/imagefolder-demo")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--train-per-class", type=int, default=16)
    parser.add_argument("--val-per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def make_image(path: Path, image_size: int, base_color: tuple[int, int, int], seed: int) -> None:
    rng = random.Random(seed)
    image = Image.new("RGB", (image_size, image_size), base_color)
    draw = ImageDraw.Draw(image)

    for _ in range(12):
        x0 = rng.randint(0, image_size - 8)
        y0 = rng.randint(0, image_size - 8)
        x1 = min(image_size - 1, x0 + rng.randint(4, 18))
        y1 = min(image_size - 1, y0 + rng.randint(4, 18))
        fill = tuple(max(0, min(255, c + rng.randint(-45, 45))) for c in base_color)
        draw.rectangle((x0, y0, x1, y1), fill=fill)

    image.save(path)


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)

    classes = {
        "red": (220, 40, 40),
        "green": (40, 180, 90),
    }
    for split, count in (("train", args.train_per_class), ("val", args.val_per_class)):
        for class_name, color in classes.items():
            class_dir = output / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                make_image(
                    class_dir / f"{class_name}_{index:03d}.png",
                    args.image_size,
                    color,
                    args.seed + index + (0 if split == "train" else 1000),
                )

    print(f"created={output}")
    print(f"layout={output}/train/<class>/*.png and {output}/val/<class>/*.png")


if __name__ == "__main__":
    main()
