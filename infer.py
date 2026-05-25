from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from mamba2_vision.checkpoint import load_model
from mamba2_vision.data import eval_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mamba-2 image inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("CUDA not visible. Use Docker with NVIDIA runtime.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, checkpoint = load_model(args.checkpoint, map_location=device)
    model.to(device).eval()

    image_size = int(checkpoint["model_config"]["image_size"])
    transform = eval_transform(image_size)
    image = Image.open(Path(args.image)).convert("RGB")
    batch = transform(image).unsqueeze(0).to(device)

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        probs = model(batch).softmax(dim=1).squeeze(0)

    k = min(args.top_k, probs.numel())
    values, indices = probs.topk(k)
    for rank, (score, index) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        print(f"{rank}\t{class_names[index]}\t{score:.6f}")


if __name__ == "__main__":
    main()

