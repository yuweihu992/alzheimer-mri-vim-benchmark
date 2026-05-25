from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .model import VimMamba2Classifier, VisionMamba2Classifier, build_image_model


ImageClassifier = VisionMamba2Classifier | VimMamba2Classifier


def save_checkpoint(
    path: str | Path,
    model: ImageClassifier,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict[str, float],
    class_names: list[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_arch": "vim" if isinstance(model, VimMamba2Classifier) else "mamba2",
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "metrics": metrics,
        "class_names": class_names,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_model(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[ImageClassifier, list[str], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    model = build_image_model(checkpoint.get("model_arch", "mamba2"), **checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint["class_names"], checkpoint
