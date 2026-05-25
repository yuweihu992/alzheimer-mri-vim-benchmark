from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.utils import save_image

from mamba2_vision.checkpoint import load_model, save_checkpoint
from mamba2_vision.model import build_image_model


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not visible inside container.")

    device = torch.device("cuda")
    out_dir = Path("/workspace/outputs/smoke")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"torch={torch.__version__}")
    print(f"cuda={torch.version.cuda}")
    print(f"gpu={torch.cuda.get_device_name(0)}")

    images = torch.randn(4, 3, 32, 32, device=device)
    targets = torch.tensor([0, 1, 2, 1], device=device)
    save_image(images[0].detach().cpu().clamp(-1, 1) * 0.5 + 0.5, out_dir / "sample.png")

    for arch in ("mamba2", "vim"):
        model = build_image_model(
            arch,
            image_size=32,
            patch_size=4,
            in_chans=3,
            num_classes=3,
            d_model=64,
            depth=1,
            d_state=16,
            d_conv=4,
            expand=2,
            headdim=32,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        checkpoint_path = out_dir / f"smoke_{arch}.pt"
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch=1,
            metrics={"loss": float(loss.item())},
            class_names=["class_0", "class_1", "class_2"],
        )

        reloaded, class_names, _ = load_model(checkpoint_path, map_location=device)
        reloaded.to(device).eval()
        with torch.no_grad():
            probs = reloaded(images[:1]).softmax(dim=1).squeeze(0)
        top_score, top_index = probs.max(dim=0)

        print(f"arch={arch} loss={loss.item():.6f}")
        print(f"arch={arch} reloaded_top1={class_names[int(top_index)]} score={float(top_score):.6f}")
        print(f"arch={arch} checkpoint={checkpoint_path}")
    print(f"sample_image={out_dir / 'sample.png'}")
    print("smoke_test=passed")


if __name__ == "__main__":
    main()
