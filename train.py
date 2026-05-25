from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from mamba2_vision.checkpoint import save_checkpoint
from mamba2_vision.data import build_dataloaders
from mamba2_vision.model import build_image_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Mamba image classifier.")
    parser.add_argument("--dataset", choices=["cifar10", "imagefolder"], required=True)
    parser.add_argument("--data-dir", default="/workspace/data")
    parser.add_argument("--train-dir")
    parser.add_argument("--val-dir")
    parser.add_argument("--output-dir", default="/workspace/outputs/run")
    parser.add_argument("--model-arch", choices=["vim", "mamba2"], default="vim")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--quick-limit", type=int)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--progress-mininterval", type=float, default=1.0)
    return parser.parse_args()


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def gpu_memory_gib(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.memory_allocated(device) / 1024**3


def optimizer_lr(optimizer: torch.optim.Optimizer | None) -> float | None:
    if optimizer is None:
        return None
    return float(optimizer.param_groups[0]["lr"])


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
    epoch: int,
    epochs: int,
    progress: bool,
    progress_mininterval: float,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_acc = 0.0
    total_items = 0
    phase = "train" if training else "val"
    started = time.perf_counter()

    progress_bar = tqdm(
        loader,
        desc=f"{phase} {epoch}/{epochs}",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
        disable=not progress,
        mininterval=progress_mininterval,
    )

    for images, targets in progress_bar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)

        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        batch = images.size(0)
        total_loss += loss.item() * batch
        total_acc += accuracy(logits.detach(), targets) * batch
        total_items += batch

        elapsed = max(time.perf_counter() - started, 1e-9)
        postfix = {
            "batch_loss": f"{loss.item():.4f}",
            "loss": f"{total_loss / total_items:.4f}",
            "acc": f"{total_acc / total_items:.4f}",
            "img/s": f"{total_items / elapsed:.1f}",
        }
        lr = optimizer_lr(optimizer)
        if lr is not None:
            postfix["lr"] = f"{lr:.2e}"
        if device.type == "cuda":
            postfix["gpu_gb"] = f"{gpu_memory_gib(device):.1f}"
        progress_bar.set_postfix(postfix)

    return total_loss / total_items, total_acc / total_items


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("CUDA not visible. Use Docker with NVIDIA runtime.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, class_names = build_dataloaders(
        dataset=args.dataset,
        image_size=args.image_size,
        batch_size=args.batch_size,
        workers=args.workers,
        data_dir=args.data_dir,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        seed=args.seed,
        quick_limit=args.quick_limit,
    )

    model = build_image_model(
        args.model_arch,
        image_size=args.image_size,
        patch_size=args.patch_size,
        in_chans=3,
        num_classes=len(class_names),
        d_model=args.d_model,
        depth=args.depth,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        headdim=args.headdim,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            args.amp,
            epoch,
            args.epochs,
            not args.no_progress,
            args.progress_mininterval,
        )
        with torch.no_grad():
            val_loss, val_acc = run_epoch(
                model,
                val_loader,
                criterion,
                None,
                scaler,
                device,
                args.amp,
                epoch,
                args.epochs,
                not args.no_progress,
                args.progress_mininterval,
            )

        metrics = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, metrics, class_names)
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, metrics, class_names)

    print(f"done best_val_acc={best_acc:.4f} output_dir={output_dir}")


if __name__ == "__main__":
    main()
