from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from tqdm import tqdm

from mamba2_vision.model import build_image_model


@dataclass
class BenchmarkResult:
    model: str
    status: str
    params_m: float
    batch_size: int
    image_size: int
    steps: int
    amp_dtype: str
    seconds: float
    img_s: float
    step_s: float
    loss: float
    peak_allocated_gb: float
    peak_reserved_gb: float
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Mamba-2 vision vs mainstream vision models.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mamba2_current", "vit_l_16"],
        help="Models to test: vim_current, mamba2_current, vit_b_16, vit_l_16, vit_h_14, convnext_large.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--mamba-patch-size", type=int, default=16)
    parser.add_argument("--mamba-d-model", type=int, default=192)
    parser.add_argument("--mamba-depth", type=int, default=6)
    parser.add_argument("--mamba-d-state", type=int, default=64)
    parser.add_argument("--mamba-headdim", type=int, default=64)
    return parser.parse_args()


def count_params(model: nn.Module) -> float:
    return sum(parameter.numel() for parameter in model.parameters()) / 1_000_000


def build_torchvision_model(name: str, image_size: int, num_classes: int) -> nn.Module:
    import torchvision.models as models

    if not hasattr(models, name):
        available = sorted(item for item in dir(models) if item.startswith(("vit_", "convnext_")))
        raise ValueError(f"torchvision model not found: {name}. Available: {available}")

    kwargs = {"weights": None, "num_classes": num_classes}
    if name.startswith("vit_"):
        kwargs["image_size"] = image_size
    return getattr(models, name)(**kwargs)


def build_model(name: str, args: argparse.Namespace) -> nn.Module:
    if name in {"mamba2_current", "vim_current"}:
        arch = "vim" if name == "vim_current" else "mamba2"
        return build_image_model(
            arch,
            image_size=args.image_size,
            patch_size=args.mamba_patch_size,
            in_chans=3,
            num_classes=args.num_classes,
            d_model=args.mamba_d_model,
            depth=args.mamba_depth,
            d_state=args.mamba_d_state,
            headdim=args.mamba_headdim,
        )
    return build_torchvision_model(name, args.image_size, args.num_classes)


def amp_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float16


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def train_step(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    amp: bool,
    dtype: torch.dtype,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=amp, dtype=dtype):
        logits = model(images)
        loss = criterion(logits, targets)

    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    return float(loss.detach().item())


def benchmark_one(name: str, args: argparse.Namespace, device: torch.device) -> BenchmarkResult:
    cleanup()
    dtype = amp_dtype(args.amp_dtype)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and dtype == torch.float16,
    )

    try:
        model = build_model(name, args).to(device)
        model.train()
        params_m = count_params(model)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        images = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
        targets = torch.randint(0, args.num_classes, (args.batch_size,), device=device)

        for _ in range(args.warmup):
            train_step(model, images, targets, criterion, optimizer, scaler, amp_enabled, dtype)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        started = time.perf_counter()
        last_loss = 0.0
        progress = tqdm(range(args.steps), desc=name, unit="step", dynamic_ncols=True)
        for _ in progress:
            last_loss = train_step(model, images, targets, criterion, optimizer, scaler, amp_enabled, dtype)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = max(time.perf_counter() - started, 1e-9)
            progress.set_postfix(
                {
                    "loss": f"{last_loss:.4f}",
                    "img/s": f"{args.batch_size * (_ + 1) / elapsed:.1f}",
                    "gpu_gb": f"{torch.cuda.max_memory_allocated() / 1024**3:.1f}" if device.type == "cuda" else "0.0",
                }
            )

        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        peak_allocated = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3 if device.type == "cuda" else 0.0
        return BenchmarkResult(
            model=name,
            status="ok",
            params_m=params_m,
            batch_size=args.batch_size,
            image_size=args.image_size,
            steps=args.steps,
            amp_dtype=args.amp_dtype if amp_enabled else "none",
            seconds=seconds,
            img_s=args.batch_size * args.steps / seconds,
            step_s=seconds / args.steps,
            loss=last_loss,
            peak_allocated_gb=peak_allocated,
            peak_reserved_gb=peak_reserved,
        )
    except torch.cuda.OutOfMemoryError as exc:
        return BenchmarkResult(
            model=name,
            status="oom",
            params_m=0.0,
            batch_size=args.batch_size,
            image_size=args.image_size,
            steps=args.steps,
            amp_dtype=args.amp_dtype if amp_enabled else "none",
            seconds=0.0,
            img_s=0.0,
            step_s=0.0,
            loss=0.0,
            peak_allocated_gb=torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0,
            peak_reserved_gb=torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0,
            error=str(exc).splitlines()[0],
        )
    finally:
        cleanup()


def print_table(results: list[BenchmarkResult]) -> None:
    headers = [
        "model",
        "status",
        "params_m",
        "batch",
        "img_s",
        "step_s",
        "peak_alloc_gb",
        "peak_reserved_gb",
    ]
    print("\t".join(headers))
    for result in results:
        print(
            "\t".join(
                [
                    result.model,
                    result.status,
                    f"{result.params_m:.1f}",
                    str(result.batch_size),
                    f"{result.img_s:.1f}",
                    f"{result.step_s:.3f}",
                    f"{result.peak_allocated_gb:.1f}",
                    f"{result.peak_reserved_gb:.1f}",
                ]
            )
        )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not visible.")

    device = torch.device("cuda")
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(
        f"batch_size={args.batch_size} image_size={args.image_size} "
        f"steps={args.steps} warmup={args.warmup} amp={args.amp} amp_dtype={args.amp_dtype}"
    )

    results = [benchmark_one(name, args, device) for name in args.models]
    print_table(results)
    for result in results:
        print(json.dumps(asdict(result), sort_keys=True))


if __name__ == "__main__":
    main()
