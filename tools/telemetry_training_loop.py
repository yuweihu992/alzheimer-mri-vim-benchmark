from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pynvml
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm


@dataclass(slots=True)
class StepMetrics:
    timestamp: float
    epoch: int
    step: int
    global_step: int
    loss: float
    lr: float
    grad_norm: float
    tokens_per_s: float
    gpu_temp_c: int
    gpu_power_w: float
    vram_used_gb: float
    vram_total_gb: float


class NvmlTelemetry:
    def __init__(self, gpu_index: int = 0) -> None:
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    def sample(self) -> tuple[int, float, float, float]:
        temp_c = pynvml.nvmlDeviceGetTemperature(
            self.handle,
            pynvml.NVML_TEMPERATURE_GPU,
        )
        power_w = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return temp_c, power_w, memory.used / 1024**3, memory.total / 1024**3

    def close(self) -> None:
        pynvml.nvmlShutdown()


class MetricsCsvWriter:
    fieldnames = [
        "timestamp",
        "epoch",
        "step",
        "global_step",
        "loss",
        "lr",
        "grad_norm",
        "tokens_per_s",
        "gpu_temp_c",
        "gpu_power_w",
        "vram_used_gb",
        "vram_total_gb",
    ]

    def __init__(self, path: Path, flush_every: int) -> None:
        self.path = path
        self.flush_every = max(1, flush_every)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.pending = 0

    def write(self, metrics: StepMetrics) -> None:
        self.writer.writerow(asdict(metrics))
        self.pending += 1
        if self.pending >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        self.handle.flush()
        self.pending = 0

    def close(self) -> None:
        self.flush()
        self.handle.close()


class TinySequenceModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_classes: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        x = self.norm(x).mean(dim=1)
        return self.head(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production-style telemetry training loop demo.")
    parser.add_argument("--output-dir", default="/workspace/outputs/telemetry-loop")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--csv-flush-every", type=int, default=10)
    parser.add_argument("--progress-mininterval", type=float, default=0.5)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="bfloat16")
    return parser.parse_args()


def progress_postfix(metrics: StepMetrics) -> str:
    return (
        f"tok/s={metrics.tokens_per_s:,.0f} | "
        f"GPU={metrics.gpu_temp_c}C {metrics.gpu_power_w:.0f}W "
        f"{metrics.vram_used_gb:.1f}/{metrics.vram_total_gb:.1f}GB | "
        f"loss={metrics.loss:.4f} lr={metrics.lr:.2e} grad={metrics.grad_norm:.2f}"
    )


def amp_dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "float16" else torch.bfloat16


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this telemetry demo.")

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda", args.gpu_index)
    output_dir = Path(args.output_dir)
    csv_writer = MetricsCsvWriter(output_dir / "metrics_history.csv", args.csv_flush_every)
    telemetry = NvmlTelemetry(args.gpu_index)

    model = TinySequenceModel(args.vocab_size, args.d_model, args.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    dtype = amp_dtype(args.amp_dtype)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.amp and dtype == torch.float16,
    )

    total_steps = args.epochs * args.steps_per_epoch
    global_step = 0
    bar_format = (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
    )

    progress = tqdm(
        total=total_steps,
        desc="train",
        dynamic_ncols=True,
        mininterval=args.progress_mininterval,
        bar_format=bar_format,
        leave=True,
    )

    try:
        for epoch in range(1, args.epochs + 1):
            for step in range(1, args.steps_per_epoch + 1):
                global_step += 1
                start = time.perf_counter()

                tokens = torch.randint(
                    0,
                    args.vocab_size,
                    (args.batch_size, args.seq_len),
                    device=device,
                )
                targets = torch.randint(0, args.num_classes, (args.batch_size,), device=device)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=args.amp, dtype=dtype):
                    logits = model(tokens)
                    loss = criterion(logits, targets)

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm_tensor = clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    grad_norm_tensor = clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                torch.cuda.synchronize(device)
                elapsed = max(time.perf_counter() - start, 1e-9)
                tokens_per_s = args.batch_size * args.seq_len / elapsed
                grad_norm = float(grad_norm_tensor)
                if not math.isfinite(grad_norm):
                    grad_norm = float("nan")
                gpu_temp_c, gpu_power_w, vram_used_gb, vram_total_gb = telemetry.sample()

                metrics = StepMetrics(
                    timestamp=time.time(),
                    epoch=epoch,
                    step=step,
                    global_step=global_step,
                    loss=float(loss.detach().item()),
                    lr=float(optimizer.param_groups[0]["lr"]),
                    grad_norm=grad_norm,
                    tokens_per_s=tokens_per_s,
                    gpu_temp_c=gpu_temp_c,
                    gpu_power_w=gpu_power_w,
                    vram_used_gb=vram_used_gb,
                    vram_total_gb=vram_total_gb,
                )
                csv_writer.write(metrics)
                progress.set_postfix_str(progress_postfix(metrics), refresh=False)
                progress.update(1)
    finally:
        progress.close()
        telemetry.close()
        csv_writer.close()

    print(f"metrics_csv={output_dir / 'metrics_history.csv'}")


if __name__ == "__main__":
    main()
