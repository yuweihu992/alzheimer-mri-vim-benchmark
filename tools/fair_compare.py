from __future__ import annotations

import argparse
import csv
import gc
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pynvml
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from mamba2_vision.data import build_dataloaders
from mamba2_vision.model import build_image_model


COLORS = {
    "mamba2": (32, 99, 223),
    "vit": (214, 88, 35),
    "train": (32, 99, 223),
    "val": (214, 88, 35),
    "grid": (222, 226, 232),
    "text": (25, 28, 35),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mamba2 then ViT under the same protocol and generate plots.")
    parser.add_argument("--dataset", choices=["cifar10", "imagefolder"], required=True)
    parser.add_argument("--data-dir", default="/workspace/data")
    parser.add_argument("--train-dir")
    parser.add_argument("--val-dir")
    parser.add_argument("--output-dir", default="/workspace/outputs/fair-compare")
    parser.add_argument("--reset-output-dir", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--quick-limit", type=int)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--progress-mininterval", type=float, default=1.0)
    parser.add_argument("--vit-model", default="vit_b_16")
    parser.add_argument("--vit-pretrained", action="store_true")
    parser.add_argument("--vit-freeze-backbone", action="store_true")
    parser.add_argument("--telemetry-csv", default="metrics_history.csv")
    parser.add_argument("--csv-flush-every", type=int, default=25)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--mamba-patch-size", type=int, default=16)
    parser.add_argument("--mamba-arch", choices=["vim", "mamba2"], default="vim")
    parser.add_argument("--mamba-d-model", type=int, default=384)
    parser.add_argument("--mamba-depth", type=int, default=12)
    parser.add_argument("--mamba-d-state", type=int, default=64)
    parser.add_argument("--mamba-headdim", type=int, default=64)
    parser.add_argument("--mamba-expand", type=int, default=2)
    return parser.parse_args()


def font(size: int = 16) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def params_m(model: nn.Module) -> float:
    return sum(parameter.numel() for parameter in model.parameters()) / 1_000_000


def trainable_params_m(model: nn.Module) -> float:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) / 1_000_000


def replace_vit_head(model: nn.Module, num_classes: int) -> None:
    if not hasattr(model, "heads") or not hasattr(model.heads, "head"):
        raise ValueError("unsupported ViT head layout")
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)


def freeze_vit_backbone(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.heads.parameters():
        parameter.requires_grad = True


def build_vit(
    name: str,
    image_size: int,
    num_classes: int,
    pretrained: bool,
    freeze_backbone: bool,
) -> nn.Module:
    import torchvision.models as models
    from torchvision.models.vision_transformer import interpolate_embeddings

    if not hasattr(models, name):
        raise ValueError(f"torchvision model not found: {name}")
    if pretrained:
        try:
            weights_enum = models.get_model_weights(name)
        except Exception as exc:
            raise ValueError(f"pretrained weights unavailable for {name}") from exc

        weights = weights_enum.DEFAULT
        kwargs = {"weights": None, "num_classes": num_classes}
        if name.startswith("vit_"):
            kwargs["image_size"] = image_size
        model = getattr(models, name)(**kwargs)

        state_dict = weights.get_state_dict(progress=True, check_hash=True)
        if name.startswith("vit_"):
            state_dict = interpolate_embeddings(
                image_size=image_size,
                patch_size=vit_patch_size(name),
                model_state=state_dict,
                reset_heads=True,
            )
        model.load_state_dict(state_dict, strict=False)
        replace_vit_head(model, num_classes)
    else:
        kwargs = {"weights": None, "num_classes": num_classes}
        if name.startswith("vit_"):
            kwargs["image_size"] = image_size
        model = getattr(models, name)(**kwargs)
    if freeze_backbone:
        freeze_vit_backbone(model)
    return model


@dataclass(slots=True)
class StepMetrics:
    timestamp: float
    model: str
    phase: str
    epoch: int
    step: int
    global_step: int
    loss: float
    acc: float
    lr: float
    grad_norm: float
    images_per_s: float
    tokens_per_s: float
    gpu_temp_c: int
    gpu_power_w: float
    vram_used_gb: float
    vram_total_gb: float


class NvmlTelemetry:
    def __init__(self, gpu_index: int) -> None:
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    def sample(self) -> tuple[int, float, float, float]:
        temp_c = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
        power_w = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return temp_c, power_w, memory.used / 1024**3, memory.total / 1024**3

    def close(self) -> None:
        pynvml.nvmlShutdown()


class MetricsHistoryWriter:
    fieldnames = list(StepMetrics.__dataclass_fields__.keys())

    def __init__(self, path: Path, flush_every: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.flush_every = max(1, flush_every)
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


def build_mamba(args: argparse.Namespace, num_classes: int) -> nn.Module:
    return build_image_model(
        args.mamba_arch,
        image_size=args.image_size,
        patch_size=args.mamba_patch_size,
        in_chans=3,
        num_classes=num_classes,
        d_model=args.mamba_d_model,
        depth=args.mamba_depth,
        d_state=args.mamba_d_state,
        headdim=args.mamba_headdim,
        expand=args.mamba_expand,
    )


def vit_patch_size(name: str) -> int:
    try:
        return int(name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 16


def tokens_per_image(model_name: str, args: argparse.Namespace) -> int:
    patch_size = args.mamba_patch_size if model_name in {"mamba2", "vim"} else vit_patch_size(model_name)
    return (args.image_size // patch_size) ** 2


def amp_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float16


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
    dtype: torch.dtype,
    model_name: str,
    epoch: int,
    epochs: int,
    mininterval: float,
    max_grad_norm: float,
    tokens_per_image: int,
    telemetry: NvmlTelemetry,
    history: MetricsHistoryWriter,
    global_step: int,
) -> tuple[float, float, float, list[int], list[int], int]:
    training = optimizer is not None
    model.train(training)
    phase = "train" if training else "val"
    total_loss = 0.0
    total_acc = 0.0
    total_items = 0
    preds_all: list[int] = []
    targets_all: list[int] = []
    started = time.perf_counter()

    progress = tqdm(
        loader,
        desc=f"{model_name} {phase} {epoch}/{epochs}",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
        mininterval=mininterval,
    )
    for batch_index, (images, targets) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp, dtype=dtype):
            logits = model(images)
            loss = criterion(logits, targets)

        grad_norm = 0.0
        if training:
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = float(clip_grad_norm_(model.parameters(), max_grad_norm))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = float(clip_grad_norm_(model.parameters(), max_grad_norm))
                optimizer.step()

        batch = images.size(0)
        acc = accuracy(logits.detach(), targets)
        total_loss += float(loss.detach().item()) * batch
        total_acc += acc * batch
        total_items += batch
        if not training:
            preds_all.extend(logits.argmax(dim=1).detach().cpu().tolist())
            targets_all.extend(targets.detach().cpu().tolist())

        elapsed = max(time.perf_counter() - started, 1e-9)
        images_per_s = total_items / elapsed
        tokens_per_s = images_per_s * tokens_per_image
        gpu_temp_c, gpu_power_w, vram_used_gb, vram_total_gb = telemetry.sample()
        lr = float(optimizer.param_groups[0]["lr"]) if optimizer is not None else 0.0
        global_step += 1
        step_metrics = StepMetrics(
            timestamp=time.time(),
            model=model_name,
            phase=phase,
            epoch=epoch,
            step=batch_index,
            global_step=global_step,
            loss=total_loss / total_items,
            acc=total_acc / total_items,
            lr=lr,
            grad_norm=grad_norm,
            images_per_s=images_per_s,
            tokens_per_s=tokens_per_s,
            gpu_temp_c=gpu_temp_c,
            gpu_power_w=gpu_power_w,
            vram_used_gb=vram_used_gb,
            vram_total_gb=vram_total_gb,
        )
        history.write(step_metrics)
        progress.set_postfix_str(
            f"tok/s={tokens_per_s:,.0f} | GPU={gpu_temp_c}C {gpu_power_w:.0f}W "
            f"{vram_used_gb:.1f}/{vram_total_gb:.1f}GB | "
            f"loss={total_loss / total_items:.4f} acc={total_acc / total_items:.4f} "
            f"lr={lr:.2e} grad={grad_norm:.2f}",
            refresh=False,
        )

    elapsed = max(time.perf_counter() - started, 1e-9)
    return total_loss / total_items, total_acc / total_items, total_items / elapsed, preds_all, targets_all, global_step


def confusion_matrix(preds: list[int], targets: list[int], classes: int) -> list[list[int]]:
    matrix = [[0 for _ in range(classes)] for _ in range(classes)]
    for target, pred in zip(targets, preds):
        matrix[target][pred] += 1
    return matrix


def train_model(
    model_name: str,
    model: nn.Module,
    args: argparse.Namespace,
    class_names: list[str],
    device: torch.device,
    telemetry: NvmlTelemetry,
    history: MetricsHistoryWriter,
    global_step: int,
) -> tuple[list[dict[str, Any]], list[list[int]], dict[str, Any], int]:
    torch.manual_seed(args.seed)
    train_loader, val_loader, _ = build_dataloaders(
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

    cleanup()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    dtype = amp_dtype(args.amp_dtype)
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and dtype == torch.float16)
    rows: list[dict[str, Any]] = []
    best_val_acc = -1.0
    final_confusion: list[list[int]] = []
    output_dir = Path(args.output_dir)
    model_tokens_per_image = tokens_per_image(model_name, args)

    print(f"=== TRAIN {model_name} ===")
    print(f"params_m={params_m(model):.2f} trainable_params_m={trainable_params_m(model):.2f}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_img_s, _, _, global_step = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp_enabled,
            dtype,
            model_name,
            epoch,
            args.epochs,
            args.progress_mininterval,
            args.max_grad_norm,
            model_tokens_per_image,
            telemetry,
            history,
            global_step,
        )
        with torch.no_grad():
            val_loss, val_acc, val_img_s, preds, targets, global_step = run_epoch(
                model,
                val_loader,
                criterion,
                None,
                scaler,
                device,
                amp_enabled,
                dtype,
                model_name,
                epoch,
                args.epochs,
                args.progress_mininterval,
                args.max_grad_norm,
                model_tokens_per_image,
                telemetry,
                history,
                global_step,
            )
        final_confusion = confusion_matrix(preds, targets, len(class_names))
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
        row = {
            "model": model_name,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "train_img_s": train_img_s,
            "val_img_s": val_img_s,
            "peak_gpu_gb": peak_gb,
            "params_m": params_m(model),
            "trainable_params_m": trainable_params_m(model),
        }
        rows.append(row)
        print(
            f"model={model_name} epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"train_img_s={train_img_s:.1f} peak_gpu_gb={peak_gb:.1f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model": model_name,
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "epoch": epoch,
                    "metrics": row,
                },
                output_dir / f"{model_name}_best.pt",
            )

    summary = {
        "model": model_name,
        "params_m": params_m(model),
        "trainable_params_m": trainable_params_m(model),
        "best_val_acc": best_val_acc,
        "final_val_acc": rows[-1]["val_acc"],
        "final_val_loss": rows[-1]["val_loss"],
        "final_train_img_s": rows[-1]["train_img_s"],
        "peak_gpu_gb": max(row["peak_gpu_gb"] for row in rows),
    }
    cleanup()
    return rows, final_confusion, summary, global_step


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def scale(values: list[float], low: int, high: int) -> list[int]:
    v_min = min(values)
    v_max = max(values)
    if abs(v_max - v_min) < 1e-12:
        return [(low + high) // 2 for _ in values]
    return [int(high - (value - v_min) / (v_max - v_min) * (high - low)) for value in values]


def draw_line_chart(path: Path, title: str, series: dict[str, list[float]], ylabel: str) -> None:
    width, height = 1200, 700
    margin_left, margin_right, margin_top, margin_bottom = 90, 40, 80, 90
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(28)
    label_font = font(18)
    draw.text((margin_left, 25), title, fill=COLORS["text"], font=title_font)
    draw.text((margin_left, height - 50), "epoch", fill=COLORS["text"], font=label_font)
    draw.text((20, margin_top), ylabel, fill=COLORS["text"], font=label_font)

    x0, y0 = margin_left, margin_top
    x1, y1 = width - margin_right, height - margin_bottom
    draw.rectangle((x0, y0, x1, y1), outline=COLORS["grid"], width=2)
    for i in range(1, 5):
        y = y0 + (y1 - y0) * i // 5
        draw.line((x0, y, x1, y), fill=COLORS["grid"])

    all_values = [value for values in series.values() for value in values]
    y_values = scale(all_values, y0 + 10, y1 - 10)
    cursor = 0
    palette = [COLORS["mamba2"], COLORS["vit"], (70, 150, 80), (150, 70, 160)]
    for index, (name, values) in enumerate(series.items()):
        ys = y_values[cursor : cursor + len(values)]
        cursor += len(values)
        if len(values) == 1:
            xs = [(x0 + x1) // 2]
        else:
            xs = [x0 + int(i * (x1 - x0) / (len(values) - 1)) for i in range(len(values))]
        points = list(zip(xs, ys))
        color = palette[index % len(palette)]
        if len(points) > 1:
            draw.line(points, fill=color, width=4)
        for point in points:
            draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=color)
        draw.text((x0 + 20 + index * 230, y1 + 25), name, fill=color, font=label_font)
    image.save(path)


def draw_bar_chart(path: Path, title: str, summaries: list[dict[str, Any]]) -> None:
    width, height = 1200, 720
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(28)
    label_font = font(18)
    draw.text((70, 25), title, fill=COLORS["text"], font=title_font)
    metrics = [
        ("best_val_acc", "Best Val Acc"),
        ("final_train_img_s", "Train img/s"),
        ("peak_gpu_gb", "Peak GPU GB"),
        ("params_m", "Params M"),
    ]
    start_y = 100
    block_h = 135
    max_name_w = 180
    palette = [COLORS["mamba2"], COLORS["vit"]]
    for metric_index, (key, label) in enumerate(metrics):
        y = start_y + metric_index * block_h
        values = [float(summary[key]) for summary in summaries]
        max_value = max(values) or 1.0
        draw.text((70, y), label, fill=COLORS["text"], font=label_font)
        for index, summary in enumerate(summaries):
            bar_y = y + 35 + index * 42
            bar_w = int((values[index] / max_value) * 760)
            color = palette[index % len(palette)]
            draw.text((95, bar_y), summary["model"], fill=COLORS["text"], font=font(15))
            draw.rectangle((70 + max_name_w, bar_y, 70 + max_name_w + bar_w, bar_y + 25), fill=color)
            draw.text(
                (70 + max_name_w + bar_w + 12, bar_y),
                f"{values[index]:.3f}" if key == "best_val_acc" else f"{values[index]:.1f}",
                fill=COLORS["text"],
                font=font(15),
            )
    image.save(path)


def draw_confusion(path: Path, title: str, matrix: list[list[int]], class_names: list[str]) -> None:
    classes = len(class_names)
    cell = max(26, min(72, 720 // max(1, classes)))
    left = 210
    top = 110
    width = left + cell * classes + 80
    height = top + cell * classes + 160
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 30), title, fill=COLORS["text"], font=font(26))
    max_count = max((max(row) for row in matrix), default=1) or 1
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            intensity = int(255 - 190 * value / max_count)
            color = (intensity, intensity, 255)
            x0 = left + col_index * cell
            y0 = top + row_index * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=color, outline=(230, 230, 230))
            if classes <= 12:
                draw.text((x0 + 6, y0 + 5), str(value), fill=COLORS["text"], font=font(13))
    for index, name in enumerate(class_names):
        label = name[:18]
        draw.text((35, top + index * cell + 5), label, fill=COLORS["text"], font=font(14))
        if classes <= 20:
            draw.text((left + index * cell + 3, top + cell * classes + 8), label[:8], fill=COLORS["text"], font=font(12))
    draw.text((40, top + cell * classes + 65), "Rows=true label, columns=predicted label", fill=COLORS["text"], font=font(16))
    image.save(path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.reset_output_dir and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not visible.")
    device = torch.device("cuda")

    _, _, class_names = build_dataloaders(
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

    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"classes={len(class_names)} output_dir={output_dir}")
    print(f"order={args.mamba_arch} then vit")

    telemetry = NvmlTelemetry(args.gpu_index)
    history = MetricsHistoryWriter(output_dir / args.telemetry_csv, args.csv_flush_every)
    global_step = 0
    vit_run_name = f"{args.vit_model}_pretrained" if args.vit_pretrained else args.vit_model
    mamba_run_name = args.mamba_arch
    models = [
        (mamba_run_name, build_mamba(args, len(class_names))),
        (
            vit_run_name,
            build_vit(
                args.vit_model,
                args.image_size,
                len(class_names),
                args.vit_pretrained,
                args.vit_freeze_backbone,
            ),
        ),
    ]
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    confusion_by_model: dict[str, list[list[int]]] = {}

    try:
        for model_name, model in models:
            rows, matrix, summary, global_step = train_model(
                model_name,
                model,
                args,
                class_names,
                device,
                telemetry,
                history,
                global_step,
            )
            all_rows.extend(rows)
            summaries.append(summary)
            confusion_by_model[model_name] = matrix
    finally:
        history.close()
        telemetry.close()

    write_csv(output_dir / "metrics.csv", all_rows)
    (output_dir / "summary.json").write_text(
        json.dumps({"summaries": summaries, "args": vars(args)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    loss_series = {f"{row['model']} train": [] for row in all_rows if row["epoch"] == 1}
    loss_series.update({f"{row['model']} val": [] for row in all_rows if row["epoch"] == 1})
    acc_series = {f"{row['model']} train": [] for row in all_rows if row["epoch"] == 1}
    acc_series.update({f"{row['model']} val": [] for row in all_rows if row["epoch"] == 1})
    for row in all_rows:
        loss_series[f"{row['model']} train"].append(float(row["train_loss"]))
        loss_series[f"{row['model']} val"].append(float(row["val_loss"]))
        acc_series[f"{row['model']} train"].append(float(row["train_acc"]))
        acc_series[f"{row['model']} val"].append(float(row["val_acc"]))

    draw_line_chart(output_dir / "loss_trend.png", "Training and Validation Loss", loss_series, "loss")
    draw_line_chart(output_dir / "accuracy_trend.png", "Training and Validation Accuracy", acc_series, "accuracy")
    draw_bar_chart(output_dir / "throughput_gpu_params.png", "Throughput, GPU Memory, Parameters", summaries)
    for model_name, matrix in confusion_by_model.items():
        draw_confusion(output_dir / f"confusion_{model_name}.png", f"Confusion Matrix: {model_name}", matrix, class_names)

    print("artifacts:")
    for path in sorted(output_dir.glob("*")):
        print(path)


if __name__ == "__main__":
    main()
