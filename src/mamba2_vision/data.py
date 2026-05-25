from __future__ import annotations

from pathlib import Path
from inspect import signature

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


def train_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def eval_transform(image_size: int):
    resize_size = int(round(image_size / 0.875))
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _limit(dataset: Dataset, max_items: int | None) -> Dataset:
    if max_items is None or max_items <= 0:
        return dataset
    return Subset(dataset, range(min(max_items, len(dataset))))


def image_folder(path: Path, transform):
    kwargs = {}
    if "allow_empty" in signature(datasets.ImageFolder).parameters:
        kwargs["allow_empty"] = True
    return datasets.ImageFolder(str(path), transform=transform, **kwargs)


def build_dataloaders(
    dataset: str,
    image_size: int,
    batch_size: int,
    workers: int,
    data_dir: str | None = None,
    train_dir: str | None = None,
    val_dir: str | None = None,
    val_fraction: float = 0.1,
    seed: int = 1337,
    quick_limit: int | None = None,
) -> tuple[DataLoader, DataLoader, list[str]]:
    pin_memory = torch.cuda.is_available()

    if dataset == "cifar10":
        root = Path(data_dir or "/workspace/data")
        train_ds = datasets.CIFAR10(
            root=str(root),
            train=True,
            download=True,
            transform=train_transform(image_size),
        )
        val_ds = datasets.CIFAR10(
            root=str(root),
            train=False,
            download=True,
            transform=eval_transform(image_size),
        )
        class_names = list(train_ds.classes)
    elif dataset == "imagefolder":
        if train_dir is None:
            raise ValueError("--train-dir is required for imagefolder")
        train_path = Path(train_dir)
        if not train_path.exists():
            raise FileNotFoundError(f"train directory not found: {train_path}")
        train_full = image_folder(train_path, transform=train_transform(image_size))
        class_names = list(train_full.classes)
        if val_dir:
            val_path = Path(val_dir)
            if not val_path.exists():
                raise FileNotFoundError(f"val directory not found: {val_path}")
            val_ds = image_folder(val_path, transform=eval_transform(image_size))
            train_ds = train_full
        else:
            val_len = max(1, int(len(train_full) * val_fraction))
            train_len = len(train_full) - val_len
            if train_len <= 0:
                raise ValueError("not enough images to split train/val")
            train_ds, val_ds = random_split(
                train_full,
                [train_len, val_len],
                generator=torch.Generator().manual_seed(seed),
            )
    else:
        raise ValueError(f"unsupported dataset: {dataset}")

    train_ds = _limit(train_ds, quick_limit)
    val_ds = _limit(val_ds, quick_limit)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, class_names
