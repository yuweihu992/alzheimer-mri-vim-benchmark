from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_mamba2():
    try:
        from mamba_ssm import Mamba2
    except Exception as exc:  # pragma: no cover - depends on CUDA extension install
        raise RuntimeError(
            "mamba-ssm with Mamba2 is required. Build and run through Docker: "
            "docker compose run --rm mamba2-vision python tools/smoke_test.py"
        ) from exc
    return Mamba2


@dataclass(frozen=True)
class VisionMamba2Config:
    image_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    num_classes: int = 1000
    d_model: int = 192
    depth: int = 6
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    dropout: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VimMamba2Config:
    image_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    num_classes: int = 1000
    d_model: int = 192
    depth: int = 6
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    dropout: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Mamba2VisionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        headdim: int,
        dropout: float,
        reverse: bool,
    ) -> None:
        super().__init__()
        Mamba2 = _load_mamba2()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )
        self.dropout = nn.Dropout(dropout)
        self.reverse = reverse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm(x)
        if self.reverse:
            y = y.flip(dims=(1,))
        y = self.mixer(y)
        if self.reverse:
            y = y.flip(dims=(1,))
        return residual + self.dropout(y)


class VimMamba2Block(nn.Module):
    """Vim-style bidirectional Mamba block for visual patch sequences."""

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        headdim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        Mamba2 = _load_mamba2()
        self.norm = nn.LayerNorm(d_model)
        self.forward_mixer = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )
        self.backward_mixer = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm(x)
        forward = self.forward_mixer(y)
        backward = self.backward_mixer(y.flip(dims=(1,))).flip(dims=(1,))
        return residual + self.dropout(0.5 * (forward + backward))


class VisionMamba2Classifier(nn.Module):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.config = VisionMamba2Config(**kwargs)
        cfg = self.config

        if cfg.image_size % cfg.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.patch_embed = nn.Conv2d(
            cfg.in_chans,
            cfg.d_model,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )
        self.grid_size = cfg.image_size // cfg.patch_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.grid_size * self.grid_size, cfg.d_model)
        )
        self.blocks = nn.ModuleList(
            [
                Mamba2VisionBlock(
                    d_model=cfg.d_model,
                    d_state=cfg.d_state,
                    d_conv=cfg.d_conv,
                    expand=cfg.expand,
                    headdim=cfg.headdim,
                    dropout=cfg.dropout,
                    reverse=bool(i % 2),
                )
                for i in range(cfg.depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def _position_embedding(self, height: int, width: int) -> torch.Tensor:
        if height == self.grid_size and width == self.grid_size:
            return self.pos_embed
        pos = self.pos_embed.reshape(
            1, self.grid_size, self.grid_size, self.config.d_model
        ).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(height, width), mode="bicubic", align_corners=False)
        return pos.permute(0, 2, 3, 1).reshape(1, height * width, self.config.d_model)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        height, width = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = x + self._position_embedding(height, width).to(dtype=x.dtype, device=x.device)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class VimMamba2Classifier(nn.Module):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.config = VimMamba2Config(**kwargs)
        cfg = self.config

        if cfg.image_size % cfg.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.patch_embed = nn.Conv2d(
            cfg.in_chans,
            cfg.d_model,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )
        self.grid_size = cfg.image_size // cfg.patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.grid_size * self.grid_size + 1, cfg.d_model)
        )
        self.blocks = nn.ModuleList(
            [
                VimMamba2Block(
                    d_model=cfg.d_model,
                    d_state=cfg.d_state,
                    d_conv=cfg.d_conv,
                    expand=cfg.expand,
                    headdim=cfg.headdim,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def _position_embedding(self, height: int, width: int) -> torch.Tensor:
        if height == self.grid_size and width == self.grid_size:
            return self.pos_embed

        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:].reshape(
            1, self.grid_size, self.grid_size, self.config.d_model
        ).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(
            1, height * width, self.config.d_model
        )
        return torch.cat((cls_pos, patch_pos), dim=1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        height, width = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self._position_embedding(height, width).to(dtype=x.dtype, device=x.device)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def build_image_model(model_arch: str, **kwargs: Any) -> nn.Module:
    if model_arch == "vim":
        return VimMamba2Classifier(**kwargs)
    if model_arch == "mamba2":
        return VisionMamba2Classifier(**kwargs)
    raise ValueError(f"unsupported model_arch: {model_arch}")
