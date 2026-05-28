from __future__ import annotations

import math
from typing import Dict, Iterable, Iterator, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAKVQKVLinear(nn.Module):
    """Wrap timm ViT qkv Linear and add LoRA only to K and V slices.

    Original qkv shape: in_dim -> 3 * dim.
    LoRA updates are applied to the K slice [dim:2*dim] and V slice [2*dim:3*dim].
    The old task LoRA is stored as dense frozen buffers for exact cumulative subtraction.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if base.out_features % 3 != 0:
            raise ValueError("qkv Linear out_features must be divisible by 3")
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.dim = base.out_features // 3
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for p in self.base.parameters():
            p.requires_grad_(False)

        self.A_k = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.B_k = nn.Parameter(torch.empty(self.dim, self.rank))
        self.A_v = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.B_v = nn.Parameter(torch.empty(self.dim, self.rank))

        self.register_buffer("old_delta_k", torch.zeros(self.dim, self.in_features))
        self.register_buffer("old_delta_v", torch.zeros(self.dim, self.in_features))

        self.old_scale = 1.0
        self.current_enabled = True
        self.reset_current_parameters()

    def reset_current_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.A_k, a=math.sqrt(5))
        nn.init.zeros_(self.B_k)
        nn.init.kaiming_uniform_(self.A_v, a=math.sqrt(5))
        nn.init.zeros_(self.B_v)

    def set_old_scale(self, scale: float) -> None:
        self.old_scale = float(scale)

    def enable_current(self, flag: bool = True) -> None:
        self.current_enabled = bool(flag)

    def _current_update(self, x: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # F.linear uses weight [out_features, in_features].
        return F.linear(F.linear(self.dropout(x), A), B) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.base(x)

        if self.old_scale != 0.0:
            k_old = F.linear(x, self.old_delta_k * self.old_scale)
            v_old = F.linear(x, self.old_delta_v * self.old_scale)
            qkv = qkv.clone()
            qkv[..., self.dim:2 * self.dim] += k_old
            qkv[..., 2 * self.dim:3 * self.dim] += v_old

        if self.current_enabled:
            k = self._current_update(x, self.A_k, self.B_k)
            v = self._current_update(x, self.A_v, self.B_v)
            qkv = qkv.clone()
            qkv[..., self.dim:2 * self.dim] += k
            qkv[..., 2 * self.dim:3 * self.dim] += v
        return qkv

    @torch.no_grad()
    def current_delta_k(self) -> torch.Tensor:
        return (self.B_k @ self.A_k) * self.scaling

    @torch.no_grad()
    def current_delta_v(self) -> torch.Tensor:
        return (self.B_v @ self.A_v) * self.scaling

    @torch.no_grad()
    def merge_current_into_old(self) -> None:
        self.old_delta_k.add_(self.current_delta_k())
        self.old_delta_v.add_(self.current_delta_v())
        self.reset_current_parameters()

    @torch.no_grad()
    def has_old_delta(self) -> bool:
        return bool(self.old_delta_k.abs().sum().item() > 0 or self.old_delta_v.abs().sum().item() > 0)

    def dense_grad_delta(self) -> torch.Tensor:
        """Approximate gradient direction of dense LoRA delta for adaptive gamma."""
        parts = []
        for A, B in [(self.A_k, self.B_k), (self.A_v, self.B_v)]:
            g = None
            if B.grad is not None:
                g = B.grad.detach() @ A.detach()
            if A.grad is not None:
                term = B.detach() @ A.grad.detach()
                g = term if g is None else g + term
            if g is None:
                g = torch.zeros_like(B.detach() @ A.detach())
            parts.append(g * self.scaling)
        return torch.cat([p.flatten() for p in parts])

    def old_delta_flat(self) -> torch.Tensor:
        return torch.cat([self.old_delta_k.flatten(), self.old_delta_v.flatten()])


def iter_lora_modules(model: nn.Module) -> Iterator[Tuple[str, LoRAKVQKVLinear]]:
    for name, module in model.named_modules():
        if isinstance(module, LoRAKVQKVLinear):
            yield name, module


def set_all_old_scales(model: nn.Module, scales: Dict[str, float] | float) -> None:
    for name, module in iter_lora_modules(model):
        scale = scales[name] if isinstance(scales, dict) and name in scales else scales
        module.set_old_scale(float(scale))


def enable_all_current(model: nn.Module, flag: bool) -> None:
    for _, module in iter_lora_modules(model):
        module.enable_current(flag)


def reset_all_current_lora(model: nn.Module) -> None:
    for _, module in iter_lora_modules(model):
        module.reset_current_parameters()


def merge_all_current_into_old(model: nn.Module) -> None:
    for _, module in iter_lora_modules(model):
        module.merge_current_into_old()


def lora_trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for _, module in iter_lora_modules(model):
        yield module.A_k
        yield module.B_k
        yield module.A_v
        yield module.B_v
