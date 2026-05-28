from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import timm

from .lora_kv import LoRAKVQKVLinear, iter_lora_modules, lora_trainable_parameters


class ContinualViT(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(feature_dim, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        if isinstance(z, (tuple, list)):
            z = z[0]
        if z.ndim == 3:
            z = z[:, 0]
        return z

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.features(x)
        logits = self.classifier(feat)
        if return_features:
            return logits, feat
        return logits


def _create_timm_model(model_cfg: Dict) -> nn.Module:
    names = [model_cfg["name"]] + list(model_cfg.get("fallback_names", []) or [])
    pretrained = bool(model_cfg.get("pretrained", True))
    allow_random = bool(model_cfg.get("allow_random_fallback", False))
    last_err = None
    for name in names:
        try:
            return timm.create_model(name, pretrained=pretrained, num_classes=0)
        except Exception as e:
            last_err = e
            continue
    if allow_random:
        return timm.create_model(model_cfg["name"], pretrained=False, num_classes=0)
    raise RuntimeError(f"Cannot create pretrained timm model from names={names}. Last error: {last_err}")


def inject_kv_lora(backbone: nn.Module, rank: int, alpha: int, dropout: float) -> int:
    count = 0
    if not hasattr(backbone, "blocks"):
        raise ValueError("Expected timm ViT model with .blocks")
    for block in backbone.blocks:
        if hasattr(block, "attn") and hasattr(block.attn, "qkv"):
            qkv = block.attn.qkv
            if isinstance(qkv, nn.Linear):
                block.attn.qkv = LoRAKVQKVLinear(qkv, rank=rank, alpha=alpha, dropout=dropout)
                count += 1
    if count == 0:
        raise RuntimeError("No attention qkv Linear modules were wrapped by LoRAKVQKVLinear")
    return count


def build_model(cfg: Dict) -> ContinualViT:
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    backbone = _create_timm_model(model_cfg)
    # Freeze everything first.
    for p in backbone.parameters():
        p.requires_grad_(False)
    inject_kv_lora(
        backbone,
        rank=int(lora_cfg.get("rank", 8)),
        alpha=int(lora_cfg.get("alpha", 16)),
        dropout=float(lora_cfg.get("dropout", 0.0)),
    )
    # Infer feature dimension.
    feature_dim = getattr(backbone, "num_features", None)
    if feature_dim is None:
        raise RuntimeError("Cannot infer backbone.num_features")
    model = ContinualViT(backbone, feature_dim=feature_dim, num_classes=int(model_cfg.get("num_classes", 100)))
    return model


def trainable_parameters_for_task(model: ContinualViT):
    # Current LoRA + classifier. Classifier rows are masked manually during training.
    params = list(lora_trainable_parameters(model)) + list(model.classifier.parameters())
    return [p for p in params if p.requires_grad]


def zero_classifier_grad_except(model: ContinualViT, current_classes: List[int]) -> None:
    cls = model.classifier
    keep = torch.zeros(cls.weight.shape[0], dtype=torch.bool, device=cls.weight.device)
    keep[torch.tensor(current_classes, device=cls.weight.device)] = True
    if cls.weight.grad is not None:
        cls.weight.grad[~keep] = 0
    if cls.bias is not None and cls.bias.grad is not None:
        cls.bias.grad[~keep] = 0


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
