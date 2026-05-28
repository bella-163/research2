from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .lora_kv import LoRAKVQKVLinear, enable_all_current, iter_lora_modules, set_all_old_scales
from .losses import masked_cross_entropy


def _collect_covariances(
    model: nn.Module,
    loader,
    device: torch.device,
    max_batches: int,
    max_rows_per_batch: int,
    desc: str,
) -> Dict[str, torch.Tensor]:
    covs: Dict[str, torch.Tensor] = {}
    counts = defaultdict(int)
    handles = []

    def make_hook(name: str):
        def hook(module, inputs):
            x = inputs[0].detach()
            x = x.reshape(-1, x.shape[-1]).float().cpu()
            if max_rows_per_batch and x.shape[0] > max_rows_per_batch:
                x = x[:max_rows_per_batch]
            if name not in covs:
                covs[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32)
            covs[name].add_(x.t().mm(x))
            counts[name] += x.shape[0]
        return hook

    for name, module in iter_lora_modules(model):
        handles.append(module.register_forward_pre_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(tqdm(loader, desc=desc, leave=False)):
            if i >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            _ = model(x)

    for h in handles:
        h.remove()

    for name in list(covs.keys()):
        covs[name] /= max(1, counts[name])
    return covs


def covariance_to_basis(cov: torch.Tensor, energy: float = 0.95, max_rank: int = 64) -> torch.Tensor:
    cov = cov.float()
    # Make it symmetric for numerical stability.
    cov = (cov + cov.t()) * 0.5
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order].clamp_min(0)
    eigvecs = eigvecs[:, order]
    total = eigvals.sum().item()
    if total <= 0:
        rank = min(max_rank, eigvecs.shape[1])
    else:
        cumsum = torch.cumsum(eigvals, dim=0) / total
        rank = int(torch.searchsorted(cumsum, torch.tensor(float(energy))).item() + 1)
        rank = max(1, min(rank, max_rank, eigvecs.shape[1]))
    return eigvecs[:, :rank].contiguous()


def build_drs_projectors(
    model: nn.Module,
    loader,
    device: torch.device,
    cfg: Dict,
    gammas: Dict[str, float],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    method = cfg["method"]
    max_batches = int(method.get("drs_batches", 8))
    max_rows = int(method.get("drs_max_rows_per_batch", 4096))
    energy = float(method.get("drs_energy", 0.95))
    max_rank = int(method.get("drs_max_rank", 64))

    # Subtracted model: W_sub = W0 - gamma_l * V_old. Current LoRA disabled.
    scales = {name: -float(gammas.get(name, 1.0)) for name, _ in iter_lora_modules(model)}
    set_all_old_scales(model, scales)
    enable_all_current(model, False)
    covs = _collect_covariances(model, loader, device, max_batches, max_rows, desc="Build DRS")
    set_all_old_scales(model, 1.0)
    enable_all_current(model, True)

    bases: Dict[str, torch.Tensor] = {}
    ranks: Dict[str, int] = {}
    for name, cov in covs.items():
        basis = covariance_to_basis(cov, energy=energy, max_rank=max_rank)
        bases[name] = basis
        ranks[name] = int(basis.shape[1])
    return bases, ranks


def project_lora_gradients(model: nn.Module, projectors: Dict[str, torch.Tensor]) -> None:
    for name, module in iter_lora_modules(model):
        if name not in projectors:
            continue
        basis = projectors[name].to(module.A_k.device, dtype=module.A_k.dtype)
        for A in [module.A_k, module.A_v]:
            if A.grad is not None:
                A.grad.data = (A.grad.data @ basis) @ basis.t()


@torch.no_grad()
def project_lora_weights(model: nn.Module, projectors: Dict[str, torch.Tensor]) -> None:
    for name, module in iter_lora_modules(model):
        if name not in projectors:
            continue
        basis = projectors[name].to(module.A_k.device, dtype=module.A_k.dtype)
        for A in [module.A_k, module.A_v]:
            A.data = (A.data @ basis) @ basis.t()


def compute_adaptive_gammas(
    model: nn.Module,
    loader,
    device: torch.device,
    cfg: Dict,
    seen_classes: List[int],
) -> Dict[str, float]:
    method = cfg["method"]
    rho = float(method.get("rho", 1.0))
    gamma_min = float(method.get("gamma_min", 0.5))
    gamma_max = float(method.get("gamma_max", 1.5))
    max_batches = int(method.get("grad_batches", 4))

    gammas = {name: 1.0 for name, _ in iter_lora_modules(model)}
    # If no old LoRA exists, gamma is irrelevant.
    if not any(m.has_old_delta() for _, m in iter_lora_modules(model)):
        return gammas

    model.train()
    set_all_old_scales(model, 1.0)
    enable_all_current(model, True)
    model.zero_grad(set_to_none=True)

    for i, (x, y) in enumerate(tqdm(loader, desc="Estimate adaptive gamma", leave=False)):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = masked_cross_entropy(logits, y, seen_classes)
        loss.backward()

    for name, module in iter_lora_modules(model):
        old = module.old_delta_flat().detach()
        if old.norm().item() == 0:
            gammas[name] = 1.0
            continue
        grad_delta = module.dense_grad_delta().detach()
        # Update direction is negative gradient.
        update_dir = -grad_delta
        denom = update_dir.norm() * old.norm() + 1e-12
        sim = float(torch.dot(update_dir.flatten(), old.flatten()) / denom)
        gamma = 1.0 - rho * sim
        gamma = max(gamma_min, min(gamma_max, gamma))
        gammas[name] = float(gamma)

    model.zero_grad(set_to_none=True)
    return gammas
