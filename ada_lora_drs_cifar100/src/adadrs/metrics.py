from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def mask_unseen_logits(logits: torch.Tensor, seen_classes: List[int]) -> torch.Tensor:
    out = torch.full_like(logits, -1e9)
    idx = torch.tensor(seen_classes, device=logits.device, dtype=torch.long)
    out[:, idx] = logits[:, idx]
    return out


@torch.no_grad()
def evaluate_on_loader(model, loader, device: torch.device, seen_classes: List[int]) -> Tuple[float, int, int]:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        logits = mask_unseen_logits(logits, seen_classes)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / max(1, total), correct, total


@torch.no_grad()
def compute_prototypes(model, loader, device: torch.device, classes: Iterable[int]) -> Dict[int, torch.Tensor]:
    model.eval()
    classes = set(int(c) for c in classes)
    sums: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits, feat = model(x, return_features=True)
        feat = F.normalize(feat.detach(), dim=1).cpu()
        for f, label in zip(feat, y):
            c = int(label)
            if c not in classes:
                continue
            if c not in sums:
                sums[c] = torch.zeros_like(f)
                counts[c] = 0
            sums[c] += f
            counts[c] += 1
    return {c: F.normalize((sums[c] / max(1, counts[c])).unsqueeze(0), dim=1).squeeze(0) for c in sums}


def compute_forgetting(acc_matrix: np.ndarray, stage: int) -> float:
    if stage <= 0:
        return 0.0
    vals = []
    for t in range(stage):
        best = np.nanmax(acc_matrix[:stage + 1, t])
        final = acc_matrix[stage, t]
        vals.append(best - final)
    return float(np.nanmean(vals)) if vals else 0.0


def prototype_drift(prev: Dict[int, torch.Tensor], cur: Dict[int, torch.Tensor], old_classes: List[int]) -> float:
    vals = []
    for c in old_classes:
        if c in prev and c in cur:
            vals.append(float(torch.norm(prev[c] - cur[c], p=2).item()))
    return float(np.mean(vals)) if vals else 0.0
