from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F


def masked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, seen_classes: List[int]) -> torch.Tensor:
    seen = torch.tensor(seen_classes, device=logits.device, dtype=torch.long)
    sub_logits = logits.index_select(1, seen)
    # Map global labels to local indices.
    mapping = {int(c): i for i, c in enumerate(seen_classes)}
    local = torch.tensor([mapping[int(y)] for y in labels.detach().cpu().tolist()], device=logits.device)
    return F.cross_entropy(sub_logits, local)


def augmented_triplet_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    old_prototypes: Dict[int, torch.Tensor],
    margin: float = 0.5,
) -> torch.Tensor:
    """A lightweight ATL-style loss.

    It pulls current samples toward same-class batch centers and pushes them away from nearest
    old-class prototypes. This is an implementation-friendly proxy for the ATL component.
    """
    if not old_prototypes:
        return features.new_tensor(0.0)

    z = F.normalize(features, dim=1)
    labels_cpu = labels.detach().cpu().tolist()
    unique = sorted(set(labels_cpu))
    centers = {}
    for c in unique:
        mask = labels == int(c)
        centers[int(c)] = F.normalize(z[mask].mean(dim=0, keepdim=True), dim=1).squeeze(0).detach()

    positives = torch.stack([centers[int(y)] for y in labels_cpu], dim=0).to(z.device)

    old_cls = sorted(old_prototypes.keys())
    old_mat = torch.stack([old_prototypes[c].to(z.device) for c in old_cls], dim=0)
    old_mat = F.normalize(old_mat, dim=1)
    sim = z @ old_mat.t()
    nearest = sim.argmax(dim=1)
    negatives = old_mat[nearest]
    return F.triplet_margin_loss(z, positives, negatives, margin=margin, p=2)
