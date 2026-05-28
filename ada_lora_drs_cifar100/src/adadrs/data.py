from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .utils import seed_worker


@dataclass
class IncrementalData:
    train_loaders: List[DataLoader]
    test_loaders: List[DataLoader]
    task_classes: List[List[int]]
    class_order: List[int]


class ClassSubset(Dataset):
    def __init__(self, base: Dataset, indices: Sequence[int]):
        self.base = base
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base[self.indices[idx]]


def _targets(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)
    raise AttributeError("Dataset has no targets attribute")


def make_class_order(num_classes: int, seed: int, mode: str = "random") -> List[int]:
    if mode == "original":
        return list(range(num_classes))
    if mode != "random":
        raise ValueError(f"Unsupported class_order mode: {mode}")
    rng = np.random.default_rng(seed)
    return rng.permutation(num_classes).tolist()


def select_indices(dataset: Dataset, classes: Sequence[int], max_per_class: int | None, seed: int) -> List[int]:
    targets = _targets(dataset)
    rng = np.random.default_rng(seed)
    indices: List[int] = []
    for c in classes:
        cls_idx = np.where(targets == c)[0]
        if max_per_class is not None and len(cls_idx) > max_per_class:
            cls_idx = rng.choice(cls_idx, size=max_per_class, replace=False)
        indices.extend(cls_idx.tolist())
    rng.shuffle(indices)
    return indices


def build_transforms(image_size: int, train: bool = True):
    # 根據使用者設定：pixel value scale 到 [0,1]，不做 ImageNet mean/std normalization。
    ts = [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    return transforms.Compose(ts)


def build_cifar100_incremental(cfg: Dict, seed: int) -> IncrementalData:
    data_cfg = cfg["data"]
    root = data_cfg.get("root", "./data")
    image_size = int(data_cfg.get("image_size", 224))
    batch_size = int(data_cfg.get("batch_size", 128))
    num_workers = int(data_cfg.get("num_workers", 4))
    num_classes = int(data_cfg.get("num_classes", 100))
    num_tasks = int(data_cfg.get("num_tasks", 10))
    classes_per_task = int(data_cfg.get("classes_per_task", num_classes // num_tasks))
    max_train = data_cfg.get("max_train_per_class", None)
    max_test = data_cfg.get("max_test_per_class", None)
    if max_train is not None:
        max_train = int(max_train)
    if max_test is not None:
        max_test = int(max_test)

    train_base = datasets.CIFAR100(root=root, train=True, download=True, transform=build_transforms(image_size, train=True))
    test_base = datasets.CIFAR100(root=root, train=False, download=True, transform=build_transforms(image_size, train=False))

    order = make_class_order(num_classes, seed=seed, mode=data_cfg.get("class_order", "random"))
    task_classes = [order[i * classes_per_task:(i + 1) * classes_per_task] for i in range(num_tasks)]

    train_loaders: List[DataLoader] = []
    test_loaders: List[DataLoader] = []
    generator = torch.Generator()
    generator.manual_seed(seed)

    for tid, classes in enumerate(task_classes):
        train_indices = select_indices(train_base, classes, max_train, seed + tid * 101)
        test_indices = select_indices(test_base, classes, max_test, seed + tid * 101 + 999)
        train_ds = ClassSubset(train_base, train_indices)
        test_ds = ClassSubset(test_base, test_indices)
        train_loaders.append(DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, worker_init_fn=seed_worker, generator=generator,
        ))
        test_loaders.append(DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=True, worker_init_fn=seed_worker,
        ))

    return IncrementalData(train_loaders, test_loaders, task_classes, order)
