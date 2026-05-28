from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .config import apply_overrides, load_config, save_config
from .data import build_cifar100_incremental
from .drs import build_drs_projectors, compute_adaptive_gammas, project_lora_gradients, project_lora_weights
from .lora_kv import enable_all_current, iter_lora_modules, merge_all_current_into_old, reset_all_current_lora, set_all_old_scales
from .losses import augmented_triplet_loss, masked_cross_entropy
from .metrics import compute_forgetting, compute_prototypes, evaluate_on_loader, prototype_drift
from .models import build_model, count_trainable_params, trainable_parameters_for_task, zero_classifier_grad_except
from .utils import AverageMeter, get_device, set_seed, write_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--method", choices=["lora_ft", "lora_drs", "ada_lora_drs"], default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--set", nargs="*", default=[])
    return p.parse_args()


def configure_method(cfg: Dict, method_name: str) -> None:
    cfg["method"]["name"] = method_name
    if method_name == "lora_ft":
        cfg["method"]["use_drs"] = False
        cfg["method"]["use_atl"] = False
    elif method_name == "lora_drs":
        cfg["method"]["use_drs"] = True
        # LoRA−DRS base includes ATL by default.
        cfg["method"]["use_atl"] = True
    elif method_name == "ada_lora_drs":
        cfg["method"]["use_drs"] = True
        cfg["method"]["use_atl"] = True
    else:
        raise ValueError(method_name)


def maybe_autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type="cuda", enabled=(enabled and device.type == "cuda"))


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.set)
    if args.method is not None:
        configure_method(cfg, args.method)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    cfg["seed"] = seed

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, work_dir / "config_resolved.yaml")

    set_seed(seed, deterministic=bool(cfg["training"].get("deterministic", False)))
    device = get_device(str(cfg["training"].get("device", "cuda")))

    data = build_cifar100_incremental(cfg, seed=seed)
    model = build_model(cfg).to(device)
    print(f"Model built. Trainable params: {count_trainable_params(model):,}")
    print(f"LoRA modules: {sum(1 for _ in iter_lora_modules(model))}")
    print(f"Class order: {data.class_order}")

    epochs = int(cfg["training"].get("epochs_per_task", 20))
    lr = float(cfg["training"].get("lr", 1e-3))
    wd = float(cfg["training"].get("weight_decay", 0.0))
    amp = bool(cfg["training"].get("amp", True))
    grad_clip = cfg["training"].get("grad_clip", None)
    if grad_clip is not None:
        grad_clip = float(grad_clip)

    method = cfg["method"]["name"]
    use_drs = bool(cfg["method"].get("use_drs", method != "lora_ft"))
    use_atl = bool(cfg["method"].get("use_atl", method != "lora_ft"))
    atl_weight = float(cfg["method"].get("atl_weight", 0.1))
    atl_margin = float(cfg["method"].get("atl_margin", 0.5))

    num_tasks = len(data.task_classes)
    acc_matrix = np.full((num_tasks, num_tasks), np.nan, dtype=np.float32)
    records = []
    stored_train_prototypes: Dict[int, torch.Tensor] = {}
    prev_eval_prototypes: Dict[int, torch.Tensor] = {}

    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    for task_id, current_classes in enumerate(data.task_classes):
        print(f"\n=== Task {task_id + 1}/{num_tasks}: classes={current_classes} | method={method} ===")
        seen_classes = [c for cls in data.task_classes[:task_id + 1] for c in cls]
        old_classes = [c for cls in data.task_classes[:task_id] for c in cls]

        reset_all_current_lora(model)
        set_all_old_scales(model, 1.0)
        enable_all_current(model, True)

        train_loader = data.train_loaders[task_id]

        gammas = {name: 1.0 for name, _ in iter_lora_modules(model)}
        drs_ranks = {}
        projectors = {}
        if use_drs and task_id > 0:
            if method == "ada_lora_drs":
                gammas = compute_adaptive_gammas(model, train_loader, device, cfg, seen_classes=seen_classes)
            projectors, drs_ranks = build_drs_projectors(model, train_loader, device, cfg, gammas)

        write_json({
            "task": task_id,
            "method": method,
            "classes": current_classes,
            "gammas": gammas,
            "drs": {"ranks": drs_ranks},
        }, work_dir / f"gammas_task_{task_id:03d}.json")

        params = trainable_parameters_for_task(model)
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)

        final_train_loss = final_ce = final_atl = 0.0
        for epoch in range(epochs):
            model.train()
            loss_meter = AverageMeter()
            ce_meter = AverageMeter()
            atl_meter = AverageMeter()
            pbar = tqdm(train_loader, desc=f"Task {task_id} epoch {epoch + 1}/{epochs}", leave=False)
            for x, y in pbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with maybe_autocast(device, amp):
                    logits, feat = model(x, return_features=True)
                    ce = masked_cross_entropy(logits, y, seen_classes)
                    atl = augmented_triplet_loss(feat, y, stored_train_prototypes, margin=atl_margin) if use_atl else feat.new_tensor(0.0)
                    loss = ce + atl_weight * atl
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                zero_classifier_grad_except(model, current_classes)
                if use_drs and projectors:
                    project_lora_gradients(model, projectors)
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                scaler.step(optimizer)
                scaler.update()
                if use_drs and projectors and bool(cfg["method"].get("drs_project_after_step", True)):
                    project_lora_weights(model, projectors)

                bs = int(y.numel())
                loss_meter.update(float(loss.detach().cpu()), bs)
                ce_meter.update(float(ce.detach().cpu()), bs)
                atl_meter.update(float(atl.detach().cpu()), bs)
                pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", ce=f"{ce_meter.avg:.4f}", atl=f"{atl_meter.avg:.4f}")

            final_train_loss = loss_meter.avg
            final_ce = ce_meter.avg
            final_atl = atl_meter.avg

        # Store current task prototypes before merging reset does not change function after old_delta update.
        cur_train_proto = compute_prototypes(model, train_loader, device, current_classes)
        stored_train_prototypes.update({c: p.cpu() for c, p in cur_train_proto.items()})

        # Merge current task LoRA into frozen cumulative old LoRA.
        merge_all_current_into_old(model)
        enable_all_current(model, False)
        set_all_old_scales(model, 1.0)

        # Evaluation across seen tasks.
        task_accs = []
        pooled_correct = 0
        pooled_total = 0
        for eval_tid in range(task_id + 1):
            acc, correct, total = evaluate_on_loader(model, data.test_loaders[eval_tid], device, seen_classes)
            acc_matrix[task_id, eval_tid] = acc
            task_accs.append(acc)
            pooled_correct += correct
            pooled_total += total
        average_accuracy = float(np.mean(task_accs))
        final_accuracy = float(pooled_correct / max(1, pooled_total))
        forgetting = compute_forgetting(acc_matrix, task_id)

        # Metric-only prototype drift on test data.
        cur_eval_prototypes: Dict[int, torch.Tensor] = {}
        for eval_tid in range(task_id + 1):
            cls = data.task_classes[eval_tid]
            cur_eval_prototypes.update(compute_prototypes(model, data.test_loaders[eval_tid], device, cls))
        drift = prototype_drift(prev_eval_prototypes, cur_eval_prototypes, old_classes)
        prev_eval_prototypes = {c: p.cpu() for c, p in cur_eval_prototypes.items()}

        rec = {
            "stage": task_id,
            "seen_classes": len(seen_classes),
            "method": method,
            "seed": seed,
            "average_accuracy": average_accuracy,
            "final_accuracy": final_accuracy,
            "forgetting": forgetting,
            "feature_drift": drift,
            "train_loss": final_train_loss,
            "ce_loss": final_ce,
            "atl_loss": final_atl,
        }
        records.append(rec)
        pd.DataFrame(records).to_csv(work_dir / "metrics.csv", index=False)
        np.save(work_dir / "accuracy_matrix.npy", acc_matrix)
        print(f"Task {task_id}: avg_acc={average_accuracy:.4f}, final_acc={final_accuracy:.4f}, forgetting={forgetting:.4f}, drift={drift:.4f}")

    if bool(cfg.get("eval", {}).get("save_checkpoint", True)):
        torch.save({
            "model": model.state_dict(),
            "config": cfg,
            "acc_matrix": acc_matrix,
        }, work_dir / "checkpoint_last.pt")

    print(f"\nDone. Results saved to: {work_dir}")


if __name__ == "__main__":
    main()
