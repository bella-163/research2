#!/usr/bin/env bash
set -euo pipefail

ROOT="outputs/cifar100_10x10_vitb16_in21k"
mkdir -p "${ROOT}/logs"

for SEED in 0 1 2 3 4; do
  for METHOD in lora_ft lora_drs ada_lora_drs; do
    echo "Running seed=${SEED}, method=${METHOD} on CIFAR-100 10x10..."
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python -m adadrs.train_cil \
      --config configs/cifar100_10x10_vitb16_in21k.yaml \
      --work-dir "${ROOT}/s${SEED}_${METHOD}" \
      --method "${METHOD}" \
      --seed "${SEED}" \
      2>&1 | tee "${ROOT}/logs/s${SEED}_${METHOD}.log"
  done
done

python scripts/summarize_results.py --root "${ROOT}"
