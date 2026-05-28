#!/usr/bin/env bash
set -euo pipefail

ROOT="outputs/quick"
mkdir -p "${ROOT}/logs"

for METHOD in lora_ft lora_drs ada_lora_drs; do
  echo "Running ${METHOD} quick experiment..."
  python -m adadrs.train_cil \
    --config configs/cifar100_quick.yaml \
    --work-dir "${ROOT}/s0_${METHOD}" \
    --method "${METHOD}" \
    --seed 0 \
    2>&1 | tee "${ROOT}/logs/s0_${METHOD}.log"
done

python scripts/summarize_results.py --root "${ROOT}"
