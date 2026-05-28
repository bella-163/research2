# Ada-LoRA-DRS on CIFAR-100

這個專案用來跑 **CIFAR-100 Class-Incremental Learning** 實驗，基準方法是 **LoRA−DRS**，並在它的 DRS construction 階段加入我們的方法：**Adaptive LoRA Subtraction**。

簡單講：

```text
LoRA−DRS：
    W_sub = W0 - V_old

Ada-LoRA−DRS：
    W_sub = W0 - gamma_l * V_old
```

其中 `gamma_l` 會根據目前 task gradient 與舊 LoRA direction 的 alignment，對每一層自動調整 subtraction strength。後面的 DRS projection 訓練流程維持 LoRA−DRS 的精神：**subtraction 只用來建立 Drift-Resistant Space，不直接拿來當訓練模型**。

---

## 1. 目前支援的實驗設定

| 項目 | 設定 |
|---|---|
| Dataset | CIFAR-100 |
| Backbone | Pre-trained ViT-B/16-IN21K |
| 訓練方式 | Frozen backbone + LoRA |
| LoRA 插入位置 | ViT attention module 的 Key / Value |
| Optimizer | Adam |
| Batch size | 128 |
| Image size | 224 × 224 |
| Normalize | ToTensor，pixel value scale 到 `[0, 1]` |
| 重複實驗 | seeds `0,1,2,3,4`，最後取 mean/std |
| 主方法 | `lora_drs` vs `ada_lora_drs` |
| 額外 baseline | `lora_ft` |

建議主表先看：

```text
LoRA-FT
LoRA−DRS
Ada-LoRA−DRS
```

真正的論文故事重點是：

```text
LoRA−DRS  vs  Ada-LoRA−DRS
```

---

## 2. 檔案結構

```text
.
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   ├── cifar100_quick.yaml
│   ├── cifar100_10x10_vitb16_in21k.yaml
│   └── cifar100_50x2_vitb16_in21k.yaml
├── scripts/
│   ├── check_timm_models.py
│   ├── run_quick.sh
│   ├── run_cifar100_10x10.sh
│   ├── run_cifar100_50x2.sh
│   └── summarize_results.py
├── src/
│   └── adadrs/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── drs.py
│       ├── lora_kv.py
│       ├── losses.py
│       ├── metrics.py
│       ├── models.py
│       ├── train_cil.py
│       └── utils.py
└── tests/
    └── test_lora_kv.py
```

---

## 3. 安裝環境

建議使用新的 Python 環境：

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
pip install -e .
```

確認 LoRA wrapper 基本測試：

```bash
python -m pytest -q
```

---

## 4. 確認 timm 裡有哪些 ViT-B/16-IN21K checkpoint

不同 timm 版本的 model name 可能不一樣。先跑：

```bash
python scripts/check_timm_models.py
```

config 預設使用：

```yaml
model:
  name: vit_base_patch16_224.augreg_in21k
```

如果你的 timm 找不到這個名字，請把 config 裡的 `model.name` 改成 `check_timm_models.py` 印出來的可用名稱，例如：

```yaml
model:
  name: vit_base_patch16_224.orig_in21k
```

或執行時直接覆寫：

```bash
python -m adadrs.train_cil \
  --config configs/cifar100_quick.yaml \
  --work-dir outputs/test \
  --method ada_lora_drs \
  --set model.name=vit_base_patch16_224.orig_in21k
```

---

## 5. 快速實驗

快速實驗只用少量 CIFAR-100 samples 和較少 epochs，目的是確認：

```text
1. pretrained ViT 可以正常載入
2. K/V LoRA 有成功插入
3. LoRA−DRS projector 有建立
4. Ada-LoRA−DRS gamma 有正常輸出
5. 訓練流程不會炸掉
```

執行：

```bash
bash scripts/run_quick.sh
```

輸出位置：

```text
outputs/quick/
```

整理結果：

```bash
python scripts/summarize_results.py --root outputs/quick
```

---

## 6. 完整實驗：CIFAR-100 10 tasks × 10 classes

這是標準 CIFAR-100 CIL setting：

```text
100 classes
10 tasks
10 classes per task
5 seeds
```

執行：

```bash
bash scripts/run_cifar100_10x10.sh
```

輸出位置：

```text
outputs/cifar100_10x10_vitb16_in21k/
```

整理結果：

```bash
python scripts/summarize_results.py --root outputs/cifar100_10x10_vitb16_in21k
```

---

## 7. 完整實驗：CIFAR-100 50 tasks × 2 classes

這是 long sequence setting，比較適合觀察 feature drift control 是否有效：

```text
100 classes
50 tasks
2 classes per task
5 seeds
```

執行：

```bash
bash scripts/run_cifar100_50x2.sh
```

輸出位置：

```text
outputs/cifar100_50x2_vitb16_in21k/
```

整理結果：

```bash
python scripts/summarize_results.py --root outputs/cifar100_50x2_vitb16_in21k
```

---

## 8. 單次手動執行方式

例如只跑 seed 0 的 Ada-LoRA−DRS：

```bash
CUDA_VISIBLE_DEVICES=0 python -m adadrs.train_cil \
  --config configs/cifar100_10x10_vitb16_in21k.yaml \
  --work-dir outputs/manual_seed0_ada \
  --method ada_lora_drs \
  --seed 0
```

跑 LoRA−DRS baseline：

```bash
CUDA_VISIBLE_DEVICES=0 python -m adadrs.train_cil \
  --config configs/cifar100_10x10_vitb16_in21k.yaml \
  --work-dir outputs/manual_seed0_drs \
  --method lora_drs \
  --seed 0
```

跑 LoRA-FT baseline：

```bash
CUDA_VISIBLE_DEVICES=0 python -m adadrs.train_cil \
  --config configs/cifar100_10x10_vitb16_in21k.yaml \
  --work-dir outputs/manual_seed0_loraft \
  --method lora_ft \
  --seed 0
```

---

## 9. 方法流程

每個 task 的流程如下：

```text
For task t:

1. 載入 frozen pretrained ViT-B/16-IN21K
2. 在 attention qkv 中只對 K/V 插入 LoRA
3. 凍結 backbone 和舊 LoRA，只訓練 current-task LoRA 與 classifier current rows
4. 若 t > 0：
   4.1 LoRA−DRS：使用 W_sub = W0 - V_old
   4.2 Ada-LoRA−DRS：使用 W_sub = W0 - gamma_l * V_old
   4.3 用 W_sub 對目前 task data 做 forward，收集各 LoRA layer 的 input covariance
   4.4 對 covariance 做 eigendecomposition，建立 DRS basis
5. 訓練 current task LoRA：
   5.1 forward 使用 W0 + V_old + current_lora
   5.2 backward 後，將 LoRA A 的 gradient 投影到 DRS
   5.3 optimizer step 後，再將 LoRA A weight 投影回 DRS
   5.4 使用 CE + optional ATL
6. task 結束後：
   6.1 將 current LoRA merge 到 frozen cumulative old LoRA buffer
   6.2 reset current LoRA
   6.3 計算目前 task prototypes
   6.4 評估所有已見 classes
```

### LoRA−DRS

```text
gamma_l = 1.0
W_sub = W0 - V_old
```

### Ada-LoRA−DRS

```text
s_l = cosine(update_direction_l, old_lora_direction_l)
gamma_l = clip(1 - rho * s_l, gamma_min, gamma_max)
W_sub = W0 - gamma_l * V_old
```

直覺：

| 情況 | gamma | 意義 |
|---|---:|---|
| 新 task update direction 與舊 LoRA direction 相近 | < 1 | 少扣，保留可轉移方向 |
| 新 task update direction 與舊 LoRA direction 衝突 | > 1 | 多扣，降低干擾 |
| 方向接近正交 | ≈ 1 | 退化回 LoRA−DRS |

---

## 10. 輸出檔案

每個 run 的資料夾會包含：

```text
metrics.csv
accuracy_matrix.npy
config_resolved.yaml
gammas_task_*.json
checkpoint_last.pt
```

### metrics.csv 欄位

| 欄位 | 意義 |
|---|---|
| stage | 第幾個 incremental task |
| seen_classes | 目前看過的 class 數 |
| average_accuracy | 到目前為止所有 seen tasks 的平均 accuracy |
| final_accuracy | 所有 seen classes pooled accuracy |
| forgetting | old task forgetting |
| feature_drift | 舊類別 prototype 在 evaluation set 上的平均漂移 |
| train_loss | 該 task 最後 epoch 的平均訓練 loss |
| ce_loss | CE loss |
| atl_loss | ATL loss |

### gammas_task_*.json

這是 Ada-LoRA−DRS 最重要的 debug 檔：

```json
{
  "task": 3,
  "method": "ada_lora_drs",
  "gammas": {
    "backbone.blocks.0.attn.qkv": 0.93,
    "backbone.blocks.1.attn.qkv": 1.12
  },
  "drs": {
    "ranks": {
      "backbone.blocks.0.attn.qkv": 32
    }
  }
}
```

如果 gamma 全部都是 1.0，代表 adaptive 幾乎退化成 LoRA−DRS。這時可以調大：

```yaml
method:
  rho: 1.5
  gamma_min: 0.3
  gamma_max: 1.7
```

---

## 11. 常見問題

### Q1：ViT-B/16-IN21K 太吃 GPU，OOM 怎麼辦？

先降低 batch size：

```bash
python -m adadrs.train_cil \
  --config configs/cifar100_10x10_vitb16_in21k.yaml \
  --work-dir outputs/oom_test \
  --method ada_lora_drs \
  --seed 0 \
  --set data.batch_size=64 method.drs_max_rows_per_batch=2048
```

還是 OOM 的話，先用 quick config debug，或暫時改小模型：

```bash
--set model.name=vit_small_patch16_224.augreg_in21k
```

正式論文主實驗仍建議回到 ViT-B/16-IN21K。

### Q2：pretrained model 下載失敗怎麼辦？

`timm` 第一次會下載 checkpoint，需要網路。若環境不能上網，請先在有網路的機器下載 HuggingFace / timm cache，再搬到伺服器。

### Q3：為什麼還要 5 seeds？這不是吃運氣嗎？

不是挑最好 seed，而是固定 seeds `0,1,2,3,4`，所有方法共用同樣 class order，最後回報 mean/std。這是為了降低單一 class order 和 initialization 的偶然性。不要只報最好看的 seed，不然就會變成「模型抽卡遊戲」，教授可能會召喚鐵拳。

### Q4：這份 code 是官方 LoRA−DRS 嗎？

不是官方 repo 的逐字重現，而是根據 LoRA−DRS 的兩階段概念做的 PyTorch/timm 實驗實作：

```text
Stage 1: LoRA subtraction 建立 DRS
Stage 2: 在 DRS 中投影 current LoRA 更新
```

我們額外加入：

```text
Adaptive layer-wise subtraction strength gamma_l
```

若要做正式論文最終數字，建議後續也和 LoRA−DRS 官方 repo 對齊超參數與結果。

---

## 12. 建議實驗順序

```text
Step 1: bash scripts/run_quick.sh
        確認流程可跑、gamma/projector 正常輸出。

Step 2: bash scripts/run_cifar100_10x10.sh
        跑標準 CIFAR-100 10-task setting。

Step 3: bash scripts/run_cifar100_50x2.sh
        跑 long sequence setting，這是方法最可能發揮的地方。

Step 4: 比較 lora_drs vs ada_lora_drs
        看 average accuracy、forgetting、feature drift。

Step 5: 若 adaptive drift 低但 accuracy 低
        調 drs_energy / drs_max_rank，避免 DRS 太保守。
```

---

## 13. 最重要的比較

正式結果不要只看 `lora_ft`。你的主要研究問題是：

```text
LoRA−DRS 的 fixed subtraction 是否有 over-subtraction / under-subtraction 問題？
Ada-LoRA−DRS 能不能讓 DRS 更 task-aware / layer-aware？
```

所以主表應該聚焦：

```text
LoRA−DRS
Ada-LoRA−DRS
```

如果 Ada-LoRA−DRS 在 50×2 long sequence 下有更低 drift 或更高 average accuracy，這個故事就比較站得住。
