# POC-5.9-v2: Scripts Reference

Active scripts for production training and evaluation.

## 📁 Scripts Overview

### Core Training Scripts

| Script | Purpose | Usage | Duration |
|--------|---------|-------|----------|
| **slurm_test.sh** | Quick 1-epoch test | `sbatch slurm_test.sh` | ~1 min |
| **slurm_train.sh** | Single model training (50 epochs) | `sbatch slurm_train.sh configs/convnext_tiny.yaml` | ~15 min |
| **slurm_train_fold.sh** | K-fold CV training | `sbatch slurm_train_fold.sh configs/convnext_tiny_shm.yaml --fold 0` | ~15 min |

### Shared Memory Scripts

| Script | Purpose | Usage | Duration |
|--------|---------|-------|----------|
| **slurm_preload_shm.sh** | Pre-load dataset to /dev/shm | `sbatch slurm_preload_shm.sh` | ~4 sec |
| **launch_cv_shm.sh** | Launch 3-fold CV with shared memory | `bash launch_cv_shm.sh configs/convnext_tiny_shm.yaml` | ~47 min |

### Utilities

| Script | Purpose | Usage |
|--------|---------|-------|
| **preload_shared_dataset.py** | Copy dataset to /dev/shm | `python preload_shared_dataset.py --data-dir ../common-data` |
| **evaluate_all.sh** | Evaluate all trained models | `bash evaluate_all.sh` |

---

## 🚀 Quick Start

### 1. Test Single Model (Fastest)

```bash
cd /opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc59-v2

# Test 1 epoch
sbatch scripts/slurm_test.sh
```

**Expected:**
- Throughput: ~80 imgs/s
- VRAM: 0.51GB (1.6%)
- Time: ~1 min

### 2. Train Single Model (No CV)

```bash
# Train ConvNeXt for 50 epochs (random split)
sbatch scripts/slurm_train.sh configs/convnext_tiny.yaml
```

**Expected:**
- mIoU: ~28-29%
- Time: ~15 min
- Output: `logs/Unet_tu-convnext_tiny/best_model.pth`

### 3. Train with 3-Fold Cross-Validation (Production)

**Option A: With Shared Memory (RECOMMENDED)**

```bash
# Preload dataset once
sbatch scripts/slurm_preload_shm.sh

# Wait ~4 seconds, then launch CV
bash scripts/launch_cv_shm.sh configs/convnext_tiny_shm.yaml
```

**Total time:** ~47 min (3 min preload + 3×15 min training)

**Option B: Without Shared Memory (Slower)**

```bash
# Launch manually
for fold in 0 1 2; do
    sbatch scripts/slurm_train_fold.sh configs/convnext_tiny.yaml --fold $fold
done
```

**Total time:** ~135 min (30 min preload × 3 + 3×15 min training)

### 4. Full Benchmark (3 Encoders × 3 Folds = 9 Jobs)

```bash
# Preload once
sbatch scripts/slurm_preload_shm.sh

# Launch all 9 jobs
for encoder in convnext_tiny segformer_b3 maxvit_tiny; do
    bash scripts/launch_cv_shm.sh configs/${encoder}_shm.yaml
done
```

**Total time:** ~2.5 hours (3 min preload + 9×15 min parallel)

---

## 📊 Monitoring

### Check Job Status

```bash
squeue -u $USER
```

### Monitor Training Output

```bash
# Find job ID from squeue
tail -f logs/train_fold_<JOB_ID>.out

# Monitor throughput
tail -f logs/train_fold_<JOB_ID>.out | grep Throughput
```

### Check /dev/shm Usage

```bash
df -h /dev/shm
ls -lh /dev/shm/artefact_cache/
```

---

## 🧹 Cleanup

### Remove Shared Memory Cache

```bash
# After all training is done
rm -rf /dev/shm/artefact_cache
```

### Clean Old Logs

```bash
cd logs/
rm -rf train_fold_*.out train_fold_*.err
```

---

## ⚙️ Script Details

### slurm_test.sh

**Purpose:** Quick validation of setup  
**Config:** Uses configs/convnext_tiny.yaml  
**Epochs:** 1  
**Key Checks:**
- VRAM usage (should be ~1.6%)
- Throughput (target: >80 imgs/s)
- Loss convergence

### slurm_train.sh

**Purpose:** Single model training with random 80/20 split  
**Arguments:** Config file path  
**Output:** `logs/<MODEL_NAME>/best_model.pth`  
**Checkpoints:** Saves best mIoU model

### slurm_train_fold.sh

**Purpose:** K-fold cross-validation training  
**Arguments:**
- Config file path
- `--fold N`: Which fold (0, 1, 2)
- `--n-folds N`: Total folds (default: 3)
- `--test-epoch`: Run only 1 epoch (for testing)

**Example:**
```bash
sbatch scripts/slurm_train_fold.sh configs/convnext_tiny_shm.yaml --fold 0
```

### slurm_preload_shm.sh

**Purpose:** Copy dataset to /dev/shm for fast access  
**Resources:**
- CPUs: 8
- RAM: 16GB
- Time: 30 min (completes in ~4 sec)

**Output:** `/dev/shm/artefact_cache/`

### launch_cv_shm.sh

**Purpose:** Orchestrate full 3-fold CV with shared memory  
**Steps:**
1. Submit preload job
2. Wait for completion
3. Submit 3 fold jobs

**Arguments:** Config file with `use_shared_memory: true`

**Example:**
```bash
bash scripts/launch_cv_shm.sh configs/segformer_b3_shm.yaml
```

---

## 🔧 Configuration Files

Active configs (use `_shm.yaml` for shared memory):

| Config | Encoder | Type | Expected mIoU |
|--------|---------|------|---------------|
| `convnext_tiny.yaml` | tu-convnext_tiny | CNN | 28-29% |
| `convnext_tiny_shm.yaml` | tu-convnext_tiny | CNN | 28-29% |
| `segformer_b3.yaml` | tu-mit_b3 | ViT | 30-32% |
| `segformer_b3_shm.yaml` | tu-mit_b3 | ViT | 30-32% |
| `maxvit_tiny.yaml` | tu-maxvit_tiny_tf_384 | Hybrid | 29-31% |
| `maxvit_tiny_shm.yaml` | tu-maxvit_tiny_tf_384 | Hybrid | 29-31% |

---

## 📈 Expected Results

### Single Training (Random Split)

```
Encoder: ConvNeXt-Tiny
mIoU: 28.5%
Throughput: 80 imgs/s
Time: 15 min
```

### 3-Fold CV (Shared Memory)

```
Encoder: ConvNeXt-Tiny
Fold 0 mIoU: 28.3%
Fold 1 mIoU: 28.7%
Fold 2 mIoU: 28.5%
Mean: 28.5 ± 0.2%
Total time: 47 min
```

---

**Last Updated:** November 16, 2025  
**POC Version:** 5.9-v2  
**Status:** ✅ Production Ready
