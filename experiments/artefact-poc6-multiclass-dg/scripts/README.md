# POC-5.9-v2: Scripts Reference

Production-ready scripts for training, evaluation, and visualization.

## 📁 Scripts Overview

### Core Pipeline Scripts

| Script | Purpose | Usage | Duration |
|--------|---------|-------|----------|
| **slurm_test.sh** | Quick 1-epoch test | `sbatch slurm_test.sh [config]` | ~1 min |
| **slurm_train.sh** | Full training (50 epochs) | `sbatch slurm_train.sh <config>` | ~42 min |
| **slurm_evaluate.sh** | Evaluate all models | `sbatch slurm_evaluate.sh` | ~5 min |
| **slurm_visualize.sh** | Generate visualizations | `sbatch slurm_visualize.sh` | ~2 min |

---

## 🚀 Quick Start

### 1. Test Single Model (Fastest)

```bash
cd /opt/home/btrigueros/HeritageArt-CNN-ViT-Hybrid/experiments/artefact-poc59-v2

# Test 1 epoch (default: ConvNeXt)
sbatch scripts/slurm_test.sh

# Test specific encoder
sbatch scripts/slurm_test.sh configs/segformer_b3.yaml
```

**Expected:**
- Throughput: ~80 imgs/s
- VRAM: 0.51GB (1.6%)
- Time: ~1 min
- Output: `logs/test_<JOB_ID>.out`

### 2. Train Single Model

```bash
# Train ConvNeXt for 50 epochs
sbatch scripts/slurm_train.sh configs/convnext_tiny.yaml

# Train SegFormer (WINNER 🏆)
sbatch scripts/slurm_train.sh configs/segformer_b3.yaml
```

**Expected:**
- mIoU: 26-38% (depending on encoder)
- Time: ~42 min (30min preload + 12min training)
- Output: `logs/models/<encoder>/best_model.pth`

### 3. Train All 3 Models (Sequential Benchmark)

```bash
# Launch with job dependencies for scientific rigor
JOB1=$(sbatch --parsable scripts/slurm_train.sh configs/convnext_tiny.yaml)
JOB2=$(sbatch --parsable --dependency=afterany:$JOB1 scripts/slurm_train.sh configs/segformer_b3.yaml)
JOB3=$(sbatch --parsable --dependency=afterany:$JOB2 scripts/slurm_train.sh configs/maxvit_tiny.yaml)

echo "Submitted jobs: $JOB1, $JOB2, $JOB3"
```

**Total time:** ~126 min (3 × 42 min sequential)

### 4. Evaluate All Models

```bash
# After training completes
sbatch scripts/slurm_evaluate.sh
```

**Outputs (per model):**
```
logs/models/{encoder}/evaluation/
├── metrics.json            # mIoU, per-class metrics
├── confusion_matrix.png    # Normalized confusion matrix
└── per_class_iou.png       # Bar chart with color coding
```

### 5. Generate Visualizations

```bash
# Create prediction visualizations (20 samples × 3 models)
sbatch scripts/slurm_visualize.sh
```

**Outputs (per model):**
```
logs/models/{encoder}/visualizations/
├── prediction_grid.png          # 20 samples × 4 columns
├── class_distribution.png       # GT vs Pred frequencies
├── error_maps.png               # Pixel correctness
└── class_{XX}_{name}.png        # TP/FP/FN analysis (6 files)
```

---

## 📊 Monitoring

### Check Job Status

```bash
squeue -u $USER
```

### Monitor Training Output

```bash
# Find job ID from squeue
tail -f logs/train_<JOB_ID>.out

# Monitor throughput
tail -f logs/train_<JOB_ID>.out | grep "imgs/s"
```

### Check VRAM Usage

```bash
# SSH to compute node
ssh <NODE_NAME>
watch -n 1 nvidia-smi
```

---

## 🧹 Cleanup

### Clean Old Logs

```bash
cd logs/training/
ls -lt  # Review logs by date

# Archive old logs (if needed)
mkdir -p ../archive/
mv evaluate_*.{out,err} ../archive/
```

### Remove Temporary Files

```bash
# Clean Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
```

---

## ⚙️ Script Details

### slurm_test.sh

**Purpose:** Quick validation of setup  
**Arguments:** Optional config file (default: `configs/convnext_tiny.yaml`)  
**Epochs:** 1  
**Resources:** 1 GPU, 32GB RAM, 2h time limit  

**Key Checks:**
- VRAM usage (target: <2%)
- Throughput (target: >80 imgs/s)
- Loss convergence

**Example:**
```bash
sbatch scripts/slurm_test.sh configs/maxvit_tiny.yaml
```

### slurm_train.sh

**Purpose:** Full training with RAM preload (50 epochs)  
**Arguments:** Config file path (required)  
**Resources:** 1 GPU, 32GB RAM, 2h time limit  
**Output:** `logs/models/<encoder>/best_model.pth`  

**Features:**
- Automatic RAM preloading (~30 min)
- Mixed precision (AMP) for 2× speedup
- OneCycleLR scheduler
- Saves best mIoU checkpoint

**Example:**
```bash
sbatch scripts/slurm_train.sh configs/segformer_b3.yaml
```

### slurm_evaluate.sh

**Purpose:** Evaluate all trained models  
**Arguments:** None (evaluates all 3 models automatically)  
**Resources:** 1 GPU, 32GB RAM, 2h time limit  

**Metrics Computed:**
- Mean IoU across all classes
- Per-class IoU, precision, recall, F1
- Confusion matrix (normalized)
- Inference time (ms/image)

**Visualizations:**
- Confusion matrix heatmap
- Per-class IoU bar chart (color-coded)

**Example:**
```bash
sbatch scripts/slurm_evaluate.sh
```

### slurm_visualize.sh

**Purpose:** Generate prediction visualizations  
**Arguments:** None (visualizes all 3 models, 20 samples each)  
**Resources:** 1 GPU, 32GB RAM, 30min time limit  
**Output:** 9 PNG files per model (27 total)

**Visualizations Generated:**
1. Prediction grid (13MB) - Input | GT | Pred | Overlay
2. Class distribution (93KB) - Bar chart comparison
3. Error maps (2.6MB) - Pixel-wise correctness
4-9. Per-class analysis (0.8-4MB each) - TP/FP/FN breakdown

**Example:**
```bash
sbatch scripts/slurm_visualize.sh
```

---

## 🔧 Configuration Files

Active configs (all production-ready):

| Config | Encoder | Type | Batch | LR | Expected mIoU |
|--------|---------|------|-------|-----|---------------|
| `convnext_tiny.yaml` | tu-convnext_tiny | CNN | 96 | 0.001 | 25-26% |
| `segformer_b3.yaml` | mit_b3 | ViT | 32 | 0.000333 | **37-38%** 🏆 |
| `maxvit_tiny.yaml` | tu-maxvit_tiny_tf_384 | Hybrid | 48 | 0.0005 | 34-35% |

**All configs include:**
- RAM preloading (`use_preload: true`)
- Mixed precision (`mixed_precision: true`)
- Balanced class weights
- OneCycleLR scheduler
- 50 epochs training

---

## 📈 Expected Results

### Complete Pipeline (Train → Eval → Viz)

```bash
# Example: Full pipeline for SegFormer
sbatch scripts/slurm_train.sh configs/segformer_b3.yaml
# Wait ~42 min

sbatch scripts/slurm_evaluate.sh
# Wait ~5 min

sbatch scripts/slurm_visualize.sh
# Wait ~2 min
```

**Total time:** ~49 minutes  
**Outputs:**
- Checkpoint: 543 MB
- Evaluation: 3 files (metrics + 2 plots)
- Visualizations: 9 PNG files

**Final Results:**
- ConvNeXt: 25.47% mIoU
- SegFormer: **37.63% mIoU** 🏆
- MaxViT: 34.58% mIoU

---

## 🎯 Production Workflow

**Recommended sequence:**

1. **Test:** Validate setup
   ```bash
   sbatch scripts/slurm_test.sh configs/segformer_b3.yaml
   ```

2. **Train:** Full 50-epoch training
   ```bash
   sbatch scripts/slurm_train.sh configs/segformer_b3.yaml
   ```

3. **Evaluate:** Generate metrics
   ```bash
   sbatch scripts/slurm_evaluate.sh
   ```

4. **Visualize:** Create samples
   ```bash
   sbatch scripts/slurm_visualize.sh
   ```

5. **Review:** Check outputs
   ```bash
   tree -L 3 logs/models/segformer_b3/
   cat logs/models/segformer_b3/evaluation/metrics.json
   ```

---

**Last Updated:** November 17, 2025  
**POC Version:** 5.9-v2  
**Status:** ✅ Production Ready - All pipelines validated
