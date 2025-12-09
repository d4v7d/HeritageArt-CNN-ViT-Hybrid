# POC-6 Implementation TODO List

Based on `documentation/POC6_PLANNING.md`.

## Day 1-2: Hierarchical Multi-Task Learning

- [x] **Copy Hierarchical Code**
    - [x] Create `src/models/hierarchical_upernet.py` (Adapt from POC-5.5 if available, or implement fresh)
    - [x] Create `src/losses/hierarchical_loss.py`

- [x] **Adapt to POC-6 Structure**
    - [x] Update `src/model_factory.py` to support hierarchical models
    - [x] Update `src/dataset.py` to provide hierarchical labels (binary, coarse, fine)
    - [x] Update configs (`configs/*.yaml`) with hierarchical parameters

- [x] **Verification**
    - [x] Run a test training for 1 epoch to verify VRAM and output shapes.

## Day 3: Progressive Curriculum Implementation

- [x] **Implement Curriculum Logic**
    - [x] Modify `src/train.py` (or create `src/train_curriculum.py`)
    - [x] Implement staging logic:
        - [x] Stage 1 (Epochs 1-20): Binary head only
        - [x] Stage 2 (Epochs 21-40): Binary + Coarse heads
        - [x] Stage 3 (Epochs 41-100): All 3 heads
    - [x] Implement checkpoint transfer logic between stages

- [x] **Logging**
    - [x] Update logging to track per-stage metrics

- [x] **Verification**
    - [x] Run a short curriculum test (e.g., 10 epochs total)

## Day 4: LOContent Splits Creation

- [x] **Split Generation**
    - [x] Create `scripts/create_locontent_splits.py`
    - [x] Generate manifests: `manifests/locontent_fold{1-4}.json`
    - [x] Verify balance (~365 samples/content) and class distribution

- [x] **Training Script**
    - [x] Create `scripts/train_locontent.sh` to loop over folds

## Day 5: Training Execution

- [ ] **Baseline Training**
    - [ ] Train ConvNeXt-Tiny (Hierarchical + Curriculum)
    - [ ] Train SegFormer-B3 (Hierarchical + Curriculum)
    - [ ] Train MaxViT-Tiny (Hierarchical + Curriculum)

- [ ] **Evaluation**
    - [ ] Evaluate Baseline models
    - [ ] Visualize results

- [x] **LOContent DG Training**
    - [x] Run LOContent folds (12 runs) - *Completed (Baseline vs Hierarchical)*
    - [x] Analyze DG Gap
    - [ ] **High-Res Experiment**: Train Fold 3 @ 768x768 (Job 2272)

## Documentation & Wrap-up

- [ ] Update `README.md`
- [ ] Generate results tables

## Results Summary (POC-6)

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | **Average** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (ConvNeXt)** | 8.60% | 9.07% | 10.18% | 8.72% | **9.14%** |
| **Hierarchical (Ours)** | 10.35% | 12.31% | 13.58% | 9.42% | **11.41%** |

**Impact**: +2.27% absolute improvement (+24.8% relative) using Hierarchical MTL.
