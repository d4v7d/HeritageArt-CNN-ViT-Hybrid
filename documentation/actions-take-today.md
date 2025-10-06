# Actions taken in the repo
1. Create folders for repo structure src/datasets, src/infer, src/models

2. Create a virtual enviroment:
    cd ~/Documents/HeritageArt-CNN-ViT-Hybrid
    python3 -m venv venv
    source venv/bin/activate

3. Created requirements.txt file and executed pip install -r requirements.txt

4. Install Tools for pre-commit
    pip install pre-commit black flake8 isort hydra-core

5. Created the file .pre-commit-config-yaml, pyproject.toml and .flake8

6. Run:
    pre-commit install
    pre-commit run --all-files

7. Created the yaml files in configs and commit chagnes

8. Organized experiments folder (2025-10-06):
    - Moved cuda-test.py, MMSeg-test.py, demo/image_demo.py → experiments/
    - Created experiments/README.md with usage instructions
    - Updated pre-commit hooks to exclude experiments/ from linting
    - Adjusted paths in moved scripts to work from new location

# Next actions:

Implement src/datasets/artefact_dataset.py using the scaffold code from the plan

Write a test in tests/test_dataset.py

# Discussion about google collab:

Can This Run in Google Colab? Yes, with Modifications
Based on the From-Paper-to-Plan.md and your current setup, yes, you can run this in Google Colab, but the plan is currently designed for local execution. Here's the breakdown:

Current Plan: Local-First Design
The plan assumes:

Local GPU (mentioned in Methods section of main.tex):

"We trained each model on an \todo{MODEL NAME} GPU (specifically, we used a \todo{AMOUNT} \todo{GPU MODEL TYPE} with \todo{XX AMOUNT} GB VRAM)"

Persistent storage:

data folder for ARTeFACT dataset (418 high-res images)
logs folder for checkpoints/metrics
configs for Hydra YAML files
Long training runs:

Phase 2: ~30-50 epochs until early stopping
Phase 3: Train 3 models (CNN/ViT/Hybrid)
Phase 4-5: Multiple LOMO/LOContent splits
Reproducibility tooling:

Git version control
Pre-commit hooks (requires local Git)
Virtual environment management
Can You Adapt It for Colab? Yes
What Works Out-of-the-Box
✅ Python code: All PyTorch/timm/albumentations code runs identically
✅ GPU access: Colab provides free T4 (16GB) or paid A100/V100
✅ Jupyter-style workflow: Colab is notebook-based (aligns with notebooks folder)

What Needs Modification
Challenge	Solution
Dataset storage	Mount Google Drive or download ARTeFACT to Colab session storage
Pre-commit hooks	Skip (Colab doesn't use Git workflows) or manually run black/flake8
Hydra configs	Works, but you'll load from Drive or hardcode paths
Long training	Risk of timeout (12hr free / 24hr paid); use checkpointing
Model checkpoints	Save to Drive to avoid losing weights on disconnect
Logs/artifacts	Stream to Weights & Biases (wandb) instead of local logs
Recommended Colab Adaptations
1. Mount Google Drive (for data + checkpoints)

from google.colab import drivedrive.mount('/content/drive')# Set pathsDATA_DIR = '/content/drive/MyDrive/HeritageArt-CNN-ViT-Hybrid/data'LOGS_DIR = '/content/drive/MyDrive/HeritageArt-CNN-ViT-Hybrid/logs'
2. Install Dependencies (in Colab cell)

!pip install -q timm albumentations hydra-core omegaconf wandb pytest black flake8
3. Skip Pre-commit (or run manually)

# Manual formatting check!black src/ --check!flake8 src/ --max-line-length=100
4. Checkpoint to Drive (prevent loss on timeout)

# In training loopif epoch % 5 == 0:    torch.save(model.state_dict(), f'{LOGS_DIR}/checkpoint_epoch{epoch}.pth')
5. Use Weights & Biases (instead of local logs)

import wandbwandb.init(project='heritage-art-cnn-vit', config=cfg)# Log metricswandb.log({'epoch': epoch, 'mIoU': miou, 'macro_f1': f1})
When to Use Colab vs Local
Use Case	Recommendation
Prototyping (Phase 1-2)	✅ Colab (free GPU, fast iteration)
Small experiments (testing dataset loader)	✅ Colab
Full training runs (Phases 3-5)	⚠️ Colab Pro+ (24hr limit) or Local preferred
Reproducibility (final paper results)	❌ Local (deterministic, version-controlled)
Team collaboration	✅ Colab (shareable notebooks)
Hybrid Workflow (Recommended)
Based on the plan's requirements, I recommend:

Phase 1-2: Colab
Develop dataset loader (src/datasets/artefact_dataset.py)
Test tiling + Hann blending
Train binary baseline quickly
Phase 3-5: Local
Train CNN/ViT/Hybrid models with full protocol
Run LOMO/LOContent splits (multiple long runs)
Generate final metrics with bootstrap CIs
Phase 6: Colab
Run Grad-CAM/attention visualizations
Create demo notebooks for release
Answer to Your Question
Is the main focus local? Yes, the plan assumes:

Local Git workflows (pre-commit, reproducible splits)
Long, uninterrupted training runs
Persistent storage for checkpoints/logs
Can you use Colab? Absolutely, but:

You'll need to adapt file I/O (Drive mounts)
Skip or manually run pre-commit checks
Use wandb for experiment tracking
Accept potential timeouts on long runs (use Colab Pro+ or local for Phases 3-5)
Best of both worlds: Prototype in Colab (free GPU), finalize training locally (reproducibility + control).

Next Steps (While Requirements Install)
✅ Finish pip install -r requirements.txt
⏭️ Create .pre-commit-config.yaml (even if you'll skip it in Colab)
⏭️ Set up configs structure (works in both environments)
⏭️ Decide: Colab for Phase 1-2 or full local workflow?
Let me know which path you prefer, and I can provide specific setup instructions!
