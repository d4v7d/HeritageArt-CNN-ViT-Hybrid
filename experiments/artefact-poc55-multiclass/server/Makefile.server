.PHONY: help verify-dataset download verify train-convnext train-swin train-maxvit \
        train-all status logs logs-live cancel clean

help:
	@echo "POC-5.5: SLURM Deployment (CITIC Server)"
	@echo "=========================================="
	@echo ""
	@echo "Setup:"
	@echo "  make verify-dataset - Check dataset"
	@echo "  make download       - Smart download"
	@echo "  make verify         - Verify environment"
	@echo ""
	@echo "Training (submit to GPU nodes):"
	@echo "  make train-convnext - ConvNeXt job"
	@echo "  make train-swin     - Swin job"
	@echo "  make train-maxvit   - MaxViT job"
	@echo "  make train-all      - All 3 parallel"
	@echo ""
	@echo "Monitoring:"
	@echo "  make status    - Job status"
	@echo "  make logs      - View logs"
	@echo "  make logs-live - Follow live"
	@echo "  make cancel    - Cancel jobs"

verify-dataset:
	@if [ -d "../data/artefact/images" ]; then \
		IMG_COUNT=$$(find ../data/artefact/images -type f | wc -l); \
		echo "✅ Dataset: $$IMG_COUNT images"; \
		du -sh ../data/artefact; \
		[ $$IMG_COUNT -eq 418 ] && echo "✅ COMPLETE" || echo "⚠️  INCOMPLETE"; \
	else \
		echo "❌ Dataset NOT FOUND"; \
	fi

download:
	@echo "📥 Smart dataset download..."
	bash download_dataset_server.sh

verify:
	@module load miniconda3 && \
	conda activate poc55 && \
	python -c "import torch, timm; print(f'✅ PyTorch: {torch.__version__}'); print(f'✅ timm: {timm.__version__}')"

train-convnext:
	@mkdir -p ../slurm_logs
	sbatch slurm_train_convnext.sh
	@echo "✅ ConvNeXt submitted!"

train-swin:
	@mkdir -p ../slurm_logs
	sbatch slurm_train_swin.sh
	@echo "✅ Swin submitted!"

train-maxvit:
	@mkdir -p ../slurm_logs
	sbatch slurm_train_maxvit.sh
	@echo "✅ MaxViT submitted!"

train-all:
	@mkdir -p ../slurm_logs
	@sbatch slurm_train_convnext.sh
	@sbatch slurm_train_swin.sh
	@sbatch slurm_train_maxvit.sh
	@echo ""
	@echo "✅ All 3 jobs submitted!"
	@echo "Monitor: make status"

status:
	@echo "📊 SLURM Jobs:"
	@squeue -u btrigueros --format="%.10i %.15j %.8T %.10M %.6D %R" || echo "None running"

logs:
	@for log in ../slurm_logs/*.out; do [ -f "$$log" ] && tail -30 "$$log"; done

logs-live:
	@tail -f $$(ls -t ../slurm_logs/*.out 2>/dev/null | head -1)

cancel:
	@scancel -u btrigueros --name=poc55-convnext
	@scancel -u btrigueros --name=poc55-swin
	@scancel -u btrigueros --name=poc55-maxvit
	@echo "✅ Jobs cancelled"

clean:
	@read -p "Remove logs? [y/N]: " c && [ "$$c" = "y" ] && rm -rf ../logs/poc55_* ../slurm_logs/*
