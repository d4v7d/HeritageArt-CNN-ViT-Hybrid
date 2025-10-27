.PHONY: help build up down shell download test-epoch train-convnext train-swin train-maxvit \
        eval-convnext eval-swin eval-maxvit eval-all compare clean clean-trainings logs

help:
	@echo "POC-5.5: Hierarchical Multi-Task Segmentation"
	@echo ""
	@echo "Available commands:"
	@echo "  make build         - Build Docker image"
	@echo "  make up            - Start container in background"
	@echo "  make down          - Stop container"
	@echo "  make shell         - Open bash shell in container"
	@echo "  make download      - Download ARTeFACT dataset"
	@echo "  make test-epoch    - Test 1 epoch with ConvNeXt (VRAM validation)"
	@echo ""
	@echo "Training:"
	@echo "  make train-convnext - Train ConvNeXt-Tiny (30 epochs)"
	@echo "  make train-swin    - Train Swin-Tiny (30 epochs)"
	@echo "  make train-maxvit  - Train MaxViT-Tiny (30 epochs)"
	@echo "  make train-all     - Train all 3 models sequentially (~2.5 hours)"
	@echo ""
	@echo "Evaluation:"
	@echo "  make eval-convnext - Evaluate ConvNeXt-Tiny"
	@echo "  make eval-swin     - Evaluate Swin-Tiny"
	@echo "  make eval-maxvit   - Evaluate MaxViT-Tiny"
	@echo "  make eval-all      - Evaluate all 3 models"
	@echo "  make compare       - Compare all models + generate report"
	@echo ""
	@echo "Other:"
	@echo "  make logs          - Show training logs"
	@echo "  make clean-trainings - Remove only training logs (no confirmation)"
	@echo "  make clean         - Remove logs and checkpoints (asks confirmation)"

build:
	@echo "🔨 Building Docker image..."
	cd docker && docker-compose build

up:
	@echo "🚀 Starting container..."
	cd docker && docker-compose up -d

down:
	@echo "🛑 Stopping container..."
	cd docker && docker-compose down

shell:
	@echo "💻 Opening shell in container..."
	docker exec -it artefact-poc55-multiclass /bin/bash

download:
	@echo "📥 Downloading ARTeFACT dataset..."
	docker exec artefact-poc55-multiclass python /workspace/scripts/download_dataset.py \
		--output-dir /workspace/data/artefact
	@echo "✅ Dataset downloaded to data/artefact/"

test-epoch:
	@echo "🧪 Testing 1 epoch with ConvNeXt (VRAM validation)..."
	docker exec artefact-poc55-multiclass python /workspace/scripts/train_poc55.py \
		--config /workspace/configs/convnext_tiny.yaml \
		--output-dir /workspace/logs \
		--test-epoch
	@echo "✅ Test epoch completed! Check logs/poc55_convnext_tiny_256px/"

train-convnext:
	@echo "🚂 Training ConvNeXt-Tiny (30 epochs, ~46 minutes on RTX 3050 6GB)..."
	docker exec artefact-poc55-multiclass python /workspace/scripts/train_poc55.py \
		--config /workspace/configs/convnext_tiny.yaml \
		--output-dir /workspace/logs

train-swin:
	@echo "🚂 Training Swin-Tiny (30 epochs, ~46 minutes)..."
	docker exec artefact-poc55-multiclass python /workspace/scripts/train_poc55.py \
		--config /workspace/configs/swin_tiny.yaml \
		--output-dir /workspace/logs

train-maxvit:
	@echo "🚂 Training MaxViT-Tiny (30 epochs, ~46 minutes)..."
	docker exec artefact-poc55-multiclass python /workspace/scripts/train_poc55.py \
		--config /workspace/configs/maxvit_tiny.yaml \
		--output-dir /workspace/logs

train-all:
	@echo "🚂 Training all 3 models sequentially (~2.5 hours total)..."
	@make train-convnext
	@make train-swin
	@make train-maxvit
	@echo "✅ All 3 models trained!"

# Evaluation commands
eval-convnext:
	@echo "📊 Evaluating ConvNeXt-Tiny..."
	docker exec artefact-poc55-multiclass python /workspace/scripts/evaluate.py \
		--checkpoint /workspace/logs/poc55_convnext_tiny_256px/checkpoints/best_model.pth \
		--config /workspace/configs/convnext_tiny.yaml \
		--output /workspace/logs/poc55_convnext_tiny_256px/evaluation

eval-swin:
	@echo "📊 Evaluating Swin-Tiny..."
	docker exec artefact-poc55-multiclass python /workspace/scripts/evaluate.py \
		--checkpoint /workspace/logs/poc55_swin_tiny_256px/checkpoints/best_model.pth \
		--config /workspace/configs/swin_tiny.yaml \
		--output /workspace/logs/poc55_swin_tiny_256px/evaluation

eval-maxvit:
	@echo "📊 Evaluating MaxViT-Tiny..."
	docker exec artefact-poc55-multiclass python /workspace/scripts/evaluate.py \
		--checkpoint /workspace/logs/poc55_maxvit_tiny_256px/checkpoints/best_model.pth \
		--config /workspace/configs/maxvit_tiny.yaml \
		--output /workspace/logs/poc55_maxvit_tiny_256px/evaluation

eval-all:
	@echo "📊 Evaluating all 3 models..."
	@make eval-convnext
	@make eval-swin
	@make eval-maxvit
	@echo "✅ All models evaluated!"

# Comparison
compare:
	@echo "📊 Comparing all models and generating report..."
	docker exec artefact-poc55-multiclass python /workspace/scripts/compare.py \
		--logs /workspace/logs \
		--output /workspace/logs/comparison
	@echo "✅ Comparison complete! Check logs/comparison/"

logs:
	@echo "📊 Training Progress:"
	@echo ""
	@for model in convnext_tiny swin_tiny maxvit_tiny; do \
		if docker exec artefact-poc55-multiclass test -f /workspace/logs/$$model/logs/training_log.csv 2>/dev/null; then \
			echo "✅ $$model:"; \
			docker exec artefact-poc55-multiclass tail -1 /workspace/logs/$$model/logs/training_log.csv | \
				awk -F',' 'NR>1 {printf "   Epoch %s/30 | Train Loss: %.4f | Val Loss: %.4f | mIoU Fine: %.4f\n", $$1, $$2, $$3, $$6}'; \
			echo ""; \
		else \
			echo "⏸️  $$model: Not started"; \
			echo ""; \
		fi \
	done
	@echo "💾 Checkpoints:"
	@docker exec artefact-poc55-multiclass bash -c "find /workspace/logs -name 'best_model.pth' 2>/dev/null | wc -l" | \
		xargs -I {} echo "   {} models saved"
	@echo ""
	@echo "Commands: make logs (refresh) | docker logs artefact-poc55-multiclass -f (live)"

clean-trainings:
	@echo "🧹 Cleaning training logs (poc55_*)..."
	docker exec artefact-poc55-multiclass bash -c "rm -rf /workspace/logs/poc55_*"
	@echo "✅ Training logs cleaned"

clean:
	@echo "🧹 Cleaning logs and checkpoints..."
	@read -p "Are you sure? This will delete all logs and checkpoints [y/N]: " confirm && \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		rm -rf logs/* data/artefact/* .cache/*; \
		echo "✅ Cleaned"; \
	else \
		echo "❌ Cancelled"; \
	fi

# Non-interactive local cleanup (safe for scripts/CI)
cleanup-local:
	@echo "🧹 Non-interactive cleanup: removing logs, checkpoints and caches..."
	@rm -rf logs/* data/artefact/* .cache/* || true
	@echo "✅ Local cleanup complete"

# Prepare a minimal transfer archive for running POC-5.5 on another machine
prepare-transfer: cleanup-local
	@echo "📦 Preparing transfer archive (poc55_transfer.tar.gz)..."
	@bash ./scripts/prepare_for_transfer.sh
	@echo "✅ Archive ready: poc55_transfer.tar.gz"

# Quick workflow for testing POC-5.5
quick-test: build up download test-epoch
	@echo ""
	@echo "✅ Quick test workflow completed!"
	@echo "   1. Docker image built"
	@echo "   2. Container started"
	@echo "   3. Dataset downloaded"
	@echo "   4. Test epoch executed"
	@echo ""
	@echo "Next steps:"
	@echo "  - Check logs/poc55_convnext_tiny/ for results"
	@echo "  - If successful, run: make train-convnext"
	@echo "  - Deploy to Dell laptop Monday for full 30-epoch training"

# Full training workflow
full-workflow: build up download train-all eval-all compare
	@echo ""
	@echo "🎉 COMPLETE POC-5.5 WORKFLOW FINISHED!"
	@echo "========================================="
	@echo ""
	@echo "✅ Training:"
	@echo "   - ConvNeXt-Tiny: Trained (30 epochs)"
	@echo "   - Swin-Tiny: Trained (30 epochs)"
	@echo "   - MaxViT-Tiny: Trained (30 epochs)"
	@echo ""
	@echo "✅ Evaluation:"
	@echo "   - All 3 models evaluated with hierarchical metrics"
	@echo "   - Check logs/*/evaluation/ for per-model results"
	@echo ""
	@echo "✅ Comparison:"
	@echo "   - Training curves: logs/comparison/training_curves.png"
	@echo "   - Hierarchical metrics: logs/comparison/hierarchical_metrics.png"
	@echo "   - Summary table: logs/comparison/comparison_table.csv"
	@echo "   - Full report: logs/comparison/summary_report.txt"
	@echo ""
	@echo "📊 Total time: ~2.7 hours (138min training + 25min eval/compare)"
	@echo "💾 VRAM usage: ~839 MB (14% of 6GB)"
