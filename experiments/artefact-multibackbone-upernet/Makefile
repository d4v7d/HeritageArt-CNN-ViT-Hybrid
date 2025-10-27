# Makefile for POC-5: Multi-backbone Comparison with UPerNet
# Architectures: ConvNeXt-Tiny (CNN), Swin-Tiny (ViT), MaxViT-Tiny (Hybrid)

.PHONY: help build train-convnext train-swin train-maxvit train-all evaluate-convnext evaluate-swin evaluate-maxvit evaluate-all compare visualize-attention clean

# Default target
help:
	@echo "POC-5: Multi-backbone ARTeFACT Damage Detection"
	@echo ""
	@echo "Available targets:"
	@echo "  build              - Build Docker image"
	@echo "  train-convnext     - Train ConvNeXt-Tiny + UPerNet"
	@echo "  train-swin         - Train Swin-Tiny + UPerNet"
	@echo "  train-maxvit       - Train MaxViT-Tiny + UPerNet"
	@echo "  train-all          - Train all 3 models sequentially"
	@echo "  evaluate-convnext  - Evaluate ConvNeXt model"
	@echo "  evaluate-swin      - Evaluate Swin model"
	@echo "  evaluate-maxvit    - Evaluate MaxViT model"
	@echo "  evaluate-all       - Evaluate all 3 models"
	@echo "  compare            - Compare all models (metrics + visualizations)"
	@echo "  visualize-attention- Generate attention maps (Swin + MaxViT)"
	@echo "  clean              - Clean logs and checkpoints"

# Build Docker image
build:
	@echo "Building Docker image for multi-backbone training..."
	docker compose -f docker/docker-compose.yml build

# Train individual models (60 epochs each, ~20-25min)
train-convnext:
	@echo "Training ConvNeXt-Tiny + UPerNet..."
	docker compose -f docker/docker-compose.yml run --rm multibackbone-upernet \
		python scripts/train.py --config configs/convnext_tiny_upernet.yaml

train-swin:
	@echo "Training Swin-Tiny + UPerNet..."
	docker compose -f docker/docker-compose.yml run --rm multibackbone-upernet \
		python scripts/train.py --config configs/swin_tiny_upernet.yaml

train-maxvit:
	@echo "Training MaxViT-Tiny + UPerNet..."
	docker compose -f docker/docker-compose.yml run --rm multibackbone-upernet \
		python scripts/train.py --config configs/maxvit_tiny_upernet.yaml

# Train all models sequentially
train-all: train-convnext train-swin train-maxvit
	@echo "✅ All 3 models trained!"

# Evaluate individual models
evaluate-convnext:
	@echo "Evaluating ConvNeXt-Tiny + UPerNet..."
	docker compose -f docker/docker-compose.yml run --rm multibackbone-upernet \
		python scripts/evaluate.py --config configs/convnext_tiny_upernet.yaml --checkpoint logs/convnext_tiny_upernet/checkpoints/best_model.pth

evaluate-swin:
	@echo "Evaluating Swin-Tiny + UPerNet..."
	docker compose -f docker/docker-compose.yml run --rm multibackbone-upernet \
		python scripts/evaluate.py --config configs/swin_tiny_upernet.yaml --checkpoint logs/swin_tiny_upernet/checkpoints/best_model.pth

evaluate-maxvit:
	@echo "Evaluating MaxViT-Tiny + UPerNet..."
	docker compose -f docker/docker-compose.yml run --rm multibackbone-upernet \
		python scripts/evaluate.py --config configs/maxvit_tiny_upernet.yaml --checkpoint logs/maxvit_tiny_upernet/checkpoints/best_model.pth

# Evaluate all models
evaluate-all: evaluate-convnext evaluate-swin evaluate-maxvit
	@echo "✅ All 3 models evaluated!"

# Compare all models (dynamic analysis with visualizations)
compare:
	@echo "Comparing all models dynamically..."
	docker compose -f docker/docker-compose.yml run --rm multibackbone-upernet \
		python scripts/compare.py --logs logs --output logs/comparison

# Generate attention maps for Swin and MaxViT
visualize-attention:
	@echo "Visualizing attention maps (Swin + MaxViT)..."
	docker compose -f docker/docker-compose.yml run --rm multibackbone-upernet \
		python scripts/visualize_attention.py \
			--config-swin configs/swin_tiny_upernet.yaml \
			--config-maxvit configs/maxvit_tiny_upernet.yaml \
			--checkpoint-swin logs/swin_tiny_upernet/checkpoints/best_model.pth \
			--checkpoint-maxvit logs/maxvit_tiny_upernet/checkpoints/best_model.pth \
			--output logs/attention

# Clean logs and checkpoints (keep directory structure)
clean:
	@echo "Cleaning logs and checkpoints..."
	rm -rf logs/*/checkpoints/*.pth
	rm -rf logs/*/training/*.csv
	rm -rf logs/*/evaluation/*.png
	rm -rf logs/comparison/*
	rm -rf logs/attention/*
	@echo "✅ Cleaned! (structure preserved)"
