.PHONY: help build train evaluate infer clean setup-data

help:
	@echo "POC-4: ARTeFACT Training Minimal"
	@echo "================================="
	@echo ""
	@echo "Available commands:"
	@echo "  make build          - Build Docker image"
	@echo "  make setup-data     - Create symlink to POC-1 demo data"
	@echo "  make train          - Train model (10 epochs baseline)"
	@echo "  make train-extended - Train model (50 epochs extended)"
	@echo "  make train-optimized - Train model (60 epochs optimized)"
	@echo "  make evaluate       - Evaluate trained model"
	@echo "  make infer          - Run single image inference demo"
	@echo "  make clean          - Clean logs and checkpoints"
	@echo "  make shell          - Open interactive shell in container"
	@echo ""
	@echo "Quick start:"
	@echo "  make build && make setup-data && make train-extended"

build:
	@echo "Building Docker image..."
	cd docker && docker-compose build

setup-data:
	@echo "Setting up data symlink..."
	@if [ -d "../artefact-data-obtention/data/demo" ]; then \
		ln -sf ../../artefact-data-obtention/data/demo data/artefact; \
		echo "Symlink created: data/artefact -> ../artefact-data-obtention/data/demo"; \
	else \
		echo "ERROR: POC-1 demo data not found!"; \
		echo "Please run POC-1 first to generate demo data."; \
		exit 1; \
	fi

train:
	@echo "Training model (baseline: 10 epochs)..."
	cd docker && docker-compose run --rm artefact-training \
		python3 scripts/train.py --config configs/train_config.yaml

train-extended:
	@echo "Training model (extended: 50 epochs)..."
	cd docker && docker-compose run --rm artefact-training \
		python3 scripts/train.py --config configs/train_config_extended.yaml

train-optimized:
	@echo "Training model (optimized: 60 epochs)..."
	cd docker && docker-compose run --rm artefact-training \
		python3 scripts/train.py --config configs/train_config_optimized.yaml

train-resume:
	@echo "Resuming training from checkpoint..."
	@read -p "Enter checkpoint path: " CKPT; \
	cd docker && docker-compose run --rm artefact-training \
		python3 scripts/train.py --config configs/train_config.yaml --resume $$CKPT

evaluate:
	@echo "Evaluating model..."
	cd docker && docker-compose run --rm artefact-training \
		python3 scripts/evaluate.py \
		--config configs/train_config.yaml \
		--checkpoint logs/checkpoints/best_model.pth \
		--output logs/evaluation \
		--max-viz 10

infer:
	@echo "Running inference demo..."
	@read -p "Enter image path (relative to workspace): " IMG; \
	cd docker && docker-compose run --rm artefact-training \
		python3 scripts/infer_demo.py \
		--image $$IMG \
		--checkpoint logs/checkpoints/best_model.pth \
		--config configs/train_config.yaml \
		--output logs/demo_inference.png

shell:
	@echo "Opening interactive shell..."
	cd docker && docker-compose run --rm artefact-training /bin/bash

clean:
	@echo "Cleaning logs and checkpoints..."
	rm -rf logs/checkpoints/*
	rm -rf logs/training/*
	rm -rf logs/visualizations/*
	rm -rf logs/evaluation/*
	@echo "Clean complete!"

clean-all: clean
	@echo "Removing Docker image..."
	docker rmi artefact-training-minimal:latest 2>/dev/null || true
	@echo "Removing data symlink..."
	rm -f data/artefact
	@echo "Deep clean complete!"
