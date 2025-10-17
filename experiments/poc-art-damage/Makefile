IMAGE_NAME=poc-art-damage:cuda126
DOCKER_BUILD_CONTEXT=./docker
WORKDIR_CONTAINER=/workspace
IMAGE?=/workspace/samples/demo.jpg

.PHONY: build bash run demo demo-build up down data train infer-w evaluate test

build:
	# Build container image
	docker build -t $(IMAGE_NAME) $(DOCKER_BUILD_CONTEXT)

bash:
	# Open an interactive shell in the container
	docker run --rm -it --gpus all -v $$(pwd)/src:$(WORKDIR_CONTAINER)/src:ro \
		-v $$(pwd)/configs:$(WORKDIR_CONTAINER)/configs:ro \
		-v $$(pwd)/samples:$(WORKDIR_CONTAINER)/samples:ro \
		-v $$(pwd)/logs:$(WORKDIR_CONTAINER)/logs \
		$(IMAGE_NAME) bash

run:
	# Run a single model. Usage: make run MODEL=upernet_swin_base IMAGE=./samples/demo.jpg
	docker run --rm --gpus all -v $$(pwd)/src:$(WORKDIR_CONTAINER)/src:ro \
		-v $$(pwd)/configs:$(WORKDIR_CONTAINER)/configs:ro \
		-v $$(pwd)/samples:$(WORKDIR_CONTAINER)/samples:ro \
		-v $$(pwd)/logs:$(WORKDIR_CONTAINER)/logs \
		$(IMAGE_NAME) bash -lc "python -m src.infer_one --config configs/$(MODEL).yaml --image $(IMAGE)"

demo:
	# Run all three models on the demo image. Usage: make demo IMAGE=./samples/demo.jpg
	# Default demo image path inside container is /workspace/samples/demo.jpg
	docker run --rm --gpus all -v $$(pwd)/src:$(WORKDIR_CONTAINER)/src:ro \
		-v $$(pwd)/configs:$(WORKDIR_CONTAINER)/configs:ro \
		-v $$(pwd)/samples:$(WORKDIR_CONTAINER)/samples:ro \
		-v $$(pwd)/logs:$(WORKDIR_CONTAINER)/logs \
		$(IMAGE_NAME) bash -lc "python -m src.run_all --image $(IMAGE)"

demo-build: build demo

up:
	docker compose -f experiments/poc-art-damage/docker-compose.yml up --build

down:
	docker compose -f experiments/poc-art-damage/docker-compose.yml down

data:
	# Download ARTeFACT via datasets and save to local folder inside logs/data
	# Use --max-samples to limit download size for testing
	docker run --rm --gpus all -v $$(pwd)/src:$(WORKDIR_CONTAINER)/src:ro \
		-v $$(pwd)/logs:$(WORKDIR_CONTAINER)/logs \
		$(IMAGE_NAME) bash -lc "python -m src.datasets.artefact --out logs/data/artefact --max-samples 50"

train:
	# Fine-tuning on ARTeFACT with specified model (default: convnext_tiny_fpn)
	docker run --rm --gpus all -v $$(pwd)/src:$(WORKDIR_CONTAINER)/src:ro \
		-v $$(pwd)/logs:$(WORKDIR_CONTAINER)/logs \
		$(IMAGE_NAME) bash -lc "python -m src.train.train_artefact --data logs/data/artefact --out logs/checkpoints --model $(MODEL) --epochs 20 --batch 4"

infer-w:
	# Inference with optional weights, e.g.: make infer-w MODEL=convnext_tiny_fpn WEIGHTS=/workspace/logs/checkpoints/convnext_tiny_fpn_artefact.pth IMAGE=/workspace/logs/data/artefact/image/0001.png
	docker run --rm --gpus all -v $$(pwd)/src:$(WORKDIR_CONTAINER)/src:ro \
		-v $$(pwd)/configs:$(WORKDIR_CONTAINER)/configs:ro \
		-v $$(pwd)/samples:$(WORKDIR_CONTAINER)/samples:ro \
		-v $$(pwd)/logs:$(WORKDIR_CONTAINER)/logs \
		$(IMAGE_NAME) bash -lc "python -m src.infer_one --config configs/$(MODEL).yaml --image $(IMAGE) --weights $(WEIGHTS)"

evaluate:
	# Evaluate trained model on test set
	docker run --rm --gpus all -v $$(pwd)/src:$(WORKDIR_CONTAINER)/src:ro \
		-v $$(pwd)/logs:$(WORKDIR_CONTAINER)/logs \
		$(IMAGE_NAME) bash -lc "python -m src.evaluate_artefact --model $(MODEL) --weights logs/checkpoints/$(MODEL)_artefact.pth --data logs/data/artefact"

test:
	# Run integration tests inside the container
	docker run --rm --gpus all -v $$(pwd)/src:$(WORKDIR_CONTAINER)/src:ro \
		-v $$(pwd)/test_artefact_integration.py:$(WORKDIR_CONTAINER)/test_artefact_integration.py:ro \
		$(IMAGE_NAME) bash -lc "python test_artefact_integration.py"

pipeline:
	# Run complete end-to-end pipeline: download data, train all 3 models, evaluate with visualizations
	# Usage: make pipeline MAX_SAMPLES=100 EPOCHS=20 MAX_EVAL=20
	docker run --rm --gpus all -v $$(pwd)/src:$(WORKDIR_CONTAINER)/src:ro \
		-v $$(pwd)/logs:$(WORKDIR_CONTAINER)/logs \
		$(IMAGE_NAME) bash -lc "python -m src.run_complete_pipeline \
		--data-dir logs/data/artefact_real \
		--output-dir logs/pipeline_results \
		--max-samples $(MAX_SAMPLES) \
		--max-eval-samples $(MAX_EVAL) \
		--epochs $(EPOCHS) \
		--batch-size 2"

pipeline-quick:
	# Quick pipeline test with existing dataset (50 samples, 10 epochs, 10 eval images)
	docker run --rm --gpus all -v $$(pwd)/src:$(WORKDIR_CONTAINER)/src:ro \
		-v $$(pwd)/logs:$(WORKDIR_CONTAINER)/logs \
		$(IMAGE_NAME) bash -lc "python -m src.run_complete_pipeline \
		--data-dir logs/data/artefact \
		--output-dir logs/pipeline_results \
		--skip-download \
		--max-eval-samples 10 \
		--epochs 10 \
		--batch-size 2"

pipeline-eval-only:
	# Run only evaluation using existing checkpoints (skip training)
	docker run --rm --gpus all -v $$(pwd)/src:$(WORKDIR_CONTAINER)/src:ro \
		-v $$(pwd)/logs:$(WORKDIR_CONTAINER)/logs \
		$(IMAGE_NAME) bash -lc "python -m src.run_complete_pipeline \
		--data-dir logs/data/artefact_real \
		--output-dir logs/pipeline_results \
		--max-eval-samples 20 \
		--skip-training"
