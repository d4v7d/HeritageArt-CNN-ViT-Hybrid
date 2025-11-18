.PHONY: help build shell download-small download-medium download-full clean-data clean-all quickstart

help: ## Show this help message
	@echo "ARTeFACT Data Obtention - Docker Commands"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Build Docker image
	cd docker && docker-compose build

shell: ## Open interactive shell in container
	cd docker && docker-compose run --rm artefact-data-obtention bash

download-small: ## Download 10 samples with visualizations (quick test)
	cd docker && docker-compose run --rm artefact-data-obtention python3 scripts/download_artefact.py --output ../artefact_test --max-samples 10

download-medium: ## Download 50 samples without visualizations (medium test)
	cd docker && docker-compose run --rm artefact-data-obtention python3 scripts/download_artefact.py --output ../artefact_test --max-samples 50 --no-viz

download-full: ## Download ALL samples to ../artefact/ (417/418, requires 16GB RAM)
	cd docker && docker-compose run --rm artefact-data-obtention python3 scripts/download_artefact.py --output ../artefact --all --no-viz

generate-augmentations: ## Generate augmented dataset ../artefact_augmented/ from ../artefact/
	cd docker && docker-compose run --rm artefact-data-obtention python3 scripts/generate_augmentations.py

full-workflow: download-full generate-augmentations ## Download original + generate augmentations (complete pipeline)
	@echo ""
	@echo "✓ Full workflow complete!"
	@echo "  ../artefact/ - 417 original samples"
	@echo "  ../artefact_augmented/ - 1458 augmented samples"

clean-data: ## Remove test downloads
	rm -rf ../artefact_test

clean-all: clean-data ## Remove all data and Docker image
	cd docker && docker-compose down
	docker rmi artefact-data-obtention:latest || true

quickstart: build download-small ## Build and download small test (10 samples)
	@echo ""
	@echo " Quickstart complete! Test data available in ../artefact_test/"
	@echo ""
	@echo "Next steps:"
	@echo "  make download-full           - Download full dataset to ../artefact/"
	@echo "  make generate-augmentations  - Generate ../artefact_augmented/"
	@echo "  make full-workflow           - Do both (complete pipeline)"
