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
	cd docker && docker-compose run --rm artefact-data-obtention python3 scripts/download_artefact.py --output data/test_small --max-samples 10

download-medium: ## Download 50 samples without visualizations (medium test)
	cd docker && docker-compose run --rm artefact-data-obtention python3 scripts/download_artefact.py --output data/test_medium --max-samples 50 --no-viz

download-full: ## Download ALL samples (417/418, requires 16GB RAM)
	cd docker && docker-compose run --rm artefact-data-obtention python3 scripts/download_artefact.py --output data/full_dataset --all --no-viz

clean-data: ## Remove all downloaded data
	rm -rf data/test_* data/full_dataset

clean-all: clean-data ## Remove all data and Docker image
	cd docker && docker-compose down
	docker rmi artefact-data-obtention:latest || true

quickstart: build download-small ## Build and download small test (10 samples)
	@echo ""
	@echo "✅ Quickstart complete! Data available in ./data/test_small"
	@echo ""
	@echo "Next steps:"
	@echo "  make download-medium  - Download 50 samples"
	@echo "  make download-full    - Download full dataset (417 samples)"
