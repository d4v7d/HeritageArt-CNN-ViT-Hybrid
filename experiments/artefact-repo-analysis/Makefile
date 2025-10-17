.PHONY: help build shell clone download process inspect visualize clean

help: ## Show this help message
	@echo "ARTeFACT Repository Analysis - Docker Commands"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Build Docker image
	cd docker && docker-compose build

shell: ## Open interactive shell in container
	cd docker && docker-compose run --rm artefact-repo-analysis bash

clone: ## Clone ARTeFACT repository (one-time setup)
	cd docker && docker-compose run --rm artefact-repo-analysis bash -c "\
		if [ ! -d artefact_repo/.git ]; then \
			cd artefact_repo && git clone https://huggingface.co/datasets/danielaivanova/damaged-media . ; \
		else \
			echo 'Repository already cloned'; \
		fi"

download: ## Download first parquet file (train-00000)
	cd docker && docker-compose run --rm artefact-repo-analysis bash -c "\
		cd artefact_repo && git lfs pull --include='data/train-00000-of-00028.parquet'"

download-all: ## Download ALL parquet files (WARNING: ~50GB+)
	cd docker && docker-compose run --rm artefact-repo-analysis bash -c "\
		cd artefact_repo && git lfs pull"

process: ## Process parquet files to extract images
	cd docker && docker-compose run --rm artefact-repo-analysis python3 scripts/process_parquet.py \
		--input "artefact_repo/data/train-00000-of-00028.parquet" \
		--output ./data/processed \
		--max-image-size 512

process-all: ## Process ALL downloaded parquet files
	cd docker && docker-compose run --rm artefact-repo-analysis python3 scripts/process_parquet.py \
		--input "artefact_repo/data/train-*.parquet" \
		--output ./data/processed_all \
		--max-image-size 512

inspect: ## Inspect parquet file structure
	cd docker && docker-compose run --rm artefact-repo-analysis python3 scripts/inspect_parquet.py \
		artefact_repo/data/train-00000-of-00028.parquet

visualize: ## Create visualizations from processed data
	cd docker && docker-compose run --rm artefact-repo-analysis python3 scripts/visualize_samples.py \
		--metadata ./data/processed/metadata.csv \
		--output ./data/visualizations

clean-data: ## Remove processed data (keeps cloned repo)
	rm -rf data/processed data/processed_all data/visualizations data/docker_test

clean-repo: ## Remove cloned repository
	rm -rf artefact_repo/*

clean-all: clean-data clean-repo ## Remove everything (data + repo)

# Quick start workflow
quickstart: build clone download process ## Complete workflow: build, clone, download, and process first file
	@echo ""
	@echo " Quickstart complete! Data available in ./data/processed"
	@echo ""
	@echo "Next steps:"
	@echo "  make visualize    - Generate sample visualizations"
	@echo "  make download-all - Download more parquet files"
	@echo "  make process-all  - Process all downloaded files"
