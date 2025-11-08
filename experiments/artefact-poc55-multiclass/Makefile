# POC-5.5: Smart Environment Router
# Auto-detects whether running on CITIC SLURM cluster or local Docker

# Detect environment based on hostname
HOSTNAME := $(shell hostname)

# Check if we're on CITIC login nodes
ifneq (,$(findstring logincluster01.citic.ucr,$(HOSTNAME)))
    MAKEFILE_PATH := server/Makefile.server
    ENV_NAME := SLURM (CITIC Server - logincluster01)
    ENV_TYPE := server
else ifneq (,$(findstring logincluster02.citic.ucr,$(HOSTNAME)))
    MAKEFILE_PATH := server/Makefile.server
    ENV_NAME := SLURM (CITIC Server - logincluster02)
    ENV_TYPE := server
else
    MAKEFILE_PATH := docker/Makefile.docker
    ENV_NAME := Docker (Local)
    ENV_TYPE := docker
endif

# Default target
.DEFAULT_GOAL := help

# Special target to show detected environment
.PHONY: env
env:
	@echo "🔍 Environment Detection"
	@echo "========================"
	@echo "Hostname:    $(HOSTNAME)"
	@echo "Detected:    $(ENV_NAME)"
	@echo "Using:       $(MAKEFILE_PATH)"
	@echo ""
	@echo "Available commands:"
	@echo "  make help          - Show available commands"
	@echo "  make env           - Show this environment info"
	@echo ""
	@echo "Common targets:"
	@echo "  make download      - Download dataset"
	@echo "  make train-convnext - Train ConvNeXt-Tiny"
	@echo "  make train-swin    - Train Swin-Tiny"
	@echo "  make train-maxvit  - Train MaxViT-Tiny"
	@echo "  make train-all     - Train all 3 models"
	@echo "  make logs          - View training logs"
	@echo "  make clean         - Clean logs/checkpoints"

# Forward all other targets to the appropriate Makefile
%:
	@echo "🔍 Detected: $(ENV_NAME)"
	@echo "🚀 Running: make -C $(dir $(MAKEFILE_PATH)) -f $(notdir $(MAKEFILE_PATH)) $@"
	@echo ""
	@$(MAKE) -C $(dir $(MAKEFILE_PATH)) -f $(notdir $(MAKEFILE_PATH)) $@
