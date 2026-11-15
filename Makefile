# ==============================================================================
# Real-Time Object Detection & Tracking Dashboard - Makefile
# Hardware Acceleration: NVIDIA CUDA (WSL2 /dev/dxg & /usr/lib/wsl)
# ==============================================================================

# Container runtime binary: detects podman.exe on WSL/Windows, falls back to podman or docker
PODMAN := $(shell which podman.exe 2>/dev/null || which podman 2>/dev/null || echo podman)

# Host volume mount path: converts to Windows path if using podman.exe under WSL
ifeq ($(findstring podman.exe,$(PODMAN)),podman.exe)
MOUNT_PATH := $(shell wslpath -w "$$(pwd)" 2>/dev/null || pwd)
else
MOUNT_PATH := $(shell pwd)
endif

# Check for WSL2 NVIDIA GPU passthrough support (/dev/dxg device + /usr/lib/wsl directory)
HAS_GPU := $(shell test -c /dev/dxg -a -d /usr/lib/wsl 2>/dev/null && echo 1 || echo 0)

ifeq ($(HAS_GPU),1)
GPU_FLAGS := --device /dev/dxg -v /usr/lib/wsl:/usr/lib/wsl:ro -e LD_LIBRARY_PATH=/usr/lib/wsl/lib
GPU_STATUS := Active (NVIDIA CUDA via WSL2 /dev/dxg)
else
GPU_FLAGS :=
GPU_STATUS := Not Detected (Falling back to CPU)
endif

IMAGE_NAME := object-tracking-dashboard-app
PORT := 8501

# Standard container runners (auto-detects GPU flags)
RUN_CONTAINER := $(PODMAN) run --rm $(GPU_FLAGS) -v "$(MOUNT_PATH):/app:Z" $(IMAGE_NAME)
RUN_INTERACTIVE := $(PODMAN) run --rm -it $(GPU_FLAGS) -v "$(MOUNT_PATH):/app:Z" $(IMAGE_NAME)
RUN_SERVER := $(PODMAN) run --rm -it -p $(PORT):8501 $(GPU_FLAGS) -v "$(MOUNT_PATH):/app:Z" $(IMAGE_NAME)

# Explicit CPU-only runners
RUN_CONTAINER_CPU := $(PODMAN) run --rm -v "$(MOUNT_PATH):/app:Z" $(IMAGE_NAME)
RUN_SERVER_CPU := $(PODMAN) run --rm -it -p $(PORT):8501 -v "$(MOUNT_PATH):/app:Z" $(IMAGE_NAME)

# Explicit GPU runners
RUN_CONTAINER_GPU := $(PODMAN) run --rm --device /dev/dxg -v /usr/lib/wsl:/usr/lib/wsl:ro -e LD_LIBRARY_PATH=/usr/lib/wsl/lib -v "$(MOUNT_PATH):/app:Z" $(IMAGE_NAME)
RUN_SERVER_GPU := $(PODMAN) run --rm -it -p $(PORT):8501 --device /dev/dxg -v /usr/lib/wsl:/usr/lib/wsl:ro -e LD_LIBRARY_PATH=/usr/lib/wsl/lib -v "$(MOUNT_PATH):/app:Z" $(IMAGE_NAME)

.PHONY: help build test test-unit test-cov test-gpu lint format check run run-gpu run-cpu dev gpu-info train-dry-run shell clean

## help: Display this help message (default target)
help:
	@echo "========================================================================"
	@echo " Real-Time Object Detection & Tracking System - Makefile"
	@echo " Runtime: $(PODMAN) | Mount: $(MOUNT_PATH)"
	@echo " GPU Acceleration: $(GPU_STATUS)"
	@echo "========================================================================"
	@echo ""
	@echo "Available Targets:"
	@echo "  make build         - Build the Podman container image"
	@echo "  make test          - Run full pytest test suite"
	@echo "  make test-gpu      - Run pytest inside GPU-enabled container"
	@echo "  make test-unit     - Run unit tests with short traceback"
	@echo "  make test-cov      - Run tests with code coverage report"
	@echo "  make lint          - Run Ruff linter"
	@echo "  make format        - Run Ruff code formatter (in-place fix)"
	@echo "  make check         - Run both Ruff linter and format check"
	@echo "  make run / dev     - Launch Live Dashboard at http://localhost:8501 (Auto GPU/CPU)"
	@echo "  make run-gpu       - Force launch Live Dashboard with CUDA GPU acceleration"
	@echo "  make run-cpu       - Force launch Live Dashboard in CPU-only mode"
	@echo "  make gpu-info      - Probe and display active GPU & VRAM telemetry"
	@echo "  make train-dry-run - Run fine-tuning pipeline smoke test (--dry-run)"
	@echo "  make shell         - Open interactive bash shell inside container"
	@echo "  make clean         - Remove temporary test/cache artifacts"
	@echo ""

## build: Build the Podman container image
build:
	$(PODMAN) build -t $(IMAGE_NAME) -f Containerfile .

## test: Run complete pytest test suite in Podman (CPU safe for CI)
test:
	$(RUN_CONTAINER_CPU) uv run pytest -v

## test-gpu: Run pytest inside GPU-enabled container
test-gpu:
	$(RUN_CONTAINER_GPU) uv run pytest -v

## test-unit: Run pytest with short traceback
test-unit:
	$(RUN_CONTAINER_CPU) uv run pytest -v --tb=short

## test-cov: Run pytest with code coverage report
test-cov:
	$(RUN_CONTAINER_CPU) uv run pytest -v --cov=src --cov-report=term-missing

## lint: Run Ruff linter in Podman
lint:
	$(RUN_CONTAINER_CPU) uv run ruff check .

## format: Run Ruff formatter in Podman (auto-formats code)
format:
	$(RUN_CONTAINER_CPU) uv run ruff format .

## check: Run both Ruff linter and format verification
check:
	$(RUN_CONTAINER_CPU) uv run ruff check .
	$(RUN_CONTAINER_CPU) uv run ruff format --check .

## run: Launch Streamlit Live Dashboard (Auto-detects GPU or CPU)
run:
	@echo "Starting Streamlit dashboard on http://localhost:$(PORT) [GPU: $(GPU_STATUS)]..."
	$(RUN_SERVER) uv run streamlit run app.py --server.port=$(PORT) --server.address=0.0.0.0 --server.headless=true

## dev: Alias for make run
dev: run

## run-gpu: Force launch Streamlit Live Dashboard with CUDA GPU acceleration
run-gpu:
	@echo "Force-starting Streamlit dashboard with CUDA GPU acceleration on http://localhost:$(PORT)..."
	$(RUN_SERVER_GPU) uv run streamlit run app.py --server.port=$(PORT) --server.address=0.0.0.0 --server.headless=true

## run-cpu: Force launch Streamlit Live Dashboard in CPU-only mode
run-cpu:
	@echo "Starting Streamlit dashboard in CPU-only mode on http://localhost:$(PORT)..."
	$(RUN_SERVER_CPU) uv run streamlit run app.py --server.port=$(PORT) --server.address=0.0.0.0 --server.headless=true

## gpu-info: Display GPU hardware & VRAM allocation details inside container
gpu-info:
	@$(RUN_CONTAINER_GPU) uv run python -m src.device

## train-dry-run: Run fine-tuning pipeline in dry-run validation mode
train-dry-run:
	$(RUN_CONTAINER) uv run python -m src.train --data SPEC.md --dry-run

## shell: Open an interactive bash shell inside the container
shell:
	$(RUN_INTERACTIVE) /bin/bash

## clean: Clean up caches and temporary directories
clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ src/__pycache__ tests/__pycache__ runs
	@echo "Clean complete."
