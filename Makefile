# ==============================================================================
# Real-Time Object Detection & Tracking Dashboard - Makefile
# ==============================================================================

# Container runtime binary: detects podman.exe on WSL/Windows, falls back to podman or docker
PODMAN := $(shell which podman.exe 2>/dev/null || which podman 2>/dev/null || echo podman)

# Host volume mount path: converts to Windows path if using podman.exe under WSL
ifeq ($(findstring podman.exe,$(PODMAN)),podman.exe)
MOUNT_PATH := $(shell wslpath -w "$$(pwd)" 2>/dev/null || pwd)
else
MOUNT_PATH := $(shell pwd)
endif

IMAGE_NAME := object-tracking-dashboard-app
PORT := 8501

RUN_CONTAINER := $(PODMAN) run --rm -v "$(MOUNT_PATH):/app:Z" $(IMAGE_NAME)
RUN_INTERACTIVE := $(PODMAN) run --rm -it -v "$(MOUNT_PATH):/app:Z" $(IMAGE_NAME)
RUN_SERVER := $(PODMAN) run --rm -it -p $(PORT):8501 -v "$(MOUNT_PATH):/app:Z" $(IMAGE_NAME)

.PHONY: help build test test-unit test-cov lint format check run dev train-dry-run shell clean

## help: Display this help message (default target)
help:
	@echo "========================================================================"
	@echo " Real-Time Object Detection & Tracking System - Makefile"
	@echo " Runtime: $(PODMAN) | Mount: $(MOUNT_PATH)"
	@echo "========================================================================"
	@echo ""
	@echo "Available Targets:"
	@echo "  make build         - Build the Podman container image"
	@echo "  make test          - Run full pytest test suite (237 tests)"
	@echo "  make test-unit     - Run unit tests with short traceback"
	@echo "  make test-cov      - Run tests with code coverage report"
	@echo "  make lint          - Run Ruff linter"
	@echo "  make format        - Run Ruff code formatter (in-place fix)"
	@echo "  make check         - Run both Ruff linter and format check"
	@echo "  make run / dev     - Launch Streamlit Live Dashboard at http://localhost:8501"
	@echo "  make train-dry-run - Run fine-tuning pipeline smoke test (--dry-run)"
	@echo "  make shell         - Open interactive bash shell inside container"
	@echo "  make clean         - Remove temporary test/cache artifacts"
	@echo ""

## build: Build the Podman container image
build:
	$(PODMAN) build -t $(IMAGE_NAME) -f Containerfile .

## test: Run complete pytest test suite in Podman
test:
	$(RUN_CONTAINER) uv run pytest -v

## test-unit: Run pytest with short traceback
test-unit:
	$(RUN_CONTAINER) uv run pytest -v --tb=short

## test-cov: Run pytest with code coverage report
test-cov:
	$(RUN_CONTAINER) uv run pytest -v --cov=src --cov-report=term-missing

## lint: Run Ruff linter in Podman
lint:
	$(RUN_CONTAINER) uv run ruff check .

## format: Run Ruff formatter in Podman (auto-formats code)
format:
	$(RUN_CONTAINER) uv run ruff format .

## check: Run both Ruff linter and format verification
check:
	$(RUN_CONTAINER) uv run ruff check .
	$(RUN_CONTAINER) uv run ruff format --check .

## run: Launch Streamlit Live Dashboard (accessible at http://localhost:8501)
run:
	@echo "Starting Streamlit dashboard on http://localhost:$(PORT)..."
	$(RUN_SERVER) uv run streamlit run app.py --server.port=$(PORT) --server.address=0.0.0.0 --server.headless=true

## dev: Alias for make run
dev: run

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
