# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:latest AS uv_bin
FROM python:3.12-slim-bookworm

# Copy official uv binaries into system path
COPY --from=uv_bin /uv /uvx /bin/

# Enforce system-wide Python environment without virtual environment nesting
ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Minimal base dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Optimize layer caching: copy dependency manifests first
COPY pyproject.toml uv.lock* ./

# Sync dependencies into system Python without installing root package
RUN uv sync --no-install-project

# Copy project source code
COPY . .

# Complete synchronization
RUN uv sync

# Default command: execute Pytest test suite
CMD ["uv", "run", "pytest", "-v"]
