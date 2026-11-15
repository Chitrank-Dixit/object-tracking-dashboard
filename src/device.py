"""Hardware abstraction and compute device management for CUDA and CPU.

Provides thread-safe hardware detection, VRAM telemetry queries,
cache management, and multi-tier graceful fallback resolution.
"""

from __future__ import annotations

import logging
from typing import Any

from src.schemas import CudaDeviceInfo, DeviceResolutionResult

logger = logging.getLogger(__name__)


def is_cuda_available() -> bool:
    """Check if CUDA runtime and compatible NVIDIA GPU are accessible.

    Returns:
        True if torch.cuda.is_available() reports True and at least 1 device exists.
    """
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception as exc:
        logger.debug("CUDA availability probe failed: %s", exc)
        return False


def get_cuda_device_count() -> int:
    """Return count of accessible CUDA devices.

    Returns:
        Integer count of devices (0 if CUDA is unavailable).
    """
    if not is_cuda_available():
        return 0
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def get_cuda_device_name(device_index: int = 0) -> str | None:
    """Retrieve human-readable name of specified CUDA device.

    Args:
        device_index: Ordinal index of GPU device.

    Returns:
        Device name string (e.g. 'NVIDIA GeForce RTX 5060 Ti') or None if unavailable.
    """
    if not is_cuda_available():
        return None
    try:
        import torch

        if device_index < 0 or device_index >= torch.cuda.device_count():
            return None
        return str(torch.cuda.get_device_name(device_index))
    except Exception as exc:
        logger.warning("Failed to retrieve CUDA device name for index %d: %s", device_index, exc)
        return None


def get_cuda_memory_info(device_index: int = 0) -> CudaDeviceInfo | None:
    """Query real-time VRAM allocation and reserved memory for CUDA device.

    Args:
        device_index: Ordinal index of GPU device.

    Returns:
        Validated CudaDeviceInfo schema or None if CUDA is unavailable or query fails.
    """
    if not is_cuda_available():
        return None

    try:
        import torch

        if device_index < 0 or device_index >= torch.cuda.device_count():
            return None

        device_name = get_cuda_device_name(device_index) or f"CUDA Device {device_index}"
        props: Any = torch.cuda.get_device_properties(device_index)
        total_bytes: int = int(props.total_memory)
        allocated_bytes: int = int(torch.cuda.memory_allocated(device_index))
        reserved_bytes: int = int(torch.cuda.memory_reserved(device_index))

        total_mb = round(total_bytes / (1024 * 1024), 2)
        alloc_mb = round(allocated_bytes / (1024 * 1024), 2)
        res_mb = round(reserved_bytes / (1024 * 1024), 2)
        free_mb = max(0.0, round(total_mb - alloc_mb, 2))
        util_pct = round((alloc_mb / total_mb) * 100.0, 2) if total_mb > 0.0 else 0.0
        util_pct = min(100.0, max(0.0, util_pct))

        return CudaDeviceInfo(
            device_name=device_name,
            device_index=device_index,
            total_memory_mb=max(1.0, total_mb),
            allocated_memory_mb=alloc_mb,
            reserved_memory_mb=res_mb,
            free_memory_mb=free_mb,
            utilization_pct=util_pct,
        )
    except Exception as exc:
        logger.warning("Failed to query CUDA memory info for index %d: %s", device_index, exc)
        return None


def resolve_compute_device(requested: str = "auto") -> DeviceResolutionResult:
    """Resolve user or configuration requested device into a concrete compute target.

    Implements graceful fallback negotiation:
    - 'auto': chooses 'cuda:0' if CUDA is accessible; otherwise 'cpu' (not considered a fallback).
    - 'cuda' or 'cuda:<idx>' or '<idx>': verifies CUDA availability. If unavailable, falls back
      to 'cpu' with is_fallback=True and diagnostic explanation.
    - 'cpu': resolves to 'cpu' with is_fallback=False.

    Args:
        requested: Requested device preference string.

    Returns:
        Validated DeviceResolutionResult contract.
    """
    req_clean = requested.strip().lower()

    if req_clean == "cpu":
        return DeviceResolutionResult(
            requested_device=requested,
            resolved_device="cpu",
            device_type="cpu",
            is_fallback=False,
            fallback_reason=None,
        )

    cuda_ok = is_cuda_available()

    if req_clean == "auto":
        if cuda_ok:
            return DeviceResolutionResult(
                requested_device=requested,
                resolved_device="cuda:0",
                device_type="cuda",
                is_fallback=False,
                fallback_reason=None,
            )
        return DeviceResolutionResult(
            requested_device=requested,
            resolved_device="cpu",
            device_type="cpu",
            is_fallback=False,
            fallback_reason="Auto-selected CPU because CUDA is not available.",
        )

    # User explicitly requested a CUDA device ('cuda', 'cuda:0', '0')
    if not cuda_ok:
        logger.warning(
            "CUDA device '%s' was requested, but CUDA is unavailable. Falling back to CPU.",
            requested,
        )
        return DeviceResolutionResult(
            requested_device=requested,
            resolved_device="cpu",
            device_type="cpu",
            is_fallback=True,
            fallback_reason=f"Requested device '{requested}' is unavailable. Operating on CPU.",
        )

    # Validate device index if specified
    import torch

    device_count = torch.cuda.device_count()
    target_idx = 0

    if req_clean.startswith("cuda:"):
        try:
            target_idx = int(req_clean.split("cuda:")[1])
        except ValueError:
            target_idx = 0
    elif req_clean.isdigit():
        target_idx = int(req_clean)

    if target_idx >= device_count:
        logger.warning(
            "Requested CUDA device index %d exceeds available devices (%d). Falling back to cuda:0.",
            target_idx,
            device_count,
        )
        return DeviceResolutionResult(
            requested_device=requested,
            resolved_device="cuda:0",
            device_type="cuda",
            is_fallback=True,
            fallback_reason=(
                f"Requested CUDA index {target_idx} is out of bounds (found {device_count} devices). "
                "Defaulting to cuda:0."
            ),
        )

    resolved_str = f"cuda:{target_idx}"
    return DeviceResolutionResult(
        requested_device=requested,
        resolved_device=resolved_str,
        device_type="cuda",
        is_fallback=False,
        fallback_reason=None,
    )


def empty_cuda_cache() -> None:
    """Safely release unused cached memory from PyTorch CUDA memory allocator."""
    if not is_cuda_available():
        return
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception as exc:
        logger.debug("torch.cuda.empty_cache() failed: %s", exc)


def print_gpu_info() -> None:
    """Print formatted hardware and VRAM diagnostic report to stdout."""
    print("=" * 65)
    print("           NVIDIA CUDA HARDWARE & VRAM TELEMETRY REPORT")
    print("=" * 65)

    if not is_cuda_available():
        print("  CUDA Available:  FALSE")
        print("  Status:          No CUDA-compatible GPU detected in container.")
        print("  Advice:          Ensure /dev/dxg and /usr/lib/wsl are mounted.")
        print("=" * 65)
        return

    import torch

    count = get_cuda_device_count()
    print("  CUDA Available:  TRUE")
    print(f"  Device Count:    {count}")
    print(f"  CUDA Version:    {torch.version.cuda}")

    for idx in range(count):
        info = get_cuda_memory_info(idx)
        print("-" * 65)
        if info is not None:
            print(f"  GPU [{idx}]:         {info.device_name}")
            print(
                f"  Total VRAM:      {info.total_memory_mb / 1024:.2f} GB ({info.total_memory_mb:.1f} MB)"
            )
            print(f"  Allocated VRAM:  {info.allocated_memory_mb:.2f} MB")
            print(f"  Reserved VRAM:   {info.reserved_memory_mb:.2f} MB")
            print(
                f"  Free VRAM:       {info.free_memory_mb / 1024:.2f} GB ({info.free_memory_mb:.1f} MB)"
            )
            print(f"  Utilization:     {info.utilization_pct:.2f}%")
        else:
            name = get_cuda_device_name(idx) or "Unknown"
            print(f"  GPU [{idx}]:         {name} (Memory telemetry unavailable)")

    print("=" * 65)


if __name__ == "__main__":
    print_gpu_info()
