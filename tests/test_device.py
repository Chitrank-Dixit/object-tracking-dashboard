"""Comprehensive unit test suite for compute device management in src.device.

Tests all hardware probing, VRAM telemetry calculations, compute device resolution
with multi-tier fallback, and cache management in 100% headless, isolated environments.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.device import (
    empty_cuda_cache,
    get_cuda_device_count,
    get_cuda_device_name,
    get_cuda_memory_info,
    is_cuda_available,
    print_gpu_info,
    resolve_compute_device,
)
from src.schemas import CudaDeviceInfo, DeviceResolutionResult

# ==============================================================================
# is_cuda_available Tests
# ==============================================================================


@pytest.mark.unit
def test_is_cuda_available_true(mocker: Any) -> None:
    """Verify is_cuda_available returns True when torch reports CUDA and >=1 device."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 1
    mocker.patch.dict("sys.modules", {"torch": mock_torch})

    assert is_cuda_available() is True


@pytest.mark.unit
def test_is_cuda_available_false_when_torch_unavailable(mocker: Any) -> None:
    """Verify is_cuda_available returns False when torch.cuda.is_available is False."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.cuda.device_count.return_value = 0
    mocker.patch.dict("sys.modules", {"torch": mock_torch})

    assert is_cuda_available() is False


@pytest.mark.unit
def test_is_cuda_available_false_when_device_count_zero(mocker: Any) -> None:
    """Verify is_cuda_available returns False when device count is 0."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 0
    mocker.patch.dict("sys.modules", {"torch": mock_torch})

    assert is_cuda_available() is False


@pytest.mark.unit
def test_is_cuda_available_handles_import_or_runtime_exception(mocker: Any) -> None:
    """Verify is_cuda_available catches any torch exception and returns False."""
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.side_effect = RuntimeError("Driver mismatch")
    mocker.patch.dict("sys.modules", {"torch": mock_torch})

    assert is_cuda_available() is False


# ==============================================================================
# get_cuda_device_count Tests
# ==============================================================================


@pytest.mark.unit
def test_get_cuda_device_count_nominal(mocker: Any) -> None:
    """Verify get_cuda_device_count returns correct device count when available."""
    mocker.patch("src.device.is_cuda_available", return_value=True)
    mock_torch = MagicMock()
    mock_torch.cuda.device_count.return_value = 2
    mocker.patch.dict("sys.modules", {"torch": mock_torch})

    assert get_cuda_device_count() == 2


@pytest.mark.unit
def test_get_cuda_device_count_when_cuda_unavailable(mocker: Any) -> None:
    """Verify get_cuda_device_count returns 0 when CUDA is not available."""
    mocker.patch("src.device.is_cuda_available", return_value=False)
    assert get_cuda_device_count() == 0


@pytest.mark.unit
def test_get_cuda_device_count_handles_exception(mocker: Any) -> None:
    """Verify get_cuda_device_count returns 0 on unexpected exception."""
    mocker.patch("src.device.is_cuda_available", return_value=True)
    mock_torch = MagicMock()
    mock_torch.cuda.device_count.side_effect = RuntimeError("Query error")
    mocker.patch.dict("sys.modules", {"torch": mock_torch})

    assert get_cuda_device_count() == 0


# ==============================================================================
# get_cuda_device_name Tests
# ==============================================================================


@pytest.mark.unit
def test_get_cuda_device_name_nominal(mocker: Any) -> None:
    """Verify get_cuda_device_name returns string device name for valid index."""
    mocker.patch("src.device.is_cuda_available", return_value=True)
    mock_torch = MagicMock()
    mock_torch.cuda.device_count.return_value = 1
    mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 5060 Ti"
    mocker.patch.dict("sys.modules", {"torch": mock_torch})

    name = get_cuda_device_name(0)
    assert name == "NVIDIA GeForce RTX 5060 Ti"
    mock_torch.cuda.get_device_name.assert_called_once_with(0)


@pytest.mark.unit
def test_get_cuda_device_name_when_cuda_unavailable(mocker: Any) -> None:
    """Verify get_cuda_device_name returns None when CUDA is unavailable."""
    mocker.patch("src.device.is_cuda_available", return_value=False)
    assert get_cuda_device_name(0) is None


@pytest.mark.unit
@pytest.mark.parametrize("invalid_index", [-1, 2, 99])
def test_get_cuda_device_name_out_of_bounds_returns_none(mocker: Any, invalid_index: int) -> None:
    """Verify get_cuda_device_name returns None for invalid or out-of-bounds index."""
    mocker.patch("src.device.is_cuda_available", return_value=True)
    mock_torch = MagicMock()
    mock_torch.cuda.device_count.return_value = 1
    mocker.patch.dict("sys.modules", {"torch": mock_torch})

    assert get_cuda_device_name(invalid_index) is None


@pytest.mark.unit
def test_get_cuda_device_name_handles_exception(mocker: Any) -> None:
    """Verify get_cuda_device_name returns None when query throws exception."""
    mocker.patch("src.device.is_cuda_available", return_value=True)
    mock_torch = MagicMock()
    mock_torch.cuda.device_count.return_value = 1
    mock_torch.cuda.get_device_name.side_effect = RuntimeError("Driver error")
    mocker.patch.dict("sys.modules", {"torch": mock_torch})

    assert get_cuda_device_name(0) is None


# ==============================================================================
# get_cuda_memory_info Tests
# ==============================================================================


@pytest.mark.unit
def test_get_cuda_memory_info_nominal(mocker: Any) -> None:
    """Verify get_cuda_memory_info returns correctly computed CudaDeviceInfo model."""
    mocker.patch("src.device.is_cuda_available", return_value=True)
    mocker.patch("src.device.get_cuda_device_name", return_value="NVIDIA GeForce RTX 5060 Ti")

    mock_torch = MagicMock()
    mock_torch.cuda.device_count.return_value = 1

    # 16 GB total = 16 * 1024 * 1024 * 1024 bytes = 17179869184
    # 2 GB allocated = 2 * 1024 * 1024 * 1024 = 2147483648
    # 3 GB reserved = 3 * 1024 * 1024 * 1024 = 3221225472
    props = MagicMock()
    props.total_memory = 16 * 1024 * 1024 * 1024
    mock_torch.cuda.get_device_properties.return_value = props
    mock_torch.cuda.memory_allocated.return_value = 2 * 1024 * 1024 * 1024
    mock_torch.cuda.memory_reserved.return_value = 3 * 1024 * 1024 * 1024
    mocker.patch.dict("sys.modules", {"torch": mock_torch})

    info = get_cuda_memory_info(0)
    assert info is not None
    assert isinstance(info, CudaDeviceInfo)
    assert info.device_name == "NVIDIA GeForce RTX 5060 Ti"
    assert info.device_index == 0
    assert info.total_memory_mb == 16384.0
    assert info.allocated_memory_mb == 2048.0
    assert info.reserved_memory_mb == 3072.0
    assert info.free_memory_mb == 14336.0
    assert info.utilization_pct == pytest.approx(12.5, rel=1e-2)


@pytest.mark.unit
def test_get_cuda_memory_info_when_cuda_unavailable(mocker: Any) -> None:
    """Verify get_cuda_memory_info returns None when CUDA is unavailable."""
    mocker.patch("src.device.is_cuda_available", return_value=False)
    assert get_cuda_memory_info(0) is None


@pytest.mark.unit
@pytest.mark.parametrize("invalid_index", [-1, 1, 10])
def test_get_cuda_memory_info_out_of_bounds_returns_none(mocker: Any, invalid_index: int) -> None:
    """Verify get_cuda_memory_info returns None for index out of bounds."""
    mocker.patch("src.device.is_cuda_available", return_value=True)
    mock_torch = MagicMock()
    mock_torch.cuda.device_count.return_value = 1
    mocker.patch.dict("sys.modules", {"torch": mock_torch})

    assert get_cuda_memory_info(invalid_index) is None


@pytest.mark.unit
def test_get_cuda_memory_info_handles_exception(mocker: Any) -> None:
    """Verify get_cuda_memory_info catches exception and returns None."""
    mocker.patch("src.device.is_cuda_available", return_value=True)
    mock_torch = MagicMock()
    mock_torch.cuda.device_count.return_value = 1
    mock_torch.cuda.get_device_properties.side_effect = RuntimeError("VRAM read error")
    mocker.patch.dict("sys.modules", {"torch": mock_torch})

    assert get_cuda_memory_info(0) is None


# ==============================================================================
# resolve_compute_device Tests
# ==============================================================================


@pytest.mark.unit
def test_resolve_compute_device_explicit_cpu() -> None:
    """Verify requesting 'cpu' directly resolves to CPU without fallback."""
    res = resolve_compute_device("cpu")
    assert isinstance(res, DeviceResolutionResult)
    assert res.requested_device == "cpu"
    assert res.resolved_device == "cpu"
    assert res.device_type == "cpu"
    assert res.is_fallback is False
    assert res.fallback_reason is None


@pytest.mark.unit
def test_resolve_compute_device_auto_when_cuda_available(mocker: Any) -> None:
    """Verify 'auto' resolves to 'cuda:0' when CUDA is available."""
    mocker.patch("src.device.is_cuda_available", return_value=True)

    res = resolve_compute_device("auto")
    assert res.requested_device == "auto"
    assert res.resolved_device == "cuda:0"
    assert res.device_type == "cuda"
    assert res.is_fallback is False
    assert res.fallback_reason is None


@pytest.mark.unit
def test_resolve_compute_device_auto_when_cuda_unavailable(mocker: Any) -> None:
    """Verify 'auto' resolves to 'cpu' when CUDA is unavailable."""
    mocker.patch("src.device.is_cuda_available", return_value=False)

    res = resolve_compute_device("auto")
    assert res.requested_device == "auto"
    assert res.resolved_device == "cpu"
    assert res.device_type == "cpu"
    assert res.is_fallback is False
    assert "Auto-selected CPU" in (res.fallback_reason or "")


@pytest.mark.unit
@pytest.mark.parametrize("requested", ["cuda", "cuda:0", "0"])
def test_resolve_compute_device_explicit_cuda_fallback_when_unavailable(
    mocker: Any, requested: str
) -> None:
    """Verify requesting CUDA when unavailable triggers Tier 1 fallback to CPU."""
    mocker.patch("src.device.is_cuda_available", return_value=False)

    res = resolve_compute_device(requested)
    assert res.requested_device == requested
    assert res.resolved_device == "cpu"
    assert res.device_type == "cpu"
    assert res.is_fallback is True
    assert "unavailable" in (res.fallback_reason or "").lower()


@pytest.mark.unit
def test_resolve_compute_device_explicit_cuda_when_available(mocker: Any) -> None:
    """Verify requesting CUDA when available resolves to valid cuda string."""
    mocker.patch("src.device.is_cuda_available", return_value=True)
    mock_torch = MagicMock()
    mock_torch.cuda.device_count.return_value = 2
    mocker.patch.dict("sys.modules", {"torch": mock_torch})

    res = resolve_compute_device("cuda:1")
    assert res.requested_device == "cuda:1"
    assert res.resolved_device == "cuda:1"
    assert res.device_type == "cuda"
    assert res.is_fallback is False


@pytest.mark.unit
def test_resolve_compute_device_integer_index_string(mocker: Any) -> None:
    """Verify requesting '0' or '1' integer string maps to 'cuda:<idx>'."""
    mocker.patch("src.device.is_cuda_available", return_value=True)
    mock_torch = MagicMock()
    mock_torch.cuda.device_count.return_value = 2
    mocker.patch.dict("sys.modules", {"torch": mock_torch})

    res = resolve_compute_device("1")
    assert res.resolved_device == "cuda:1"
    assert res.device_type == "cuda"
    assert res.is_fallback is False


@pytest.mark.unit
def test_resolve_compute_device_index_out_of_bounds_fallback(mocker: Any) -> None:
    """Verify requesting out-of-bounds CUDA index defaults to cuda:0 with fallback."""
    mocker.patch("src.device.is_cuda_available", return_value=True)
    mock_torch = MagicMock()
    mock_torch.cuda.device_count.return_value = 1
    mocker.patch.dict("sys.modules", {"torch": mock_torch})

    res = resolve_compute_device("cuda:4")
    assert res.requested_device == "cuda:4"
    assert res.resolved_device == "cuda:0"
    assert res.device_type == "cuda"
    assert res.is_fallback is True
    assert "out of bounds" in (res.fallback_reason or "").lower()


# ==============================================================================
# empty_cuda_cache Tests
# ==============================================================================


@pytest.mark.unit
def test_empty_cuda_cache_invoked_when_cuda_available(mocker: Any) -> None:
    """Verify empty_cuda_cache calls torch.cuda.empty_cache() when CUDA is available."""
    mocker.patch("src.device.is_cuda_available", return_value=True)
    mock_torch = MagicMock()
    mocker.patch.dict("sys.modules", {"torch": mock_torch})

    empty_cuda_cache()
    mock_torch.cuda.empty_cache.assert_called_once()


@pytest.mark.unit
def test_empty_cuda_cache_noop_when_cuda_unavailable(mocker: Any) -> None:
    """Verify empty_cuda_cache is a safe no-op when CUDA is unavailable."""
    mocker.patch("src.device.is_cuda_available", return_value=False)
    mock_torch = MagicMock()
    mocker.patch.dict("sys.modules", {"torch": mock_torch})

    empty_cuda_cache()
    mock_torch.cuda.empty_cache.assert_not_called()


@pytest.mark.unit
def test_empty_cuda_cache_handles_exception_gracefully(mocker: Any) -> None:
    """Verify empty_cuda_cache does not propagate exceptions from empty_cache."""
    mocker.patch("src.device.is_cuda_available", return_value=True)
    mock_torch = MagicMock()
    mock_torch.cuda.empty_cache.side_effect = RuntimeError("CUDA driver lost")
    mocker.patch.dict("sys.modules", {"torch": mock_torch})

    # Should not raise
    empty_cuda_cache()


# ==============================================================================
# print_gpu_info Tests
# ==============================================================================


@pytest.mark.unit
def test_print_gpu_info_cuda_unavailable(mocker: Any, capsys: Any) -> None:
    """Verify print_gpu_info prints expected message when CUDA is unavailable."""
    mocker.patch("src.device.is_cuda_available", return_value=False)

    print_gpu_info()
    captured = capsys.readouterr().out
    assert "CUDA Available:  FALSE" in captured
    assert "No CUDA-compatible GPU detected" in captured


@pytest.mark.unit
def test_print_gpu_info_cuda_available(mocker: Any, capsys: Any) -> None:
    """Verify print_gpu_info prints full telemetry report when CUDA is available."""
    mocker.patch("src.device.is_cuda_available", return_value=True)
    mocker.patch("src.device.get_cuda_device_count", return_value=1)
    mock_torch = MagicMock()
    mock_torch.version.cuda = "12.8"
    mocker.patch.dict("sys.modules", {"torch": mock_torch})

    mock_info = CudaDeviceInfo(
        device_name="NVIDIA GeForce RTX 5060 Ti",
        device_index=0,
        total_memory_mb=16384.0,
        allocated_memory_mb=1024.0,
        reserved_memory_mb=2048.0,
        free_memory_mb=15360.0,
        utilization_pct=6.25,
    )
    mocker.patch("src.device.get_cuda_memory_info", return_value=mock_info)

    print_gpu_info()
    captured = capsys.readouterr().out
    assert "CUDA Available:  TRUE" in captured
    assert "Device Count:    1" in captured
    assert "CUDA Version:    12.8" in captured
    assert "NVIDIA GeForce RTX 5060 Ti" in captured
    assert "16.00 GB" in captured
    assert "6.25%" in captured
