"""Real-Time Object Detection, Tracking, and Spatial Analytics Package."""

from typing import Any

from src.counter import SpatialAnalytics
from src.device import (
    empty_cuda_cache,
    get_cuda_device_name,
    get_cuda_memory_info,
    is_cuda_available,
    resolve_compute_device,
)
from src.schemas import (
    DEFAULT_LINES_JSON,
    DEFAULT_ZONES_JSON,
    AnalyticsSummary,
    CountingLine,
    CountingZone,
    CudaDeviceInfo,
    DashboardConfig,
    DetectorConfig,
    DeviceResolutionResult,
    FrameResult,
    LineCrossingEvent,
    SpatialAnalyticsConfig,
    TrackedObject,
    TrainingConfig,
    TrainingResult,
)
from src.tracker import ObjectTracker


def run_training(*args: Any, **kwargs: Any) -> Any:
    """Execute training pipeline (lazy-imported to avoid runpy collision)."""
    from src.train import run_training as _run_training

    return _run_training(*args, **kwargs)


def download_roboflow_dataset(*args: Any, **kwargs: Any) -> Any:
    """Download Roboflow dataset (lazy-imported to avoid runpy collision)."""
    from src.train import download_roboflow_dataset as _download_roboflow_dataset

    return _download_roboflow_dataset(*args, **kwargs)


__all__ = [
    "DEFAULT_LINES_JSON",
    "DEFAULT_ZONES_JSON",
    "AnalyticsSummary",
    "CountingLine",
    "CountingZone",
    "CudaDeviceInfo",
    "DashboardConfig",
    "DetectorConfig",
    "DeviceResolutionResult",
    "FrameResult",
    "LineCrossingEvent",
    "ObjectTracker",
    "SpatialAnalytics",
    "SpatialAnalyticsConfig",
    "TrackedObject",
    "TrainingConfig",
    "TrainingResult",
    "download_roboflow_dataset",
    "empty_cuda_cache",
    "get_cuda_device_name",
    "get_cuda_memory_info",
    "is_cuda_available",
    "resolve_compute_device",
    "run_training",
]
