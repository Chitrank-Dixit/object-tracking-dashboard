"""Real-Time Object Detection, Tracking, and Spatial Analytics Package."""

from typing import Any

from src.counter import SpatialAnalytics
from src.schemas import (
    DEFAULT_LINES_JSON,
    DEFAULT_ZONES_JSON,
    AnalyticsSummary,
    CountingLine,
    CountingZone,
    DashboardConfig,
    DetectorConfig,
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
    "DashboardConfig",
    "DetectorConfig",
    "FrameResult",
    "LineCrossingEvent",
    "ObjectTracker",
    "SpatialAnalytics",
    "SpatialAnalyticsConfig",
    "TrackedObject",
    "TrainingConfig",
    "TrainingResult",
    "download_roboflow_dataset",
    "run_training",
]
