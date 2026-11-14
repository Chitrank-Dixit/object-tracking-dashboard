"""Pytest fixtures and synthetic test utilities for object tracking tests.

Zero hardware dependencies: all video frames and bounding boxes are generated
synthetically with NumPy arrays and deterministic coordinate fixtures.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from src.schemas import TrackedObject


@pytest.fixture
def sample_synthetic_frame() -> np.ndarray:
    """Generates a standard 640x480 3-channel synthetic black frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_frame_with_box() -> np.ndarray:
    """Generates a synthetic frame with a high-contrast box pattern."""
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    frame[100:200, 100:200] = 255
    return frame


@pytest.fixture
def valid_tracked_object() -> TrackedObject:
    """Generates a valid nominal TrackedObject instance."""
    return TrackedObject(
        track_id=1,
        class_id=0,
        class_name="person",
        confidence=0.88,
        bbox_xyxy=(100.0, 150.0, 200.0, 250.0),
    )


class MockBoxes:
    """Mock container mimicking Ultralytics Results.boxes."""

    def __init__(
        self,
        xyxy: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
        ids: np.ndarray | None = None,
    ) -> None:
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls
        self.id = ids

    def __len__(self) -> int:
        return len(self.xyxy)


class MockResult:
    """Mock container mimicking Ultralytics Results object."""

    def __init__(
        self,
        boxes: MockBoxes | None,
        names: dict[int, str] | None = None,
        speed: dict[str, float] | None = None,
    ) -> None:
        self.boxes = boxes
        self.names = names if names is not None else {0: "person", 1: "bicycle", 2: "car"}
        self.speed = (
            speed
            if speed is not None
            else {
                "preprocess": 1.0,
                "inference": 15.0,
                "postprocess": 1.0,
            }
        )


@pytest.fixture
def mock_yolo_result_factory() -> Callable[..., MockResult]:
    """Factory fixture creating mock Ultralytics Results objects."""

    def _factory(
        boxes_xyxy: list[list[float]] | np.ndarray | None = None,
        confidences: list[float] | np.ndarray | None = None,
        class_ids: list[int] | np.ndarray | None = None,
        track_ids: list[int] | np.ndarray | None = None,
        names: dict[int, str] | None = None,
        speed: dict[str, float] | None = None,
    ) -> MockResult:
        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            mock_boxes = MockBoxes(
                xyxy=np.empty((0, 4), dtype=np.float32),
                conf=np.empty((0,), dtype=np.float32),
                cls=np.empty((0,), dtype=np.int32),
                ids=None,
            )
            return MockResult(boxes=mock_boxes, names=names, speed=speed)

        xyxy_arr = np.asarray(boxes_xyxy, dtype=np.float32)
        n = len(xyxy_arr)
        conf_arr = (
            np.asarray(confidences, dtype=np.float32)
            if confidences is not None
            else np.ones(n, dtype=np.float32)
        )
        cls_arr = (
            np.asarray(class_ids, dtype=np.int32)
            if class_ids is not None
            else np.zeros(n, dtype=np.int32)
        )
        id_arr = np.asarray(track_ids, dtype=np.float32) if track_ids is not None else None

        mock_boxes = MockBoxes(xyxy=xyxy_arr, conf=conf_arr, cls=cls_arr, ids=id_arr)
        return MockResult(boxes=mock_boxes, names=names, speed=speed)

    return _factory


@pytest.fixture
def mock_yolo_model(mock_yolo_result_factory: Callable[..., MockResult]) -> Any:
    """Creates a mock YOLO model instance with controlled track() outputs."""
    model = MagicMock()
    model.names = {0: "person", 1: "bicycle", 2: "car"}
    model.predictor = MagicMock()
    model.predictor.trackers = ["active_tracker_0"]

    # Nominal default tracking result with 2 tracked objects
    default_result = mock_yolo_result_factory(
        boxes_xyxy=[[100.0, 150.0, 200.0, 250.0], [300.0, 350.0, 450.0, 420.0]],
        confidences=[0.88, 0.92],
        class_ids=[0, 2],
        track_ids=[1, 2],
        names=model.names,
    )
    model.track.return_value = [default_result]
    return model


@pytest.fixture
def synthetic_video_file(tmp_path: Path) -> Generator[Path, None, None]:
    """Generates a temporary 5-frame synthetic MP4 video file and cleans up after."""
    video_path = tmp_path / "test_synth_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (640, 480))
    for i in range(5):
        frame = np.full((480, 640, 3), fill_value=i * 40, dtype=np.uint8)
        out.write(frame)
    out.release()

    yield video_path

    if video_path.exists():
        video_path.unlink()
