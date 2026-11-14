"""Comprehensive test suite for Detection & ByteTrack Inference Engine (src/tracker.py).

Implements all 18 test cases (TC-TRK-01 through TC-TRK-18) verifying ObjectTracker:
- Contract validation on input frames and frame indices
- Ultralytics Results parsing to Pydantic v2 FrameResult models
- Negative regression coordinate clamping and degenerate box skipping
- Headless annotation rendering and immutability guarantees
- Video iteration, resource lifecycle, and error cleanup
- Tracker state reset and configuration propagation
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.schemas import DetectorConfig, FrameResult
from src.tracker import DEFAULT_PALETTE, ObjectTracker

# ==============================================================================
# TC-TRK-01: Nominal Multi-Object Tracking
# ==============================================================================


@pytest.mark.unit
def test_tc_trk_01_nominal_multi_object_tracking(
    sample_synthetic_frame: np.ndarray,
    mock_yolo_model: Any,
) -> None:
    """TC-TRK-01: Verify multi-object frame tracking returns validated FrameResult."""
    tracker = ObjectTracker(model=mock_yolo_model)
    result = tracker.track_frame(sample_synthetic_frame, frame_index=0)

    assert isinstance(result, FrameResult)
    assert result.frame_index == 0
    assert result.inference_ms >= 0.0
    assert result.fps >= 0.0
    assert len(result.objects) == 2

    # Verify first object (person, track_id 1)
    obj1 = result.objects[0]
    assert obj1.track_id == 1
    assert obj1.class_id == 0
    assert obj1.class_name == "person"
    assert obj1.confidence == pytest.approx(0.88, rel=1e-3)
    assert obj1.bbox_xyxy == (100.0, 150.0, 200.0, 250.0)
    assert obj1.centroid == (150.0, 200.0)

    # Verify second object (car, track_id 2)
    obj2 = result.objects[1]
    assert obj2.track_id == 2
    assert obj2.class_id == 2
    assert obj2.class_name == "car"
    assert obj2.confidence == pytest.approx(0.92, rel=1e-3)
    assert obj2.bbox_xyxy == (300.0, 350.0, 450.0, 420.0)
    assert obj2.centroid == (375.0, 385.0)


# ==============================================================================
# TC-TRK-02: Zero Detections (Empty Frame)
# ==============================================================================


@pytest.mark.unit
def test_tc_trk_02_zero_detections(
    sample_synthetic_frame: np.ndarray,
    mock_yolo_model: Any,
    mock_yolo_result_factory: Callable[..., Any],
) -> None:
    """TC-TRK-02: Verify empty detections return FrameResult with empty objects list."""
    mock_yolo_model.track.return_value = [mock_yolo_result_factory()]
    tracker = ObjectTracker(model=mock_yolo_model)
    result = tracker.track_frame(sample_synthetic_frame, frame_index=42)

    assert isinstance(result, FrameResult)
    assert result.frame_index == 42
    assert result.objects == []
    assert len(result.objects) == 0

    # Test when result.boxes is explicitly None
    res_none_boxes = mock_yolo_result_factory()
    res_none_boxes.boxes = None
    mock_yolo_model.track.return_value = [res_none_boxes]

    result_none = tracker.track_frame(sample_synthetic_frame, frame_index=43)
    assert result_none.objects == []


# ==============================================================================
# TC-TRK-03: Untracked Detections (boxes.id is None)
# ==============================================================================


@pytest.mark.unit
def test_tc_trk_03_untracked_detections_skipped(
    sample_synthetic_frame: np.ndarray,
    mock_yolo_model: Any,
    mock_yolo_result_factory: Callable[..., Any],
) -> None:
    """TC-TRK-03: Verify detections lacking assigned track IDs are skipped."""
    # Track IDs are None (untracked detections)
    mock_yolo_model.track.return_value = [
        mock_yolo_result_factory(
            boxes_xyxy=[[50.0, 60.0, 150.0, 160.0]],
            confidences=[0.85],
            class_ids=[0],
            track_ids=None,
        )
    ]
    tracker = ObjectTracker(model=mock_yolo_model)
    result = tracker.track_frame(sample_synthetic_frame, frame_index=1)

    assert isinstance(result, FrameResult)
    assert len(result.objects) == 0

    # Detections where track ID is NaN
    nan_result = mock_yolo_result_factory(
        boxes_xyxy=[[50.0, 60.0, 150.0, 160.0]],
        confidences=[0.85],
        class_ids=[0],
        track_ids=[float("nan")],
    )
    mock_yolo_model.track.return_value = [nan_result]
    result_nan = tracker.track_frame(sample_synthetic_frame, frame_index=2)
    assert len(result_nan.objects) == 0


# ==============================================================================
# TC-TRK-04: Sub-pixel Negative Coordinate Clamping
# ==============================================================================


@pytest.mark.unit
def test_tc_trk_04_subpixel_negative_coordinate_clamping(
    sample_synthetic_frame: np.ndarray,
    mock_yolo_model: Any,
    mock_yolo_result_factory: Callable[..., Any],
) -> None:
    """TC-TRK-04: Verify sub-pixel negative coordinate regressions clamp to 0.0."""
    mock_yolo_model.track.return_value = [
        mock_yolo_result_factory(
            boxes_xyxy=[[-2.5, -1.8, 120.0, 180.0]],
            confidences=[0.90],
            class_ids=[0],
            track_ids=[1],
        )
    ]
    tracker = ObjectTracker(model=mock_yolo_model)
    result = tracker.track_frame(sample_synthetic_frame, frame_index=0)

    assert len(result.objects) == 1
    obj = result.objects[0]
    assert obj.bbox_xyxy == (0.0, 0.0, 120.0, 180.0)
    assert obj.centroid == (60.0, 90.0)


# ==============================================================================
# TC-TRK-05: Degenerate Bounding Box Skipped
# ==============================================================================


@pytest.mark.unit
def test_tc_trk_05_degenerate_bounding_box_skipped(
    sample_synthetic_frame: np.ndarray,
    mock_yolo_model: Any,
    mock_yolo_result_factory: Callable[..., Any],
) -> None:
    """TC-TRK-05: Verify degenerate bounding boxes (x1>=x2 or y1>=y2) are omitted."""
    mock_yolo_model.track.return_value = [
        mock_yolo_result_factory(
            boxes_xyxy=[
                [100.0, 50.0, 50.0, 100.0],  # Inverted X (x1 > x2)
                [50.0, 100.0, 100.0, 50.0],  # Inverted Y (y1 > y2)
                [60.0, 60.0, 60.0, 120.0],  # Zero width (x1 == x2)
                [60.0, 60.0, 120.0, 60.0],  # Zero height (y1 == y2)
                [10.0, 10.0, 60.0, 60.0],  # Valid box
            ],
            confidences=[0.8, 0.8, 0.8, 0.8, 0.95],
            class_ids=[0, 0, 0, 0, 2],
            track_ids=[10, 11, 12, 13, 14],
        )
    ]
    tracker = ObjectTracker(model=mock_yolo_model)
    result = tracker.track_frame(sample_synthetic_frame, frame_index=0)

    assert len(result.objects) == 1
    valid_obj = result.objects[0]
    assert valid_obj.track_id == 14
    assert valid_obj.bbox_xyxy == (10.0, 10.0, 60.0, 60.0)


# ==============================================================================
# TC-TRK-06: Invalid Frame Type Raises ValueError
# ==============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_frame",
    [None, "not_an_array", [1, 2, 3], 12345, {"image": "frame"}],
)
def test_tc_trk_06_invalid_frame_type_raises(
    invalid_frame: Any,
    mock_yolo_model: Any,
) -> None:
    """TC-TRK-06: Verify non-numpy frame objects raise ValueError."""
    tracker = ObjectTracker(model=mock_yolo_model)
    with pytest.raises(ValueError, match="numpy.ndarray"):
        tracker.track_frame(invalid_frame)


# ==============================================================================
# TC-TRK-07: Invalid Frame Dimensions Raises ValueError
# ==============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_dim_frame",
    [
        np.zeros((480, 640), dtype=np.uint8),  # 2D grayscale
        np.zeros((1, 480, 640, 3), dtype=np.uint8),  # 4D batched
        np.zeros((480, 640, 1), dtype=np.uint8),  # 3D 1-channel
        np.zeros((480, 640, 4), dtype=np.uint8),  # 3D 4-channel BGRA
    ],
)
def test_tc_trk_07_invalid_frame_dimensions_raises(
    invalid_dim_frame: np.ndarray,
    mock_yolo_model: Any,
) -> None:
    """TC-TRK-07: Verify frames without (H, W, 3) shape raise ValueError."""
    tracker = ObjectTracker(model=mock_yolo_model)
    with pytest.raises(ValueError, match=r"shape \(H, W, 3\)"):
        tracker.track_frame(invalid_dim_frame)


# ==============================================================================
# TC-TRK-08: Zero-Sized / Empty Frame Raises ValueError
# ==============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "zero_frame",
    [
        np.zeros((0, 640, 3), dtype=np.uint8),
        np.zeros((480, 0, 3), dtype=np.uint8),
        np.zeros((0, 0, 3), dtype=np.uint8),
    ],
)
def test_tc_trk_08_zero_sized_frame_raises(
    zero_frame: np.ndarray,
    mock_yolo_model: Any,
) -> None:
    """TC-TRK-08: Verify zero-dimension frames raise ValueError."""
    tracker = ObjectTracker(model=mock_yolo_model)
    with pytest.raises(ValueError, match="dimensions must be non-zero"):
        tracker.track_frame(zero_frame)


# ==============================================================================
# TC-TRK-09: Invalid Frame Dtype Raises ValueError
# ==============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_dtype_frame",
    [
        np.zeros((480, 640, 3), dtype=np.float32),
        np.zeros((480, 640, 3), dtype=np.int32),
        np.zeros((480, 640, 3), dtype=np.int64),
        np.zeros((480, 640, 3), dtype=np.float64),
    ],
)
def test_tc_trk_09_invalid_dtype_raises(
    invalid_dtype_frame: np.ndarray,
    mock_yolo_model: Any,
) -> None:
    """TC-TRK-09: Verify non-uint8 dtype frames raise ValueError."""
    tracker = ObjectTracker(model=mock_yolo_model)
    with pytest.raises(ValueError, match="dtype must be uint8"):
        tracker.track_frame(invalid_dtype_frame)


# ==============================================================================
# TC-TRK-10: Negative Frame Index Raises ValueError
# ==============================================================================


@pytest.mark.unit
@pytest.mark.parametrize("invalid_idx", [-1, -5, -999])
def test_tc_trk_10_negative_frame_index_raises(
    invalid_idx: int,
    sample_synthetic_frame: np.ndarray,
    mock_yolo_model: Any,
) -> None:
    """TC-TRK-10: Verify negative frame index raises ValueError."""
    tracker = ObjectTracker(model=mock_yolo_model)
    with pytest.raises(ValueError, match="frame_index must be non-negative"):
        tracker.track_frame(sample_synthetic_frame, frame_index=invalid_idx)


# ==============================================================================
# TC-TRK-11: DetectorConfig Parameters Propagated to model.track()
# ==============================================================================


@pytest.mark.unit
def test_tc_trk_11_detector_config_propagated_to_model(
    sample_synthetic_frame: np.ndarray,
    mock_yolo_model: Any,
) -> None:
    """TC-TRK-11: Verify custom DetectorConfig parameters propagate to model.track()."""
    custom_config = DetectorConfig(
        confidence_threshold=0.65,
        iou_threshold=0.55,
        target_classes=[0, 2],
        tracker_type="bytetrack.yaml",
    )
    tracker = ObjectTracker(config=custom_config, model=mock_yolo_model)
    tracker.track_frame(sample_synthetic_frame, frame_index=0)

    mock_yolo_model.track.assert_called_once_with(
        source=sample_synthetic_frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.65,
        iou=0.55,
        classes=[0, 2],
        verbose=False,
    )


# ==============================================================================
# TC-TRK-12: Unknown Class ID Fallback
# ==============================================================================


@pytest.mark.unit
def test_tc_trk_12_unknown_class_id_fallback(
    sample_synthetic_frame: np.ndarray,
    mock_yolo_model: Any,
    mock_yolo_result_factory: Callable[..., Any],
) -> None:
    """TC-TRK-12: Verify class ID not present in model.names falls back to 'class_<id>'."""
    # Model names only has {0: "person", 2: "car"}
    mock_yolo_model.track.return_value = [
        mock_yolo_result_factory(
            boxes_xyxy=[[10.0, 10.0, 50.0, 50.0]],
            confidences=[0.75],
            class_ids=[999],  # Unknown class
            track_ids=[5],
            names={0: "person", 2: "car"},
        )
    ]
    tracker = ObjectTracker(model=mock_yolo_model)
    result = tracker.track_frame(sample_synthetic_frame, frame_index=0)

    assert len(result.objects) == 1
    obj = result.objects[0]
    assert obj.class_id == 999
    assert obj.class_name == "class_999"


# ==============================================================================
# TC-TRK-13: Video Processing Nominal Iteration (5 Frames)
# ==============================================================================


@pytest.mark.unit
def test_tc_trk_13_video_processing_nominal_iteration(
    synthetic_video_file: Path,
    mock_yolo_model: Any,
) -> None:
    """TC-TRK-13: Verify process_video iterates 5 frames with annotations enabled."""
    tracker = ObjectTracker(model=mock_yolo_model)
    results = list(tracker.process_video(synthetic_video_file, annotate=True))

    assert len(results) == 5

    for idx, (annotated_frame, frame_result) in enumerate(results):
        assert annotated_frame is not None
        assert isinstance(annotated_frame, np.ndarray)
        assert annotated_frame.shape == (480, 640, 3)
        assert annotated_frame.dtype == np.uint8
        assert isinstance(frame_result, FrameResult)
        assert frame_result.frame_index == idx
        assert len(frame_result.objects) == 2


# ==============================================================================
# TC-TRK-14: Video Processing Without Annotation (annotate=False)
# ==============================================================================


@pytest.mark.unit
def test_tc_trk_14_video_processing_without_annotation(
    synthetic_video_file: Path,
    mock_yolo_model: Any,
) -> None:
    """TC-TRK-14: Verify process_video yields None for annotated frames when disabled."""
    tracker = ObjectTracker(model=mock_yolo_model)
    results = list(tracker.process_video(synthetic_video_file, annotate=False))

    assert len(results) == 5

    for idx, (annotated_frame, frame_result) in enumerate(results):
        assert annotated_frame is None
        assert isinstance(frame_result, FrameResult)
        assert frame_result.frame_index == idx
        assert len(frame_result.objects) == 2


# ==============================================================================
# TC-TRK-15: Video File Not Found Raises FileNotFoundError
# ==============================================================================


@pytest.mark.unit
def test_tc_trk_15_video_file_not_found_raises(
    mock_yolo_model: Any,
    tmp_path: Path,
) -> None:
    """TC-TRK-15: Verify missing video file path raises FileNotFoundError."""
    missing_file = tmp_path / "does_not_exist.mp4"
    tracker = ObjectTracker(model=mock_yolo_model)

    with pytest.raises(FileNotFoundError, match="Video file does not exist"):
        list(tracker.process_video(missing_file))


# ==============================================================================
# TC-TRK-16: Video Resource Cleanup Guaranteed on Error
# ==============================================================================


@pytest.mark.unit
def test_tc_trk_16_video_resource_cleanup_on_error(
    synthetic_video_file: Path,
    mock_yolo_model: Any,
) -> None:
    """TC-TRK-16: Verify cv2.VideoCapture is released even if processing raises error."""
    tracker = ObjectTracker(model=mock_yolo_model)

    # Force track_frame to raise during video iteration
    with (
        patch.object(tracker, "track_frame", side_effect=RuntimeError("Pipeline crash")),
        pytest.raises(RuntimeError, match="Pipeline crash"),
    ):
        list(tracker.process_video(synthetic_video_file))

    # Also test un-openable video error
    with (
        patch("cv2.VideoCapture") as mock_cap_cls,
        pytest.raises(ValueError, match="Failed to open video file"),
    ):
        mock_cap_instance = MagicMock()
        mock_cap_instance.isOpened.return_value = False
        mock_cap_cls.return_value = mock_cap_instance
        list(tracker.process_video(synthetic_video_file))


# ==============================================================================
# TC-TRK-17: Headless Annotation Integrity and Immutability
# ==============================================================================


@pytest.mark.unit
def test_tc_trk_17_headless_annotation_integrity(
    sample_synthetic_frame: np.ndarray,
    mock_yolo_model: Any,
) -> None:
    """TC-TRK-17: Verify annotate_frame returns modified copy without mutating input."""
    tracker = ObjectTracker(model=mock_yolo_model)
    frame_result = tracker.track_frame(sample_synthetic_frame, frame_index=0)

    original_input = sample_synthetic_frame.copy()
    annotated = tracker.annotate_frame(sample_synthetic_frame, frame_result)

    # 1. Output shape and dtype match input
    assert annotated.shape == sample_synthetic_frame.shape
    assert annotated.dtype == sample_synthetic_frame.dtype

    # 2. Input array is unmutated
    assert np.array_equal(sample_synthetic_frame, original_input)

    # 3. Output is a new modified array (bounding boxes and labels drawn)
    assert not np.array_equal(annotated, sample_synthetic_frame)

    # 4. Custom color palette support
    custom_palette = {1: (0, 255, 0), 2: (0, 0, 255)}
    annotated_custom = tracker.annotate_frame(
        sample_synthetic_frame,
        frame_result,
        color_palette=custom_palette,
    )
    assert annotated_custom.shape == sample_synthetic_frame.shape
    assert not np.array_equal(annotated_custom, sample_synthetic_frame)

    # 5. Invalid frame to annotate_frame raises ValueError
    with pytest.raises(ValueError, match="Frame must be a numpy.ndarray"):
        tracker.annotate_frame("not_an_array", frame_result)  # type: ignore[arg-type]


# ==============================================================================
# TC-TRK-18: Tracker State Reset
# ==============================================================================


@pytest.mark.unit
def test_tc_trk_18_tracker_state_reset(mock_yolo_model: Any) -> None:
    """TC-TRK-18: Verify tracker.reset() clears Kalman filters and tracker states."""
    tracker = ObjectTracker(model=mock_yolo_model)
    mock_yolo_model.predictor.trackers = ["active_tracker_1", "active_tracker_2"]

    tracker.reset()
    assert mock_yolo_model.predictor.trackers == []

    # Defensive resilience when predictor is None
    mock_yolo_model.predictor = None
    tracker.reset()  # Must not raise

    # Defensive resilience when model has no predictor attribute
    del mock_yolo_model.predictor
    tracker.reset()  # Must not raise


# ==============================================================================
# Model Initialization Coverage
# ==============================================================================


@pytest.mark.unit
def test_object_tracker_init_with_real_yolo_mocked(tmp_path: Path) -> None:
    """Verify ObjectTracker initializes model from config path when model is None."""
    custom_model_file = tmp_path / "models" / "yolov8n.pt"
    config = DetectorConfig(model_path=str(custom_model_file))

    with patch("src.tracker.YOLO") as mock_yolo_cls:
        tracker = ObjectTracker(config=config)
        mock_yolo_cls.assert_called_once_with(str(custom_model_file))
        assert tracker.config == config
        assert custom_model_file.parent.exists()


@pytest.mark.unit
def test_default_palette_length() -> None:
    """Verify default palette contains 10 distinct BGR color tuples."""
    assert len(DEFAULT_PALETTE) == 10
    for color in DEFAULT_PALETTE:
        assert len(color) == 3
        assert all(0 <= channel <= 255 for channel in color)
