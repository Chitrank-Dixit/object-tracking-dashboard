"""Exhaustive unit test suite for Pydantic v2 domain contracts in src.schemas.

Validates all 33 test cases (TC-01 through TC-33) covering DetectorConfig,
TrackedObject, FrameResult, CountingLine, CountingZone, and AnalyticsSummary.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas import (
    AnalyticsSummary,
    CountingLine,
    CountingZone,
    DashboardConfig,
    DetectorConfig,
    FrameResult,
    TrackedObject,
)

# ==============================================================================
# DetectorConfig Tests (TC-01 to TC-10)
# ==============================================================================


@pytest.mark.unit
def test_tc01_detector_config_nominal_defaults() -> None:
    """TC-01: Verify DetectorConfig initializes with expected default values."""
    config = DetectorConfig()

    assert config.model_path == "yolov8n.pt"
    assert config.confidence_threshold == 0.25
    assert config.iou_threshold == 0.45
    assert config.target_classes is None
    assert config.tracker_type == "bytetrack.yaml"


@pytest.mark.unit
def test_tc02_detector_config_custom_valid() -> None:
    """TC-02: Verify DetectorConfig accepts valid custom parameters."""
    config = DetectorConfig(
        model_path="yolov8m.pt",
        confidence_threshold=0.6,
        iou_threshold=0.5,
        target_classes=[0, 2],
        tracker_type="botsort.yaml",
    )

    assert config.model_path == "yolov8m.pt"
    assert config.confidence_threshold == 0.6
    assert config.iou_threshold == 0.5
    assert config.target_classes == [0, 2]
    assert config.tracker_type == "botsort.yaml"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("conf", "iou"),
    [
        (0.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (1.0, 0.0),
    ],
)
def test_tc03_detector_config_boundary_thresholds(conf: float, iou: float) -> None:
    """TC-03: Verify DetectorConfig accepts exact boundary threshold values 0.0 and 1.0."""
    config = DetectorConfig(confidence_threshold=conf, iou_threshold=iou)
    assert config.confidence_threshold == conf
    assert config.iou_threshold == iou


@pytest.mark.unit
@pytest.mark.parametrize("invalid_conf", [-0.01, -0.5, -1.0])
def test_tc04_detector_config_negative_confidence_raises(invalid_conf: float) -> None:
    """TC-04: Verify DetectorConfig rejects negative confidence thresholds."""
    with pytest.raises(ValidationError) as exc_info:
        DetectorConfig(confidence_threshold=invalid_conf)
    assert "greater_than_equal" in str(exc_info.value) or "ge" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.parametrize("invalid_conf", [1.01, 1.5, 2.0])
def test_tc05_detector_config_over_unity_confidence_raises(invalid_conf: float) -> None:
    """TC-05: Verify DetectorConfig rejects confidence thresholds greater than 1.0."""
    with pytest.raises(ValidationError) as exc_info:
        DetectorConfig(confidence_threshold=invalid_conf)
    assert "less_than_equal" in str(exc_info.value) or "le" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.parametrize("invalid_path", ["", "   ", "\t\n"])
def test_tc06_detector_config_empty_model_path_raises(invalid_path: str) -> None:
    """TC-06: Verify DetectorConfig rejects empty or whitespace-only model paths."""
    with pytest.raises(ValidationError) as exc_info:
        DetectorConfig(model_path=invalid_path)
    assert "model_path cannot be an empty or whitespace string" in str(
        exc_info.value
    ) or "string_too_short" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_classes",
    [
        [-1],
        [0, -1],
        [2, 3, -10],
    ],
)
def test_tc07_detector_config_negative_target_class_id_raises(
    invalid_classes: list[int],
) -> None:
    """TC-07: Verify DetectorConfig rejects negative target class IDs."""
    with pytest.raises(ValidationError) as exc_info:
        DetectorConfig(target_classes=invalid_classes)
    assert "Class IDs must be non-negative integers" in str(exc_info.value)


@pytest.mark.unit
def test_tc08_detector_config_empty_target_classes_raises() -> None:
    """TC-08: Verify DetectorConfig rejects empty list for target_classes (must be None or populated)."""
    with pytest.raises(ValidationError) as exc_info:
        DetectorConfig(target_classes=[])
    assert "target_classes cannot be an empty list; use None to track all classes" in str(
        exc_info.value
    )


@pytest.mark.unit
@pytest.mark.parametrize("invalid_tracker", ["deepsort.yaml", "sort.yaml", "custom", ""])
def test_tc09_detector_config_unsupported_tracker_type_raises(
    invalid_tracker: str,
) -> None:
    """TC-09: Verify DetectorConfig rejects tracker types other than bytetrack.yaml or botsort.yaml."""
    with pytest.raises(ValidationError) as exc_info:
        DetectorConfig(tracker_type=invalid_tracker)  # type: ignore[arg-type]
    assert "Input should be 'bytetrack.yaml' or 'botsort.yaml'" in str(exc_info.value)


@pytest.mark.unit
def test_tc10_detector_config_frozen_immutability_and_extra_forbid() -> None:
    """TC-10: Verify DetectorConfig enforces frozen immutability and forbids undeclared fields."""
    config = DetectorConfig()

    # Frozen immutability verification
    with pytest.raises(ValidationError):
        config.confidence_threshold = 0.5  # type: ignore[misc]

    # Extra attribute prohibition verification
    with pytest.raises(ValidationError) as exc_info:
        DetectorConfig(extra_arg=123)  # type: ignore[call-arg]
    assert "extra_forbidden" in str(exc_info.value)


# ==============================================================================
# TrackedObject Tests (TC-11 to TC-18)
# ==============================================================================


@pytest.mark.unit
def test_tc11_tracked_object_nominal_auto_centroid() -> None:
    """TC-11: Verify TrackedObject automatically computes centroid from bbox_xyxy when omitted."""
    obj = TrackedObject(
        track_id=1,
        class_id=0,
        class_name="person",
        confidence=0.90,
        bbox_xyxy=(100.0, 150.0, 200.0, 250.0),
    )

    assert obj.centroid == (150.0, 200.0)


@pytest.mark.unit
def test_tc12_tracked_object_explicit_valid_centroid() -> None:
    """TC-12: Verify TrackedObject accepts explicit valid centroid within 1e-2 tolerance."""
    # Exact match
    obj_exact = TrackedObject(
        track_id=1,
        class_id=0,
        class_name="car",
        confidence=0.85,
        bbox_xyxy=(10.0, 20.0, 30.0, 40.0),
        centroid=(20.0, 30.0),
    )
    assert obj_exact.centroid == (20.0, 30.0)

    # Within 1e-2 numerical tolerance
    obj_tol = TrackedObject(
        track_id=2,
        class_id=0,
        class_name="car",
        confidence=0.85,
        bbox_xyxy=(10.0, 20.0, 30.0, 40.0),
        centroid=(20.005, 29.995),
    )
    assert obj_tol.centroid == (20.005, 29.995)


@pytest.mark.unit
def test_tc13_tracked_object_centroid_mismatch_raises() -> None:
    """TC-13: Verify TrackedObject rejects explicit centroid deviating beyond tolerance."""
    with pytest.raises(ValidationError) as exc_info:
        TrackedObject(
            track_id=1,
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox_xyxy=(10.0, 20.0, 30.0, 40.0),
            centroid=(25.0, 35.0),
        )
    assert "does not match bbox center" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_bbox",
    [
        (200.0, 100.0, 200.0, 200.0),  # x1 == x2
        (250.0, 100.0, 200.0, 200.0),  # x1 > x2
    ],
)
def test_tc14_tracked_object_inverted_x_coords_raises(
    invalid_bbox: tuple[float, float, float, float],
) -> None:
    """TC-14: Verify TrackedObject rejects bounding boxes where x1 >= x2."""
    with pytest.raises(ValidationError) as exc_info:
        TrackedObject(
            track_id=1,
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox_xyxy=invalid_bbox,
        )
    assert "x1" in str(exc_info.value) and "must be strictly less than x2" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_bbox",
    [
        (100.0, 200.0, 200.0, 200.0),  # y1 == y2
        (100.0, 250.0, 200.0, 200.0),  # y1 > y2
    ],
)
def test_tc15_tracked_object_inverted_y_coords_raises(
    invalid_bbox: tuple[float, float, float, float],
) -> None:
    """TC-15: Verify TrackedObject rejects bounding boxes where y1 >= y2."""
    with pytest.raises(ValidationError) as exc_info:
        TrackedObject(
            track_id=1,
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox_xyxy=invalid_bbox,
        )
    assert "y1" in str(exc_info.value) and "must be strictly less than y2" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_bbox",
    [
        (-1.0, 10.0, 50.0, 100.0),
        (10.0, -5.0, 50.0, 100.0),
        (10.0, 10.0, -50.0, 100.0),
        (10.0, 10.0, 50.0, -100.0),
    ],
)
def test_tc16_tracked_object_negative_coordinates_raises(
    invalid_bbox: tuple[float, float, float, float],
) -> None:
    """TC-16: Verify TrackedObject rejects bounding boxes containing negative coordinates."""
    with pytest.raises(ValidationError) as exc_info:
        TrackedObject(
            track_id=1,
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox_xyxy=invalid_bbox,
        )
    assert "Bounding box coordinates must be non-negative" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("track_id", "class_id"),
    [
        (-1, 0),
        (1, -1),
        (-2, -2),
    ],
)
def test_tc17_tracked_object_negative_ids_raises(track_id: int, class_id: int) -> None:
    """TC-17: Verify TrackedObject rejects negative track_id or class_id."""
    with pytest.raises(ValidationError) as exc_info:
        TrackedObject(
            track_id=track_id,
            class_id=class_id,
            class_name="person",
            confidence=0.9,
            bbox_xyxy=(10.0, 10.0, 50.0, 50.0),
        )
    assert "greater_than_equal" in str(exc_info.value) or "ge" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.parametrize("invalid_name", ["", "   ", "\t\n"])
def test_tc18_tracked_object_empty_class_name_raises(invalid_name: str) -> None:
    """TC-18: Verify TrackedObject rejects empty or whitespace-only class names."""
    with pytest.raises(ValidationError) as exc_info:
        TrackedObject(
            track_id=1,
            class_id=0,
            class_name=invalid_name,
            confidence=0.9,
            bbox_xyxy=(10.0, 10.0, 50.0, 50.0),
        )
    assert "class_name cannot be empty or whitespace" in str(
        exc_info.value
    ) or "string_too_short" in str(exc_info.value)


# ==============================================================================
# FrameResult Tests (TC-19 to TC-22)
# ==============================================================================


@pytest.mark.unit
def test_tc19_frame_result_nominal_multi_object(valid_tracked_object: TrackedObject) -> None:
    """TC-19: Verify FrameResult initializes correctly with multiple tracked objects."""
    second_object = TrackedObject(
        track_id=2,
        class_id=2,
        class_name="car",
        confidence=0.95,
        bbox_xyxy=(300.0, 200.0, 400.0, 300.0),
    )

    result = FrameResult(
        frame_index=1,
        inference_ms=15.4,
        fps=58.5,
        objects=[valid_tracked_object, second_object],
    )

    assert result.frame_index == 1
    assert result.inference_ms == 15.4
    assert result.fps == 58.5
    assert len(result.objects) == 2
    assert result.objects[0].track_id == 1
    assert result.objects[1].track_id == 2


@pytest.mark.unit
def test_tc20_frame_result_empty_detections() -> None:
    """TC-20: Verify FrameResult supports empty detections lists (e.g. frame with no objects)."""
    # Explicit empty list
    result_explicit = FrameResult(frame_index=0, inference_ms=5.0, fps=30.0, objects=[])
    assert result_explicit.objects == []

    # Default factory empty list
    result_default = FrameResult(frame_index=0, inference_ms=5.0, fps=30.0)
    assert result_default.objects == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("frame_idx", "inference_ms", "fps"),
    [
        (-1, 10.0, 30.0),
        (0, -1.0, 30.0),
        (0, 10.0, -0.1),
    ],
)
def test_tc21_frame_result_negative_metrics_raises(
    frame_idx: int, inference_ms: float, fps: float
) -> None:
    """TC-21: Verify FrameResult rejects negative frame index, inference duration, or FPS."""
    with pytest.raises(ValidationError) as exc_info:
        FrameResult(frame_index=frame_idx, inference_ms=inference_ms, fps=fps)
    assert "greater_than_equal" in str(exc_info.value) or "ge" in str(exc_info.value)


@pytest.mark.unit
def test_tc22_frame_result_json_round_trip(valid_tracked_object: TrackedObject) -> None:
    """TC-22: Verify FrameResult serializes to JSON and deserializes identically."""
    result = FrameResult(
        frame_index=42,
        inference_ms=8.5,
        fps=58.2,
        objects=[valid_tracked_object],
    )

    serialized = result.model_dump_json()
    assert isinstance(serialized, str)

    restored = FrameResult.model_validate_json(serialized)
    assert restored == result
    assert restored.objects[0].track_id == 1
    assert restored.objects[0].centroid == (150.0, 200.0)


# ==============================================================================
# CountingLine Tests (TC-23 to TC-26)
# ==============================================================================


@pytest.mark.unit
def test_tc23_counting_line_nominal() -> None:
    """TC-23: Verify CountingLine initializes with valid coordinates and name."""
    line = CountingLine(name="entry_gate", start_point=(0, 240), end_point=(640, 240))

    assert line.name == "entry_gate"
    assert line.start_point == (0, 240)
    assert line.end_point == (640, 240)


@pytest.mark.unit
def test_tc24_counting_line_degenerate_raises() -> None:
    """TC-24: Verify CountingLine rejects degenerate lines where start_point equals end_point."""
    with pytest.raises(ValidationError) as exc_info:
        CountingLine(name="point_line", start_point=(100, 100), end_point=(100, 100))
    assert "cannot be identical" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("start", "end"),
    [
        ((-10, 50), (100, 50)),
        ((0, -5), (100, 50)),
        ((10, 50), (-100, 50)),
        ((10, 50), (100, -50)),
    ],
)
def test_tc25_counting_line_negative_coordinates_raises(
    start: tuple[int, int], end: tuple[int, int]
) -> None:
    """TC-25: Verify CountingLine rejects negative coordinates for start or end points."""
    with pytest.raises(ValidationError) as exc_info:
        CountingLine(name="entry", start_point=start, end_point=end)
    assert "Line coordinates must be non-negative" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.parametrize("invalid_name", ["", "   ", "\t\n"])
def test_tc26_counting_line_empty_name_raises(invalid_name: str) -> None:
    """TC-26: Verify CountingLine rejects empty or whitespace-only name strings."""
    with pytest.raises(ValidationError) as exc_info:
        CountingLine(name=invalid_name, start_point=(0, 100), end_point=(100, 100))
    assert "CountingLine name cannot be empty or whitespace" in str(
        exc_info.value
    ) or "string_too_short" in str(exc_info.value)


# ==============================================================================
# CountingZone Tests (TC-27 to TC-31)
# ==============================================================================


@pytest.mark.unit
def test_tc27_counting_zone_nominal() -> None:
    """TC-27: Verify CountingZone initializes correctly with a valid polygon."""
    zone = CountingZone(
        name="loading_dock",
        polygon=[(10, 10), (100, 10), (100, 100), (10, 100)],
    )

    assert zone.name == "loading_dock"
    assert len(zone.polygon) == 4
    assert zone.polygon[0] == (10, 10)


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_polygon",
    [
        [],
        [(10, 10)],
        [(10, 10), (50, 50)],
    ],
)
def test_tc28_counting_zone_fewer_than_3_vertices_raises(
    invalid_polygon: list[tuple[int, int]],
) -> None:
    """TC-28: Verify CountingZone rejects polygons with fewer than 3 vertices."""
    with pytest.raises(ValidationError) as exc_info:
        CountingZone(name="invalid_poly", polygon=invalid_polygon)
    assert "Polygon must have at least 3 vertices" in str(exc_info.value) or "too_short" in str(
        exc_info.value
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "duplicate_polygon",
    [
        [(10, 10), (10, 10), (50, 50)],
        [(5, 5), (5, 5), (5, 5)],
    ],
)
def test_tc29_counting_zone_duplicate_vertices_raises(
    duplicate_polygon: list[tuple[int, int]],
) -> None:
    """TC-29: Verify CountingZone rejects polygons having fewer than 3 unique vertices."""
    with pytest.raises(ValidationError) as exc_info:
        CountingZone(name="dup_zone", polygon=duplicate_polygon)
    assert "Polygon must contain at least 3 unique vertices" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    "collinear_polygon",
    [
        [(0, 0), (10, 10), (20, 20)],  # Diagonal collinear
        [(0, 50), (50, 50), (100, 50)],  # Horizontal collinear
        [(50, 0), (50, 50), (50, 100)],  # Vertical collinear
    ],
)
def test_tc30_counting_zone_collinear_vertices_raises(
    collinear_polygon: list[tuple[int, int]],
) -> None:
    """TC-30: Verify CountingZone rejects degenerate polygons where all vertices are collinear."""
    with pytest.raises(ValidationError) as exc_info:
        CountingZone(name="collinear_zone", polygon=collinear_polygon)
    assert "all vertices are collinear (zero area)" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_coords",
    [
        [(-10, 0), (10, 0), (10, 10)],
        [(0, -5), (10, 0), (10, 10)],
    ],
)
def test_tc31_counting_zone_negative_coordinates_raises(
    invalid_coords: list[tuple[int, int]],
) -> None:
    """TC-31: Verify CountingZone rejects negative coordinates in polygon vertices."""
    with pytest.raises(ValidationError) as exc_info:
        CountingZone(name="negative_zone", polygon=invalid_coords)
    assert "Polygon vertex coordinates must be non-negative" in str(exc_info.value)


# ==============================================================================
# AnalyticsSummary Tests (TC-32 to TC-33)
# ==============================================================================


@pytest.mark.unit
def test_tc32_analytics_summary_nominal() -> None:
    """TC-32: Verify AnalyticsSummary nominal defaults and custom assignments."""
    # Defaults
    summary_default = AnalyticsSummary()
    assert summary_default.total_in == 0
    assert summary_default.total_out == 0
    assert summary_default.zone_occupancy == {}
    assert summary_default.class_counts == {}

    # Custom
    summary_custom = AnalyticsSummary(
        total_in=10,
        total_out=5,
        zone_occupancy={"dock_a": 3, "dock_b": 0},
        class_counts={"person": 2, "car": 1},
    )
    assert summary_custom.total_in == 10
    assert summary_custom.total_out == 5
    assert summary_custom.zone_occupancy == {"dock_a": 3, "dock_b": 0}
    assert summary_custom.class_counts == {"person": 2, "car": 1}


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_kwargs",
    [
        {"total_in": -1},
        {"total_out": -1},
        {"zone_occupancy": {"dock_a": -1}},
        {"class_counts": {"car": -5}},
    ],
)
def test_tc33_analytics_summary_negative_counts_raises(
    invalid_kwargs: dict[str, object],
) -> None:
    """TC-33: Verify AnalyticsSummary rejects negative counts across all fields."""
    with pytest.raises(ValidationError) as exc_info:
        AnalyticsSummary(**invalid_kwargs)
    assert (
        "greater_than_equal" in str(exc_info.value)
        or "ge" in str(exc_info.value)
        or "Occupancy and class counts cannot be negative" in str(exc_info.value)
    )


# ==============================================================================
# DashboardConfig Tests (TC-34 to TC-37)
# ==============================================================================


@pytest.mark.unit
def test_tc34_dashboard_config_nominal_defaults() -> None:
    """TC-34: Verify DashboardConfig initializes with deterministic defaults."""
    config = DashboardConfig()
    assert config.model_path == "yolov8n.pt"
    assert config.confidence_threshold == 0.25
    assert config.iou_threshold == 0.45
    assert config.tracker_type == "bytetrack.yaml"
    assert config.target_classes is None
    assert config.source_type == "Sample Video"
    assert config.source_path == "data/sample_video.mp4"
    assert config.lines == []
    assert config.zones == []
    assert config.draw_boxes is True
    assert config.draw_trajectories is True
    assert config.draw_lines is True
    assert config.draw_zones is True
    assert config.draw_hud is True


@pytest.mark.unit
def test_tc35_dashboard_config_frozen_immutability_and_extra_forbid() -> None:
    """TC-35: Verify DashboardConfig is immutable and rejects undeclared fields."""
    config = DashboardConfig()
    with pytest.raises(ValidationError):
        config.model_path = "yolov8s.pt"  # type: ignore[misc]

    with pytest.raises(ValidationError):
        DashboardConfig(unknown_param=123)  # type: ignore[call-arg]


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_kwargs",
    [
        {"confidence_threshold": -0.01},
        {"confidence_threshold": 1.01},
        {"iou_threshold": -0.01},
        {"iou_threshold": 1.01},
    ],
)
def test_tc36_dashboard_config_threshold_boundaries(invalid_kwargs: dict[str, float]) -> None:
    """TC-36: Verify DashboardConfig enforces strict [0.0, 1.0] threshold ranges."""
    with pytest.raises(ValidationError):
        DashboardConfig(**invalid_kwargs)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "val"),
    [
        ("model_path", ""),
        ("model_path", "   "),
        ("source_path", ""),
        ("source_path", "   \t"),
    ],
)
def test_tc37_dashboard_config_empty_paths_raises(field: str, val: str) -> None:
    """TC-37: Verify DashboardConfig rejects empty or whitespace-only paths."""
    with pytest.raises(ValidationError) as exc_info:
        DashboardConfig(**{field: val})
    assert "cannot be empty or whitespace" in str(exc_info.value) or "string_too_short" in str(
        exc_info.value
    )
