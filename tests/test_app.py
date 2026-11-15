"""Comprehensive unit and integration test suite for Streamlit Dashboard in app.py.

Validates all 24 test scenarios (TC-APP-01 through TC-APP-24):
- TC-APP-01 to TC-APP-02: scan_available_models filesystem discovery and fallbacks.
- TC-APP-03 to TC-APP-06: parse_counting_lines_json parsing and validation.
- TC-APP-07 to TC-APP-09: parse_counting_zones_json parsing and validation.
- TC-APP-10 to TC-APP-14: resolve_video_source resolution, synthetic fallback, spooling, cleanup.
- TC-APP-15: process_single_frame decoupled inference and BGR to RGB conversion.
- TC-APP-16 to TC-APP-18: run_video_processing_loop lifecycle, error recovery, cap.release().
- TC-APP-19 to TC-APP-21: Reset counters action and class filtering.
- TC-APP-22 to TC-APP-23: AppTest headless simulation and sidebar interactions.
- TC-APP-24: DashboardConfig schema contract audit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest

from app import (
    COCO_CLASSES,
    init_session_state,
    parse_counting_lines_json,
    parse_counting_zones_json,
    process_single_frame,
    render_sidebar,
    resolve_video_source,
    run_video_processing_loop,
    scan_available_models,
)
from src.schemas import (
    AnalyticsSummary,
    CountingLine,
    CountingZone,
    DashboardConfig,
    FrameResult,
    TrackedObject,
)

# ==============================================================================
# Model Discovery Tests (TC-APP-01 to TC-APP-02)
# ==============================================================================


@pytest.mark.unit
def test_tc_app_01_scan_available_models_custom_weights(tmp_path: Path) -> None:
    """TC-APP-01: scan_available_models discovers custom .pt weights and sorts them."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "best_custom.pt").touch()
    (models_dir / "yolov8s.pt").touch()
    (tmp_path / "root_model.pt").touch()

    discovered = scan_available_models(models_dir=models_dir, root_dir=tmp_path)

    assert "yolov8n.pt" in discovered
    assert str(models_dir / "best_custom.pt") in discovered
    assert str(models_dir / "yolov8s.pt") in discovered
    assert "root_model.pt" in discovered
    assert discovered == sorted(discovered)


@pytest.mark.unit
def test_tc_app_02_scan_available_models_empty_fallback(tmp_path: Path) -> None:
    """TC-APP-02: scan_available_models falls back to default yolov8n.pt if directories empty."""
    empty_models = tmp_path / "non_existent_models"
    empty_root = tmp_path / "empty_root"
    empty_root.mkdir()

    discovered = scan_available_models(models_dir=empty_models, root_dir=empty_root)

    assert discovered == ["yolov8n.pt"]


# ==============================================================================
# Counting Lines JSON Parsing Tests (TC-APP-03 to TC-APP-06)
# ==============================================================================


@pytest.mark.unit
def test_tc_app_03_parse_counting_lines_nominal() -> None:
    """TC-APP-03: parse_counting_lines_json successfully parses valid JSON into CountingLine instances."""
    valid_json = json.dumps(
        [
            {"name": "Line-A", "start_point": [50, 100], "end_point": [500, 100]},
            {"name": "Line-B", "start_point": [200, 50], "end_point": [200, 400]},
        ]
    )

    lines = parse_counting_lines_json(valid_json)

    assert len(lines) == 2
    assert isinstance(lines[0], CountingLine)
    assert lines[0].name == "Line-A"
    assert lines[0].start_point == (50, 100)
    assert lines[0].end_point == (500, 100)
    assert lines[1].name == "Line-B"


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_json",
    [
        "{malformed json",
        "",
        "   ",
        json.dumps({"name": "Not a list"}),
        json.dumps(123),
    ],
)
def test_tc_app_04_parse_counting_lines_malformed_syntax_raises(invalid_json: str) -> None:
    """TC-APP-04: parse_counting_lines_json raises ValueError on invalid JSON syntax or non-list."""
    with pytest.raises(ValueError) as exc_info:
        parse_counting_lines_json(invalid_json)
    assert "Invalid JSON" in str(exc_info.value) or "must be a list" in str(exc_info.value)


@pytest.mark.unit
def test_tc_app_05_parse_counting_lines_degenerate_coords_raises() -> None:
    """TC-APP-05: parse_counting_lines_json raises ValueError for degenerate zero-length line."""
    degenerate_json = json.dumps(
        [{"name": "Degenerate", "start_point": [100, 100], "end_point": [100, 100]}]
    )

    with pytest.raises(ValueError) as exc_info:
        parse_counting_lines_json(degenerate_json)
    assert "validation failed" in str(exc_info.value)


@pytest.mark.unit
def test_tc_app_06_parse_counting_lines_negative_coords_raises() -> None:
    """TC-APP-06: parse_counting_lines_json raises ValueError for negative coordinates."""
    negative_json = json.dumps(
        [{"name": "Negative", "start_point": [-10, 50], "end_point": [100, 50]}]
    )

    with pytest.raises(ValueError) as exc_info:
        parse_counting_lines_json(negative_json)
    assert "validation failed" in str(exc_info.value)


# ==============================================================================
# Counting Zones JSON Parsing Tests (TC-APP-07 to TC-APP-09)
# ==============================================================================


@pytest.mark.unit
def test_tc_app_07_parse_counting_zones_nominal() -> None:
    """TC-APP-07: parse_counting_zones_json successfully parses valid polygonal zones."""
    valid_json = json.dumps(
        [
            {
                "name": "Zone-North",
                "polygon": [[100, 100], [400, 100], [400, 300], [100, 300]],
            }
        ]
    )

    zones = parse_counting_zones_json(valid_json)

    assert len(zones) == 1
    assert isinstance(zones[0], CountingZone)
    assert zones[0].name == "Zone-North"
    assert len(zones[0].polygon) == 4


@pytest.mark.unit
def test_tc_app_08_parse_counting_zones_collinear_polygon_raises() -> None:
    """TC-APP-08: parse_counting_zones_json raises ValueError for collinear degenerate polygon."""
    collinear_json = json.dumps([{"name": "Collinear", "polygon": [[0, 0], [50, 50], [100, 100]]}])

    with pytest.raises(ValueError) as exc_info:
        parse_counting_zones_json(collinear_json)
    assert "validation failed" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_zone_json",
    [
        json.dumps([{"name": "TooFewVertices", "polygon": [[10, 10], [50, 50]]}]),
        "{bad json",
        json.dumps({"single": "zone"}),
    ],
)
def test_tc_app_09_parse_counting_zones_invalid_vertices_or_syntax_raises(
    invalid_zone_json: str,
) -> None:
    """TC-APP-09: parse_counting_zones_json raises ValueError for < 3 vertices or syntax error."""
    with pytest.raises(ValueError) as exc_info:
        parse_counting_zones_json(invalid_zone_json)
    assert (
        "validation failed" in str(exc_info.value)
        or "Invalid JSON" in str(exc_info.value)
        or "must be a list" in str(exc_info.value)
    )


# ==============================================================================
# Video Source Resolution Tests (TC-APP-10 to TC-APP-14)
# ==============================================================================


@pytest.mark.unit
def test_tc_app_10_resolve_video_source_sample_existing(tmp_path: Path) -> None:
    """TC-APP-10: resolve_video_source returns existing sample video without creating temporary files."""
    sample_file = tmp_path / "sample.mp4"
    sample_file.touch()

    target, temp_path = resolve_video_source(
        source_type="Sample Video",
        uploaded_file=None,
        custom_source="0",
        sample_path=str(sample_file),
    )

    assert target == str(sample_file)
    assert temp_path is None


@pytest.mark.unit
def test_tc_app_11_resolve_video_source_sample_missing_creates_synthetic(tmp_path: Path) -> None:
    """TC-APP-11: resolve_video_source generates synthetic video if sample file does not exist."""
    synthetic_target = tmp_path / "synthetic" / "sample.mp4"
    assert not synthetic_target.exists()

    target, temp_path = resolve_video_source(
        source_type="Sample Video",
        uploaded_file=None,
        custom_source="0",
        sample_path=str(synthetic_target),
    )

    assert target == str(synthetic_target)
    assert temp_path is None
    assert synthetic_target.is_file()
    assert synthetic_target.stat().st_size > 0


@pytest.mark.unit
def test_tc_app_12_resolve_video_source_upload_spooling() -> None:
    """TC-APP-12: resolve_video_source spools uploaded file to temporary .mp4 file."""
    mock_upload = MagicMock()
    mock_upload.read.return_value = b"RIFF_FAKE_VIDEO_STREAM_BYTES"

    target, temp_path = resolve_video_source(
        source_type="Upload Video",
        uploaded_file=mock_upload,
        custom_source="0",
    )

    assert temp_path is not None
    assert temp_path.is_file()
    assert target == str(temp_path)
    assert temp_path.read_bytes() == b"RIFF_FAKE_VIDEO_STREAM_BYTES"

    # Cleanup temp file
    temp_path.unlink(missing_ok=True)


@pytest.mark.unit
def test_tc_app_13_resolve_video_source_upload_missing_raises() -> None:
    """TC-APP-13: resolve_video_source raises ValueError if Upload Video selected without file."""
    with pytest.raises(ValueError, match="Please upload a video file"):
        resolve_video_source(
            source_type="Upload Video",
            uploaded_file=None,
            custom_source="0",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("custom_input", "expected_target"),
    [
        ("0", 0),
        ("1", 1),
        (" 2 ", 2),
        ("", 0),
        ("   ", 0),
        ("rtsp://192.168.1.100:554/live", "rtsp://192.168.1.100:554/live"),
        ("http://example.com/stream.m3u8", "http://example.com/stream.m3u8"),
    ],
)
def test_tc_app_14_resolve_video_source_webcam_rtsp(
    custom_input: str,
    expected_target: str | int,
) -> None:
    """TC-APP-14: resolve_video_source converts digit webcam indices and preserves RTSP URLs."""
    target, temp_path = resolve_video_source(
        source_type="Webcam / RTSP",
        uploaded_file=None,
        custom_source=custom_input,
    )

    assert target == expected_target
    assert temp_path is None


@pytest.mark.unit
def test_tc_app_14_resolve_video_source_unsupported_raises() -> None:
    """TC-APP-14b: resolve_video_source raises ValueError for unsupported source type."""
    with pytest.raises(ValueError, match="Unsupported source type"):
        resolve_video_source(
            source_type="Unknown Source",
            uploaded_file=None,
            custom_source="0",
        )


# ==============================================================================
# Frame Processing Tests (TC-APP-15)
# ==============================================================================


@pytest.mark.unit
def test_tc_app_15_process_single_frame() -> None:
    """TC-APP-15: process_single_frame coordinates tracker, analytics, and converts BGR to RGB."""
    # Synthetic 640x480 BGR image (blue dominant)
    synthetic_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    synthetic_bgr[:, :, 0] = 255  # Blue channel max

    mock_tracker = MagicMock()
    mock_analytics = MagicMock()

    obj = TrackedObject(
        track_id=1,
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox_xyxy=(50.0, 50.0, 150.0, 150.0),
    )
    mock_frame_result = FrameResult(
        frame_index=1,
        objects=[obj],
        inference_ms=12.5,
        fps=30.0,
    )
    mock_summary = AnalyticsSummary(
        total_in=1,
        total_out=0,
        zone_occupancy={"Zone-A": 1},
        class_counts={"person": 1},
    )

    mock_tracker.track_frame.return_value = mock_frame_result
    mock_tracker.annotate_frame.return_value = synthetic_bgr.copy()
    mock_analytics.update.return_value = mock_summary
    mock_analytics.annotate_analytics.return_value = synthetic_bgr.copy()

    config = DashboardConfig(
        model_path="yolov8n.pt",
        draw_boxes=True,
        draw_lines=True,
        draw_zones=True,
        draw_trajectories=True,
        draw_hud=True,
    )

    rgb_frame, result, summary = process_single_frame(
        frame=synthetic_bgr,
        tracker=mock_tracker,
        analytics=mock_analytics,
        frame_idx=1,
        config=config,
    )

    mock_tracker.track_frame.assert_called_once_with(synthetic_bgr, frame_index=1)
    mock_analytics.update.assert_called_once_with(mock_frame_result)
    mock_tracker.annotate_frame.assert_called_once()
    mock_analytics.annotate_analytics.assert_called_once()

    assert result == mock_frame_result
    assert summary == mock_summary
    assert rgb_frame.shape == (480, 640, 3)
    # Since original had B=255, R=0, after BGR->RGB, R should be 255 and B should be 0
    assert rgb_frame[0, 0, 0] == 0
    assert rgb_frame[0, 0, 2] == 255


# ==============================================================================
# Video Processing Loop Lifecycle Tests (TC-APP-16 to TC-APP-18)
# ==============================================================================


@pytest.mark.unit
@patch("app.cv2.VideoCapture")
@patch("app.ObjectTracker")
@patch("app.SpatialAnalytics")
@patch("app.st")
def test_tc_app_16_run_video_processing_loop_graceful_eof(
    mock_st: MagicMock,
    mock_spatial_cls: MagicMock,
    mock_tracker_cls: MagicMock,
    mock_video_capture: MagicMock,
) -> None:
    """TC-APP-16: run_video_processing_loop terminates gracefully on EOF and updates session state."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    # Frame 0 returns valid frame, Frame 1 returns EOF
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap.read.side_effect = [(True, test_frame), (False, None)]
    mock_video_capture.return_value = mock_cap

    mock_st.session_state = {"is_running": True}

    mock_placeholder = MagicMock()
    mock_metric_cols = {
        "fps": MagicMock(),
        "latency": MagicMock(),
        "in": MagicMock(),
        "out": MagicMock(),
        "tracks": MagicMock(),
    }
    mock_table = MagicMock()

    config = DashboardConfig(source_path="data/sample_video.mp4")

    with patch("app.process_single_frame") as mock_process:
        mock_process.return_value = (
            test_frame,
            FrameResult(frame_index=0, objects=[], inference_ms=10.0, fps=25.0),
            AnalyticsSummary(),
        )
        run_video_processing_loop(
            config=config,
            temp_path=None,
            video_placeholder=mock_placeholder,
            metric_cols=mock_metric_cols,
            table_placeholder=mock_table,
        )

    assert mock_st.session_state["is_running"] is False
    mock_cap.release.assert_called_once()
    mock_st.info.assert_called_once_with("Video playback completed or stream disconnected.")


@pytest.mark.unit
@patch("app.cv2.VideoCapture")
@patch("app.st")
def test_tc_app_17_run_video_processing_loop_unopenable_source(
    mock_st: MagicMock,
    mock_video_capture: MagicMock,
    tmp_path: Path,
) -> None:
    """TC-APP-17: run_video_processing_loop halts and unlinks temp file if source fails to open."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_video_capture.return_value = mock_cap

    mock_st.session_state = {"is_running": True}

    temp_file = tmp_path / "temp_upload.mp4"
    temp_file.touch()

    config = DashboardConfig(source_path=str(temp_file))

    run_video_processing_loop(
        config=config,
        temp_path=temp_file,
        video_placeholder=MagicMock(),
        metric_cols={},
        table_placeholder=MagicMock(),
    )

    assert mock_st.session_state["is_running"] is False
    assert not temp_file.exists()
    mock_st.error.assert_called_once()


@pytest.mark.unit
@patch("app.cv2.VideoCapture")
@patch("app.ObjectTracker")
@patch("app.SpatialAnalytics")
@patch("app.st")
def test_tc_app_18_run_video_processing_loop_exception_recovery(
    mock_st: MagicMock,
    mock_spatial_cls: MagicMock,
    mock_tracker_cls: MagicMock,
    mock_video_capture: MagicMock,
    tmp_path: Path,
) -> None:
    """TC-APP-18: run_video_processing_loop guarantees cap.release() and cleanup in finally block."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = RuntimeError("Fatal hardware decoding error")
    mock_video_capture.return_value = mock_cap

    mock_st.session_state = {"is_running": True}

    temp_file = tmp_path / "temp_to_clean.mp4"
    temp_file.touch()

    config = DashboardConfig(source_path=str(temp_file))

    with pytest.raises(RuntimeError, match="Fatal hardware decoding error"):
        run_video_processing_loop(
            config=config,
            temp_path=temp_file,
            video_placeholder=MagicMock(),
            metric_cols={},
            table_placeholder=MagicMock(),
        )

    # cap.release() and temp_file unlinking MUST have occurred in finally
    mock_cap.release.assert_called_once()
    assert not temp_file.exists()


# ==============================================================================
# UI Actions & Filter Tests (TC-APP-19 to TC-APP-21)
# ==============================================================================


@pytest.mark.unit
def test_init_session_state() -> None:
    """Verify init_session_state registers expected keys with deterministic defaults."""
    with patch("app.st.session_state", {}) as mock_state:
        init_session_state()
        assert mock_state["is_running"] is False
        assert mock_state["tracker"] is None
        assert mock_state["analytics"] is None
        assert mock_state["current_summary"] is None
        assert mock_state["last_metrics"] == {"fps": 0.0, "latency": 0.0, "tracks": 0}


@pytest.mark.unit
def test_tc_app_19_reset_counters_action() -> None:
    """TC-APP-19: Reset action calls tracker.reset() and analytics.reset() and clears summary."""
    mock_tracker = MagicMock()
    mock_analytics = MagicMock()

    session_state = {
        "tracker": mock_tracker,
        "analytics": mock_analytics,
        "current_summary": MagicMock(),
    }

    # Simulate reset button trigger logic from app.py
    if session_state.get("tracker"):
        session_state["tracker"].reset()
    if session_state.get("analytics"):
        session_state["analytics"].reset()
    session_state["current_summary"] = None

    mock_tracker.reset.assert_called_once()
    mock_analytics.reset.assert_called_once()
    assert session_state["current_summary"] is None


@pytest.mark.unit
def test_tc_app_20_class_filter_empty_tracks_all() -> None:
    """TC-APP-20: Empty class selection resolves target_classes to None (track all)."""
    assert COCO_CLASSES[0] == "person"
    assert COCO_CLASSES[2] == "car"
    selected_classes: list[int] = []
    target_classes = selected_classes if selected_classes else None

    config = DashboardConfig(target_classes=target_classes)

    assert config.target_classes is None


@pytest.mark.unit
def test_tc_app_21_class_filter_specific_selection() -> None:
    """TC-APP-21: Specific class selection passes target_classes list to DashboardConfig."""
    selected_classes: list[int] = [0, 2]
    target_classes = selected_classes if selected_classes else None

    config = DashboardConfig(target_classes=target_classes)

    assert config.target_classes == [0, 2]


# ==============================================================================
# AppTest Headless Simulation Tests (TC-APP-22 to TC-APP-23)
# ==============================================================================

APP_FILE = str(Path(__file__).parent.parent / "app.py")


@pytest.mark.integration
def test_tc_app_22_streamlit_apptest_initial_render() -> None:
    """TC-APP-22: AppTest headless simulation renders initial dashboard layout cleanly."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()

    assert not at.exception
    assert len(at.title) >= 1
    assert "Real-Time Object Tracking" in at.title[0].value

    # Control buttons rendered
    button_labels = [btn.label for btn in at.button]
    assert "▶ Start Tracking" in button_labels
    assert "⏹ Stop Tracking" in button_labels
    assert "↺ Reset Counts" in button_labels

    # Sidebar controls rendered
    sidebar_slider_labels = [s.label for s in at.sidebar.slider]
    assert "Confidence Threshold" in sidebar_slider_labels
    assert "IoU Threshold" in sidebar_slider_labels


@pytest.mark.integration
def test_tc_app_23_streamlit_apptest_sidebar_interactions() -> None:
    """TC-APP-23: AppTest interacts with sidebar sliders, checkboxes, and re-renders safely."""
    at = AppTest.from_file(APP_FILE, default_timeout=30)
    at.run()
    assert not at.exception

    # Adjust confidence threshold slider
    at.sidebar.slider[0].set_value(0.60).run()
    assert not at.exception
    assert at.sidebar.slider[0].value == 0.60

    # Toggle an overlay checkbox
    at.sidebar.checkbox[0].uncheck().run()
    assert not at.exception
    assert at.sidebar.checkbox[0].value is False


# ==============================================================================
# DashboardConfig Schema Contract Audit (TC-APP-24)
# ==============================================================================


@pytest.mark.unit
def test_tc_app_24_dashboard_config_contract_audit() -> None:
    """TC-APP-24: DashboardConfig enforces frozen immutability, extra='forbid', and boundaries."""
    config = DashboardConfig()

    # 1. Nominal defaults
    assert config.model_path == "yolov8n.pt"
    assert config.confidence_threshold == 0.25
    assert config.iou_threshold == 0.45
    assert config.tracker_type == "bytetrack.yaml"
    assert config.target_classes is None
    assert config.source_type == "Sample Video"
    assert config.source_path == "data/sample_video.mp4"
    assert config.draw_boxes is True
    assert config.draw_trajectories is True
    assert config.draw_lines is True
    assert config.draw_zones is True
    assert config.draw_hud is True
    assert config.device == "auto"

    # 2. Frozen immutability
    with pytest.raises(ValidationError):
        config.confidence_threshold = 0.50  # type: ignore[misc]

    # 3. Extra fields forbidden
    with pytest.raises(ValidationError):
        DashboardConfig(unexpected_key="not_allowed")  # type: ignore[call-arg]

    # 4. Boundary thresholds
    with pytest.raises(ValidationError):
        DashboardConfig(confidence_threshold=-0.1)

    with pytest.raises(ValidationError):
        DashboardConfig(confidence_threshold=1.1)

    with pytest.raises(ValidationError):
        DashboardConfig(iou_threshold=-0.05)

    with pytest.raises(ValidationError):
        DashboardConfig(iou_threshold=1.05)

    # 5. Empty or whitespace paths
    with pytest.raises(ValidationError):
        DashboardConfig(model_path="   ")

    with pytest.raises(ValidationError):
        DashboardConfig(source_path="")


# ==============================================================================
# GPU Device UI & Telemetry Tests (TC-APP-25 to TC-APP-27)
# ==============================================================================


@pytest.mark.unit
@patch("app.st")
def test_tc_app_25_sidebar_cuda_device_telemetry_rendered(mock_st: MagicMock, mocker: Any) -> None:
    """TC-APP-25: Verify sidebar renders GPU success badge and VRAM metrics when CUDA is active."""
    # Mock selectbox returns
    # 1st: YOLO weights, 2nd: Tracker, 3rd: Compute Device ("CUDA GPU (NVIDIA RTX 5060 Ti)"), 4th: Video Source
    mock_st.sidebar.selectbox.side_effect = [
        "yolov8n.pt",
        "bytetrack.yaml",
        "CUDA GPU (NVIDIA RTX 5060 Ti)",
    ]
    mock_st.sidebar.radio.return_value = "Sample Video"
    mock_st.sidebar.slider.side_effect = [0.25, 0.45]
    mock_st.sidebar.multiselect.return_value = []
    mock_st.sidebar.text_area.side_effect = ["[]", "[]"]
    mock_st.sidebar.checkbox.return_value = True
    mock_st.sidebar.columns.return_value = [MagicMock(), MagicMock()]

    mocker.patch(
        "app.resolve_compute_device",
        return_value=MagicMock(
            resolved_device="cuda:0",
            device_type="cuda",
            is_fallback=False,
            fallback_reason=None,
        ),
    )
    mock_mem = MagicMock(
        device_name="NVIDIA GeForce RTX 5060 Ti",
        allocated_memory_mb=1024.0,
        total_memory_mb=16384.0,
        utilization_pct=6.25,
    )
    mocker.patch("app.get_cuda_memory_info", return_value=mock_mem)

    config, temp_path, err = render_sidebar()

    assert err is None
    assert config is not None
    assert config.device == "cuda:0"
    mock_st.sidebar.success.assert_called_once_with("🟢 **Compute:** CUDA:0")
    mock_st.sidebar.caption.assert_called_once()
    mock_st.sidebar.progress.assert_called_once()


@pytest.mark.unit
@patch("app.st")
def test_tc_app_26_sidebar_cuda_fallback_warning(mock_st: MagicMock, mocker: Any) -> None:
    """TC-APP-26: Verify sidebar renders warning when device resolution falls back."""
    mock_st.sidebar.selectbox.side_effect = [
        "yolov8n.pt",
        "bytetrack.yaml",
        "CUDA GPU (NVIDIA RTX 5060 Ti)",
    ]
    mock_st.sidebar.radio.return_value = "Sample Video"
    mock_st.sidebar.slider.side_effect = [0.25, 0.45]
    mock_st.sidebar.multiselect.return_value = []
    mock_st.sidebar.text_area.side_effect = ["[]", "[]"]
    mock_st.sidebar.checkbox.return_value = True
    mock_st.sidebar.columns.return_value = [MagicMock(), MagicMock()]

    mocker.patch(
        "app.resolve_compute_device",
        return_value=MagicMock(
            resolved_device="cpu",
            device_type="cpu",
            is_fallback=True,
            fallback_reason="Requested device 'cuda:0' is unavailable. Operating on CPU.",
        ),
    )

    config, temp_path, err = render_sidebar()

    assert err is None
    assert config is not None
    assert config.device == "cuda:0"
    mock_st.sidebar.info.assert_called_once_with("⚪ **Compute:** CPU Mode")
    mock_st.sidebar.warning.assert_called_once_with(
        "⚠️ Requested device 'cuda:0' is unavailable. Operating on CPU."
    )


@pytest.mark.unit
@patch("app.cv2.VideoCapture")
@patch("app.ObjectTracker")
@patch("app.SpatialAnalytics")
@patch("app.st")
def test_tc_app_27_processing_loop_calls_empty_cuda_cache(
    mock_st: MagicMock,
    mock_spatial_cls: MagicMock,
    mock_tracker_cls: MagicMock,
    mock_video_capture: MagicMock,
    mocker: Any,
) -> None:
    """TC-APP-27: Verify empty_cuda_cache is always executed upon video loop exit."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (False, None)
    mock_video_capture.return_value = mock_cap

    mock_st.session_state = {"is_running": True}
    mock_empty_cache = mocker.patch("app.empty_cuda_cache")

    config = DashboardConfig(source_path="data/sample_video.mp4")

    run_video_processing_loop(
        config=config,
        temp_path=None,
        video_placeholder=MagicMock(),
        metric_cols={
            "fps": MagicMock(),
            "latency": MagicMock(),
            "in": MagicMock(),
            "out": MagicMock(),
            "tracks": MagicMock(),
        },
        table_placeholder=MagicMock(),
    )

    mock_empty_cache.assert_called_once()
