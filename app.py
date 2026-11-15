"""Real-Time Object Detection, Tracking, and Spatial Analytics Streamlit Dashboard.

Provides an interactive web dashboard for real-time video inference, multi-object tracking,
directional line-crossing tripwires, polygonal zone occupancy, and analytics summaries.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st

from src.counter import SpatialAnalytics
from src.device import empty_cuda_cache, get_cuda_memory_info, resolve_compute_device
from src.schemas import (
    DEFAULT_LINES_JSON,
    DEFAULT_ZONES_JSON,
    AnalyticsSummary,
    CountingLine,
    CountingZone,
    DashboardConfig,
    DetectorConfig,
    FrameResult,
)
from src.tracker import ObjectTracker

logger = logging.getLogger(__name__)

# Standard COCO category mapping for class filtering
COCO_CLASSES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    11: "stop sign",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    39: "bottle",
    56: "chair",
    62: "tv",
    63: "laptop",
    67: "cell phone",
}


def scan_available_models(
    models_dir: str | Path = "models",
    root_dir: str | Path = ".",
) -> list[str]:
    """Scan filesystem for available YOLO .pt weights.

    Args:
        models_dir: Primary directory containing fine-tuned or downloaded weights.
        root_dir: Workspace root directory.

    Returns:
        Sorted list of discovered model weights, ensuring yolov8n.pt is present.
    """
    discovered: set[str] = {"yolov8n.pt"}

    models_path = Path(models_dir)
    if models_path.is_dir():
        for p in models_path.glob("*.pt"):
            discovered.add(str(p))

    root_path = Path(root_dir)
    if root_path.is_dir():
        for p in root_path.glob("*.pt"):
            discovered.add(p.name)

    return sorted(discovered)


def parse_counting_lines_json(raw_json: str) -> list[CountingLine]:
    """Parse and validate JSON string into a list of CountingLine domain models.

    Args:
        raw_json: JSON string representing an array of line definitions.

    Returns:
        List of validated CountingLine instances.

    Raises:
        ValueError: If JSON syntax is invalid or schema validation fails.
    """
    try:
        data = json.loads(raw_json)
    except Exception as exc:
        raise ValueError(f"Invalid JSON syntax for Counting Lines: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError("Counting Lines JSON must be a list of line objects")

    lines: list[CountingLine] = []
    for idx, item in enumerate(data):
        try:
            line = CountingLine(**item)
            lines.append(line)
        except Exception as exc:
            raise ValueError(f"Line item #{idx + 1} validation failed: {exc}") from exc

    return lines


def parse_counting_zones_json(raw_json: str) -> list[CountingZone]:
    """Parse and validate JSON string into a list of CountingZone domain models.

    Args:
        raw_json: JSON string representing an array of zone definitions.

    Returns:
        List of validated CountingZone instances.

    Raises:
        ValueError: If JSON syntax is invalid or schema validation fails.
    """
    try:
        data = json.loads(raw_json)
    except Exception as exc:
        raise ValueError(f"Invalid JSON syntax for Counting Zones: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError("Counting Zones JSON must be a list of zone objects")

    zones: list[CountingZone] = []
    for idx, item in enumerate(data):
        try:
            zone = CountingZone(**item)
            zones.append(zone)
        except Exception as exc:
            raise ValueError(f"Zone item #{idx + 1} validation failed: {exc}") from exc

    return zones


def resolve_video_source(
    source_type: str,
    uploaded_file: Any | None,
    custom_source: str,
    sample_path: str = "data/sample_video.mp4",
) -> tuple[str | int, Path | None]:
    """Resolve OpenCV VideoCapture target and manage temporary spooling.

    Args:
        source_type: One of 'Sample Video', 'Upload Video', or 'Webcam / RTSP'.
        uploaded_file: Streamlit UploadedFile object or None.
        custom_source: Custom camera index or RTSP stream string.
        sample_path: Path to bundled sample video.

    Returns:
        Tuple of (capture_target, temporary_file_path_or_None).

    Raises:
        ValueError: If required upload is missing or source specification is invalid.
    """
    if source_type == "Sample Video":
        sample = Path(sample_path)
        if not sample.is_file():
            # Create a 30-frame synthetic test video if sample video is missing
            sample.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(sample), fourcc, 10.0, (640, 480))
            for i in range(30):
                frame = np.full((480, 640, 3), 40, dtype=np.uint8)
                cv2.circle(frame, (100 + i * 10, 100 + i * 8), 20, (0, 255, 0), -1)
                out.write(frame)
            out.release()
        return str(sample), None

    elif source_type == "Upload Video":
        if uploaded_file is None:
            raise ValueError("Please upload a video file (.mp4, .avi, .mov)")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(uploaded_file.read())
            tfile.flush()
            file_name = tfile.name
        return file_name, Path(file_name)

    elif source_type == "Webcam / RTSP":
        stripped = custom_source.strip()
        if not stripped:
            return 0, None
        if stripped.isdigit():
            return int(stripped), None
        return stripped, None

    raise ValueError(f"Unsupported source type: {source_type}")


def init_session_state() -> None:
    """Initialize Streamlit session state keys with deterministic defaults."""
    st.session_state.setdefault("is_running", False)
    st.session_state.setdefault("tracker", None)
    st.session_state.setdefault("analytics", None)
    st.session_state.setdefault("current_summary", None)
    st.session_state.setdefault("last_metrics", {"fps": 0.0, "latency": 0.0, "tracks": 0})


def render_sidebar() -> tuple[DashboardConfig | None, Path | None, str | None]:
    """Render sidebar controls and validate configuration into DashboardConfig.

    Returns:
        Tuple of (DashboardConfig_or_None, temp_cleanup_path_or_None, error_message_or_None).
    """
    st.sidebar.title("Configuration")

    # 1. Model & Tracker Selection
    st.sidebar.subheader("Model & Inference")
    available_models = scan_available_models()
    selected_model = st.sidebar.selectbox("YOLO Weights", available_models, index=0)

    tracker_type = st.sidebar.selectbox("Tracker", ["bytetrack.yaml", "botsort.yaml"], index=0)

    # Compute Device Selection & Telemetry
    device_choices = [
        "Auto-Detect (GPU if available)",
        "CUDA GPU (NVIDIA RTX 5060 Ti)",
        "CPU Only",
    ]
    selected_device_choice = st.sidebar.selectbox("Compute Device", device_choices, index=0)
    if "CUDA GPU" in selected_device_choice:
        requested_device = "cuda:0"
    elif "CPU Only" in selected_device_choice:
        requested_device = "cpu"
    else:
        requested_device = "auto"

    dev_res = resolve_compute_device(requested_device)
    if dev_res.device_type == "cuda":
        st.sidebar.success(f"🟢 **Compute:** {dev_res.resolved_device.upper()}")
        mem_info = get_cuda_memory_info(0)
        if mem_info is not None:
            st.sidebar.caption(
                f"**GPU:** {mem_info.device_name}\n\n"
                f"**VRAM:** {mem_info.allocated_memory_mb:.1f} MB / {mem_info.total_memory_mb / 1024:.1f} GB"
            )
            st.sidebar.progress(
                min(1.0, max(0.0, mem_info.utilization_pct / 100.0)),
                text=f"VRAM Utilization: {mem_info.utilization_pct:.1f}%",
            )
    else:
        st.sidebar.info("⚪ **Compute:** CPU Mode")

    if dev_res.is_fallback and dev_res.fallback_reason:
        st.sidebar.warning(f"⚠️ {dev_res.fallback_reason}")

    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.25, 0.05)
    iou_thresh = st.sidebar.slider("IoU Threshold", 0.05, 1.0, 0.45, 0.05)

    # Class filter multiselect
    class_options = list(COCO_CLASSES.keys())
    selected_classes = st.sidebar.multiselect(
        "Filter Classes (Empty = Track All)",
        options=class_options,
        format_func=lambda x: f"{COCO_CLASSES[x]} (ID:{x})",
        default=[0, 2, 3, 5, 7],  # Common transport/pedestrian defaults
    )
    target_classes = selected_classes if selected_classes else None

    # 2. Source Selection
    st.sidebar.subheader("Video Source")
    source_type = st.sidebar.radio(
        "Source Type",
        ["Sample Video", "Upload Video", "Webcam / RTSP"],
        index=0,
    )

    uploaded_file = None
    custom_source = "0"
    if source_type == "Upload Video":
        uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
    elif source_type == "Webcam / RTSP":
        custom_source = st.sidebar.text_input("Device Index or RTSP URL", value="0")

    # 3. Spatial Analytics Tripwires & Zones
    st.sidebar.subheader("Spatial Analytics Geometry")
    raw_lines_json = st.sidebar.text_area(
        "Tripwire Lines (JSON)", value=DEFAULT_LINES_JSON, height=140
    )
    raw_zones_json = st.sidebar.text_area(
        "Polygonal Zones (JSON)", value=DEFAULT_ZONES_JSON, height=140
    )

    # 4. Display Overlay Toggles
    st.sidebar.subheader("Overlays")
    col1, col2 = st.sidebar.columns(2)
    draw_boxes = col1.checkbox("BBoxes & IDs", value=True)
    draw_trajectories = col2.checkbox("Trajectories", value=True)
    draw_lines = col1.checkbox("Tripwires", value=True)
    draw_zones = col2.checkbox("Zones", value=True)
    draw_hud = st.sidebar.checkbox("Analytics HUD", value=True)

    # Validation
    try:
        lines = parse_counting_lines_json(raw_lines_json)
        zones = parse_counting_zones_json(raw_zones_json)
        source_target, temp_path = resolve_video_source(source_type, uploaded_file, custom_source)

        config = DashboardConfig(
            model_path=selected_model,
            confidence_threshold=conf_thresh,
            iou_threshold=iou_thresh,
            tracker_type=tracker_type,
            target_classes=target_classes,
            source_type=source_type,
            source_path=str(source_target),
            lines=lines,
            zones=zones,
            draw_boxes=draw_boxes,
            draw_trajectories=draw_trajectories,
            draw_lines=draw_lines,
            draw_zones=draw_zones,
            draw_hud=draw_hud,
            device=requested_device,
        )
        return config, temp_path, None
    except Exception as exc:
        return None, None, str(exc)


def process_single_frame(
    frame: np.ndarray,
    tracker: ObjectTracker,
    analytics: SpatialAnalytics,
    frame_idx: int,
    config: DashboardConfig,
) -> tuple[np.ndarray, FrameResult, AnalyticsSummary]:
    """Execute decoupled tracking and spatial analytics on a single frame.

    Args:
        frame: Input BGR image array (H, W, 3).
        tracker: ObjectTracker instance.
        analytics: SpatialAnalytics instance.
        frame_idx: Monotonic sequence index.
        config: DashboardConfig controlling visualization switches.

    Returns:
        Tuple of (RGB_annotated_frame, FrameResult, AnalyticsSummary).
    """
    frame_result = tracker.track_frame(frame, frame_index=frame_idx)
    summary = analytics.update(frame_result)

    annotated = tracker.annotate_frame(frame, frame_result) if config.draw_boxes else frame.copy()
    annotated = analytics.annotate_analytics(
        annotated,
        summary,
        draw_lines=config.draw_lines,
        draw_zones=config.draw_zones,
        draw_trajectories=config.draw_trajectories,
        draw_hud=config.draw_hud,
    )

    # Convert BGR to RGB for Streamlit display
    rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return rgb_frame, frame_result, summary


def run_video_processing_loop(
    config: DashboardConfig,
    temp_path: Path | None,
    video_placeholder: Any,
    metric_cols: dict[str, Any],
    table_placeholder: Any,
) -> None:
    """Manage real-time frame ingestion, inference loop, and graceful shutdown."""
    target = int(config.source_path) if config.source_path.isdigit() else config.source_path
    cap = cv2.VideoCapture(target)

    if not cap.isOpened():
        st.error(f"Failed to open video source: {config.source_path}")
        st.session_state["is_running"] = False
        if temp_path:
            temp_path.unlink(missing_ok=True)
        return

    # Instantiate or reuse tracker & analytics
    detector_config = DetectorConfig(
        model_path=config.model_path,
        confidence_threshold=config.confidence_threshold,
        iou_threshold=config.iou_threshold,
        target_classes=config.target_classes,
        tracker_type=config.tracker_type,
        device=config.device,
    )
    tracker = ObjectTracker(config=detector_config)
    analytics = SpatialAnalytics(lines=config.lines, zones=config.zones)

    if tracker.is_fallback and tracker.fallback_reason:
        st.sidebar.warning(f"⚠️ {tracker.fallback_reason}")

    st.session_state["tracker"] = tracker
    st.session_state["analytics"] = analytics

    frame_idx = 0
    try:
        while st.session_state.get("is_running", False):
            ret, frame = cap.read()
            if not ret or frame is None:
                st.info("Video playback completed or stream disconnected.")
                st.session_state["is_running"] = False
                break

            rgb_frame, frame_result, summary = process_single_frame(
                frame=frame,
                tracker=tracker,
                analytics=analytics,
                frame_idx=frame_idx,
                config=config,
            )

            # Update live image viewport
            video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            # Update metrics cards
            metric_cols["fps"].metric("FPS", f"{frame_result.fps:.1f}")
            metric_cols["latency"].metric("Latency", f"{frame_result.inference_ms:.1f} ms")
            metric_cols["in"].metric("Total IN", summary.total_in)
            metric_cols["out"].metric("Total OUT", summary.total_out)
            metric_cols["tracks"].metric("Active Tracks", len(frame_result.objects))

            # Update zone occupancy & class counts table
            zone_data = {
                "Zone": list(summary.zone_occupancy.keys()),
                "Current Occupancy": list(summary.zone_occupancy.values()),
            }
            table_placeholder.dataframe(zone_data, use_container_width=True, hide_index=True)

            frame_idx += 1
    finally:
        cap.release()
        empty_cuda_cache()
        if temp_path:
            temp_path.unlink(missing_ok=True)


def main() -> None:
    """Primary application layout and state manager."""
    st.set_page_config(
        page_title="Real-Time Object Tracking & Analytics",
        page_icon="🎯",
        layout="wide",
    )

    init_session_state()

    st.title("🎯 Real-Time Object Tracking & Spatial Analytics")
    st.caption("Powered by Ultralytics YOLO, ByteTrack, OpenCV, Pydantic v2, and Streamlit")

    # Render Sidebar and extract validated configuration
    config, temp_path, error_msg = render_sidebar()

    if error_msg:
        st.sidebar.error(error_msg)

    # Main Action Controls
    ctrl_col1, ctrl_col2, ctrl_col3, _ = st.columns([1, 1, 1, 5])
    start_btn = ctrl_col1.button(
        "▶ Start Tracking", type="primary", disabled=st.session_state["is_running"]
    )
    stop_btn = ctrl_col2.button("⏹ Stop Tracking", disabled=not st.session_state["is_running"])
    reset_btn = ctrl_col3.button("↺ Reset Counts")

    if start_btn:
        if config is None:
            st.error("Cannot start: please fix configuration errors in the sidebar.")
        else:
            st.session_state["is_running"] = True

    if stop_btn:
        st.session_state["is_running"] = False

    if reset_btn:
        if st.session_state.get("tracker"):
            st.session_state["tracker"].reset()
        if st.session_state.get("analytics"):
            st.session_state["analytics"].reset()
        st.session_state["current_summary"] = None
        st.success("Spatial counters and tracker state successfully reset.")

    # Top KPI Metrics Cards
    metric_cols = {
        "fps": st.empty(),
        "latency": st.empty(),
        "in": st.empty(),
        "out": st.empty(),
        "tracks": st.empty(),
    }
    m1, m2, m3, m4, m5 = st.columns(5)
    metric_cols["fps"] = m1
    metric_cols["latency"] = m2
    metric_cols["in"] = m3
    metric_cols["out"] = m4
    metric_cols["tracks"] = m5

    # Initialize metric displays with neutral default state
    metric_cols["fps"].metric("FPS", "0.0")
    metric_cols["latency"].metric("Latency", "0.0 ms")
    metric_cols["in"].metric("Total IN", 0)
    metric_cols["out"].metric("Total OUT", 0)
    metric_cols["tracks"].metric("Active Tracks", 0)

    # Primary Canvas & Side Analytics View
    vid_col, data_col = st.columns([3, 1])
    video_placeholder = vid_col.empty()

    with data_col:
        st.subheader("Zone Occupancy")
        table_placeholder = st.empty()

    # Active Execution
    if st.session_state["is_running"] and config is not None:
        run_video_processing_loop(
            config=config,
            temp_path=temp_path,
            video_placeholder=video_placeholder,
            metric_cols=metric_cols,
            table_placeholder=table_placeholder,
        )
    else:
        video_placeholder.info("Click 'Start Tracking' to launch inference stream.")


if __name__ == "__main__":
    main()
