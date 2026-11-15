"""Detection and ByteTrack multi-object tracking inference engine.

Provides the ObjectTracker class wrapping Ultralytics YOLO with ByteTrack,
strictly validating outputs into Pydantic v2 FrameResult models.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from src.device import empty_cuda_cache, resolve_compute_device
from src.schemas import DetectorConfig, DeviceResolutionResult, FrameResult, TrackedObject

logger = logging.getLogger(__name__)

# Standard 10-color BGR palette for track visualization
DEFAULT_PALETTE: list[tuple[int, int, int]] = [
    (255, 56, 56),  # Red
    (255, 157, 151),  # Coral
    (255, 112, 31),  # Orange
    (255, 178, 29),  # Yellow
    (207, 210, 49),  # Lime
    (72, 249, 10),  # Green
    (146, 204, 23),  # Turquoise
    (61, 219, 134),  # Aqua
    (26, 147, 247),  # Sky Blue
    (118, 93, 255),  # Violet
]


class ObjectTracker:
    """Multi-object tracker encapsulating Ultralytics YOLO with ByteTrack."""

    def __init__(
        self,
        config: DetectorConfig | None = None,
        model: Any | None = None,
    ) -> None:
        """Initialize tracker with configuration and optional injected YOLO model.

        Args:
            config: Validated DetectorConfig settings. Defaults to standard configuration.
            model: Optional pre-loaded or mocked YOLO instance for offline testing.
        """
        self.config: DetectorConfig = config if config is not None else DetectorConfig()
        self.resolution: DeviceResolutionResult = resolve_compute_device(self.config.device)
        self.resolved_device: str = self.resolution.resolved_device
        self.is_fallback: bool = self.resolution.is_fallback
        self.fallback_reason: str | None = self.resolution.fallback_reason

        if model is not None:
            self.model: Any = model
        else:
            model_path = Path(self.config.model_path)
            if model_path.parent and not model_path.parent.exists():
                model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model = YOLO(self.config.model_path)

        # Tier 2 Model device placement with graceful fallback protection
        if self.resolved_device != "cpu" and hasattr(self.model, "to"):
            try:
                self.model.to(self.resolved_device)
            except Exception as exc:
                logger.warning(
                    "Failed to move model to %s (%s). Falling back to CPU.",
                    self.resolved_device,
                    exc,
                )
                self.resolved_device = "cpu"
                self.is_fallback = True
                self.fallback_reason = (
                    f"Model placement on {self.resolution.resolved_device} failed: {exc}. "
                    "Operating on CPU."
                )
                try:
                    self.model.to("cpu")
                except Exception:
                    pass

    def _validate_frame(self, frame: Any) -> None:
        """Validate that input frame meets format, dimension, and dtype requirements.

        Args:
            frame: Object to validate as an image frame.

        Raises:
            ValueError: If frame is not a valid 3-channel uint8 numpy ndarray.
        """
        if not isinstance(frame, np.ndarray):
            raise ValueError(f"Frame must be a numpy.ndarray; received {type(frame).__name__}")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Frame must be a 3D array with shape (H, W, 3); received shape {frame.shape}"
            )
        if frame.shape[0] == 0 or frame.shape[1] == 0 or frame.size == 0:
            raise ValueError(f"Frame dimensions must be non-zero; received shape {frame.shape}")
        if frame.dtype != np.uint8:
            raise ValueError(f"Frame dtype must be uint8; received {frame.dtype}")

    def track_frame(self, frame: np.ndarray, frame_index: int = 0) -> FrameResult:
        """Execute tracking inference on a single BGR image frame.

        Args:
            frame: Input 3-channel BGR uint8 NumPy image array (H, W, 3).
            frame_index: Non-negative integer index of the frame in sequence.

        Returns:
            Validated FrameResult containing TrackedObject items and timing metrics.

        Raises:
            ValueError: If frame is invalid or frame_index < 0.
        """
        self._validate_frame(frame)
        if frame_index < 0:
            raise ValueError(f"frame_index must be non-negative; received {frame_index}")

        start_time = time.perf_counter()

        # Execute tracking with persistence across consecutive frames
        try:
            raw_results = self.model.track(
                source=frame,
                persist=True,
                tracker=self.config.tracker_type,
                conf=self.config.confidence_threshold,
                iou=self.config.iou_threshold,
                classes=self.config.target_classes,
                device=self.resolved_device,
                verbose=False,
            )
        except Exception as exc:
            if self.resolved_device != "cpu":
                logger.warning(
                    "Inference failed on %s (%s). Triggering mid-stream fallback to CPU.",
                    self.resolved_device,
                    exc,
                )
                self.resolved_device = "cpu"
                self.is_fallback = True
                self.fallback_reason = (
                    f"Mid-stream fallback on frame {frame_index}: {exc}. Switched to CPU."
                )
                empty_cuda_cache()
                raw_results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker=self.config.tracker_type,
                    conf=self.config.confidence_threshold,
                    iou=self.config.iou_threshold,
                    classes=self.config.target_classes,
                    device="cpu",
                    verbose=False,
                )
            else:
                raise

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return self._parse_results(
            raw_results=raw_results,
            frame_index=frame_index,
            elapsed_ms=elapsed_ms,
        )

    def _parse_results(
        self,
        raw_results: Any,
        frame_index: int,
        elapsed_ms: float,
    ) -> FrameResult:
        """Parse raw Ultralytics Results into validated Pydantic FrameResult.

        Args:
            raw_results: Ultralytics Results list or object returned by model.track().
            frame_index: Current frame index.
            elapsed_ms: Inference duration in milliseconds.

        Returns:
            Validated FrameResult instance.
        """
        tracked_objects: list[TrackedObject] = []

        # Ultralytics track returns a list of Results objects (one per input frame)
        result = raw_results[0] if isinstance(raw_results, (list, tuple)) else raw_results

        if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            has_ids = hasattr(boxes, "id") and boxes.id is not None

            # Get class name ontology from model
            names: dict[int, str] = {}
            if hasattr(self.model, "names") and isinstance(self.model.names, dict):
                names = self.model.names
            elif hasattr(self.model, "names") and isinstance(self.model.names, (list, tuple)):
                names = {idx: name for idx, name in enumerate(self.model.names)}

            # Extract coordinates, confidences, class IDs, and track IDs
            xyxy_tensor = boxes.xyxy
            conf_tensor = boxes.conf
            cls_tensor = boxes.cls
            id_tensor = boxes.id if has_ids else None

            # Convert to CPU/numpy for iteration
            xyxy_array = (
                xyxy_tensor.cpu().numpy()
                if hasattr(xyxy_tensor, "cpu")
                else np.asarray(xyxy_tensor)
            )
            conf_array = (
                conf_tensor.cpu().numpy()
                if hasattr(conf_tensor, "cpu")
                else np.asarray(conf_tensor)
            )
            cls_array = (
                cls_tensor.cpu().numpy() if hasattr(cls_tensor, "cpu") else np.asarray(cls_tensor)
            )
            id_array = None
            if id_tensor is not None:
                id_array = (
                    id_tensor.cpu().numpy() if hasattr(id_tensor, "cpu") else np.asarray(id_tensor)
                )

            for i in range(len(xyxy_array)):
                # If tracker hasn't assigned an ID yet, skip this detection
                if id_array is None or id_array[i] is None or np.isnan(id_array[i]):
                    continue

                track_id = int(id_array[i])
                class_id = int(cls_array[i])
                confidence = float(conf_array[i])
                class_name = names.get(class_id, f"class_{class_id}")

                # Clamp sub-pixel negative coordinate regressions to non-negative floats
                raw_x1, raw_y1, raw_x2, raw_y2 = xyxy_array[i][:4]
                x1 = max(0.0, float(raw_x1))
                y1 = max(0.0, float(raw_y1))
                x2 = max(0.0, float(raw_x2))
                y2 = max(0.0, float(raw_y2))

                # Skip degenerate bounding boxes (width or height <= 0)
                if x1 >= x2 or y1 >= y2:
                    logger.warning(
                        "Skipping degenerate bounding box: (%.2f, %.2f, %.2f, %.2f)", x1, y1, x2, y2
                    )
                    continue

                try:
                    tracked_obj = TrackedObject(
                        track_id=track_id,
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox_xyxy=(x1, y1, x2, y2),
                        centroid=None,  # Auto-computed by schema validator
                    )
                    tracked_objects.append(tracked_obj)
                except Exception as exc:
                    logger.warning("Failed to validate TrackedObject: %s", exc)
                    continue

        fps = 1000.0 / elapsed_ms if elapsed_ms > 0.0 else 0.0

        return FrameResult(
            frame_index=frame_index,
            inference_ms=elapsed_ms,
            fps=fps,
            objects=tracked_objects,
        )

    def annotate_frame(
        self,
        frame: np.ndarray,
        frame_result: FrameResult,
        color_palette: dict[int, tuple[int, int, int]] | None = None,
    ) -> np.ndarray:
        """Render tracking bounding boxes, IDs, labels, and confidences headlessly.

        Args:
            frame: Original BGR image frame (H, W, 3).
            frame_result: Tracked objects and inference metadata for this frame.
            color_palette: Optional mapping of track_id to BGR color tuples.

        Returns:
            New annotated copy of the frame as np.ndarray.
        """
        self._validate_frame(frame)
        annotated = frame.copy()

        for obj in frame_result.objects:
            x1, y1, x2, y2 = (int(round(coord)) for coord in obj.bbox_xyxy)
            cx, cy = (int(round(coord)) for coord in obj.centroid) if obj.centroid else (0, 0)

            # Assign color by track_id
            if color_palette and obj.track_id in color_palette:
                color = color_palette[obj.track_id]
            else:
                color = DEFAULT_PALETTE[obj.track_id % len(DEFAULT_PALETTE)]

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw centroid marker
            cv2.circle(annotated, (cx, cy), 4, color, -1)

            # Construct label
            label = f"#{obj.track_id} {obj.class_name} {obj.confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            label_y1 = max(0, y1 - label_h - baseline - 4)
            label_y2 = y1

            # Draw label background box
            cv2.rectangle(
                annotated,
                (x1, label_y1),
                (x1 + label_w + 4, label_y2),
                color,
                -1,
            )

            # Draw text label in white/black
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - baseline - 2),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                lineType=cv2.LINE_AA,
            )

        return annotated

    def process_video(
        self,
        video_path: str | Path,
        annotate: bool = True,
    ) -> Generator[tuple[np.ndarray | None, FrameResult], None, None]:
        """Process a video file sequentially frame-by-frame.

        Args:
            video_path: File path to the video source.
            annotate: Whether to generate and yield annotated image frames.

        Yields:
            Tuple of (annotated_frame_or_None, FrameResult) per frame.

        Raises:
            FileNotFoundError: If video file does not exist.
            ValueError: If video file cannot be opened.
        """
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {path}")

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_result = self.track_frame(frame, frame_index=frame_idx)
                annotated_frame = self.annotate_frame(frame, frame_result) if annotate else None

                yield (annotated_frame, frame_result)
                frame_idx += 1
        finally:
            cap.release()

    def reset(self) -> None:
        """Reset internal tracker state (Kalman filters and track IDs) and flush GPU cache."""
        if (
            hasattr(self.model, "predictor")
            and self.model.predictor is not None
            and hasattr(self.model.predictor, "trackers")
        ):
            self.model.predictor.trackers = []
        empty_cuda_cache()
